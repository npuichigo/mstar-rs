//! mstar-py: the PyO3 boundary. Exposes the Rust runtime to Python as
//! `mstar_rs._core`.
//!
//! Only descriptors cross this boundary. The Python side keeps a
//! `uuid -> torch.Tensor` object store; tensor refs travel as
//! `(uuid, dims, dtype)` tuples to keep the FFI surface flat and cheap.

use std::collections::BTreeMap;
use std::os::raw::{c_int, c_void};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use std::collections::BTreeMap as StdBTreeMap;

use mstar_comm::ShmArena;
use mstar_core::{IncomingInput, TensorRef};
use mstar_runtime::{Event, KvCacheConfig, Runtime, RuntimeError};

fn to_py_err(e: RuntimeError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// (uuid, dims, dtype) <-> TensorRef
fn tensor_ref_from_py(obj: &Bound<'_, PyAny>) -> PyResult<TensorRef> {
    let (uuid, dims, dtype): (u64, Vec<i64>, String) = obj.extract()?;
    Ok(TensorRef::new(uuid, dims, dtype))
}

fn tensor_ref_to_py(py: Python<'_>, t: &TensorRef) -> PyResult<PyObject> {
    Ok((t.uuid, t.dims.clone(), t.dtype.clone()).into_pyobject(py)?.unbind().into())
}

fn tensors_to_py(py: Python<'_>, tensors: &[TensorRef]) -> PyResult<PyObject> {
    let list = PyList::empty(py);
    for t in tensors {
        list.append(tensor_ref_to_py(py, t)?)?;
    }
    Ok(list.unbind().into())
}

/// name -> [TensorRef] mapping from a Python dict.
fn named_tensors_from_py(obj: &Bound<'_, PyAny>) -> PyResult<BTreeMap<String, Vec<TensorRef>>> {
    let dict = obj.downcast::<PyDict>()?;
    let mut out = BTreeMap::new();
    for (k, v) in dict.iter() {
        let name: String = k.extract()?;
        let mut tensors = Vec::new();
        for item in v.try_iter()? {
            tensors.push(tensor_ref_from_py(&item?)?);
        }
        out.insert(name, tensors);
    }
    Ok(out)
}

#[pyclass(name = "Batch")]
struct PyBatch {
    #[pyo3(get)]
    batch_id: u64,
    #[pyo3(get)]
    node: String,
    #[pyo3(get)]
    walk: String,
    /// {request_id: {name: [(uuid, dims, dtype), ...]}}
    #[pyo3(get)]
    inputs: PyObject,
    /// For KV nodes: {request_id: {"label", "pages", "seq_pos", "append_len"}}
    #[pyo3(get)]
    kv: PyObject,
}

#[pyclass(name = "Runtime")]
struct PyRuntime {
    inner: Runtime,
}

#[pymethods]
impl PyRuntime {
    /// walks_json: either a bare walk-set `{"walk_name": <Section>, ...}`
    /// or a full model spec `{"walks": {...}, "partitions": [...],
    /// "connections": [...]}` (streaming models).
    #[new]
    fn new(walks_json: &str) -> PyResult<Self> {
        let inner = if walks_json.contains("\"walks\"") {
            Runtime::from_spec_json(walks_json)
        } else {
            Runtime::from_walks_json(walks_json)
        }
        .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    fn new_uuid(&mut self) -> u64 {
        self.inner.new_uuid()
    }

    fn add_request(&mut self) -> u64 {
        self.inner.add_request()
    }

    /// configs: [(label, num_pages, page_size)]; node_labels: {node: label}.
    fn configure_kv(
        &mut self,
        configs: Vec<(String, u32, u32)>,
        node_labels: StdBTreeMap<String, String>,
    ) {
        self.inner.configure_kv(
            configs
                .into_iter()
                .map(|(label, num_pages, page_size)| KvCacheConfig {
                    label,
                    num_pages,
                    page_size,
                })
                .collect(),
            node_labels,
        );
    }

    /// (pages, seq_pos) for a request's cache label.
    fn kv_state(&self, request_id: u64, label: &str) -> (Vec<u32>, u64) {
        let st = self.inner.kv_state(request_id, label);
        (st.pages, st.seq_pos)
    }

    /// inputs: [(node, name, [(uuid, dims, dtype), ...]), ...]
    /// kv_appends: {label: tokens appended per execution of that label's KV
    /// node in this walk} (omit or 0 for read-only).
    /// kv_scratch: {label: transient tokens past the committed sequence each
    /// execution also needs pages for} (never committed).
    #[pyo3(signature = (request_id, walk, inputs, kv_appends = None, kv_scratch = None))]
    fn start_walk(
        &mut self,
        request_id: u64,
        walk: &str,
        inputs: Vec<(String, String, Vec<(u64, Vec<i64>, String)>)>,
        kv_appends: Option<StdBTreeMap<String, u64>>,
        kv_scratch: Option<StdBTreeMap<String, u64>>,
    ) -> PyResult<()> {
        let seeded = inputs
            .into_iter()
            .map(|(node, name, tensors)| IncomingInput {
                node,
                name,
                tensors: tensors
                    .into_iter()
                    .map(|(u, d, t)| TensorRef::new(u, d, t))
                    .collect(),
            })
            .collect();
        self.inner
            .start_walk_with_kv(
                request_id,
                walk,
                seeded,
                kv_appends.unwrap_or_default(),
                kv_scratch.unwrap_or_default(),
            )
            .map_err(to_py_err)
    }

    fn signal_loop_finish(&mut self, request_id: u64, loop_name: &str) -> PyResult<()> {
        self.inner
            .signal_loop_finish(request_id, loop_name)
            .map_err(to_py_err)
    }

    /// Returns a Batch or None. max_batch_size=0 means unbounded.
    fn next_batch(&mut self, py: Python<'_>, max_batch_size: usize) -> PyResult<Option<PyBatch>> {
        let Some(batch) = self.inner.next_batch(max_batch_size).map_err(to_py_err)? else {
            return Ok(None);
        };
        let inputs = PyDict::new(py);
        for (rid, named) in &batch.inputs {
            let per_req = PyDict::new(py);
            for (name, tensors) in named {
                per_req.set_item(name, tensors_to_py(py, tensors)?)?;
            }
            inputs.set_item(rid, per_req)?;
        }
        let kv = PyDict::new(py);
        for (rid, view) in &batch.kv {
            let v = PyDict::new(py);
            v.set_item("label", &view.label)?;
            v.set_item("pages", view.pages.clone())?;
            v.set_item("seq_pos", view.seq_pos)?;
            v.set_item("append_len", view.append_len)?;
            v.set_item("scratch_len", view.scratch_len)?;
            kv.set_item(rid, v)?;
        }
        Ok(Some(PyBatch {
            batch_id: batch.batch_id,
            node: batch.node,
            walk: batch.walk,
            inputs: inputs.unbind().into(),
            kv: kv.unbind().into(),
        }))
    }

    /// outputs: {request_id: {name: [(uuid, dims, dtype), ...]}}.
    /// Returns a list of event dicts:
    ///   {"type": "emission", "request_id", "name", "modality", "tensors"}
    ///   {"type": "walk_done", "request_id", "walk", "fwd_index",
    ///    "persist": {name: tensors}}
    fn complete_batch(
        &mut self,
        py: Python<'_>,
        batch_id: u64,
        outputs: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let out_dict = outputs.downcast::<PyDict>()?;
        let mut parsed = BTreeMap::new();
        for (k, v) in out_dict.iter() {
            let rid: u64 = k.extract()?;
            parsed.insert(rid, named_tensors_from_py(&v)?);
        }
        let events = self
            .inner
            .complete_batch(batch_id, parsed)
            .map_err(to_py_err)?;

        let list = PyList::empty(py);
        for event in events {
            let d = PyDict::new(py);
            match event {
                Event::Emission {
                    request_id,
                    partition,
                    name,
                    modality,
                    tensors,
                } => {
                    d.set_item("type", "emission")?;
                    d.set_item("request_id", request_id)?;
                    d.set_item("partition", partition)?;
                    d.set_item("name", name)?;
                    d.set_item("modality", modality)?;
                    d.set_item("tensors", tensors_to_py(py, &tensors)?)?;
                }
                Event::WalkDone {
                    request_id,
                    partition,
                    walk,
                    fwd_index,
                    persist,
                    stream_done,
                } => {
                    d.set_item("type", "walk_done")?;
                    d.set_item("request_id", request_id)?;
                    d.set_item("partition", partition)?;
                    d.set_item("walk", walk)?;
                    d.set_item("fwd_index", fwd_index)?;
                    d.set_item("stream_done", stream_done)?;
                    let p = PyDict::new(py);
                    for (name, tensors) in persist {
                        p.set_item(name, tensors_to_py(py, &tensors)?)?;
                    }
                    d.set_item("persist", p)?;
                }
            }
            list.append(d)?;
        }
        Ok(list.unbind().into())
    }

    /// Mark a partition done (signals producer-done on its outgoing
    /// streams). Returns true when ALL partitions of the request are done.
    fn finish_partition(&mut self, request_id: u64, partition: &str) -> PyResult<bool> {
        self.inner
            .finish_partition(request_id, partition)
            .map_err(to_py_err)
    }

    fn finish_request(&mut self, request_id: u64) -> PyResult<()> {
        self.inner.finish_request(request_id).map_err(to_py_err)
    }

    fn idle(&self) -> bool {
        self.inner.idle()
    }
}

/// Shared-memory tensor arena for cross-process transport. Producer:
/// `create(name, size)` -> `reserve(nbytes)` -> `torch.frombuffer(
/// memoryview(arena)[off:off+n], dtype=..).copy_(cpu_tensor)`; send the
/// offset descriptor; `free(off)` on ACK. Consumer: `open(name)` and
/// `torch.frombuffer(memoryview(arena)[off:off+n], ..)` (then H2D).
#[pyclass(name = "ShmArena")]
struct PyShmArena {
    arena: ShmArena,
}

#[pymethods]
impl PyShmArena {
    #[staticmethod]
    fn create(name: &str, size: usize) -> PyResult<Self> {
        Ok(Self {
            arena: ShmArena::create(name, size).map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
    }

    #[staticmethod]
    fn open(name: &str) -> PyResult<Self> {
        Ok(Self {
            arena: ShmArena::open(name).map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        })
    }

    fn reserve(&self, nbytes: usize) -> PyResult<usize> {
        self.arena
            .reserve(nbytes)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    }

    fn free(&self, offset: usize) -> bool {
        self.arena.free(offset)
    }

    #[getter]
    fn size(&self) -> usize {
        self.arena.size()
    }

    #[getter]
    fn bytes_free(&self) -> usize {
        self.arena.bytes_free()
    }

    fn close(&mut self) {
        self.arena.close();
    }

    /// Whole arena as a writable memoryview -> zero-copy `torch.frombuffer`.
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyValueError::new_err("null buffer view"));
        }
        let borrow = slf.borrow();
        let ptr = borrow.arena.as_mut_ptr() as *mut c_void;
        let len = borrow.arena.size() as ffi::Py_ssize_t;
        let ret = ffi::PyBuffer_FillInfo(view, slf.as_ptr(), ptr, len, 0, flags);
        if ret != 0 {
            Err(PyErr::fetch(slf.py()))
        } else {
            Ok(())
        }
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRuntime>()?;
    m.add_class::<PyBatch>()?;
    m.add_class::<PyShmArena>()?;
    m.add("EMIT_TO_CLIENT", mstar_core::EMIT_TO_CLIENT)?;
    m.add("EMPTY_DESTINATION", mstar_core::EMPTY_DESTINATION)?;
    Ok(())
}
