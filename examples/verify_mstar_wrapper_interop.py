"""Interop proof for the transport wrapper: mstar's pyzmq
``ZMQCommunicator`` and the Rust-backed ``RustZMQCommunicator`` (the draft
wrapper in mstar's tree) exchanging PICKLED messages over the SAME endpoints —
so migration can proceed one process at a time, wrapped and unwrapped entities
coexisting.

Checks:
  1. pyzmq -> Rust: an object sent with ``send_pyobj`` decodes on the wrapper
  2. Rust -> pyzmq: the wrapper's pickle frame decodes with ``recv_pyobj``
  3. eventfd wakeup: an ``EventWakeup`` fire cuts ``wait_for_work`` short
  4. buffered readiness: ``poll_for_messages`` consuming a frame does not
     drop or reorder it for ``get_all_new_messages``

Run with an environment that has BOTH mstar and pyzmq (e.g. mstar's venv):

    python examples/verify_mstar_wrapper_interop.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO.parent / "mstar"))

from mstar.communication.communicator import ZMQCommunicator  # noqa: E402
from mstar.communication.event import EventWakeup  # noqa: E402
from mstar.communication.rust_communicator import RustZMQCommunicator  # noqa: E402


def wait_msgs(comm, n=1, timeout=5.0):
    out = []
    deadline = time.time() + timeout
    while len(out) < n and time.time() < deadline:
        out.extend(comm.get_all_new_messages())
        time.sleep(0.005)
    return out


def main() -> int:
    prefix = tempfile.mkdtemp(prefix="mstar_wrap_")
    orig = ZMQCommunicator("orig", push_ids=["rust"], ipc_socket_path_prefix=prefix)
    rust = RustZMQCommunicator("rust", push_ids=["orig"], ipc_socket_path_prefix=prefix)

    ok = True

    # 1) pyzmq (send_pyobj / pickle) -> Rust wrapper
    payload = {"op": "execute", "rids": [1, 2, 3], "nested": {"f": 1.5}}
    orig.send("rust", payload)
    got = wait_msgs(rust)
    good = got == [payload]
    print(f"  pyzmq -> rust : {got!r} {'OK' if good else 'FAIL'}")
    ok &= good

    # 2) Rust wrapper -> pyzmq (recv_pyobj)
    reply = ("done", 42)
    rust.send("orig", reply)
    got = wait_msgs(orig)
    good = got == [reply]
    print(f"  rust -> pyzmq : {got!r} {'OK' if good else 'FAIL'}")
    ok &= good

    # 3) eventfd wakeup cuts wait_for_work short
    ev = EventWakeup()
    rust.register_event_for_poll(ev)
    threading.Thread(target=lambda: (time.sleep(0.05),
                                     os.eventfd_write(ev.fd, 1))).start()
    t0 = time.time()
    rust.wait_for_work(timeout_ms=2000)
    waited = time.time() - t0
    good = waited < 0.5
    print(f"  eventfd wake  : {waited*1000:.0f}ms (timeout 2000ms) "
          f"{'OK' if good else 'FAIL'}")
    ok &= good

    # 4) readiness poll must not drop/reorder the consumed frame
    orig.send("rust", "first")
    ready = False
    for _ in range(500):
        if rust.poll_for_messages(timeout_ms=10):
            ready = True
            break
    orig.send("rust", "second")
    time.sleep(0.1)
    got = rust.get_all_new_messages()
    good = ready and got == ["first", "second"]
    print(f"  buffered poll : ready={ready} msgs={got!r} {'OK' if good else 'FAIL'}")
    ok &= good

    print(f"\nMSTAR WRAPPER INTEROP {'OK' if ok else 'FAILED'} "
          f"(pyzmq <-> Rust transport on the same mesh, pickle wire, eventfd wake)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
