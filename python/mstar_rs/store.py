"""The data plane's object store: uuid -> torch.Tensor.

The Rust control plane only ever sees ``(uuid, dims, dtype)`` descriptor
tuples; this store is where the actual tensors live. Mirrors the role of
mstar's ``TensorStore`` in ``communication/tensors.py`` for the in-process
case (no transport, no refcounts yet — tensors are freed per request when
the driver finishes it).
"""

from __future__ import annotations

import torch

TensorRefTuple = tuple[int, list[int], str]


class TensorStore:
    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self._tensors: dict[int, torch.Tensor] = {}
        self._request_uuids: dict[int, list[int]] = {}

    def put(self, tensor: torch.Tensor, request_id: int | None = None) -> TensorRefTuple:
        uuid = self._runtime.new_uuid()
        self._tensors[uuid] = tensor
        if request_id is not None:
            self._request_uuids.setdefault(request_id, []).append(uuid)
        dtype = str(tensor.dtype).removeprefix("torch.")
        return (uuid, list(tensor.shape), dtype)

    def get(self, ref: TensorRefTuple) -> torch.Tensor:
        return self._tensors[ref[0]]

    def get_all(self, refs: list[TensorRefTuple]) -> list[torch.Tensor]:
        return [self.get(r) for r in refs]

    def free_request(self, request_id: int) -> None:
        for uuid in self._request_uuids.pop(request_id, []):
            self._tensors.pop(uuid, None)

    def __len__(self) -> int:
        return len(self._tensors)
