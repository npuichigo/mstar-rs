"""Verify the mstar-comm transport capabilities from Python — the exact shapes
mstar's worker needs to wrap this communicator:

  1. eventfd wakeup: register an eventfd alongside the PULL inbox; a write
     from another thread (mstar: a completed compute future) wakes
     `recv_or_wake` immediately — NOT on the poll timeout. Done wrong, this
     silently stalls the async worker, so latency is asserted.
  2. TCP endpoints: bind on an OS-assigned port, exchange with an explicitly
     registered peer (the multi-node scheme).
  3. Raw byte frames: what one side sends is exactly what the other receives —
     no transport framing, so pickle/msgpack blobs pass through untouched.

    python examples/verify_comm_wakeup_tcp.py
"""

from __future__ import annotations

import ctypes
import os
import pickle
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mstar_rs._core import ZmqCommunicator  # noqa: E402


def _eventfd() -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.eventfd(0, 0)
    assert fd >= 0, "eventfd failed"
    return fd


def check_wakeup() -> bool:
    d = tempfile.mkdtemp(prefix="mstar_comm_wake_")
    a = ZmqCommunicator("a", d)
    efd = _eventfd()
    a.register_wakeup_fd(efd)

    def fire() -> None:
        time.sleep(0.05)
        os.write(efd, (1).to_bytes(8, "little"))

    t = threading.Thread(target=fire)
    t.start()
    t0 = time.time()
    kind, payload = a.recv_or_wake(2000)   # 2 s timeout; wake must beat it
    waited = time.time() - t0
    t.join()
    ok = kind == "wake" and payload is None and waited < 0.5
    print(f"  wakeup: kind={kind!r} in {waited*1000:.0f}ms (timeout was 2000ms) "
          f"{'OK' if ok else 'FAIL'}")

    os.read(efd, 8)   # clear (level-triggered)
    kind2, _ = a.recv_or_wake(30)
    ok2 = kind2 == "timeout"
    print(f"  cleared fd -> {kind2!r} {'OK' if ok2 else 'FAIL'}")
    os.close(efd)
    return ok and ok2


def check_tcp() -> bool:
    a = ZmqCommunicator.bind_endpoint("a", "tcp://127.0.0.1:*")
    b = ZmqCommunicator.bind_endpoint("b", "tcp://127.0.0.1:*")
    a_ep, b_ep = a.last_endpoint(), b.last_endpoint()
    a.register_peer("b", b_ep)
    b.register_peer("a", a_ep)
    a.send("b", b"over tcp")
    for _ in range(500):
        m = b.try_recv()
        if m is not None:
            ok = m == b"over tcp"
            print(f"  tcp: {a_ep} -> {b_ep} delivered {'OK' if ok else 'FAIL'}")
            return ok
        time.sleep(0.004)
    print("  tcp: FAIL (never delivered)")
    return False


def check_raw_passthrough() -> bool:
    # A pickle blob must survive byte-identical — the encoding seam mstar
    # needs (their mesh starts on pickle, migrates to msgpack; the transport
    # must not care).
    d = tempfile.mkdtemp(prefix="mstar_comm_raw_")
    a = ZmqCommunicator("a", d)
    b = ZmqCommunicator("b", d)
    blob = pickle.dumps({"op": "execute", "ids": list(range(16)), "f": 1.5})
    a.send("b", blob)
    for _ in range(500):
        m = b.try_recv()
        if m is not None:
            ok = m == blob and pickle.loads(m)["op"] == "execute"
            print(f"  raw: {len(blob)}-byte pickle round-trips byte-identical "
                  f"{'OK' if ok else 'FAIL'}")
            return ok
        time.sleep(0.004)
    print("  raw: FAIL (never delivered)")
    return False


def main() -> int:
    print("[1/3] eventfd wakeup (async-worker wake pattern)")
    ok = check_wakeup()
    print("[2/3] tcp endpoints (multi-node scheme)")
    ok &= check_tcp()
    print("[3/3] raw byte frames (pickle/msgpack seam)")
    ok &= check_raw_passthrough()
    print(f"\nCOMM WAKEUP+TCP+SEAM {'OK' if ok else 'FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
