"""Pure-CPU check of the pad-to-bucket index logic used by batched CUDA-graph
capture (the GPU capture/replay + batched attention are verified separately by
examples/verify_batched_capture.py on a GPU).

    python examples/verify_batched_graph_buckets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


def main() -> int:
    from mstar_rs.batched_graph import DEFAULT_CAPTURE_BATCH_SIZES as B
    from mstar_rs.batched_graph import padded_bucket

    assert padded_bucket(1, B) == 1
    assert padded_bucket(3, B) == 4
    assert padded_bucket(5, B) == 8
    assert padded_bucket(8, B) == 8
    assert padded_bucket(9, B) == 16
    assert padded_bucket(64, B) == 64
    for bad in (0, -1, 65, 1000):
        try:
            padded_bucket(bad, B)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass
    print(f"BATCHED-BUCKETS OK — pad-to-bucket over {B} (3->4, 5->8, 9->16); "
          f"rejects 0 and >max")
    return 0


if __name__ == "__main__":
    sys.exit(main())
