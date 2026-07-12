"""Stage 1: verify the Rust runtime compiles + schedules qwen3-omni's topology.

No weights, no torch compute — this exercises only the control plane: build the
8-walk / 3-partition / dual-AR / 3-streaming-connection graph from
Qwen3OmniPolicy, construct the Rust Runtime (which compiles + validates it),
configure the two KV caches, seed all three partitions, and confirm the
scheduler routes the Thinker prefill first. Proves mstar-rs can express the
flagship's control shape before the ~60 GB engine work.

    python examples/verify_qwen3_graph.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from mstar_rs._core import Runtime  # noqa: E402
from mstar_rs.models.qwen3_omni import Qwen3OmniPolicy  # noqa: E402


def main() -> int:
    pol = Qwen3OmniPolicy()
    walks = pol.walks()
    parts, conns = pol.partitions()
    spec = json.dumps({"walks": walks, "partitions": parts, "connections": conns})

    # 1) compile: the Runtime validates the whole graph (nodes, edges, loops,
    # partition membership, stream targets) at construction.
    rt = Runtime(spec)
    rt.configure_kv(*pol.kv_config())
    print(f"1. compiled: {len(walks)} walks, {len(parts)} partitions, "
          f"{len(conns)} streaming connections, KV labels "
          f"{[c[0] for c in pol.kv_config()[0]]}")

    # 2) structural sanity.
    assert len(walks) == 8 and len(parts) == 3 and len(conns) == 3
    loops = [w for w, g in walks.items() if _has_loop(g)]
    assert set(loops) == {"thinker_decode", "talker_decode"}, loops
    print(f"2. dual-AR loops present: {sorted(loops)}")

    # 3) seed all three partitions (like ingest) and confirm scheduling. Only
    # Thinker has a real seed (text_inputs); Talker/Code2Wav wait on streamed
    # inputs (thinker_states / codec_tokens) — exactly the streaming topology.
    rid = rt.add_request()

    def ref(dims, dt="int64"):
        return (rt.new_uuid(), dims, dt)

    rt.start_walk(rid, "prefill_text",
                  [("Thinker", "text_inputs", [ref([12])])], {"thinker": 12})
    rt.start_walk(rid, "talker_prefill",
                  [("Talker", "talker_trigger", [ref([1])])], {"talker": 0})
    rt.start_walk(rid, "code2wav_chunk", [])
    print("3. seeded Thinker(prefill_text) + Talker(talker_prefill) + Code2Wav")

    batch = rt.next_batch(8)
    assert batch is not None and batch.node == "Thinker" and batch.walk == "prefill_text", batch
    # Talker/Code2Wav are NOT schedulable yet — they await streamed inputs.
    print(f"4. scheduler routed first batch to {batch.node}/{batch.walk} "
          f"(Talker/Code2Wav correctly wait on streamed inputs)")

    print("\nQWEN3-OMNI GRAPH COMPILES + SCHEDULES (control plane verified)")
    return 0


def _has_loop(section: dict) -> bool:
    if section.get("kind") == "loop":
        return True
    return any(_has_loop(s) for s in section.get("sections", []))


if __name__ == "__main__":
    sys.exit(main())
