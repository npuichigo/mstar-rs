"""Concurrent-load client for a served mstar-rs stack: N simultaneous HTTP
requests with DISTINCT prompts (mixed text chat + TTS when the model serves
speech), verifying every response is correct, complete, and free of
cross-request contamination (answer i must match prompt i — a scheduler or
KV mix-up shows up as crossed answers or truncated/garbled audio).

Run against a launched stack (either serving mode):

    python -m mstar_rs.launch --speech [--decentralized] --tp 2 --port 8000 &
    python examples/client_concurrent.py --port 8000 --n 4 --speech
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
import urllib.request
import wave

CAPITALS = {
    "France": "Paris",
    "Japan": "Tokyo",
    "Italy": "Rome",
    "Spain": "Madrid",
    "Germany": "Berlin",
    "Canada": "Ottawa",
    "Egypt": "Cairo",
    "Peru": "Lima",
}


def post(base: str, path: str, body: dict, timeout: float = 600.0):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        ctype = r.headers.get("content-type", "")
        data = r.read()
    return ctype, data


def wav_seconds(wav_bytes: bytes) -> float:
    try:
        with wave.open(io.BytesIO(wav_bytes)) as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--n", type=int, default=4, help="concurrent requests")
    ap.add_argument("--speech", action="store_true",
                    help="interleave /v1/audio/speech requests (speech stack)")
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    for _ in range(600):
        try:
            urllib.request.urlopen(base + "/health", timeout=1)
            break
        except Exception:
            time.sleep(1)
    else:
        print("server never came up")
        return 1
    print("server up", flush=True)

    countries = list(CAPITALS)[: args.n]
    results: dict[int, tuple[bool, str]] = {}

    def chat_worker(i: int, country: str) -> None:
        t0 = time.time()
        try:
            _, data = post(base, "/v1/chat/completions", {
                "messages": [{"role": "user", "content":
                              f"What is the capital of {country}? Answer in one sentence."}],
            })
            text = json.loads(data)["choices"][0]["message"]["content"]
            ok = CAPITALS[country].lower() in text.lower()
            # cross-talk check: no OTHER capital may appear
            crossed = [c for k, c in CAPITALS.items()
                       if k != country and c.lower() in text.lower()]
            if crossed:
                ok = False
            results[i] = (ok, f"chat[{country}] {time.time()-t0:.1f}s -> {text!r}"
                              + (f" CROSSED:{crossed}" if crossed else ""))
        except Exception as e:  # noqa: BLE001
            results[i] = (False, f"chat[{country}] ERROR {e}")

    def tts_worker(i: int, text: str) -> None:
        t0 = time.time()
        try:
            ctype, data = post(base, "/v1/audio/speech",
                               {"input": text, "voice": "chelsie"})
            secs = wav_seconds(data)
            ok = data[:4] == b"RIFF" and secs > 0.3
            results[i] = (ok, f"tts[{text[:20]!r}] {time.time()-t0:.1f}s -> "
                              f"{len(data)}B {secs:.2f}s audio ({ctype})")
        except Exception as e:  # noqa: BLE001
            results[i] = (False, f"tts ERROR {e}")

    threads = []
    for i, country in enumerate(countries):
        if args.speech and i % 2 == 1:
            t = threading.Thread(target=tts_worker,
                                 args=(i, f"The capital of {country} is {CAPITALS[country]}."))
        else:
            t = threading.Thread(target=chat_worker, args=(i, country))
        threads.append(t)

    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=600)
    wall = time.time() - t0

    ok = len(results) == len(threads) and all(v[0] for v in results.values())
    for i in sorted(results):
        good, line = results[i]
        print(f"  [{'OK ' if good else 'BAD'}] {line}")
    print(f"\n{len(threads)} concurrent requests in {wall:.1f}s wall")
    print(f"\nCONCURRENT SERVING {'OK' if ok else 'FAILED'} "
          f"(distinct answers, no cross-request contamination)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
