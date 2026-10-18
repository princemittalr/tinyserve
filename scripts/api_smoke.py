"""Start scripts/serve.py, talk to it over real HTTP, stop it (M6 demo and smoke test).

Usage: python scripts/api_smoke.py [serve.py arguments, e.g. --device cuda --attention triton]

Prints a streamed chat answer as it arrives, a plain completion, and two refused requests.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

URL = "http://127.0.0.1:8000"


def main() -> None:
    serve = Path(__file__).with_name("serve.py")
    server = subprocess.Popen([sys.executable, str(serve), *sys.argv[1:]])
    try:
        deadline = time.monotonic() + 300  # loading the weights takes a while
        while True:
            try:
                if httpx.get(f"{URL}/health").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if server.poll() is not None or time.monotonic() > deadline:
                sys.exit("server did not come up")
            time.sleep(0.5)

        chat = {
            "messages": [{"role": "user", "content": "Name three planets."}],
            "max_tokens": 40,
            "temperature": 0,
            "stream": True,
        }
        print("stream:", end=" ", flush=True)
        start, first = time.perf_counter(), None
        with httpx.stream("POST", f"{URL}/v1/chat/completions", json=chat, timeout=120) as r:
            for line in r.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                choice = json.loads(line[6:])["choices"][0]
                first = first or time.perf_counter()
                print(choice["delta"].get("content", ""), end="", flush=True)
                if choice["finish_reason"]:
                    print(f"\n  finish_reason={choice['finish_reason']}, first chunk after "
                          f"{(first - start) * 1e3:.0f} ms, all after "
                          f"{(time.perf_counter() - start) * 1e3:.0f} ms")  # fmt: skip

        body = {"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}
        r = httpx.post(f"{URL}/v1/completions", json=body, timeout=120)
        out = r.json()
        print(f"completion: {out['choices'][0]['text']!r} usage={out['usage']}")

        for bad in ({"prompt": "x", "max_tokens": 100_000}, {"prompt": "x", "n": 2}):
            r = httpx.post(f"{URL}/v1/completions", json=bad, timeout=120)
            print(f"refused {bad}: {r.status_code} {r.json()['error']['message']}")
    finally:
        server.terminate()
        server.wait()


if __name__ == "__main__":
    main()
