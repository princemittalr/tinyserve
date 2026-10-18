"""Serve Qwen3-0.6B over an OpenAI-compatible HTTP API (M6); the code is in
src/tinyserve/server.py. Usage: python scripts/serve.py --help
"""

from tinyserve.server import main

if __name__ == "__main__":
    main()
