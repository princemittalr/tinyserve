"""Serve Qwen3-0.6B over an OpenAI-compatible HTTP API (M6).

Usage: python scripts/serve.py [--device cuda --attention triton --prefix-cache]
       python scripts/serve.py --device cuda --attention triton_grouped --cuda-graphs \\
           --prefill-attention batched        (the fastest M8 configuration)
   or: python -m tinyserve.server [...]       (what tinyserve.client starts)

    curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \\
      -d '{"messages": [{"role": "user", "content": "Hi"}], "stream": true}'

It binds to 127.0.0.1, so only this machine can reach it: there is no authentication.
Another host needs --host 0.0.0.0 AND --allow-remote, and then anyone who can reach the
port can use the GPU.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import uvicorn
from tokenizers import Tokenizer

from tinyserve.api import Limits, create_app, is_loopback
from tinyserve.config import ModelConfig
from tinyserve.cuda_graphs import DecodeGraphs
from tinyserve.engine import Engine
from tinyserve.kv_cache import KVCache
from tinyserve.model import DECODE_ATTENTIONS, PREFILL_ATTENTIONS, Qwen3
from tinyserve.serving import AsyncEngine
from tinyserve.weights import load_weights


class QwenTokenizer:
    """The HF tokenizers library behind the API's encode/decode interface."""

    def __init__(self, path: Path) -> None:
        self.tok = Tokenizer.from_file(str(path))

    def encode(self, text: str) -> list[int]:
        # Special tokens in the text (<|im_start|> from the chat template) map to their
        # ids; nothing else is added (Qwen3 adds no BOS).
        return list(self.tok.encode(text, add_special_tokens=False).ids)

    def decode(self, ids: list[int]) -> str:
        return str(self.tok.decode(ids, skip_special_tokens=True))


def eos_ids(model_dir: Path) -> frozenset[int]:
    path = model_dir / "generation_config.json"
    if not path.exists():
        return frozenset()
    eos = json.loads(path.read_text(encoding="utf-8")).get("eos_token_id", [])
    return frozenset(eos if isinstance(eos, list) else [eos])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/Qwen3-0.6B"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--allow-remote", action="store_true",
                        help="required to bind to anything but loopback (no auth!)")  # fmt: skip
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16"], default=None,
                        help="default: float16 on cuda, float32 on cpu")  # fmt: skip
    parser.add_argument("--attention", choices=DECODE_ATTENTIONS, default="torch")
    parser.add_argument("--prefill-attention", choices=PREFILL_ATTENTIONS, default="per_seq",
                        help="per sequence, or one call per layer (M8)")  # fmt: skip
    parser.add_argument("--cuda-graphs", action="store_true",
                        help="replay CUDA graphs for decode steps (M8; needs a triton "
                             "attention)")  # fmt: skip
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--kv-gib", type=float, default=2.0, help="KV cache size")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-batch-tokens", type=int, default=2048)
    parser.add_argument("--chunked-prefill", action="store_true",
                        help="prompts in chunks of --max-batch-tokens per step (M9)")  # fmt: skip
    parser.add_argument("--max-requests", type=int, default=256, help="in flight; then 429")
    parser.add_argument("--max-prompt-tokens", type=int, default=Limits.max_prompt_tokens)
    parser.add_argument("--max-tokens-cap", type=int, default=Limits.max_tokens_cap)
    parser.add_argument("--seed", type=int, default=0, help="for requests without a seed")
    parser.add_argument("--ignore-eos", action="store_true",
                        help="benchmarks only: every request runs to max_tokens")  # fmt: skip
    parser.add_argument("--trace-requests", action="store_true",
                        help="profiling: time requests and steps, serve /debug/trace")  # fmt: skip
    parser.add_argument("--switch-interval-ms", type=float, default=None,
                        help="profiling: how often Python offers the GIL to another "
                             "thread (default 5 ms)")  # fmt: skip
    args = parser.parse_args(argv)
    if args.cuda_graphs and args.attention == "torch":
        parser.error("--cuda-graphs needs --attention triton or triton_grouped")

    if not is_loopback(args.host) and not args.allow_remote:
        sys.exit(f"refusing to bind to {args.host}: the API has no authentication. "
                 "Pass --allow-remote if you really want that.")  # fmt: skip

    if args.switch_interval_ms is not None:
        sys.setswitchinterval(args.switch_interval_ms / 1e3)
    dtype = getattr(torch, args.dtype or ("float32" if args.device == "cpu" else "float16"))
    cfg = ModelConfig.from_file(args.model_dir / "config.json")
    weights = load_weights(args.model_dir, cfg, dtype, args.device)
    model = Qwen3(cfg, weights, args.attention, prefill_attention=args.prefill_attention)
    elem = torch.empty((), dtype=dtype).element_size()
    per_block = 2 * cfg.num_layers * args.block_size * cfg.num_kv_heads * cfg.head_dim * elem
    num_blocks = int(args.kv_gib * 2**30 // per_block)
    cache = KVCache(cfg, num_blocks, args.block_size, dtype, args.device)
    # Captured before serving starts, for batches up to --max-num-seqs.
    graphs = DecodeGraphs(model, cache, args.max_num_seqs) if args.cuda_graphs else None
    engine = Engine(model, cache, args.max_num_seqs, args.max_batch_tokens,
                    prefix_cache=args.prefix_cache, seed=args.seed,
                    trace_steps=args.trace_requests, decode_graphs=graphs,
                    chunked_prefill=args.chunked_prefill)  # fmt: skip
    limits = Limits(max_prompt_tokens=args.max_prompt_tokens, max_tokens_cap=args.max_tokens_cap)
    tok = QwenTokenizer(args.model_dir / "tokenizer.json")
    ae = AsyncEngine(engine, max_requests=args.max_requests, trace=args.trace_requests)
    eos = frozenset() if args.ignore_eos else eos_ids(args.model_dir)
    app = create_app(ae, tok, args.model_dir.name, cfg.vocab_size, eos, limits)
    graph_note = "off"
    if graphs is not None:
        graph_note = (
            f"buckets up to {graphs.max_batch}, captured in {graphs.capture_seconds:.1f} s"
            if graphs.graphs
            else "eager (no CUDA)"
        )
    print(f"{args.device} {dtype}, {args.attention} decode attention, "
          f"{args.prefill_attention} prefill attention, CUDA graphs {graph_note}, chunked "
          f"prefill {'on' if args.chunked_prefill else 'off'} ({args.max_batch_tokens} tokens "
          f"per step), prefix cache "
          f"{'on' if args.prefix_cache else 'off'}, KV cache {num_blocks} blocks x "
          f"{args.block_size} tokens ({cache.memory_bytes / 2**30:.2f} GiB)"
          f"{', EOS ignored' if args.ignore_eos else ''}"
          f"{', tracing' if args.trace_requests else ''}, "
          f"GIL switch interval {sys.getswitchinterval() * 1e3:g} ms; "
          f"serving {args.model_dir.name!r} on http://{args.host}:{args.port}",
          flush=True)  # fmt: skip
    ae.start()
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        ae.shutdown()


if __name__ == "__main__":
    main()
