"""M6 parity: greedy answers over HTTP vs Hugging Face generate(), and the chat template.

1. The chat endpoint's prompt (``api.qwen3_chat_prompt``) must equal HF's own Qwen3 chat
   template with ``enable_thinking=False``, for a few conversations.
2. Start scripts/serve.py (fp32 by default, with the kernel and the prefix cache as
   asked), send every prompt at once as token ids, temperature 0: half streamed, half
   plain, so both paths are checked and the requests are batched by the server. Then
   stop the server, run HF ``generate()`` on the same ids, and compare the text: HF's
   tokens decoded with the same tokenizer (special tokens skipped) must equal the API's
   text exactly.

Usage: python scripts/compare_api_hf.py --device cuda --attention triton --prefix-cache
"""

import argparse
import asyncio
import datetime as dt
import json
from pathlib import Path
from typing import Any

import httpx
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from tinyserve.api import ChatMessage, qwen3_chat_prompt
from tinyserve.client import running_server, stream_completion
from tinyserve.model import DECODE_ATTENTIONS, PREFILL_ATTENTIONS
from tinyserve.server import eos_ids

CONVERSATIONS: list[list[dict[str, str]]] = [
    [{"role": "user", "content": "What is the capital of France?"}],
    [
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "Name a prime number."},
    ],
    [
        {"role": "system", "content": "Answer in one word."},
        {"role": "user", "content": "Colour of the sky?"},
        {"role": "assistant", "content": "Blue."},
        {"role": "user", "content": "And grass?"},
    ],
    [{"role": "user", "content": "Écris « bonjour » 🙂\nwith a newline"}],
]


def check_chat_template(model_dir: Path) -> int:
    hf_tok: Any = AutoTokenizer.from_pretrained(model_dir)
    bad = 0
    for conv in CONVERSATIONS:
        ours = qwen3_chat_prompt([ChatMessage.model_validate(m) for m in conv])
        theirs = hf_tok.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ok = ours == theirs
        bad += not ok
        print(f"chat template {'OK  ' if ok else 'DIFF'} {len(conv)} messages")
        if not ok:
            print(f"  ours:   {ours!r}\n  theirs: {theirs!r}")
    return bad


async def ask_all(url: str, prompts: list[list[int]], max_tokens: int) -> list[str]:
    async with httpx.AsyncClient(timeout=None) as client:

        async def one(i: int, ids: list[int]) -> str:
            body = {"prompt": ids, "max_tokens": max_tokens, "temperature": 0}
            if i % 2:  # odd prompts: plain JSON answer
                r = await client.post(f"{url}/v1/completions", json=body)
                r.raise_for_status()
                return str(r.json()["choices"][0]["text"])
            return (await stream_completion(client, url, body)).text

        return list(await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/Qwen3-0.6B"))
    parser.add_argument("--prompts", type=Path, default=Path("benchmarks/prompts/m1_greedy.txt"))
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--attention", choices=DECODE_ATTENTIONS, default="torch")
    parser.add_argument("--prefill-attention", choices=PREFILL_ATTENTIONS, default="per_seq")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    dev = args.device.split(":")[0]
    milestone = "m8" if args.cuda_graphs else "m6"
    out = args.out or Path(f"results/{milestone}_api_hf_compare_{args.dtype}_{dev}.json")

    template_diffs = check_chat_template(args.model_dir)

    tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    lines = args.prompts.read_text(encoding="utf-8").splitlines()
    texts = [ln.replace("\\n", "\n") for ln in lines if ln.strip()]
    prompts = [tok.encode(t).ids for t in texts]
    serve_args = [
        "--model-dir", str(args.model_dir), "--device", args.device, "--dtype", args.dtype,
        "--attention", args.attention, "--kv-gib", "2", "--max-num-seqs", "64",
        "--prefill-attention", args.prefill_attention,
    ] + (["--prefix-cache"] if args.prefix_cache else []) + (
        ["--cuda-graphs"] if args.cuda_graphs else [])  # fmt: skip
    with running_server(serve_args, args.port) as url:
        api = asyncio.run(ask_all(url, prompts, args.max_new_tokens))

    # The server process has exited, so its GPU memory is free for HF.
    dtype = getattr(torch, args.dtype)
    hf: Any = AutoModelForCausalLM.from_pretrained(args.model_dir, dtype=dtype)
    hf.to(args.device).eval()
    eos = sorted(eos_ids(args.model_dir))
    rows = []
    for n, (text, ids, got) in enumerate(zip(texts, prompts, api, strict=True), 1):
        x = torch.tensor([ids], device=args.device)
        with torch.inference_mode():
            full = hf.generate(x, max_new_tokens=args.max_new_tokens, do_sample=False,
                               eos_token_id=eos or None,
                               pad_token_id=eos[0] if eos else None)  # fmt: skip
        want = tok.decode(full[0, len(ids) :].tolist(), skip_special_tokens=True)
        rows.append({"prompt": text, "match": got == want, "api": got, "hf": want,
                     "path": "plain" if (n - 1) % 2 else "stream"})  # fmt: skip
        print(f"[{n}/{len(texts)}] {'OK  ' if got == want else 'DIFF'} {text[:50]!r}")

    mismatches = [r for r in rows if not r["match"]]
    summary = {
        "date": dt.date.today().isoformat(),
        "hardware": torch.cuda.get_device_name() if dev == "cuda" else "CPU",
        "torch": torch.__version__,
        "dtype": args.dtype,
        "attention": args.attention,
        "prefill_attention": args.prefill_attention,
        "cuda_graphs": args.cuda_graphs,
        "prefix_cache": args.prefix_cache,
        "transport": "HTTP /v1/completions, token-id prompts, half SSE half plain, all at once",
        "prompts": len(rows),
        "max_new_tokens": args.max_new_tokens,
        "text_mismatches": len(mismatches),
        "chat_template_conversations": len(CONVERSATIONS),
        "chat_template_diffs": template_diffs,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    for r in mismatches:
        print(f"MISMATCH ({r['path']}) {r['prompt']!r}:\n  api: {r['api']!r}\n  hf:  {r['hf']!r}")


if __name__ == "__main__":
    main()
