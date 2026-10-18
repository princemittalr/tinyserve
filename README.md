# TinyServe

A from-scratch LLM inference engine for a single small GPU (NVIDIA T4, 16 GB, Turing).
Target model: Qwen3-0.6B. Written to be read and understood, not to compete with vLLM.

> Status: **M1 done.** Greedy output is token-identical to Hugging Face `generate()` on 120 prompts (fp32, CPU). Next: paged KV cache (M2). See [PROGRESS.md](PROGRESS.md).

## Planned features

1. Model forward pass loaded from safetensors (Hugging Face used only as a test reference)
2. Paged KV cache with a block allocator
3. Continuous-batching scheduler with preemption
4. Radix-tree prefix cache with LRU eviction
5. Custom Triton paged decode-attention kernel
6. Batched sampling and an OpenAI-compatible streaming endpoint

## Quickstart (development)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pre-commit install          # run ruff and mypy on every commit
ruff check . && ruff format --check . && mypy && pytest
```

## Benchmarks

Every number here comes from a logged run with date and hardware; see [PROGRESS.md](PROGRESS.md).

- Correctness (2026-10-18, laptop CPU, fp32): 0 greedy mismatches vs HF over 120 prompts x 16 tokens.
- No throughput numbers yet: M1 has no KV cache, so speed is not meaningful until M3.

## How it's tested

- CPU tests run in GitHub Actions on Python 3.11, 3.12 and 3.14 on every push; the job fails if
  any test fails. The same lint and type checks run locally as a pre-commit hook.
- The weight loader is tested on small synthetic checkpoints (missing, unexpected, misshapen
  and fp16-overflowing tensors). `tests/test_real_checkpoint.py` checks the real Qwen3-0.6B
  files when `TINYSERVE_MODEL_DIR` is set.
- `tests/test_hf_parity.py` builds a tiny random Qwen3 with Hugging Face transformers, saves it,
  loads it through TinyServe's own loader, and checks logits and greedy tokens match (runs in CI).
- `scripts/compare_hf.py` compares greedy tokens with HF `generate()` on the real Qwen3-0.6B over
  120 prompts (`benchmarks/prompts/m1_greedy.txt`).
- GPU tests are marked `@pytest.mark.gpu` and run on a Kaggle/Colab T4.

## Limitations

- Single GPU only. No tensor parallelism, no quantization, not for deployment.
- Only Qwen3 dense checkpoints are supported; other architectures are rejected at load time.
- Weights are loaded in fp16 (the T4 has no fast bf16), not the checkpoint's native bf16.
- No KV cache yet: generation re-runs the whole sequence every step, so it is slow.

## Prior art

vLLM (PagedAttention, SOSP 2023), nano-vllm, Mini-SGLang.
