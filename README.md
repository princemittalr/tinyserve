# TinyServe

A from-scratch LLM inference engine for a single small GPU (NVIDIA T4, 16 GB, Turing).
Target model: Qwen3-0.6B. Written to be read and understood, not to compete with vLLM.

> Status: **M3 done.** A continuous-batching scheduler (with preemption by recompute) runs many requests through the paged KV cache at once, and its greedy output is token-identical to Hugging Face `generate()` on 120 prompts (fp32, CPU). On a T4 it is still 5x slower than HF `generate()` at 64 concurrent requests. Next: the Triton paged-attention kernel (M4). See [PROGRESS.md](PROGRESS.md).

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

- Correctness (2026-10-18, laptop CPU, fp32): 0 greedy mismatches vs HF over 120 prompts x 16 tokens,
  without a KV cache (M1), with the paged KV cache (M2), and with all 120 prompts batched
  through the continuous-batching engine (M3).
- Throughput (2026-10-18, Kaggle T4, fp16, Qwen3-0.6B, 128-token random prompts, 128 output
  tokens each, same prompts for both):

  | Concurrency | TinyServe tok/s | HF `generate()` tok/s | TinyServe TTFT p50 / p99 | TinyServe TPOT p50 / p99 |
  |---|---|---|---|---|
  | 1 | 25.5 | 25.2 | 40 / 66 ms | 39.2 / 39.7 ms |
  | 8 | 70.6 | 193.3 | 134 / 147 ms | 113.0 / 113.8 ms |
  | 32 | 88.0 | 415.6 | 366 / 721 ms | 363.1 / 364.1 ms |
  | 64 | 90.4 | 451.3 | 701 / 2137 ms | 708.3 / 718.0 ms |

  TinyServe is slower than HF from 8 concurrent requests on. Its attention still runs one
  sequence at a time and copies each sequence's keys and values out of the cache (inferred
  from TPOT growing linearly with the batch; not yet profiled). HF uses static batches, so
  it reports no TTFT or TPOT. Raw data: `results/m3_bench_cuda.json`, `results/m3_bench_hf_cuda.json`.

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
- The block allocator and block tables have Hypothesis property tests: random allocate/free
  traffic must never hand out a block twice, must reject double frees, and must not leak blocks.
- Paged attention and the cached model are checked against the no-cache M1 path (logits and
  greedy tokens), including blocks scattered in random order and blocks reused with stale data.
- The continuous-batching scheduler has a Hypothesis test over random workloads (every request
  finishes with the right output, no leaked blocks, no livelock), and the engine's greedy output
  matches the single-sequence path with and without preemptions.
- `tests/test_devices.py` runs the model on PyTorch's `meta` device, so a tensor created on the
  wrong device fails on CPU instead of on the GPU.
- GPU tests are marked `@pytest.mark.gpu` and run on a Kaggle/Colab T4.

## Limitations

- Single GPU only. No tensor parallelism, no quantization, not for deployment.
- Only Qwen3 dense checkpoints are supported; other architectures are rejected at load time.
- Weights are loaded in fp16 (the T4 has no fast bf16), not the checkpoint's native bf16.
- Greedy decoding only; temperature and top-p sampling arrive with the API in M6.
- Attention runs one sequence at a time inside each batched step (the M4 kernel does all
  sequences in one launch); the linear layers and MLP are batched.
- Paged attention copies each sequence's keys and values out of the cache before the math.
  It is simple and testable, not fast; the M4 Triton kernel will read the blocks in place.

## Prior art

vLLM (PagedAttention, SOSP 2023), nano-vllm, Mini-SGLang.
