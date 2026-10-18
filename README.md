# TinyServe

A from-scratch LLM inference engine for a single small GPU (NVIDIA T4, 16 GB, Turing).
Target model: Qwen3-0.6B. Written to be read and understood, not to compete with vLLM.

> Status: **M5 done.** A radix-tree prefix cache lets requests that share a prompt prefix reuse its keys and values instead of recomputing them. On a T4, with 4 shared 1,024-token prefixes, it cuts median time to first token from 232 to 42 ms for one user and the 99th percentile from 8.9 to 2.0 s at 32 concurrent users, and raises throughput 2.24x there (the median at 32 users got worse; see Benchmarks). M4's Triton kernel runs decode attention for the whole batch in one launch per layer (2.59x HF `generate()` throughput at 64 concurrent requests, 1.72x at 32; the 3x target at 32 is not met yet). Greedy output is token-identical to HF in fp32, on CPU and on the T4, with and without the prefix cache; in fp16 a few near-ties flip. Next: the streaming API (M6). See [PROGRESS.md](PROGRESS.md).

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
- Correctness on the T4 (2026-10-18, fp16, same 120 prompts, batched engine): 7 prompts diverge
  from HF fp16 with the Triton kernel and 6 without it (the same 6 plus one). Every divergence is a
  near-tie: HF's top two logits differ by at most 0.0156, one fp16 rounding step.
- Exactness of the kernel on the T4 (2026-10-18, fp32, same 120 prompts, batched engine with the
  Triton kernel): 0 greedy mismatches / 1,920 tokens; max prefill logit difference 9.7e-05.
- Throughput (2026-10-18, Kaggle T4, fp16, Qwen3-0.6B, 128-token random prompts, 128 output
  tokens each, same prompts for both):

  | Concurrency | M4 (Triton kernel) tok/s | M3 (PyTorch attention) tok/s | HF `generate()` tok/s | M4 / HF | M4 TTFT p50 / p99 | M4 TPOT p50 / p99 |
  |---|---|---|---|---|---|---|
  | 1 | 28.7 | 25.5 | 25.2 | 1.14x | 44 / 70 ms | 34.8 / 36.3 ms |
  | 8 | 216.1 | 70.6 | 193.3 | 1.12x | 134 / 142 ms | 36.4 / 36.8 ms |
  | 32 | 716.0 | 88.0 | 415.6 | 1.72x | 330 / 602 ms | 42.4 / 44.7 ms |
  | 64 | 1168.2 | 90.4 | 451.3 | 2.59x | 337 / 1294 ms | 52.5 / 54.5 ms |

  M3 ran attention one sequence at a time: profiled at 64 sequences, that was 94% of a decode
  step and ~30,600 kernel launches, with the GPU idle 82% of the time. The M4 kernel does all
  sequences in one launch per layer (decode step 740 -> 40 ms). What is left is mostly launch
  overhead of the small non-attention kernels (GPU busy 44% of a step). HF uses static batches,
  so it reports no TTFT or TPOT. Raw data: `results/m4_bench_cuda.json`,
  `results/m3_bench_cuda.json`, `results/m3_bench_hf_cuda.json`, `results/m4_profile*_cuda.json`.
- Prefix cache (2026-10-18, Kaggle T4, fp16, Triton kernel): 4 random 1,024-token prefixes, each
  request one of them plus its own 64-token suffix, 64 output tokens, closed loop, cache off vs on
  on the same prompts (`scripts/bench_prefix.py`):

  | Concurrency | tok/s off / on | TTFT p50 off / on | TTFT p99 off / on | TPOT p50 off / on | prompt tokens from cache |
  |---|---|---|---|---|---|
  | 1 | 27.6 / 29.5 | 232 / 42 ms | 234 / 236 ms | 33.1 / 32.9 ms | 71% |
  | 8 | 121.4 / 188.0 | 266 / 111 ms | 1,944 / 1,138 ms | 61.7 / 37.9 ms | 82% |
  | 32 | 149.4 / 335.1 | 346 / 704 ms | 8,928 / 1,994 ms | 207.3 / 81.7 ms | 91% |

  Only the first request of each prefix misses. p99 at C=1 is one of those misses. At C=32 the
  median TTFT got worse; the likely cause (not yet profiled) is that hits arrive in large
  synchronized waves whose prefills each attend over 1,088 cached tokens one sequence at a time.
  fp32 greedy parity with the cache on 24 chat prompts sharing a system prompt: 0 mismatches /
  384 tokens. Raw data: `results/m5_prefix_bench_cuda.json`, `results/m5_hf_compare_fp32_cuda.json`.

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
  With sharing (M5), every block's reference count must equal the number of owners holding it.
- The radix prefix cache has a Hypothesis test that plays the scheduler with colliding random
  prompts and random eviction: every match must return blocks computed for exactly the query's
  prefix, blocks a running sequence uses are never evicted, and no block leaks.
- With the prefix cache on, the scheduler and engine must give exactly the same outputs as without
  it (fake model, and M1 greedy on the tiny Qwen3, incl. tight caches with eviction and
  preemption and the Triton kernel path), and run fewer prompt tokens.
- Paged attention and the cached model are checked against the no-cache M1 path (logits and
  greedy tokens), including blocks scattered in random order and blocks reused with stale data.
- The continuous-batching scheduler has a Hypothesis test over random workloads (every request
  finishes with the right output, no leaked blocks, no livelock), and the engine's greedy output
  matches the single-sequence path with and without preemptions.
- `tests/test_devices.py` runs the model on PyTorch's `meta` device, so a tensor created on the
  wrong device fails on CPU instead of on the GPU.
- The Triton decode kernel is compared with the PyTorch reference (within 1e-5 in fp32, 1e-3 in
  fp16) across head groupings, block sizes, sequence lengths and scattered blocks; every cache slot
  no sequence owns holds NaN, so any out-of-range read fails. Without a GPU (CI) the kernel runs in
  Triton's interpreter; on the T4 the compiled kernel runs. The engine with the kernel matches the
  PyTorch path's greedy tokens, with and without preemptions.
- Sampling is checked statistically: over 100,000 draws, token frequencies must match
  softmax(logits / temperature) and the renormalised top-p nucleus, a Hypothesis test checks the
  top-p set is the smallest most likely one, and a seeded request must give the same tokens
  alone, in a mixed batch, and when it is preempted and recomputed.
- GPU tests are marked `@pytest.mark.gpu` and run on a Kaggle/Colab T4.

## Limitations

- Single GPU only. No tensor parallelism, no quantization, not for deployment.
- Only Qwen3 dense checkpoints are supported; other architectures are rejected at load time.
- Weights are loaded in fp16 (the T4 has no fast bf16), not the checkpoint's native bf16.
- The prefix cache reuses whole blocks only (16 tokens by default), and never a prompt's last
  token; eviction drops whole leaves of the tree.
- Sampling supports temperature and top-p only (no top-k, repetition penalties or logprobs).
  Requests without a seed are reproducible per run, not per request.
- The Triton kernel covers decode only. Prefill attention still runs one sequence at a time in
  PyTorch and copies that sequence's keys and values out of the cache first.
- The kernel runs one program per (sequence, query head), so the two query heads of a GQA group
  each load the same keys and values; no split-K over long contexts. Tuned only for the T4.
- Triton has no Windows wheels: on Windows the kernel tests skip and the PyTorch path is used.
- The decode step is still launch-bound (Python launches every small kernel); no CUDA graphs yet.

## Prior art

vLLM (PagedAttention, SOSP 2023), nano-vllm, Mini-SGLang.
