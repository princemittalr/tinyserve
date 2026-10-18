# TinyServe

A from-scratch LLM inference engine for a single small GPU (NVIDIA T4, 16 GB, Turing).
Target model: Qwen3-0.6B. Written to be read and understood, not to compete with vLLM.

> Status: **M7 done.** TinyServe serves Qwen3-0.6B over an OpenAI-compatible HTTP API (`/v1/completions`, `/v1/chat/completions`, streaming with Server-Sent Events) with a paged KV cache, continuous batching, a radix prefix cache and a Triton decode kernel. On a T4 it delivers 766 tok/s at 32 concurrent sequences (1.84x HF `generate()`, 69% of vLLM 0.32 on the same workload) and 1,206 tok/s at 64 (2.67x HF, 87% of vLLM). The 3x-HF target at 32 is not met; vLLM itself reaches 2.65x there. Profiling shows the decode step is launch-bound (the GPU is busy about half of each step). [docs/WRITEUP.md](docs/WRITEUP.md) has the roofline argument, where the time goes, and the comparison. See [PROGRESS.md](PROGRESS.md).

## Features

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

## Serving (OpenAI-compatible)

```bash
python scripts/serve.py --device cuda --attention triton --prefix-cache   # 127.0.0.1:8000
curl http://127.0.0.1:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hi"}], "stream": true, "temperature": 0}'
```

Any OpenAI client works with `base_url="http://127.0.0.1:8000/v1"`. Endpoints: `/v1/completions`,
`/v1/chat/completions` (both with `"stream": true`), `/v1/models`, `/health`. Requests are
limited in body size, prompt length, `max_tokens` and requests in flight (413 / 400 / 429).

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
  median TTFT got worse. Profiled in M7: hits arrive in waves of up to 25 prefills in one step
  (645-685 ms), and 77% of such a step is prefill attention over the 1,024 cached tokens, run one
  sequence at a time in PyTorch.
  fp32 greedy parity with the cache on 24 chat prompts sharing a system prompt: 0 mismatches /
  384 tokens. In fp32, outputs with the cache on and off are identical for 128 / 128 requests at
  C=32 (fp16: 124 / 128, rounding). Raw data: `results/m5_prefix_bench_cuda.json`,
  `results/m5_hf_compare_fp32_cuda.json`, `results/m5_prefix_bench_fp32_cuda.json`.

- HTTP API (2026-10-18, Kaggle T4, fp16, Triton kernel): the M3/M4 workload sent to
  `scripts/serve.py` as streaming `/v1/completions` requests (128-token prompts, 128 output tokens,
  closed loop, client on the same machine), next to the engine alone in the same session
  (`scripts/bench_api.py`, `scripts/benchmark.py`):

  | Concurrency | tok/s engine / HTTP | HTTP cost | TTFT p50 engine / HTTP | TTFT p99 engine / HTTP | TPOT p50 engine / HTTP |
  |---|---|---|---|---|---|
  | 1 | 30.7 / 30.0 | 2.3% | 41 / 44 ms | 102 / 104 ms | 32.4 / 33.2 ms |
  | 8 | 231.1 / 221.7 | 4.1% | 121 / 203 ms | 124 / 261 ms | 33.9 / 34.6 ms |
  | 32 | 766.4 / 723.8 | 5.6% | 315 / 498 ms | 586 / 5,607 ms | 39.5 / 40.5 ms |
  | 64 | 1,205.9 / 1,124.0 | 6.8% | 326 / 667 ms | 1,242 / 7,294 ms | 50.4 / 52.2 ms |

  Streaming adds about 1 ms per token. Time to first token is the weak spot under load. Profiled
  in M7 (`scripts/profile_api.py`), it is not the GIL: the event loop is never more than 9 ms
  late. The median gap is requests arriving mid-step and waiting for it to end (~100 ms at 32
  streams). The p99 is a few requests on random-token prompts whose whole output decodes to empty
  text, so their only chunk arrives with their last token (confirmed: 3 of 128 at 32 streams,
  1 chunk each, empty text). Measured to the first token instead, TTFT p99 is 731 ms at 32
  streams and 1,426 ms at 64. Greedy text through the API (fp32, kernel, prefix cache, 120 prompts at once, half
  streamed) is identical to HF `generate()`, and the chat template is identical to Qwen3's own.
  Raw data: `results/m6_api_bench_cuda.json`, `results/m6_engine_bench_cuda.json`,
  `results/m6_api_hf_compare_float32_cuda.json`.

- Where a decode step's time goes (2026-10-18, Kaggle T4, fp16, Triton kernel, M3 workload,
  `scripts/profile_engine.py`): the step is nearly flat in batch size, 33.4 / 34.4 / 37.8 / 43.4 ms
  at C = 1 / 8 / 32 / 64, while the roofline floor (reading 1.19 GB of weights plus the k/v at
  320 GB/s) is 3.8 / 4.3 / 5.9 / 8.1 ms. At C=32 the GPU runs kernels for 16.7 ms of a 34.2 ms
  forward pass (1,851 kernel launches per step), so about half the time it waits for Python to
  launch the next one; `build_batch` adds 3.2 ms. The GEMMs reach 176 GB/s (55% of the T4's
  spec), the decode kernel 89 GB/s (it loads each k/v twice). Raw data:
  `results/m7_engine_profile_cuda.json`, `results/m7_prefix_steps_cuda.json`.

- Comparison with HF `generate()` and vLLM (2026-10-18, Kaggle T4, fp16, the M3 workload: the
  same seeded 128-token random prompts, 128 greedy output tokens, EOS ignored; closed loop at
  C, HF in static batches of C; output tok/s; `scripts/bench_vllm.py`):

  | Concurrency | HF `generate()` | TinyServe engine | TinyServe over HTTP | vLLM 0.32 eager | vLLM 0.32 (CUDA graphs) | TinyServe / vLLM |
  |---|---|---|---|---|---|---|
  | 1 | 25.2 | 30.7 | 30.0 | not run | 142.4 | 22% |
  | 8 | 193.3 | 231.1 | 221.7 | not run | 682.0 | 34% |
  | 32 | 415.6 | 766.4 | 723.8 | 814.3 | 1,103.2 | 69% |
  | 64 | 451.3 | 1,205.9 | 1,124.0 | not run | 1,384.5 | 87% |

  vLLM without CUDA graphs is within session noise of TinyServe at C=32 (814 vs 766 tok/s;
  TinyServe's own C=32 result varied 716-766 over three sessions), and CUDA graphs make vLLM
  1.35x faster than its eager mode: the gap is mostly kernel-launch overhead. The runs were in
  different Kaggle sessions, vLLM on torch 2.13.0+cu130 and the rest on 2.11.0+cu128. Raw data:
  `results/m7_vllm_bench_cuda.json`, `results/m3_bench_hf_cuda.json`,
  `results/m6_engine_bench_cuda.json`, `results/m6_api_bench_cuda.json`.

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
- The serving loop runs real asyncio clients against the engine thread: streamed tokens equal
  greedy output, concurrent clients share steps, a client that disconnects (or is still queued)
  is aborted and every block comes back, and a crashed engine fails every open request.
  Streamed text must equal the full decode for random UTF-8 cut into random byte tokens.
- The HTTP API is tested end to end on the tiny model: greedy text through both endpoints,
  streamed and not, equals the engine's; every limit returns OpenAI's error shape with the right
  status; a client that disconnects mid-stream or before its answer frees every block.
- `tests/test_server_e2e.py` starts the real server process on a tiny checkpoint and checks,
  over a socket, that streamed and plain answers equal the offline engine's greedy text.
- The roofline model's byte counts are checked against the real checkpoint's 596,049,920
  parameters and against every tensor of the tiny test checkpoint; the engine's step trace must
  not change any output.
- GPU tests are marked `@pytest.mark.gpu` and run on a Kaggle/Colab T4.

## Limitations

- Single GPU only. No tensor parallelism, no quantization, not for deployment.
- Only Qwen3 dense checkpoints are supported; other architectures are rejected at load time.
- Weights are loaded in fp16 (the T4 has no fast bf16), not the checkpoint's native bf16.
- The prefix cache reuses whole blocks only (16 tokens by default), and never a prompt's last
  token; eviction drops whole leaves of the tree.
- The API has no authentication (it binds to 127.0.0.1), serves one model, and does not
  implement `n > 1`, `stop`, logprobs, `echo` or penalties (requests using them get a 400).
  Chat content must be plain text; the chat template turns Qwen3's thinking off.
- Sampling supports temperature and top-p only (no top-k, repetition penalties or logprobs).
  Requests without a seed are reproducible per run, not per request.
- The Triton kernel covers decode only. Prefill attention still runs one sequence at a time in
  PyTorch and copies that sequence's keys and values out of the cache first.
- The kernel runs one program per (sequence, query head), so the two query heads of a GQA group
  each load the same keys and values; no split-K over long contexts. Tuned only for the T4.
- Triton has no Windows wheels: on Windows the kernel tests skip and the PyTorch path is used.
- The decode step is still launch-bound (Python launches every small kernel); no CUDA graphs yet.
- Benchmarked on one synthetic workload (random-token prompts, fixed lengths, EOS ignored) on
  one GPU type. TTFT over HTTP is measured to the first text chunk, which a request whose output
  decodes to empty text gets only at its end.

## Prior art

vLLM (PagedAttention, SOSP 2023), nano-vllm, Mini-SGLang.
