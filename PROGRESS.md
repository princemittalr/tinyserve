# TinyServe progress

## Milestones

| Milestone | Status |
|---|---|
| M0 Repo scaffold | done (2026-10-17) |
| M1 Forward pass, token-identical to HF | done (2026-10-18): 0 mismatches / 1,920 tokens |
| M2 Paged KV cache and allocator | done (2026-10-18): property tests pass; 0 mismatches / 1,920 tokens with the cache |
| M3 Continuous batching and benchmark harness | done (2026-10-18): 0 mismatches / 1,920 tokens batched; T4 benchmark recorded (below HF throughput, see results) |
| M4 Triton paged-attention kernel | in progress |
| M5 Radix prefix cache | not started |
| M6 Streaming API | not started |
| M7 Profiling and write-up | not started |

## Step log

- M0 step 1: pyproject, ruff, mypy (strict), pytest + Hypothesis, CI, README, PROGRESS.
- M0 step 2: pre-commit hook, `.gitattributes`, CI on 3.14, first push to GitHub.
- M0 step 3: CI `fail-fast: false`. Verified CI goes red on a failing test (canary PR #1, closed).
- M1 step 1: `ModelConfig` parsed from config.json, unsupported features rejected. Verified on the real Qwen3-0.6B config.json.
- M1 step 2: torch added; strict safetensors loader, synthetic-checkpoint tests, `scripts/inspect_weights.py`. Real checkpoint (2026-10-17, laptop): 310 tensors, 596,049,920 parameters (440,467,456 without the embedding), all finite in fp16; largest |weight| 192.0.
- M1 step 3: spec tests and skeletons for `rms_norm`, `rope_cos_sin`, `apply_rope` in `layers.py`. I attempted it, got stuck, and used the mentor's solution; I can explain each line.
- M1 step 4: SwiGLU MLP (`swiglu_mlp`), 5 spec tests; the three blanks were answered by the mentor.
- M1 step 5: reference causal grouped-query attention (`attention.py`), 8 spec tests incl. a match against PyTorch SDPA. Blank answers came from the mentor.
- M1 step 6: `Qwen3` forward pass (28 layers, no KV cache) and `greedy_generate`; tiny-random-model tests; `scripts/generate.py` demo. First real run (2026-10-18, laptop CPU, fp32): "The capital of France is" -> " Paris. The capital of Italy is Rome. ...", 20 tokens in 6.8 s.
- M1 step 7: HF parity tests on a tiny random Qwen3 (run in CI); `scripts/compare_hf.py` + 120 prompts for the real-model acceptance run. Config now also reads the newer `rope_parameters` layout.
- M1 closed: acceptance run passed (see Benchmark results).
- M2 step 1: `BlockAllocator` (free stack + used set) with 6 blanks; 14 unit tests and a Hypothesis state machine (random allocate/free traffic; checks no double hand-out, double free rejected, no leaked blocks). The blank answers came from the mentor.
- M2 step 2: `BlockTable` (a sequence's blocks, token position -> KV-cache slot) with 5 blanks; 18 unit tests and a Hypothesis state machine (several sequences on one allocator; no two tokens share a slot, no leak). I got stuck and the mentor gave the full file.
- M2 step 3: `KVCache` (per-layer key/value tensors of token slots, written by the mentor) and `paged_gqa_attention` in `attention.py` (4 blanks; the answers were given by the mentor with the step). 13 tests compare it with the M1 reference: scattered blocks, decode, chunked prefill, a Hypothesis test over random block orders, and two sequences sharing one cache.
- M2 step 4: `Qwen3.forward_paged` (runs only the new tokens; writes their k/v to the cache, reads history through the block table) and `greedy_generate_paged`, written by the mentor (wiring, no new algorithm). 15 tests: logits and greedy tokens identical to the M1 no-cache path for block sizes 1/4/16, blocks returned after generation and after OutOfBlocksError, stale data in reused blocks has no effect. `generate.py` and `compare_hf.py` use the cache by default (`--no-cache` / `--engine m1` for the old path).
- M2 closed: allocator and block-table property tests pass (no double hand-out, double free rejected, no leaked blocks); greedy tokens with the paged cache match HF on all 120 prompts (see Benchmark results).
- M3 step 1: `Batch` / `build_batch` (`batch.py`) flatten many sequences' new tokens into one `[T]` batch, and `Qwen3.forward_batch` runs them in one pass, returning only each sequence's last-token logits. Written by the mentor (wiring, no new algorithm). 8 tests: a mixed prefill + decode batch, chunked prefill (Hypothesis) and 8 lockstep greedy steps all match M1's no-cache path; batch order does not matter.
- M3 step 2: `Sequence` state (`sequence.py`, written by the mentor) and the continuous-batching `Scheduler` (`scheduler.py`, 9 blanks; the mentor gave the answers and then the full file): FCFS admission under max_num_seqs, a token budget and free blocks; preemption by recompute when the cache is full. 15 tests: admission limits, no queue-jumping, newest-first eviction, a Hypothesis test over random workloads (every request finishes with the right output, no leaked blocks, no livelock), and the tiny model run through the scheduler matches M1 greedy with and without preemptions.
- M3 step 3: `Engine` (`engine.py`, written by the mentor; serving-loop plumbing): each step runs `schedule()`, one `forward_batch` and `update()`, and records when every output token came out (TTFT, TPOT) plus KV-cache utilization per step. 8 tests: outputs match M1 greedy with a roomy cache, one sequence at a time, and a tight cache that forces preemptions; requests can join mid-run; finished requests leave at once; EOS; timing bookkeeping. `compare_hf.py --engine batched` runs every prompt through the engine at once.
- M3 step 3 check (real model): all 120 prompts through the engine at once (max 32 in flight) give HF's greedy tokens exactly (see Benchmark results).
- M3 step 4: benchmark harness, written by the mentor (benchmark plumbing): `bench.py` (closed-loop driver at a fixed concurrency; output tok/s, TTFT and TPOT p50/p99, KV utilization mean/max; `percentile` checked against numpy by Hypothesis) and `scripts/benchmark.py` (synthetic random-token workload, concurrency 1/8/32/64, JSON report). 10 tests. No GPU numbers yet: the first T4 run is the next step.
- M3 step 5: `scripts/bench_hf.py`, the HF `generate()` baseline on the same seeded prompts (static batches of C, fp16 on the T4), and Kaggle notebook cells for the first T4 run. `synthetic_prompts` and `num_requests_for` moved into `bench.py` so both scripts run exactly the same inputs; requests per level default to 4 per concurrency slot (at least 16). Written by the mentor (benchmark plumbing). 2 new tests.
- M3 step 6: first T4 run (Kaggle). HF baseline ran; our benchmark crashed: `rope_cos_sin` built its frequency table on the CPU while the positions were on the GPU ("Expected all tensors to be on the same device"). Cause: `torch.arange` without `device=`; CPU-only tests could not see it. Fix (mentor): create it on `positions.device`. New `tests/test_devices.py` runs the forward paths on PyTorch's `meta` device, which refuses to mix with CPU tensors, so a missing `device=` now fails on the laptop and in CI; plus a `gpu` test that runs the engine on CUDA and compares with CPU greedy.
- M3 step 6 result (Kaggle T4, after the fix): 177 passed, 2 skipped, including the `gpu` engine test. First TinyServe T4 benchmark recorded (see Benchmark results).
- M3 closed: the harness reports tok/s at concurrency 1/8/32/64, TTFT and TPOT p50/p99 and KV utilization; batched greedy output matches HF on all 120 prompts. Throughput is far below HF at C >= 8 (0.20x at C=64); see the analysis below the results table. M4's kernel targets exactly that.
- M4 step 1: `scripts/profile_step.py` and `profiling.py` (written by the mentor; measurement plumbing) to confirm the bottleneck before writing the kernel. On a decode batch of C sequences with 192 cached tokens each, it times `build_batch`, `forward_batch`, and `forward_batch` with attention replaced by a stand-in that returns q (the difference is what attention costs as written), at C = 1/8/32/64; then one C=64 forward under torch.profiler (op calls, kernel launches, GPU busy fraction, top operators). 6 tests.
- M4 step 1 result (Kaggle T4, fp16, 192 cached tokens per sequence): attention is 32% / 74% / 91% / 94% of a forward pass at C = 1 / 8 / 32 / 64 (forward 41.2 / 113.1 / 377.3 / 713.6 ms; without attention 28.2 / 29.2 / 35.3 / 42.5 ms). One C=64 forward: 227,499 aten op calls, 30,603 kernel launches, GPU busy 18% of wall time. The M3 inference is confirmed: per-sequence attention dominates, and the step is launch-bound (the GPU mostly waits for Python), not compute-bound. `build_batch` costs 6.1 ms at C=64.
- M4 step 2: Triton paged decode-attention kernel (`triton_attention.py`): one launch per layer for the whole decode batch, one program per (sequence, query head), walking the sequence's cache blocks with an online softmax. The launcher and argument checks were written by the mentor; the kernel body has 10 blanks and the mentor gave the answers with the step. Reference: `paged_decode_attention_ref` (attention.py), M2's attention run per sequence. 18 tests (17 run on CPU in Triton's interpreter, 1 `gpu`): Qwen3 shapes across block boundaries, MHA / GQA / MQA groupings, block sizes 1-32, fp16, scores large enough to overflow exp() without the running max, sequences isolated from each other, Hypothesis over random shapes; unused cache slots are NaN so a read past a sequence's end fails. CI installs Triton for the interpreter.

## Design decisions

- `src/` layout: tests import the installed package, so packaging bugs show up early.
- mypy `strict` from day one: cheaper than retrofitting types later.
- GPU tests use a `gpu` marker and auto-skip without CUDA: one suite for laptop, CI and Kaggle.
- pre-commit hooks are `local`: they reuse the venv's ruff/mypy, so local and CI versions never drift.
- `.gitattributes` forces LF: development is on Windows, CI is Linux; avoids whole-file CRLF diffs.
- CI tests 3.11, 3.12 and 3.14: Kaggle/Colab run 3.11-3.12, the laptop runs 3.14.
- CI matrix `fail-fast: false`: one red version should not hide the others' results.
- `xfail_strict = true`: a test marked xfail that starts passing fails the run, so stale xfail markers can't hide fixed or changed behaviour.
- Unsupported config features (bias, rope_scaling, non-SiLU) are rejected, not ignored: fail at load, not mid-generation.
- `head_dim` is read from config, never derived: Qwen3-0.6B uses 16 heads x 128 = 2048, not hidden_size 1024.
- No `transformers` at runtime: config.json is plain JSON, read with the standard library. huggingface_hub is only the download client.
- Weights default to fp16: Turing tensor cores do fp16, not bf16. bf16->fp16 can overflow, so the loader checks for inf/nan.
- Strict loader (missing, unexpected or misshapen tensor = error): a silent mismatch would surface much later as wrong tokens.
- RMSNorm and RoPE do their math in fp32 and cast back: fp16 squares overflow past 255, and RoPE angles reach 40960 rad.
- RoPE uses the rotate-half layout (pair i with i + D/2): it is what Qwen3 was trained with; interleaved pairs give wrong tokens.
- RoPE takes explicit positions per token: in a continuous batch every sequence is at a different position.
- Linear layers use `F.linear(x, w)` on the checkpoint's [out, in] weights: no transposed copies, so memory stays at 1x.
- Reference attention is plain PyTorch (einsum + mask + softmax in fp32): the paged cache (M2) and Triton kernel (M4) are tested against it.
- GQA maps query head h to kv head h // (Hq/Hkv) (`repeat_interleave`, not `repeat`): that is how Qwen3 groups heads.
- M1 has no KV cache: each step re-runs the whole sequence (O(n^2)). Simplest correct baseline; M2 replaces it.
- Qwen3 applies q_norm/k_norm per head before RoPE; the output head reuses the embedding matrix (tied).
- M1 acceptance is measured in fp32 on CPU: it isolates algorithm bugs from fp16 kernel rounding, which is a separate question for the GPU milestones.
- `transformers` is a dev-only dependency used as the reference in tests; tinyserve never imports it.
- CI installs CPU-only torch: no GPU there, and the default Linux wheel pulls ~3 GB of CUDA libraries.
- Block allocator is plain Python with no tensors: bookkeeping is separate from the KV memory, so it can be property-tested in milliseconds.
- Free blocks are a list used as a stack: allocate and free are O(1) per block, and a just-freed block is reused first.
- Used blocks are tracked in a set: the double-free check is O(1) instead of scanning the free list.
- allocate() and free() are all-or-nothing: on an error nothing changes, so the M3 scheduler can catch OutOfBlocksError and preempt safely.
- Slot of position p = blocks[p // block_size] * block_size + p % block_size: the KV cache is addressed as one flat array of token slots, so writes in step 3 are a single scatter.
- A block table never holds an empty block (blocks = ceil(tokens / block_size)): blocks are only taken when a token needs one.
- KV cache stored per layer as flat token slots [num_slots, Hkv, D]: writing new tokens is one indexed assignment, and the same memory viewed as [num_blocks, block_size, Hkv, D] is the block layout the M4 kernel reads.
- Paged attention takes the query positions as the LAST n of the sequence: one function covers prefill (n = total), decode (n = 1) and chunked prefill.
- Paged attention gathers k/v into a contiguous copy before the math: slow, but obviously equal to the M1 reference. M4's kernel reads blocks in place instead.
- The KV cache stores fp16 by default (fp32 on CPU in tests): it is the largest memory user after the weights.
- The model has one layer stack and takes attention as a function: the no-cache path (M1) and the paged path share every other line, so they cannot drift apart.
- `greedy_generate_paged` releases the sequence's blocks in a `finally`: an error mid-generation cannot leak cache memory.
- Batched forward flattens new tokens with no padding (sequence after sequence, `[T]`): linear layers and the MLP run on all rows at once and no compute goes to padding; only attention splits by sequence.
- `forward_batch` computes the output head only for each sequence's last new token: the 151,936 x 1,024 head is the largest matmul, and the other prefill rows are never sampled.
- All new tokens' k/v are written to the cache in one scatter per layer before attention: a sequence only ever reads its own slots, so writing everyone's first is safe.
- Scheduler re-plans before every step (continuous batching, Orca): finished sequences leave and waiting ones join at once, instead of a static batch waiting for its slowest member.
- Running sequences get their next slot before any new sequence is admitted: work already in progress is never starved by new arrivals.
- Preemption evicts the newest running sequence: the oldest always keeps its blocks, so it always makes progress and preemption cannot livelock.
- Preemption by recompute (free the blocks, re-run prompt + output as one prefill), not swapping k/v to CPU: no extra memory or copy path; the cost is repeated prefill work.
- A preempted sequence goes to the front of the waiting queue: it arrived before everything behind it.
- Admission is strict FCFS and stops at the first request that doesn't fit: no skipping ahead, so long requests are not starved by short ones.
- Requests whose longest possible length exceeds the batch token budget or the whole cache are rejected at `add()`: after a preemption they could never be re-admitted.
- The engine records one timestamp per output token, not just per request: TTFT, TPOT and their percentiles all come from the same raw data.
- The engine's clock is injectable: timing tests use a fake counter instead of sleeping or asserting on real time.
- Benchmark is closed-loop at a fixed concurrency (a new request as soon as one finishes): concurrency is then a controlled variable, matching the M3 criteria (1/8/32/64).
- Benchmark requests ignore EOS and all produce exactly `--output-len` tokens: runs at different concurrencies (and against HF) do the same work.
- Benchmark prompts are random token ids from a fixed seed: the work per request is exact and reproducible; the text content does not change the cost of a forward pass.
- TPOT is averaged per request first, then p50/p99 are taken across requests: one number per request, so long requests don't dominate the percentiles.
- HF baseline is static batching (batches of C run to completion) with the same prompts and fixed output length: that is what `generate()` offers, and identical inputs make the throughput comparison fair. Only tok/s is compared; HF's per-token times are not observable.
- Requests per benchmark level scale with concurrency (4 per slot, at least 16): concurrency 64 runs several waves, and concurrency 1 does not take an hour.
- Device bugs are caught on CPU by running the model on the `meta` device: it has no data, but it refuses to mix with CPU tensors, so any tensor created without `device=` fails a laptop test instead of a GPU run.
- Attention's cost is measured by replacing it with a stand-in (returns q) and timing the difference: a direct ablation is easier to trust than adding up profiler rows, and it includes the Python launch overhead the profiler splits across many ops.
- The decode kernel runs one program per (sequence, query head) and loops over that sequence's blocks: no padding work for short sequences, and the block table is read in place, so there is no gather copy. Sharing each loaded k/v block between the 2 query heads of a GQA group is a later optimisation, to be measured.
- Online softmax (running max, sum and weighted sum, rescaled when the max grows): one pass over the blocks with O(D) state per program, and no overflow because exp() only ever sees score - max <= 0.
- The kernel takes block ids (`[B, max_blocks]` int32) and lengths, not per-token slots: one int per block instead of one per token, and slot = block * block_size + offset is computed in the kernel.
- Kernel tests fill every slot no sequence owns with NaN: any out-of-range read shows up as NaN instead of passing by luck on zeros.
- Without a GPU, kernel tests run in Triton's interpreter (`TRITON_INTERPRET=1`, set in conftest.py): CI checks the kernel's results on CPU. Triton has no Windows wheels, so on the laptop these tests skip.
- M2 starts without reference counts; the M5 prefix cache, where sequences share blocks, will add them.

## Environment

- Laptop: Windows, PowerShell, Python 3.14.8 (CPU only).
- GPU: Kaggle and Colab free T4.

## Benchmark results

| Date | Hardware | What | Result |
|---|---|---|---|
| 2026-10-18 | Windows AMD64 laptop CPU, torch 2.14.1+cpu, fp32 | M1 greedy parity vs HF `generate()`: 120 prompts x 16 tokens | 0 mismatches / 1,920 tokens; max prefill logit abs diff 1.23e-4; 1047.5 s total (no KV cache) |
| 2026-10-18 | same | `scripts/generate.py`, 5-token prompt, 20 new tokens | 6.8 s (no KV cache; correctness baseline, not a speed claim) |
| 2026-10-18 | same | M2 greedy parity vs HF `generate()` with the paged KV cache (block size 16): 120 prompts x 16 tokens | 0 mismatches / 1,920 tokens; max prefill logit abs diff 1.23e-4; 692.9 s total (includes HF's own generate time) |
| 2026-10-18 | same | `scripts/generate.py`, same prompt, 20 new tokens, paged KV cache | 3.2 s vs 6.8 s without the cache, identical token ids (single run; not a benchmark) |
| 2026-10-18 | same | M3 greedy parity vs HF `generate()`, all 120 prompts through the continuous-batching engine at once (max 32 in flight, block size 16): 120 prompts x 16 tokens | 0 mismatches / 1,920 tokens; max prefill logit abs diff 1.23e-4; engine run 45.2 s, 64 steps, 0 preemptions (single run; CPU, not a benchmark) |
| 2026-10-18 | Kaggle Tesla T4, torch 2.11.0+cu128, transformers 5.16.1, fp16, sdpa | HF `generate()` baseline, static batches of C, 128-token random prompts, 128 output tokens each (`scripts/bench_hf.py`) | C=1: 25.2 tok/s (16 requests, 81.4 s); C=8: 193.3 tok/s (32, 21.2 s); C=32: 415.6 tok/s (128, 39.4 s); C=64: 451.3 tok/s (256, 72.6 s) |
| 2026-10-18 | Kaggle Tesla T4, torch 2.11.0+cu128, fp16, block size 16, 1,024 blocks (1.75 GiB KV cache), max 2,048 batch tokens | TinyServe engine, same workload, closed loop at C (`scripts/benchmark.py`) | C=1: 25.5 tok/s, TTFT p50/p99 40/66 ms, TPOT p50/p99 39.2/39.7 ms, KV util mean/max 1%/2%; C=8: 70.6 tok/s, TTFT 134/147 ms, TPOT 113.0/113.8 ms, KV 10%/12%; C=32: 88.0 tok/s, TTFT 366/721 ms, TPOT 363.1/364.1 ms, KV 39%/50%; C=64: 90.4 tok/s, TTFT 701/2137 ms, TPOT 708.3/718.0 ms, KV 77%/100%; 0 preemptions at every level |
| 2026-10-18 | Kaggle Tesla T4, torch 2.11.0+cu128, fp16, block size 16 | M4 step 1 profile: one decode forward, every sequence with 192 cached tokens (`scripts/profile_step.py`) | forward / without attention: C=1 41.2 / 28.2 ms (attention 32%); C=8 113.1 / 29.2 ms (74%); C=32 377.3 / 35.3 ms (91%); C=64 713.6 / 42.5 ms (94%). C=64 under torch.profiler: 227,499 aten op calls, 30,603 kernel launches, GPU busy 241.8 ms of 1,353.6 ms wall (18%) |

Raw data: `results/m1_hf_compare.json`, `results/m2_hf_compare.json`, `results/m3_hf_compare.json`, `results/m3_bench_cuda.json`, `results/m3_bench_hf_cuda.json`, `results/m4_profile_cuda.json` (the HF file was transcribed from the notebook screenshot; its tok/s values equal tokens / seconds).

M3 T4 analysis (inferred from the numbers above, not yet profiled; M7 profiles):
- TinyServe / HF throughput: 1.01x at C=1, 0.37x at C=8, 0.21x at C=32, 0.20x at C=64. The target (>= 3x HF at C=32) is not met.
- Our TPOT grows almost linearly with the batch: 39, 113, 363, 708 ms at C=1, 8, 32, 64, i.e. about 11 ms per extra sequence per step. HF's batch time grows far more slowly. The likely cause is attention: `forward_batch` calls `paged_gqa_attention` once per sequence per layer (28 x C calls per step), and each call gathers and copies that sequence's k/v. The batched linear layers and MLP should cost little extra per sequence. This is what the M4 Triton kernel replaces with one launch per layer.
- TTFT p99 at C=64 (2.1 s) is queueing: with a 2,048-token step budget, only about 15 new 128-token prompts fit in one step, and each step already takes ~0.7 s.

## Open questions

- Resolved 2026-10-18: the block allocator (core algorithm) is scaffolded as plain-Python blanks for me to fill, like M1.
- Resolved 2026-10-18 (kernel tolerance): the Triton kernel must match the PyTorch reference within atol = rtol = 1e-5 in fp32 and 1e-3 in fp16; both accumulate in fp32 and round once.
- Greedy parity in fp16 on the T4 with the kernel in the engine: expect rare near-tie flips; measure and decide in M4 step 3.
- Profile one C=64 step (torch profiler) at the start of M4 to confirm that per-sequence attention dominates before writing the kernel. Resolved 2026-10-18 (M4 step 1): confirmed, see Benchmark results.

- Resolved 2026-10-17: torch 2.14.1 on PyPI has a cp314 win_amd64 wheel.
- Dev tools use `>=` ranges, so CI may get a newer ruff than the laptop. Pin exact versions if that ever breaks CI.
- Resolved 2026-10-17: tensor names and shapes verified, `test_real_checkpoint.py` passes on the real files.
- Resolved 2026-10-18: Kaggle's preinstalled torch 2.11.0+cu128 (CUDA 12.8) runs on the T4 (compute capability 7.5); install tinyserve with `--no-deps` so it doesn't replace that torch.
