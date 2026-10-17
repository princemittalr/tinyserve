# TinyServe progress

## Milestones

| Milestone | Status |
|---|---|
| M0 Repo scaffold | in progress (step 1 done) |
| M1 Forward pass, token-identical to HF | not started |
| M2 Paged KV cache and allocator | not started |
| M3 Continuous batching and benchmark harness | not started |
| M4 Triton paged-attention kernel | not started |
| M5 Radix prefix cache | not started |
| M6 Streaming API | not started |
| M7 Profiling and write-up | not started |

## Step log

- M0 step 1: pyproject, ruff, mypy (strict), pytest + Hypothesis, CI, README, PROGRESS.

## Design decisions

- `src/` layout: tests import the installed package, so packaging bugs show up early.
- mypy `strict` from day one: cheaper than retrofitting types later.
- No runtime dependencies yet: torch arrives in M1, so CI stays fast until it is needed.
- GPU tests use a `gpu` marker and auto-skip without CUDA: one suite for laptop, CI and Kaggle.

## Benchmark results

None yet.

## Open questions

- Pin a CPU-only torch wheel in CI (M1) to keep install time down?
