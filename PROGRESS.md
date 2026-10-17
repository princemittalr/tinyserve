# TinyServe progress

## Milestones

| Milestone | Status |
|---|---|
| M0 Repo scaffold | in progress (step 2 done) |
| M1 Forward pass, token-identical to HF | not started |
| M2 Paged KV cache and allocator | not started |
| M3 Continuous batching and benchmark harness | not started |
| M4 Triton paged-attention kernel | not started |
| M5 Radix prefix cache | not started |
| M6 Streaming API | not started |
| M7 Profiling and write-up | not started |

## Step log

- M0 step 1: pyproject, ruff, mypy (strict), pytest + Hypothesis, CI, README, PROGRESS.
- M0 step 2: pre-commit hook, `.gitattributes`, CI on 3.14, first push to GitHub.

## Design decisions

- `src/` layout: tests import the installed package, so packaging bugs show up early.
- mypy `strict` from day one: cheaper than retrofitting types later.
- No runtime dependencies yet: torch arrives in M1, so CI stays fast until it is needed.
- GPU tests use a `gpu` marker and auto-skip without CUDA: one suite for laptop, CI and Kaggle.
- pre-commit hooks are `local`: they reuse the venv's ruff/mypy, so local and CI versions never drift.
- `.gitattributes` forces LF: development is on Windows, CI is Linux; avoids whole-file CRLF diffs.
- CI tests 3.11, 3.12 and 3.14: Kaggle/Colab run 3.11-3.12, the laptop runs 3.14.

## Environment

- Laptop: Windows, PowerShell, Python 3.14.8 (CPU only).
- GPU: Kaggle and Colab free T4.

## Benchmark results

None yet.

## Open questions

- Does the PyTorch release we pick in M1 ship Windows wheels for Python 3.14? Check before M1.
- Pin a CPU-only torch wheel in CI (M1) to keep install time down?
