"""Batched sampling: greedy, temperature and top-p, one step for the whole batch (M6).

The model gives one row of logits per sequence. Each request brings its own
``SamplingParams``, so one batch can mix greedy and sampled requests:

    temperature == 0   greedy: argmax of the logits (exactly the M1-M5 path)
    temperature  > 0   sample from softmax(logits / temperature)
    top_p < 1          first keep only the smallest set of most likely tokens whose
                       probabilities add up to at least top_p ("nucleus", Holtzman 2019)

Sampling uses the exponential race instead of ``torch.multinomial``: draw E_i ~ Exp(1)
for every token and pick argmax(p_i / E_i). E_i / p_i is exponential with rate p_i, and
the smallest of independent exponentials is i with probability p_i / sum(p), so this
samples exactly from p, needs no renormalisation after top-p (dropped tokens have
p_i = 0 and never win), and is one elementwise op and one argmax for the whole batch.

Logprobs (M9): a request with ``logprobs=k`` also gets, for every token it samples, that
token's log-probability and the k most likely tokens with theirs (``token_logprobs``).
They come from the model's own distribution, log_softmax of the raw logits, before
temperature and top-p, which is what vLLM reports by default.

Randomness: a request with a ``seed`` draws its noise from its own generator, so its
output depends only on its prompt, its parameters and its seed, not on which other
requests happen to share its batches. Requests without a seed share the engine's
generator (reproducible for a whole run, not per request).
"""

import math
from dataclasses import dataclass

import torch

MAX_LOGPROBS = 20  # OpenAI's limit for top_logprobs


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0  # 0 = greedy, the default, so nothing changes unless asked
    top_p: float = 1.0
    seed: int | None = None
    logprobs: int | None = None  # None: none; k: the sampled token's and the top k (M9)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.temperature) and self.temperature >= 0):
            raise ValueError(f"temperature must be finite and >= 0, got {self.temperature}")
        if not (0 < self.top_p <= 1):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.logprobs is not None and not 0 <= self.logprobs <= MAX_LOGPROBS:
            raise ValueError(f"logprobs must be in [0, {MAX_LOGPROBS}], got {self.logprobs}")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0


def top_p_filter(probs: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """Zero every token outside each row's nucleus; ``probs`` is ``[B, V]``, ``top_p`` ``[B]``.

    A token is kept if the probability of all tokens ranked above it is still below
    top_p. The most likely token is always kept (nothing ranks above it), so a row never
    ends up empty. The result is not renormalised. Cost: one sort per row, O(V log V).
    """
    sorted_p, order = probs.sort(dim=-1, descending=True)
    mass_above = sorted_p.cumsum(dim=-1) - sorted_p
    drop_sorted = mass_above >= top_p.unsqueeze(-1)
    drop = drop_sorted.scatter(-1, order, drop_sorted)  # back to vocabulary order
    return probs.masked_fill(drop, 0.0)


def sample(
    logits: torch.Tensor,
    params: list[SamplingParams],
    generators: list[torch.Generator | None],
    shared: torch.Generator,
) -> torch.Tensor:
    """One next token per row of ``logits`` (``[B, V]`` float32); returns ``[B]`` int64.

    ``generators[b]`` is row b's own generator (a seeded request) or None (use ``shared``).
    """
    if len(params) != logits.shape[0] or len(generators) != logits.shape[0]:
        raise ValueError(f"{logits.shape[0]} rows, {len(params)} params, {len(generators)} gens")
    greedy_tokens = logits.argmax(dim=-1)
    if all(p.greedy for p in params):
        return greedy_tokens  # no softmax, no noise: bit-for-bit the old argmax path

    dev = logits.device
    greedy = torch.tensor([p.greedy for p in params], device=dev)
    # Greedy rows get temperature 1 only to keep the division finite; they are
    # overwritten by their argmax at the end.
    temps = torch.tensor([p.temperature or 1.0 for p in params], device=dev)
    probs = torch.softmax(logits / temps.unsqueeze(-1), dim=-1)
    if any(p.top_p < 1 for p in params):
        probs = top_p_filter(probs, torch.tensor([p.top_p for p in params], device=dev))

    noise = torch.empty_like(probs).exponential_(generator=shared)
    for b, g in enumerate(generators):
        if g is not None and not params[b].greedy:
            noise[b].exponential_(generator=g)  # this request's own stream
    # A draw of exactly 0 would make a dropped token's 0 / 0 = NaN, and argmax picks NaN.
    noise.clamp_(min=torch.finfo(noise.dtype).tiny)
    sampled = (probs / noise).argmax(dim=-1)
    return torch.where(greedy, greedy_tokens, sampled)


@dataclass(frozen=True)
class TokenLogprobs:
    """One sampled token's log-probability and the most likely alternatives (M9)."""

    logprob: float
    top: list[tuple[int, float]]  # (token id, log-probability), most likely first


def token_logprobs(
    logits: torch.Tensor, tokens: torch.Tensor, ks: list[int]
) -> list[TokenLogprobs]:
    """For each row: ``log_softmax(logits)`` at ``tokens[b]``, and the top ``ks[b]``.

    ``logits`` is ``[B, V]`` (the raw logits the tokens were sampled from), ``tokens``
    ``[B]``. One log_softmax and one topk for the whole batch, then one copy to the host.
    """
    lp = torch.log_softmax(logits.float(), dim=-1)
    chosen = lp.gather(-1, tokens[:, None])[:, 0].tolist()
    top_v, top_i = lp.topk(max(ks, default=0), dim=-1)
    vals, ids = top_v.tolist(), top_i.tolist()
    return [
        TokenLogprobs(chosen[b], list(zip(ids[b][:k], vals[b][:k], strict=True)))
        for b, k in enumerate(ks)
    ]
