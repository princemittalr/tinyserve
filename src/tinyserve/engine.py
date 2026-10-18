"""The serving loop: the scheduler and the model, one step at a time.

    engine = Engine(model, cache)
    ids = [engine.add_request(p, max_new_tokens=64) for p in prompts]
    while engine.has_work():
        engine.step()          # requests may be added between steps (continuous batching)

Every step: the scheduler picks the sequences and reserves their cache slots, the model
runs all their pending tokens in one ``forward_batch``, and each sequence gets its next
token (greedy for now; M6 adds sampling). The engine also records when each token came
out, which is what the M3 benchmark turns into TTFT and TPOT.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from tinyserve.batch import build_batch
from tinyserve.block_allocator import BlockAllocator
from tinyserve.kv_cache import KVCache
from tinyserve.model import Qwen3
from tinyserve.scheduler import Scheduler
from tinyserve.sequence import Sequence


@dataclass
class RequestTimes:
    """Wall-clock times (seconds, from the engine's clock) of one request."""

    arrival: float
    tokens: list[float] = field(default_factory=list)  # when each output token came out

    @property
    def ttft(self) -> float:
        """Time to first token: queueing + prefill."""
        return self.tokens[0] - self.arrival

    @property
    def tpot(self) -> float | None:
        """Mean time per output token after the first; None with a single token."""
        if len(self.tokens) < 2:
            return None
        return (self.tokens[-1] - self.tokens[0]) / (len(self.tokens) - 1)


class Engine:
    def __init__(
        self,
        model: Qwen3,
        cache: KVCache,
        max_num_seqs: int = 64,
        max_batch_tokens: int = 2048,
        clock: Callable[[], float] = time.perf_counter,
        prefix_cache: bool = False,
    ) -> None:
        self.model = model
        self.cache = cache
        self.scheduler = Scheduler(
            BlockAllocator(cache.num_blocks),
            cache.block_size,
            max_num_seqs,
            max_batch_tokens,
            prefix_cache=prefix_cache,
        )
        self.clock = clock
        self.seqs: dict[int, Sequence] = {}
        self.times: dict[int, RequestTimes] = {}
        self.num_steps = 0
        self.kv_utilization: list[float] = []  # sampled once per step, after scheduling
        self._next_id = 0

    def add_request(
        self, prompt: list[int], max_new_tokens: int, eos_ids: frozenset[int] = frozenset()
    ) -> int:
        seq_id = self._next_id
        self._next_id += 1
        self.seqs[seq_id] = self.scheduler.add(seq_id, prompt, max_new_tokens, eos_ids)
        self.times[seq_id] = RequestTimes(arrival=self.clock())
        return seq_id

    def has_work(self) -> bool:
        return self.scheduler.has_work()

    def step(self) -> list[Sequence]:
        """Run one model step. Returns the sequences that finished in it."""
        s = self.scheduler
        seqs = s.schedule()
        if not seqs:
            # add() rejects requests that can never fit, so this would be a scheduler bug.
            raise RuntimeError("scheduler made no progress with requests still queued")
        self.kv_utilization.append(s.kv_utilization)
        # Each sequence's input is what schedule() just appended to its table: a whole
        # prompt, the part after a cached prefix (M5), or one decode token.
        items = [(q.tokens[-q.num_scheduled :], q.table) for q in seqs]
        batch = build_batch(items, device=self.cache.k.device)
        next_tokens = self.model.forward_batch(batch, self.cache).argmax(-1).tolist()
        now = self.clock()
        for q in seqs:
            self.times[q.seq_id].tokens.append(now)
        self.num_steps += 1
        return s.update(seqs, next_tokens)

    def generate(
        self,
        prompts: list[list[int]],
        max_new_tokens: int,
        eos_ids: frozenset[int] = frozenset(),
    ) -> list[list[int]]:
        """Run all prompts to completion together; outputs in the order of ``prompts``."""
        ids = [self.add_request(p, max_new_tokens, eos_ids) for p in prompts]
        while self.has_work():
            self.step()
        return [self.seqs[i].output for i in ids]
