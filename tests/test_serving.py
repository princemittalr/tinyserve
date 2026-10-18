"""Spec for aborting requests and for the async serving loop (M6 step 2).

The serving tests run real asyncio clients against a real engine thread with the tiny
random Qwen3 on CPU. Each test runs its own event loop with ``asyncio.run``.
"""

import asyncio
import time
from collections.abc import Callable

import pytest
import torch

from tests.test_model import CFG, tiny_model
from tinyserve.engine import Engine
from tinyserve.kv_cache import KVCache
from tinyserve.model import Qwen3, greedy_generate
from tinyserve.sampling import SamplingParams
from tinyserve.sequence import SeqStatus
from tinyserve.serving import AsyncEngine, EngineDeadError, QueueFullError, StreamOutput

PROMPTS = [[1], [5, 9, 2], [3, 8, 11, 2, 7, 7, 1], [4, 4, 6, 1, 2], [9, 30, 12, 5, 6, 2, 8]]
NUM_BLOCKS = 64


def engine(
    num_blocks: int = NUM_BLOCKS, max_num_seqs: int = 8, prefix_cache: bool = False
) -> Engine:
    cache = KVCache(CFG, num_blocks, 4, dtype=torch.float32)
    return Engine(tiny_model(), cache, max_num_seqs=max_num_seqs, prefix_cache=prefix_cache)


def expected(prompt: list[int], n: int, eos: frozenset[int] = frozenset()) -> list[int]:
    return greedy_generate(tiny_model(), prompt, n, eos)


def wait_until(cond: Callable[[], bool], timeout: float = 10.0) -> None:
    """Poll a condition the engine thread will make true (from the test's thread)."""
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "timed out"
        time.sleep(0.005)


# --- abort in the scheduler and engine --------------------------------------------------


def test_abort_running_and_waiting_frees_everything() -> None:
    e = engine(max_num_seqs=1)
    a = e.add_request(PROMPTS[2], 10)
    b = e.add_request(PROMPTS[1], 10)
    e.step()
    e.step()
    assert e.seqs[a].status is SeqStatus.RUNNING and e.seqs[b].status is SeqStatus.WAITING
    e.abort(a)
    e.abort(b)
    assert not e.has_work()
    assert e.scheduler.allocator.num_free == NUM_BLOCKS
    e.abort(a)  # aborting twice (or after finishing) is a no-op
    e.release(a)
    e.release(b)
    assert not e.seqs and not e.times and not e.sampling


def test_others_are_unaffected_by_an_abort() -> None:
    e = engine()
    ids = [e.add_request(p, 10) for p in PROMPTS]
    for _ in range(3):
        e.step()
    e.abort(ids[1])
    while e.has_work():
        e.step()
    for i, p in zip(ids, PROMPTS, strict=True):
        if i != ids[1]:
            assert e.seqs[i].output == expected(p, 10)
    assert len(e.seqs[ids[1]].output) == 3


def test_aborted_request_is_cached_like_a_finished_one() -> None:
    e = engine(prefix_cache=True)
    prompt = list(range(1, 14))  # 13 tokens
    a = e.add_request(prompt, 10)
    for _ in range(4):
        e.step()  # the cache now holds 13 + 3 = 16 tokens: 4 full blocks
    assert e.scheduler.cache is not None
    assert e.scheduler.cache.num_cached_blocks == 3  # the prompt, cached after prefill
    retry = prompt + e.seqs[a].output[:3] + [5]
    e.abort(a)
    assert e.scheduler.cache.num_cached_blocks == 4  # its decoded block too
    b = e.add_request(retry, 4)  # a retry of the conversation reuses all of it
    while e.has_work():
        e.step()
    assert e.seqs[b].output == expected(retry, 4)
    assert e.scheduler.num_cached_tokens == 16  # the first request had nothing to reuse


def test_release_refuses_a_live_request() -> None:
    e = engine()
    a = e.add_request(PROMPTS[0], 5)
    with pytest.raises(ValueError):
        e.release(a)


# --- the async serving loop -------------------------------------------------------------


async def collect(ae: AsyncEngine, prompt: list[int], n: int, **kw: object) -> list[StreamOutput]:
    return [out async for out in ae.generate(prompt, n, **kw)]  # type: ignore[arg-type]


def serve(e: Engine, **kw: int) -> AsyncEngine:
    ae = AsyncEngine(e, **kw)
    ae.start()
    return ae


def test_one_request_streams_its_greedy_tokens() -> None:
    ae = serve(engine())
    outs = asyncio.run(collect(ae, PROMPTS[2], 10))
    ae.shutdown()
    assert [t for o in outs for t in o.token_ids] == expected(PROMPTS[2], 10)
    assert len(outs) == 10  # one output per engine step
    assert all(o.finish_reason is None for o in outs[:-1])
    assert outs[-1].finish_reason == "length"
    assert outs[-1].num_prompt_tokens == len(PROMPTS[2]) and outs[-1].num_output_tokens == 10


def test_eos_finishes_with_stop() -> None:
    eos = frozenset({expected(PROMPTS[1], 3)[2]})
    ae = serve(engine())
    outs = asyncio.run(collect(ae, PROMPTS[1], 10, eos_ids=eos))
    ae.shutdown()
    assert [t for o in outs for t in o.token_ids] == expected(PROMPTS[1], 10, eos)
    assert outs[-1].finish_reason == "stop"


def test_concurrent_clients_share_engine_steps() -> None:
    e = engine()
    ae = serve(e)

    async def main() -> list[list[StreamOutput]]:
        return await asyncio.gather(*(collect(ae, p, 12) for p in PROMPTS))

    results = asyncio.run(main())
    ae.shutdown()
    for p, outs in zip(PROMPTS, results, strict=True):
        assert [t for o in outs for t in o.token_ids] == expected(p, 12)
    # Batched: far fewer steps than the 60 one-at-a-time serving would take.
    assert e.num_steps < 30
    assert not e.seqs  # every finished request was released
    assert e.scheduler.allocator.num_free == NUM_BLOCKS


def test_seeded_sampling_matches_the_offline_engine() -> None:
    hot = SamplingParams(temperature=1.5, top_p=0.9, seed=3)
    offline = engine().generate([PROMPTS[3]], 12, sampling=hot)[0]
    ae = serve(engine())

    async def main() -> list[list[StreamOutput]]:
        noise = SamplingParams(temperature=1.5)
        others = [collect(ae, p, 12, sampling=noise) for p in PROMPTS]
        return await asyncio.gather(collect(ae, PROMPTS[3], 12, sampling=hot), *others)

    outs = asyncio.run(main())[0]
    ae.shutdown()
    assert [t for o in outs for t in o.token_ids] == offline


def test_client_that_stops_reading_is_aborted() -> None:
    async def main() -> None:
        stream = ae.generate(PROMPTS[2], 900)
        got = [await anext(stream), await anext(stream)]
        assert len(got) == 2
        await stream.aclose()  # what a disconnect does to the streaming response

    big = Engine(tiny_model(), KVCache(CFG, 256, 4, dtype=torch.float32))
    ae = serve(big)
    asyncio.run(main())
    wait_until(lambda: not big.seqs)
    assert big.num_steps < 500  # stopped early, not run to its 900 tokens
    assert big.scheduler.allocator.num_free == 256
    assert ae.num_requests == 0
    ae.shutdown()


def test_cancelled_task_is_aborted() -> None:
    e = Engine(tiny_model(), KVCache(CFG, 256, 4, dtype=torch.float32), max_num_seqs=1)
    ae = serve(e)

    async def main() -> None:
        seen: list[StreamOutput] = []

        async def consume(prompt: list[int]) -> None:
            async for out in ae.generate(prompt, 900):
                seen.append(out)

        first = asyncio.create_task(consume(PROMPTS[2]))
        second = asyncio.create_task(consume(PROMPTS[1]))  # waits behind the first
        while not seen:
            await asyncio.sleep(0.001)
        first.cancel()
        second.cancel()
        for t in (first, second):
            with pytest.raises(asyncio.CancelledError):
                await t

    asyncio.run(main())
    wait_until(lambda: not e.seqs)
    assert e.num_steps < 500  # neither request ran to its 900 tokens
    assert e.scheduler.allocator.num_free == 256
    assert not e.has_work()
    ae.shutdown()


def test_request_that_can_never_fit_is_rejected_alone() -> None:
    ae = serve(engine(num_blocks=4))  # 16 tokens of cache

    async def main() -> list[StreamOutput]:
        with pytest.raises(ValueError, match="KV cache"):
            await collect(ae, PROMPTS[2], 50)
        return await collect(ae, PROMPTS[0], 5)  # the engine is still fine

    outs = asyncio.run(main())
    ae.shutdown()
    assert [t for o in outs for t in o.token_ids] == expected(PROMPTS[0], 5)


def test_too_many_requests_in_flight() -> None:
    ae = serve(engine(), max_requests=2)

    async def main() -> None:
        a, b = ae.generate(PROMPTS[0], 50), ae.generate(PROMPTS[1], 50)
        await anext(a)
        await anext(b)
        with pytest.raises(QueueFullError):
            await collect(ae, PROMPTS[2], 5)
        await a.aclose()
        assert len(await collect(ae, PROMPTS[2], 5)) == 5  # a slot is free again
        await b.aclose()

    asyncio.run(main())
    ae.shutdown()


class Exploding(Qwen3):
    def forward_batch(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise RuntimeError("CUDA error: out of memory (simulated)")


def test_engine_crash_fails_every_request_and_later_ones() -> None:
    m = tiny_model()
    bad = Exploding(m.cfg, m.w)
    ae = serve(Engine(bad, KVCache(CFG, 64, 4, dtype=torch.float32)))

    async def main() -> None:
        results = await asyncio.gather(
            *(collect(ae, p, 5) for p in PROMPTS), return_exceptions=True
        )
        assert all(isinstance(r, EngineDeadError) for r in results)
        with pytest.raises(EngineDeadError):
            await collect(ae, PROMPTS[0], 5)

    asyncio.run(main())
    assert not ae.is_alive


def test_shutdown_fails_open_requests() -> None:
    e = Engine(tiny_model(), KVCache(CFG, 256, 4, dtype=torch.float32))
    ae = serve(e)

    async def main() -> None:
        stream = ae.generate(PROMPTS[0], 900)
        await anext(stream)
        await asyncio.to_thread(ae.shutdown)
        with pytest.raises(EngineDeadError):
            async for _ in stream:
                pass

    asyncio.run(main())
