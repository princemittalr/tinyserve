"""The roofline: the fastest a decode step could possibly run on a given GPU (M7).

A kernel is limited either by arithmetic (FLOPs / peak FLOP/s) or by memory traffic
(bytes / peak bandwidth), whichever takes longer. Their ratio, FLOPs per byte, is the
arithmetic intensity; a GPU's "ridge point" is peak FLOP/s / peak bandwidth. Below the
ridge a kernel is memory-bound, above it compute-bound.

One decode step for B sequences, in the model's dtype (``e`` bytes per element):
    linear layers: every weight is read once and used for B rows: 2 * P * B FLOPs over
                   P * e bytes, an intensity of 2B / e (= B in fp16). The T4's fp16
                   ridge is 65e12 / 320e9 = 203 FLOP/byte, so they stay memory-bound up
                   to B of about 200: reading the weights is the floor.
    attention:     each sequence reads its own k/v (nothing is shared between rows), and
                   each k/v element is used by Hq / Hkv query heads at 2 FLOPs each:
                   for Qwen3-0.6B (16 / 8 heads) 4 FLOPs per element, 2 per byte in
                   fp16. Memory-bound at any batch size.
So the floor of a decode step is (weight bytes + k/v bytes) / bandwidth. Real kernels
reach maybe 70-85% of the spec bandwidth; the gap between this floor and a measured
step is time the GPU spends waiting (launches, Python) or running slower kernels.
"""

from dataclasses import dataclass

from tinyserve.config import ModelConfig


@dataclass(frozen=True)
class GPUSpec:
    name: str
    mem_gbps: float  # peak DRAM bandwidth, GB/s (10^9 bytes)
    fp16_tflops: float  # peak dense fp16 tensor-core throughput
    fp32_tflops: float

    def peak_flops(self, elem_bytes: int) -> float:
        return (self.fp16_tflops if elem_bytes == 2 else self.fp32_tflops) * 1e12

    def ridge(self, elem_bytes: int) -> float:
        """FLOPs per byte above which a kernel is compute-bound."""
        return self.peak_flops(elem_bytes) / (self.mem_gbps * 1e9)


# NVIDIA T4 datasheet: 320 GB/s GDDR6, 65 TFLOPS mixed fp16/fp32, 8.1 TFLOPS fp32.
T4 = GPUSpec("Tesla T4", mem_gbps=320.0, fp16_tflops=65.0, fp32_tflops=8.1)


def layer_matmul_params(cfg: ModelConfig) -> int:
    """Weights of one layer's linears: q, k, v, o projections and the 3 MLP matrices."""
    h = cfg.hidden_size
    attn = h * cfg.q_size + 2 * h * cfg.kv_size + cfg.q_size * h
    mlp = 3 * h * cfg.intermediate_size
    return attn + mlp


def weight_bytes(cfg: ModelConfig, elem_bytes: int) -> int:
    """Bytes of weights one forward pass must read: every layer's linears and norms, the
    final norm and the output head. Only a few embedding rows are read (counted per
    token in ``decode_step_cost``), but with tied embeddings the head IS that matrix."""
    norms = cfg.num_layers * (2 * cfg.hidden_size + 2 * cfg.head_dim) + cfg.hidden_size
    head = cfg.vocab_size * cfg.hidden_size
    return (cfg.num_layers * layer_matmul_params(cfg) + norms + head) * elem_bytes


def kv_bytes_per_token(cfg: ModelConfig, elem_bytes: int) -> int:
    """K and V of one token across all layers (what the cache holds per token)."""
    return 2 * cfg.num_layers * cfg.kv_size * elem_bytes


@dataclass(frozen=True)
class StepCost:
    flops: float
    weight_bytes: float
    kv_bytes: float  # k/v read by attention, each element once (the minimum)

    @property
    def bytes(self) -> float:
        return self.weight_bytes + self.kv_bytes

    @property
    def intensity(self) -> float:
        return self.flops / self.bytes

    def floor_ms(self, gpu: GPUSpec, elem_bytes: int) -> float:
        """Roofline time: the larger of the memory time and the compute time."""
        memory = self.bytes / (gpu.mem_gbps * 1e9)
        compute = self.flops / gpu.peak_flops(elem_bytes)
        return max(memory, compute) * 1e3

    def memory_bound(self, gpu: GPUSpec, elem_bytes: int) -> bool:
        return self.intensity < gpu.ridge(elem_bytes)


def decode_step_cost(
    cfg: ModelConfig, num_seqs: float, context_tokens: float, elem_bytes: int
) -> StepCost:
    """Cost of one all-decode step: ``num_seqs`` sequences bring one token each, and
    ``context_tokens`` is the sum of their context lengths (new token included), i.e.
    how many cached (key, value) pairs attention reads in each layer."""
    layer_flops = 2 * layer_matmul_params(cfg) * num_seqs
    # q.k and p.v: 2 FLOPs per element each, for every query head and every cached token.
    attn_flops = 4 * cfg.num_heads * cfg.head_dim * context_tokens
    head_flops = 2 * cfg.vocab_size * cfg.hidden_size * num_seqs
    flops = cfg.num_layers * (layer_flops + attn_flops) + head_flops
    embed_rows = num_seqs * cfg.hidden_size * elem_bytes
    return StepCost(
        flops=flops,
        weight_bytes=weight_bytes(cfg, elem_bytes) + embed_rows,
        kv_bytes=context_tokens * kv_bytes_per_token(cfg, elem_bytes),
    )
