"""The KV cache memory: one key tensor and one value tensor shared by every sequence.

Each layer's cache is a flat array of token slots, ``[num_slots, Hkv, D]`` with
``num_slots = num_blocks * block_size``. A ``BlockTable`` turns a token position into a
slot; this class only stores and returns the vectors at those slots. Slot s belongs to
block ``s // block_size``, so the same memory viewed as ``[num_blocks, block_size, Hkv, D]``
is the block layout the Triton kernel (M4) will read.
"""

import torch

from tinyserve.config import ModelConfig


class KVCache:
    def __init__(
        self,
        cfg: ModelConfig,
        num_blocks: int,
        block_size: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_slots = num_blocks * block_size
        shape = (cfg.num_layers, self.num_slots, cfg.num_kv_heads, cfg.head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)

    @property
    def memory_bytes(self) -> int:
        """Bytes held by the key and value tensors together."""
        return 2 * self.k.numel() * self.k.element_size()

    def write(self, layer: int, slots: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store new keys and values: ``k[i]`` and ``v[i]`` go to slot ``slots[i]``.

        Args:
            slots: ``[T]`` int64, from ``BlockTable.append``.
            k, v: ``[T, Hkv, D]``, already RoPE-rotated (k); cast to the cache dtype.
        """
        self.k[layer, slots] = k.to(self.k.dtype)
        self.v[layer, slots] = v.to(self.v.dtype)
