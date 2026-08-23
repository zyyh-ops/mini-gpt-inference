from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dataclasses import dataclass
from minivllm.kv_runtime import KVRuntimeContext
from paged_attention_v1_reference import paged_attention_online


@dataclass
class KVCache:
    k: Tensor
    v: Tensor
    length: int


def create_kv_cache(
    batch_size,
    n_head,
    block_size,
    head_size,
    device,
    dtype,
):
    shape = (
        batch_size,
        n_head,
        block_size,
        head_size,
    )

    return KVCache(
        k = torch.empty(
            shape,
            device=device,
            dtype=dtype,
        ),
        v = torch.empty(
            shape,
            device=device,
            dtype=dtype,
        ),
        length=0,
    )


class KVCacheCausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        context: KVRuntimeContext,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        if n_embd % n_head != 0:
            raise ValueError(
                f"n_embd {n_embd} is not divisible by n_head {n_head}"
            )
        
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.block_size = block_size
        self.context = context
        self.dropout = dropout

        self.qkv = nn.Linear(
            n_embd,
            3 * n_embd,
            bias=False,
        )

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.attn_dropout = nn.Dropout(dropout)

        self.project = nn.Linear(
            n_embd,
            n_embd,
        )

        self.resid_dropout = nn.Dropout(dropout)

    def forward_decode(
        self,
        x: Tensor,
        kv_context: KVRuntimeContext | None = None,
    ):
        B, T, C = x.shape

        past_len = 0 if kv_context is None else kv_context.seq_state.length

        if past_len + T > self.block_size:
            raise ValueError(
                f"sequence length {past_len + T} exceeds "
                f"block_size {self.block_size}"
            )

        if kv_context is not None:
            end = kv_context.seq_state.length + T

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q, k, v = (
            t.view(B, T, self.n_head, self.head_size).transpose(1, 2)
            for t in (q, k, v)
        )

        slot = self.context.block_manager.append_slot(
            self.context.seq_state
        )

        physical_block = slot // self.context.block_manager.block_size
        offset = slot % self.context.block_manager.block_size

        self.context.kv_pool.write(
            physical_block=physical_block,
            block_offset=offset,
            k_new=k,
            v_new=v,
        )

        attention_out = paged_attention_online(
            q=q,
            pool=self.context.kv_pool,
            seq=self.context.seq_state,
            layer_idx=self.context.layer_idx,
        )

        return attention_out