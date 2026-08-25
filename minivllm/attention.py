from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dataclasses import dataclass
from kv_runtime import KVRuntimeContext
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

    def forward_prefill(
        self,
        x: Tensor,
        kv_contexts: list[KVRuntimeContext] | None = None,
    ):
        outputs = []

        for t in range(x.shape[1]):
            xt = x[:, t:t+1]

            out = self.forward_decode(
                xt, kv_contexts[t]
            )

            outputs.append(out)

        return torch.cat(
            outputs, dim=-1,
        )

    def forward_decode(
        self,
        x: Tensor,
        kv_context: KVRuntimeContext | None = None,
    ):
        B, T, C = x.shape

        assert B == 1
        assert T == 1

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q, k, v = (
            t.view(B, T, self.n_head, self.head_size).transpose(1, 2)
            for t in (q, k, v)
        )

        q = q[0, :, 0, :]
        k = k[0, :, 0, :]
        v = v[0, :, 0, :]


        kv_context.kv_pool.write_layer(
            layer_idx=kv_context.layer_idx,
            physical_block=kv_context.physical_block,
            block_offset=kv_context.block_offset,
            k_new=k,
            v_new=v,
        )

        attention_out = paged_attention_online(
            q=q,
            pool=kv_context.kv_pool,
            seq=kv_context.seq_state,
            layer_idx=kv_context.layer_idx,
            context_length=kv_context.context_length,
        )

        attention_out = attention_out.reshape(1, 1, self.n_embd)

        attention_out = self.project(attention_out)

        attention_out = self.resid_dropout(attention_out)

        return attention_out

    def forward(
        self,
        x: Tensor,
        kv_cache: KVCache | None = None,
    ):
        B, T, C = x.shape

        past_len = 0 if kv_cache is None else kv_cache.length

        if past_len + T > self.block_size:
            raise ValueError(
                f"sequence length {past_len + T} exceeds "
                f"block_size {self.block_size}"
            )

        if kv_cache is not None:
            end = kv_cache.length + T

            if end > kv_cache.k.size(2):
                raise ValueError(
                    f"KV cache capacity exceeded: "
                    f"end={end}, capacity={kv_cache.k.size(2)}"
                )

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q, k, v = (
            t.view(B, T, self.n_head, self.head_size).transpose(1, 2)
            for t in (q, k, v)
        )

        if kv_cache is not None:
            start = kv_cache.length
            end = start + T

            kv_cache.k[:, :, start:end, :].copy_(k)
            kv_cache.v[:, :, start:end, :].copy_(v)

            k_all = kv_cache.k[:, :, :end, :]
            v_all = kv_cache.v[:, :, :end, :]

            kv_cache.length = end
        else:
            k_all = k
            v_all = v

        wei = q @ k_all.transpose(-2, -1)

        wei = wei * (self.head_size ** (-0.5))

        if kv_cache is not None or T > 1:
            causal_mask = self.tril[
                past_len: past_len + T,
                :past_len + T,
            ]

            wei = wei.masked_fill(
                causal_mask == 0,
                float("-inf"),
            )

        wei = F.softmax(wei, dim=-1)

        wei = self.attn_dropout(wei)

        out = wei @ v_all

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.project(out)

        out = self.resid_dropout(out)

        return out