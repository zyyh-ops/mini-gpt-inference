from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


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

    def forward(
        self,
        x: Tensor,
        past_kv: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ):
        B, T, C = x.shape

        past_len = 0 if past_kv is None else past_kv[0].size(2)

        if past_len + T > self.block_size:
            raise ValueError(
                f"sequence length {past_len + T} exceeds "
                f"block_size {self.block_size}"
            )

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q, k, v = (
            t.view(B, T, self.n_head, self.head_size).transpose(1, 2)
            for t in (q, k, v)
        )
        
        if past_kv is not None:
            past_k, past_v = past_kv

            k = torch.cat(
                [past_k, k],
                dim=2,
            )

            v = torch.cat(
                [past_v, v],
                dim=2,
            )

        wei = q @ k.transpose(-2, -1)

        wei = wei * (self.head_size ** (-0.5))
        
        causal_mask = self.tril[
            past_len:past_len + T,
            :past_len + T,
        ]

        wei = wei.masked_fill(
            causal_mask == 0,
            float("-inf"),
        )

        wei = F.softmax(wei, dim=-1)

        wei = self.attn_dropout(wei)

        out = wei @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.project(out)

        out = self.resid_dropout(out)

        present_kv = (k, v) if use_cache else None
        
        return out, present_kv