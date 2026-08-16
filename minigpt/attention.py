from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SDPACausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if n_embd % n_head != 0:
            raise ValueError(
                f"n_embd {n_embd} not divisible by n_head {n_head}"
            )

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.block_size = block_size
        self.attn_dropout_p = dropout

        self.qkv = nn.Linear(
            n_embd,
            3 * n_embd,
            bias=False,
        )

        self.project = nn.Linear(
            n_embd,
            n_embd,
        )

        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape

        if T > self.block_size:
            raise ValueError(
                f"the length of T is greater than {self.block_size}"
            )

        qkv = self.qkv(x)

        q, k, v = qkv.split(
            self.n_embd,
            dim=-1,
        )

        q = (
            q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        )

        k = (
            k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        )

        v = (
            v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        )

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=(
                self.attn_dropout_p
                if self.train
                else 0.0
            ),
            is_causal=True,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.project(out)

        out = self.resid_dropout(out)

        return out


class VectorizedMultiHeadCausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if n_embd % n_head != 0:
            raise ValueError(
                f"n_embd {n_embd} not divisible by n_head {n_head}"
            )

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.block_size = block_size

        self.qkv = nn.Linear(
            n_embd,
            3 * n_embd,
            bias=False,
        )

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size)),
        )

        self.attn_dropout = nn.Dropout(dropout)

        self.project = nn.Linear(
            n_embd,
            n_embd,
        )

        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape

        if T > self.block_size:
            raise ValueError(
                f"the length of T is greater than {self.block_size}"
            )

        qkv = self.qkv(x)

        q, k, v = qkv.split(
            self.n_embd,
            dim=-1,
        )

        q = (
            q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        )

        k = (
            k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        )

        v = (
            v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        )

        wei = q @ k.transpose(-2, -1)

        wei = wei * (self.head_size ** (-0.5))

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf"),
        )

        wei = F.softmax(
            wei,
            dim=-1,
        )

        wei = self.attn_dropout(wei)

        out = wei @ v

        out = (
            out.transpose(1, 2).contiguous().view(B, T, C)
        )

        out = self.project(out)

        out = self.resid_dropout(out)

        return out


class CausalSelfAttentionCore(nn.Module):
    def __init__(
        self,
        head_size: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.head_size = head_size
        self.block_size = block_size

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
    ) -> Tensor:
        _, T, _ = q.shape

        if T > self.block_size:
            raise ValueError(
                f"the length of T is greater than {self.block_size}"
            )

        wei = q @ k.transpose(-2, -1)

        wei = wei * (self.head_size ** (-0.5))

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        wei = F.softmax(wei, dim=-1)

        wei = self.dropout(wei)

        out = wei @ v

        return out


class FusedQKVMultiHeadCausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if n_embd % n_head != 0:
            raise ValueError(
                f"n_embd {n_embd} not divisible by n_head {n_head}"
            )

        self.n_embd = n_embd
        self.n_head = n_head
        self.block_size = block_size
        self.head_size = n_embd // n_head

        self.qkv = nn.Linear(
            n_embd,
            3 * n_embd,
            bias=False,
        )

        self.heads = nn.ModuleList([
            CausalSelfAttentionCore(
                head_size=self.head_size,
                block_size=self.block_size,
                dropout=dropout,
            )
            for _ in range(n_head)
        ])

        self.project = nn.Linear(n_embd, n_embd)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        qkv = self.qkv(x)

        q, k, v = qkv.split(self.n_embd, dim=-1)

        head_outs = []

        for i, head in enumerate(self.heads):
            start = i * self.head_size
            end = start + self.head_size

            q_i = q[:, :, start:end]
            k_i = k[:, :, start:end]
            v_i = v[:, :, start:end]

            head_outs.append(head(q_i, k_i, v_i))

        out = torch.cat(head_outs, dim=-1)

        out = self.project(out)

        out = self.dropout(out)

        return out


class CausalSelfAttentionHead(nn.Module):
    def __init__(
        self,
        n_embd: int,
        head_size: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.head_size = head_size

        self.block_size = block_size

        self.query = nn.Linear(n_embd, head_size, bias=False)

        self.key = nn.Linear(n_embd, head_size, bias=False)

        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        _, T, _ = x.shape

        if T > self.block_size:
            raise ValueError(f"the length of T is greater than {self.block_size}")

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1)

        wei = wei * (self.head_size ** (-0.5))

        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        wei = F.softmax(wei, dim=-1)

        wei = self.dropout(wei)

        out = wei @ v

        return out


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if n_embd % n_head != 0:
            raise ValueError(f"n_embd {n_embd} not divisible by n_head {n_head}")

        self.n_head = n_head

        self.block_size = block_size

        self.head_size = n_embd // n_head

        self.heads = nn.ModuleList([
            CausalSelfAttentionHead(
                n_embd=n_embd,
                head_size=self.head_size,
                block_size=self.block_size,
                dropout=dropout)
            for _ in range(n_head)
        ])

        self.project = nn.Linear(n_embd, n_embd)

        self.dropout = nn.Dropout(dropout)


    def forward(self, x: Tensor) -> Tensor:
        head_outs = [
            head(x)
            for head in self.heads
        ]

        out = torch.cat(head_outs, dim=-1)

        out = self.project(out)

        out = self.dropout(out)

        return out