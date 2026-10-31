from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from dataclasses import dataclass
from .kv_runtime import KVRuntimeContext, PrefillRuntimeContext, PrefillChunkContext
from .paged_attention_v1_reference import paged_attention_online
from .decode_batch_metadata import DecodeBatchMetadata
from .kv_block_pool import KVBlockPool
from .paged_attention_batch import paged_attention_batch_reference, paged_attention_batch_online_v1


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
        # Step 4: opt-in first-chunk SDPA fast path; default stays reference.
        self.prefill_fastpath_enabled = False
        # v2 opt-in for chunks with a cached KV prefix.
        self.prefill_prefix_fastpath_enabled = False
        # Decode-only, explicit opt-in. State dict and Prefill remain unchanged.
        self.decode_attention_backend = "reference"

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
        context: PrefillRuntimeContext,
    ):
        import math

        B, T, C = x.shape

        assert B == 1
        assert len(context.slot_mapping) == T

        q, k, v = self.qkv(x).chunk(
            3,
            dim=-1,
        )

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_size)

        mask = torch.tril(
            torch.ones(
                T, T, 
                device=x.device, 
                dtype=torch.bool,
            )
        )

        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

        probs = torch.softmax(scores, dim=-1)

        out = probs @ v

        k_tokens = k[0].transpose(0, 1)
        v_tokens = v[0].transpose(0, 1)

        for token_idx, slot in enumerate(context.slot_mapping):
            physical_block = slot // context.kv_pool.block_size

            block_offset = slot % context.kv_pool.block_size

            context.kv_pool.write_layer(
                layer_idx=context.layer_idx,

                physical_block=physical_block,

                block_offset=block_offset,

                k_new=k_tokens[token_idx],

                v_new=v_tokens[token_idx],
            )

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        out = self.project(out)

        out = self.resid_dropout(out)

        return out


    def forward_prefill_chunk(
        self,
        x: Tensor,
        context: PrefillChunkContext,
    ):
        B, T, C = x.shape

        assert B == 1
        assert T == context.num_tokens

        q, k, v = self.qkv(x).chunk(
            3, dim=-1
        )

        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        k_tokens = k[0].transpose(0, 1)
        v_tokens = v[0].transpose(0, 1)

        for i, slot in enumerate(context.slot_mapping):
            physical_block = slot // context.kv_pool.block_size

            block_offset = slot % context.kv_pool.block_size

            context.kv_pool.write_layer(
                layer_idx=context.layer_idx,

                physical_block=physical_block,

                block_offset=block_offset,

                k_new=k_tokens[i],

                v_new=v_tokens[i],
            )

        # The first chunk has no prefix: q and k have the same absolute
        # starting position, so SDPA's upper-left causal alignment is valid.
        # Keep the existing KV writes above unchanged for Paged KV semantics.
        # Later chunks / recompute prefixes have start_pos > 0 and must keep
        # the reference path until an offset-aware mask is implemented.
        if (
            self.prefill_fastpath_enabled
            and context.start_pos == 0
            and q.dtype == torch.float32
        ):
            attention_out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )
            out = attention_out.transpose(1, 2).contiguous().view(B, T, C)
            return self.resid_dropout(self.project(out))

        if (
            self.prefill_fastpath_enabled
            and self.prefill_prefix_fastpath_enabled
            and context.start_pos > 0
            and q.dtype == torch.float32
        ):
            # Prefix-Aware SDPA v2: gather the *valid* historical KV tokens
            # by logical order. Physical blocks need not be adjacent.
            # The current chunk has already been written into the paged pool,
            # but its q/k/v tensors can be used directly without gathering it.
            prefix_len = context.start_pos
            bs = context.kv_pool.block_size
            prefix_blocks = (prefix_len + bs - 1) // bs
            physical_ids = context.seq_state.block_table[:prefix_blocks]
            if len(physical_ids) != prefix_blocks:
                raise RuntimeError('incomplete KV block table for prefix')
            block_ids = torch.tensor(physical_ids, device=q.device, dtype=torch.long)
            prefix_k = context.kv_pool.k[context.layer_idx].index_select(0, block_ids)
            prefix_v = context.kv_pool.v[context.layer_idx].index_select(0, block_ids)
            # [num_blocks, bs, H, D] -> [1, H, prefix_len, D]
            prefix_k = (prefix_k.reshape(-1, self.n_head, self.head_size)
                        [:prefix_len].transpose(0, 1).unsqueeze(0))
            prefix_v = (prefix_v.reshape(-1, self.n_head, self.head_size)
                        [:prefix_len].transpose(0, 1).unsqueeze(0))
            all_k = torch.cat((prefix_k, k), dim=2)
            all_v = torch.cat((prefix_v, v), dim=2)

            # Query i is at absolute position (prefix_len + i).
            # SDPA is_causal=True gives upper-left alignment and is *wrong*
            # when Q_len < KV_len. Explicit mask uses absolute positions.
            query_positions = prefix_len + torch.arange(T, device=q.device)
            key_positions = torch.arange(prefix_len + T, device=q.device)
            allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            attention_out = F.scaled_dot_product_attention(
                q, all_k, all_v, attn_mask=allowed,
                dropout_p=0.0, is_causal=False,
            )
            out = attention_out.transpose(1, 2).contiguous().view(B, T, C)
            return self.resid_dropout(self.project(out))

        outputs = []

        for i in range(T):
            q_i = q[0, :, i, :]

            logical_pos = context.start_pos + i

            current_context_length = logical_pos + 1

            out_i = paged_attention_online(
                q=q_i,
                pool=context.kv_pool,
                seq=context.seq_state,
                layer_idx=context.layer_idx,
                context_length=current_context_length,
            )

            outputs.append(out_i)

        out = torch.stack(
            outputs,
            dim=0,
        )

        out = out.unsqueeze(0).contiguous().view(B, T, C)

        out = self.project(out)

        out = self.resid_dropout(out)

        return out


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


    def forward_decode_batch(
        self,
        x: torch.Tensor,
        metadata: DecodeBatchMetadata,
        kv_pool: KVBlockPool,
        layer_idx: int,
    ):
        B, T, C = x.shape

        assert T == 1

        q, k, v = self.qkv(x).chunk(
            3, dim=-1,
        )

        q = q.view(B, 1, self.n_head, self.head_size).transpose(1, 2)[:, :, 0, :]

        k = k.view(B, 1, self.n_head, self.head_size).transpose(1, 2)[:, :, 0, :]

        v = v.view(B, 1, self.n_head, self.head_size).transpose(1, 2)[:, :, 0, :]

        slots = metadata.slot_mapping

        physical_blocks = torch.div(
            slots,
            kv_pool.block_size,
            rounding_mode="floor",
        )

        block_offsets = (
            slots % kv_pool.block_size
        )

        kv_pool.k[
            layer_idx,
            physical_blocks,
            block_offsets,
        ] = k

        kv_pool.v[
            layer_idx,
            physical_blocks,
            block_offsets,
        ] = v

        # The newly appended K/V MUST be visible before either backend reads
        # the KV pool. Triton is an attention-only kernel: it does not write KV.
        if self.decode_attention_backend == "reference":
            attention_out = paged_attention_batch_online_v1(
                q=q, pool=kv_pool,
                block_tables=metadata.block_tables,
                context_lengths=metadata.context_lengths,
                layer_idx=layer_idx,
            )
        elif self.decode_attention_backend == "triton_v0":
            if torch.is_grad_enabled() and q.requires_grad:
                raise RuntimeError("triton_v0 is inference-only; run under torch.inference_mode()")
            # Lazy import: CPU-only tests and default Runtime do not require
            # the Triton dependency. Unsupported dtype/device fails explicitly.
            from .triton_paged_decode import paged_attention_batch_triton_v0
            attention_out = paged_attention_batch_triton_v0(
                q=q, pool=kv_pool,
                block_tables=metadata.block_tables,
                context_lengths=metadata.context_lengths,
                layer_idx=layer_idx,
            )
        else:
            raise ValueError(f"unknown decode attention backend: {self.decode_attention_backend}")

        attention_out = attention_out.reshape(B, 1, self.n_embd)

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