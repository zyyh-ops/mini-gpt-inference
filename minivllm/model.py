import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from .block import TransformerBlock
from .attention import KVCache
from .kv_runtime import KVRuntimeContext, PrefillRuntimeContext, PrefillChunkContext
from .sequence import SequenceState
from .decode_batch_metadata import DecodeBatchMetadata
from .kv_block_pool import KVBlockPool



class MiniGPT(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        vocab_size: int,
        block_size: int,
        dropout: float = 0.0,
        n_layer: int = 1,
        enable_activation_checkpointing: bool = False,
    ):
        super().__init__()

        self.block_size = block_size

        self.vocab_size = vocab_size

        self.n_layer = n_layer

        self.token_embedding_table = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=n_embd,
        )

        self.position_embedding_table = nn.Embedding(
            num_embeddings=block_size,
            embedding_dim=n_embd,
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    n_embd=n_embd,
                    n_head=n_head,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(n_layer)
            ]
        )

        self.layer_norm = nn.LayerNorm(n_embd)

        self.language_model_head = nn.Linear(n_embd, vocab_size)

        self.enable_activation_checkpointing = enable_activation_checkpointing

    def set_prefill_fastpath(self, enabled: bool) -> None:
        """Keep the original v1 boolean API: only first-chunk SDPA."""
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        for block in self.blocks:
            block.attn.prefill_fastpath_enabled = enabled
            block.attn.prefill_prefix_fastpath_enabled = False

    def set_prefill_fastpath_mode(self, mode: str) -> None:
        """Choose reference, initial-only SDPA v1, or prefix-aware SDPA v2."""
        if mode not in ('reference', 'sdpa_v1', 'sdpa_v2'):
            raise ValueError('mode must be reference, sdpa_v1 or sdpa_v2')
        for block in self.blocks:
            block.attn.prefill_fastpath_enabled = mode != 'reference'
            block.attn.prefill_prefix_fastpath_enabled = mode == 'sdpa_v2'

    def forward(
        self,
        idx: Tensor,
        kv_caches: list[KVCache] | None = None,
        targets: Tensor | None = None
    ):
        B, T = idx.shape

        past_len = kv_caches[0].length if kv_caches is not None else 0

        if past_len + T > self.block_size:
            raise ValueError(
                f"sequence length {past_len + T} "
                f"exceeds block_size {self.block_size}"
            )

        positions = torch.arange(past_len, past_len + T, device=idx.device)

        position_embeddings = self.position_embedding_table(positions)

        token_embeddings = self.token_embedding_table(idx)

        x = position_embeddings + token_embeddings

        for layer_idx, block in enumerate(self.blocks):
            should_checkpoint = (
                self.enable_activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
            )

            if should_checkpoint:
                x, _ = checkpoint(
                    block,
                    x,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
                continue

            past_kv_cache = (
                kv_caches[layer_idx]
                if kv_caches is not None
                else None
            )

            x = block(
                x,
                kv_cache=past_kv_cache,
            )
                

        x = self.layer_norm(x)

        logits = self.language_model_head(x)

        B, T, C = logits.shape

        loss = None

        if targets is not None:
            logits_flat = logits.reshape(B * T, C)

            targets_flat = targets.reshape(B * T)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss


    def prefill(
        self,
        input_ids: torch.Tensor,
        contexts: list[PrefillRuntimeContext],
    ):
        B, T = input_ids.shape

        start_pos = contexts[0].start_pos

        positions = torch.arange(start_pos, start_pos + T, device=input_ids.device)

        x = (
            self.token_embedding_table(input_ids) +
            self.position_embedding_table(positions)
        )

        for layer_idx, block in enumerate(self.blocks):
            x = block.forward_prefill(
                x,
                contexts[layer_idx]
            )

        x = self.layer_norm(x)

        logits = self.language_model_head(x)

        return logits


    def prefill_chunk(
        self,
        input_ids: torch.Tensor,
        contexts: list[PrefillChunkContext],
    ):
        B, T = input_ids.shape

        start_pos = contexts[0].start_pos

        positions = torch.arange(start_pos, start_pos + T, device=input_ids.device)

        x = (
            self.token_embedding_table(input_ids)
            + self.position_embedding_table(positions)
        )

        for layer_idx, block in enumerate(self.blocks):
            x = block.forward_prefill_chunk(
                x,
                contexts[layer_idx],
            )

        x = self.layer_norm(x)

        logits = self.language_model_head(x)

        return logits


    def decode_batch(
        self,
        input_ids: Tensor,
        metadata: DecodeBatchMetadata,
        kv_pool: KVBlockPool,
    ):
        B = input_ids.shape[0]

        positions = metadata.positions.unsqueeze(1)

        x = (
            self.token_embedding_table(input_ids)
            + self.position_embedding_table(positions)
        )

        for layer_idx, block in enumerate(self.blocks):
            x = block.forward_decode_batch(
                x,

                metadata=metadata,

                kv_pool=kv_pool,

                layer_idx=layer_idx,
            )

        x = self.layer_norm(x)

        logits = self.language_model_head(x)

        return logits


    def decode(
        self,
        input_ids: torch.Tensor,
        contexts: list[KVRuntimeContext],
    ):
        assert len(contexts) == self.n_layer
        
        position = contexts[0].logical_pos

        positions = torch.tensor(
            [[position]],
            device=input_ids.device,
            dtype=torch.long,
        )
        
        x = (
            self.token_embedding_table(input_ids)
            + self.position_embedding_table(positions)
        )

        for layer_idx, block in enumerate(self.blocks):

            ctx = contexts[layer_idx]

            x = block.forward_decode(
                x, ctx
            )

        x = self.layer_norm(x)

        logits = self.language_model_head(x)

        return logits