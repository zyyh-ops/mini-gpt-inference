import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from minigpt.block import TransformerBlock
from minigpt.attention import KVCache

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

    def forward(
        self,
        idx: Tensor,
        kv_caches: list[KVCache] | None = None,
        use_cache: bool = False,
        targets: Tensor | None = None
    ):
        B, T = idx.shape

        if T > self.block_size:
            raise ValueError(f"{T} must be smaller than {self.block_size}")

        past_len = kv_caches[0].length if kv_caches is not None else 0

        positions = torch.arange(past_len, past_len + T, device=idx.device)

        position_embeddings = self.position_embedding_table(positions)

        token_embeddings = self.token_embedding_table(idx)

        x = position_embeddings + token_embeddings

        present_key_values = [] if use_cache else None

        for idx, block in enumerate(self.blocks):
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
                kv_caches[idx]
                if kv_caches is not None
                else None
            )

            x, present_kv = block(
                x,
                kv_cache=past_kv_cache,
                use_cache=use_cache,
            )

            if use_cache:
                present_key_values.append(present_kv)
                

        x = self.layer_norm(x)

        logits = self.language_model_head(x)

        B, T, C = logits.shape

        loss = None

        if targets is not None:
            logits_flat = logits.reshape(B * T, C)

            targets_flat = targets.reshape(B * T)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss, present_key_values