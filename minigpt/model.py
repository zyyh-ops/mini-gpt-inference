import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from minigpt.block import TransformerBlock

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
        atten_impl: str = "naive",
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
                    atten_impl=atten_impl,
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
        targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        B, T = idx.shape

        if T > self.block_size:
            raise ValueError(f"{T} must be smaller than {self.block_size}")

        positions = torch.arange(T, device=idx.device)

        position_embeddings = self.position_embedding_table(positions)

        token_embeddings = self.token_embedding_table(idx)

        x = position_embeddings + token_embeddings

        for block in self.blocks:
            should_checkpoint = (
                self.enable_activation_checkpointing
                and self.training
                and torch.is_grad_enabled()
            )

            if should_checkpoint:
                x = checkpoint(
                    block,
                    x,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                x = block(x)

        x = self.layer_norm(x)

        logits = self.language_model_head(x)

        B, T, C = logits.shape

        loss = None

        if targets is not None:
            logits_flat = logits.reshape(B * T, C)

            targets_flat = targets.reshape(B * T)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Tensor:
        was_training = self.training

        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]

            logits = logits / temperature

            if top_k is not None:
                actual_k = min(top_k, self.vocab_size)

                top_values, _ = torch.topk(
                    logits,
                    k=actual_k,
                    dim=-1
                )

                thresold = top_values[:, -1:]

                logits = logits.masked_fill(
                    logits < thresold,
                    float('-inf')
                )

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=-1)

        return idx