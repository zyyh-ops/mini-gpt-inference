from torch import nn, Tensor

from minigpt.attention import (
    MultiHeadCausalSelfAttention, 
    FusedQKVMultiHeadCausalSelfAttention, 
    VectorizedMultiHeadCausalSelfAttention,
    SDPACausalSelfAttention
)
from minigpt.feed_forward import FeedForward
from torch.nn.attention import (
    SDPBackend,
    sdpa_kernel,
)

ATTENTION_IMPLS = {
    "naive": MultiHeadCausalSelfAttention,
    "fused_qkv": FusedQKVMultiHeadCausalSelfAttention,
    "vectorized": VectorizedMultiHeadCausalSelfAttention,
    "sdpa": SDPACausalSelfAttention,
}

class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
        atten_impl: str = "naive",
    ):
        super(TransformerBlock, self).__init__()

        if n_embd <= 0:
            raise ValueError("n_embd 必须大于 0")

        if n_head <= 0:
            raise ValueError("n_head 必须大于 0")

        if atten_impl not in ATTENTION_IMPLS:
            raise ValueError(
                f"未知 attention_impl: {atten_impl}"
            )

        self.atten_impl = atten_impl

        atten_cls = ATTENTION_IMPLS[atten_impl]

        self.attn = atten_cls(n_embd, n_head, block_size, dropout)

        self.ffn = FeedForward(n_embd, dropout)

        self.layer_norm_1 = nn.LayerNorm(n_embd)

        self.layer_norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.layer_norm_1(x))

        out = x + self.ffn(self.layer_norm_2(x))

        return out