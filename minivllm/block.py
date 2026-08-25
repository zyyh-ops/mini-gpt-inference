from torch import nn, Tensor

from attention import KVCacheCausalSelfAttention, KVCache
from feed_forward import FeedForward
from kv_runtime import KVRuntimeContext


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        dropout: float = 0.0,
    ):
        super(TransformerBlock, self).__init__()

        if n_embd <= 0:
            raise ValueError("n_embd 必须大于 0")

        if n_head <= 0:
            raise ValueError("n_head 必须大于 0")

        self.attn = KVCacheCausalSelfAttention(n_embd, n_head, block_size, dropout)

        self.ffn = FeedForward(n_embd, dropout)

        self.layer_norm_1 = nn.LayerNorm(n_embd)

        self.layer_norm_2 = nn.LayerNorm(n_embd)


    def forward_prefill(
        self,
        x: Tensor,
        kv_contexts: list[KVRuntimeContext],
    ) -> Tensor:
        x = x + self.attn.forward_prefill(self.layer_norm_1(x), kv_contexts)

        x = x + self.ffn(self.layer_norm_2(x))

        return x

    def forward_decode(
        self, 
        x: Tensor,
        kv_context: KVRuntimeContext,
    ) -> Tensor:

        x = x + self.attn.forward_decode(self.layer_norm_1(x), kv_context=kv_context)

        x = x + self.ffn(self.layer_norm_2(x))

        return x

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        x = x + self.attn(self.layer_norm_1(x))

        out = x + self.ffn(self.layer_norm_2(x))