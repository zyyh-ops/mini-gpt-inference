from torch import nn, Tensor

from minigpt.attention import KVCacheCausalSelfAttention
from minigpt.feed_forward import FeedForward

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

    def forward(
        self, 
        x: Tensor,
        past_kv: tuple[Tensor, Tensor] | None = None,
        use_cache: bool = False,
    ) -> Tensor:
        attn_out, present_kv = self.attn(
            self.layer_norm_1(x),
            past_kv=past_kv,
            use_cache=use_cache,
        )

        x = x + attn_out
        x = x + self.ffn(self.layer_norm_2(x))

        return x, present_kv