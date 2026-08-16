from torch import nn, Tensor

class FeedForward(nn.Module):
    def __init__(
            self,
            n_embd: int,
            dropout: float = 0.0,
    ):
        super(FeedForward, self).__init__()

        if n_embd <= 0:
            raise ValueError('n_embd must be > 0')

        if not 0.0 <= dropout <= 1.0:
            raise ValueError('dropout must be in [0, 1]')

        hidden_size = 4 * n_embd


        self.network = nn.Sequential(
            nn.Linear(n_embd, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

