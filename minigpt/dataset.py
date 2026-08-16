import torch
# from collections import Sequence

class LanguageModelDataset:
    def __init__(
        self,
        token_ids: torch.Tensor,
        block_size: int,
        train_ratio: float = 0.9,
        seed: int = 42,
    ):
        super().__init__()

        self.tokens = torch.as_tensor(token_ids, dtype=torch.long)

        self.block_size = block_size

        n = int(len(token_ids) * train_ratio)

        self.train_data = self.tokens[:n]
        self.val_data = self.tokens[n:]

        self.validate_split_length(
            self.train_data,
            "train",
        )
        self.validate_split_length(
            self.val_data,
            "val"
        )

        self.generator = torch.Generator().manual_seed(seed)

    def validate_split_length(
        self,
        data: torch.Tensor,
        split_name: str,
    ) -> None:
        required_length = self.block_size + 1

        if len(data) < required_length:
            raise ValueError(
                f"{split_name} requires at least {required_length} tokens."
                f"{required_length} 个 Token, "
                f"实际只有 {len(data)} 个"
            )

    def get_batch(
        self,
        split: str,
        batch_size: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if split == "train":
            data = self.train_data
        elif split == "val":
            data = self.val_data
        else:
            raise ValueError(
                "split 只能是 'train' 或者 'val'"
            )

        ix = torch.randint(0, len(data) - self.block_size, (batch_size,), generator=self.generator)

        x = torch.stack([
            data[i: i + self.block_size]
            for i in ix.tolist()
        ])

        y = torch.stack([
            data[i + 1: i + 1 + self.block_size]
            for i in ix.tolist()
        ])

        if device is not None:
            x = x.to(device)
            y = y.to(device)

        return x, y