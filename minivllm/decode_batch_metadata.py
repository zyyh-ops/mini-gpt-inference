from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class DecodeBatchMetadata:
    positions: Tensor

    context_lengths: Tensor

    block_tables: Tensor

    slot_mapping: Tensor

    