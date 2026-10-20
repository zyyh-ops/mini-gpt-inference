from dataclasses import dataclass
from .sequence import SequenceState
from .block_manager import BlockManager
from .kv_block_pool import KVBlockPool


@dataclass
class KVRuntimeContext:

    seq_state: SequenceState

    block_manager: BlockManager

    kv_pool: KVBlockPool

    layer_idx: int

    logical_pos: int
    context_length: int

    physical_block: int
    block_offset: int
    physical_slot: int


@dataclass
class PrefillRuntimeContext:

    seq_state: SequenceState

    kv_pool: KVBlockPool

    layer_idx: int

    start_pos: int

    context_length: int

    slot_mapping: list[int]


@dataclass
class PrefillChunkContext:

    seq_state: SequenceState

    kv_pool: KVBlockPool

    layer_idx: int

    start_pos: int

    num_tokens: int

    slot_mapping: list[int]