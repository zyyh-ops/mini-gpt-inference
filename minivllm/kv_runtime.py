from dataclasses import dataclass
from sequence import SequenceState
from block_manager import BlockManager
from kv_block_pool import KVBlockPool


@dataclass
class KVRuntimeContext:

    seq_state: SequenceState

    block_manager: BlockManager

    kv_pool: KVBlockPool

    layer_idx: int