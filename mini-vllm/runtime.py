from scheduler import Scheduler
from block_manager import BlockManager
from sequence import SequenceState
from kv_block_pool import KVBlockPool, make_kv


def step():
    num_blocks = 8
    block_size = 4

    max_batch_size = 2
    
    block_manager = BlockManager(
        num_blocks=num_blocks, 
        block_size=block_size
    )

    scheduler = Scheduler(
        block_manager=block_manager,
        max_batch_size=max_batch_size,
    )

    A = SequenceState()