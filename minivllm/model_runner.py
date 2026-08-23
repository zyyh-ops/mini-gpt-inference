from minigpt.model import MiniGPT
from kv_block_pool import KVBlockPool
from block_manager import BlockManager
from sequence import SequenceState
from kv_runtime import KVRuntimeContext
import torch


class ModelRunner:
    def __init__(
        self,
        model: MiniGPT,
        kv_pool: KVBlockPool,
        block_manager: BlockManager,
    ):
        self.model = model

        self.kv_pool = kv_pool

        self.block_manager = block_manager


    def decode_one(
        self,
        seq: SequenceState,
    ):
        input_id = torch.tensor(
            [
                [seq.last_token]
            ]
        )

        contexts = []

        for layer_idx in range(self.model.n_layer):
            contexts.append(
                KVRuntimeContext(
                    seq_state=seq,
                    block_manager=self.block_manager,
                    kv_pool=self.kv_pool,
                    layer_idx=layer_idx,
                )
            )

        logits = self.model.decode(
            input_ids=input_id,
            contexts=contexts,
        )

        next_token = torch.argmax(logits[:, -1], dim=-1)

        return next_token.item()