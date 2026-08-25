from model import MiniGPT
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

    def prefill(
        self,
        seq: SequenceState,
    ):
        input_ids = torch.tensor([seq.prompt_tokens])



    def decode_one(
        self,
        seq: SequenceState,
    ):
        (
            physical_block,
            block_offset,
            physical_slot,
        ) = self.block_manager.reserve_append_slot(seq)

        logical_pos = seq.length
        context_length = seq.length + 1

        contexts = []

        for layer_idx in range(self.model.n_layer):
            contexts.append(
                KVRuntimeContext(
                    seq_state=seq,
                    block_manager=self.block_manager,
                    kv_pool=self.kv_pool,

                    layer_idx=layer_idx,

                    logical_pos=logical_pos,
                    context_length=context_length,

                    physical_block=physical_block,
                    block_offset=block_offset,
                    physical_slot=physical_slot,
                )
            )

        device = next(self.model.parameters()).device

        input_ids = torch.tensor(
            [[seq.last_token]],
            device=device,
            dtype=torch.long,
        )

        logits = self.model.decode(
            input_ids=input_ids,
            contexts=contexts,
        )

        seq.length += 1

        next_token = torch.argmax(logits[:, -1], dim=-1)

        return next_token.item()
    