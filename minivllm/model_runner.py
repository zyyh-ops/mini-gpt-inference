from model import MiniGPT
from kv_block_pool import KVBlockPool
from block_manager import BlockManager
from sequence import SequenceState
from kv_runtime import KVRuntimeContext, PrefillRuntimeContext
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
        device = next(self.model.parameters()).device

        input_ids = torch.tensor(
            [seq.prompt_tokens],
            device=device,
            dtype=torch.long,
        )

        T = len(seq.prompt_tokens)

        start_pos = seq.length

        slot_mapping = self.block_manager.reserve_slots(seq, T)

        contexts = []

        for layer_idx in range(self.model.n_layer):
            contexts.append(
                PrefillRuntimeContext(
                    seq_state=seq,
                    kv_pool=self.kv_pool,
                    layer_idx=layer_idx,
                    start_pos=start_pos,
                    context_length=start_pos + T,
                    slot_mapping=slot_mapping,
                )
            )

        logits = self.model.prefill(input_ids, contexts)

        seq.length += T

        next_token = torch.argmax(
            logits[:, -1],
            dim=-1,
        ).item()

        seq.generated_tokens.append(next_token)

        seq.last_token = next_token

        return next_token


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
    