from model import MiniGPT
from kv_block_pool import KVBlockPool
from block_manager import BlockManager
from sequence import SequenceState
from kv_runtime import KVRuntimeContext, PrefillRuntimeContext, PrefillChunkContext
from decode_batch_metadata import DecodeBatchMetadata
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

        return next_token


    def prefill_chunk(
        self,
        seq: SequenceState,
        start_pos: int,
        num_tokens: int,
    ):
        assert start_pos == seq.prompt_pos
        assert start_pos == seq.length

        end_pos = start_pos + num_tokens

        device = next(self.model.parameters()).device

        tokens = seq.prompt_tokens[start_pos:end_pos]

        input_ids = torch.tensor(
            [tokens],
            device=device,
            dtype=torch.long,
        )

        slot_mapping = (
            self.block_manager.reserve_slots(
                seq=seq,
                num_tokens=num_tokens,
            )
        )

        contexts = []

        for layer_idx in range(self.model.n_layer):
            contexts.append(
                PrefillChunkContext(
                    seq_state=seq,
                    kv_pool=self.kv_pool,
                    layer_idx=layer_idx,
                    start_pos=start_pos,
                    num_tokens=num_tokens,
                    slot_mapping=slot_mapping,
                )
            )

        logits = self.model.prefill_chunk(
            input_ids=input_ids,
            contexts=contexts,
        )

        seq.length += num_tokens

        return logits


    def _prepare_decode_batch(
        self,
        sequences: list[SequenceState],
    ):
        device = next(
            self.model.parameters()
        ).device

        B = len(sequences)

        assert B > 0

        input_ids = torch.tensor(
            [
                [seq.last_token]
                for seq in sequences
            ],
            device=device,
            dtype=torch.long,
        )

        slots = []

        for seq in sequences:
            (
                physical_block,
                block_offset,
                physical_slot,
            ) = (
                self.block_manager
                .reserve_append_slot(seq)
            )

            slots.append(physical_slot)

        positions = torch.tensor(
            [
                seq.length
                for seq in sequences
            ],
            device=device,
            dtype=torch.long,
        )

        context_lengths = torch.tensor(
            [
                seq.length + 1
                for seq in sequences
            ],
            device=device,
            dtype=torch.long,
        )

        max_num_blocks = max(
            len(seq.block_table)
            for seq in sequences
        )

        block_tables = torch.full(
            (
                B,
                max_num_blocks,
            ),
            -1,
            device=device,
            dtype=torch.long,
        )

        for b, seq in enumerate(sequences):
            table = torch.tensor(
                seq.block_table,
                device=device,
                dtype=torch.long,
            )

            block_tables[
                b,
                :len(seq.block_table),
            ] = table

        slot_mapping = torch.tensor(
            slots,
            device=device,
            dtype=torch.long,
        )

        metadata = DecodeBatchMetadata(
            positions=positions,
            context_lengths=context_lengths,
            block_tables=block_tables,
            slot_mapping=slot_mapping,
        )

        return input_ids, metadata


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


    def decode_batch(
        self,
        sequences: list[SequenceState],
    ):
        if not sequences:
            return []

        input_ids, metadata = (
            self._prepare_decode_batch(
                sequences=sequences
            )
        )

        logits = self.model.decode_batch(
            input_ids=input_ids,

            metadata=metadata,

            kv_pool=self.kv_pool,
        )

        for seq in sequences:
            seq.length += 1

        next_tokens = torch.argmax(
            logits[:, -1, :],
            dim=-1,
        )

        return (
            logits[:, -1, :],
            next_tokens.tolist(),
        )