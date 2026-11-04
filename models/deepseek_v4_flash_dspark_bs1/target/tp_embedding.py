# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl

import pypto.language.distributed as pld

from .config import DECODE_TOKENS, FLASH as M, TP

T = DECODE_TOKENS

D = M.hidden_size

VOCAB = M.vocab_size

N_RANKS = TP

VOCAB_PER_TP = VOCAB // N_RANKS

HIDDEN_TILE = 512

@pl.jit.inline
def embed_shard_gather(
    input_ids: pl.Tensor[[T], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    owned_rows: pl.Tensor[[T, D], pl.BF16],
    vocab_base: pl.Scalar[pl.INT32],
):
    """Copy the rows this rank owns; rows it does not own are left untouched."""
    for block in pl.spmd(T * (D // HIDDEN_TILE), name_hint="embed_shard_gather", allow_early_resolve=True):
        token = block // (D // HIDDEN_TILE)
        hidden_offset = (block % (D // HIDDEN_TILE)) * HIDDEN_TILE
        token_id = pl.cast(pl.tensor.read(input_ids, [token]), pl.INT32)
        local_id = token_id - vocab_base
        if local_id >= 0:
            if local_id < VOCAB_PER_TP:
                shard_row = pl.cast(local_id, pl.INDEX)
                shard_tile = embed_weight[shard_row : shard_row + 1, hidden_offset : hidden_offset + HIDDEN_TILE]
                owned_rows[token : token + 1, hidden_offset : hidden_offset + HIDDEN_TILE] = shard_tile

@pl.jit.inline
def all_gather_embedding(
    input_ids: pl.Tensor[[T], pl.INT64],
    owned_rows: pl.Tensor[[T, D], pl.BF16],
    hidden_states: pl.Tensor[[T, D], pl.BF16],
    gather_window: pld.DistributedTensor[[T, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    vocab_base: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    # 1-based call id; gather_signal is monotonic so waits use `>= gather_epoch`.
    gather_epoch: pl.Scalar[pl.INT32],
):
    """Publish each owned row to every peer, barrier, then read the assembled window."""
    with pl.spmd(T, name_hint="embed_publish", allow_early_resolve=True) as publish_tid:
        token = pl.tile.get_block_idx()
        token_id = pl.cast(pl.tensor.read(input_ids, [token]), pl.INT32)
        local_id = token_id - vocab_base
        if local_id >= 0:
            if local_id < VOCAB_PER_TP:
                for peer in pl.range(N_RANKS):
                    pld.tensor.put(
                        dst=gather_window,
                        peer=peer,
                        src=owned_rows,
                        dst_offsets=[token, 0],
                        src_offsets=[token, 0],
                        shape=[1, D],
                    )

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="embed_barrier", allow_early_resolve=True, deps=[publish_tid]) as barrier_tid:
        for peer in pl.range(N_RANKS):
            pld.system.notify(
                target=gather_signal,
                peer=peer,
                offsets=[my_rank, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )
        for src in pl.range(N_RANKS):
            pld.system.wait(
                signal=gather_signal,
                offsets=[src, 0],
                expected=gather_epoch,
                cmp=pld.WaitCmp.Ge,
            )

    with pl.spmd(T * (D // HIDDEN_TILE), name_hint="embed_readback", allow_early_resolve=True, deps=[barrier_tid]) as _readback_tid:
        block = pl.tile.get_block_idx()
        token = block // (D // HIDDEN_TILE)
        hidden_offset = (block % (D // HIDDEN_TILE)) * HIDDEN_TILE
        gathered = pl.load(gather_window, [token, hidden_offset], [1, HIDDEN_TILE])
        pl.store(gathered, [token, hidden_offset], hidden_states)

@pl.jit.inline
def clear_embedding_signals(
    completion_anchor: pl.Tensor[[T, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    """Clear this rank's embedding counters after its forward completes."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="embed_signal_clear") as s6_embed_cleanup_tid:
        _completion_anchor = pl.read(completion_anchor, [0, 0])
        zero = pl.cast(0, pl.INT32)
        for src in pl.range(N_RANKS):
            pl.write(gather_signal, [src, 0], zero)
    return completion_anchor, s6_embed_cleanup_tid

@pl.jit.inline
def tp_lookup_embedding(
    input_ids: pl.Tensor[[T], pl.INT64],
    embed_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    hidden_states: pl.Tensor[[T, D], pl.BF16],
    gather_window: pld.DistributedTensor[[T, D], pl.BF16],
    gather_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    gather_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, D], pl.BF16]:
    vocab_base = pl.cast(my_rank, pl.INT32) * VOCAB_PER_TP
    if N_RANKS == 1:
        embed_shard_gather(input_ids, embed_weight, hidden_states, vocab_base)
    else:
        owned_rows = pl.create_tensor([T, D], dtype=pl.BF16)
        embed_shard_gather(input_ids, embed_weight, owned_rows, vocab_base)
        all_gather_embedding(
            input_ids, owned_rows, hidden_states,
            gather_window, gather_signal,
            vocab_base, my_rank, gather_epoch,
        )
    return hidden_states
