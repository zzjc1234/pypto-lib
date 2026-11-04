# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys

import pypto.language as pl

import pypto.language.distributed as pld

from .config import DECODE_TOKENS, FLASH as M, TP as CONFIG_TP

from .sample import sample_from_greedy_partials

T_DYN = pl.dynamic('NATIVE_S_LM_HEAD_T_DYN')

D = M.hidden_size

VOCAB = M.vocab_size

_TP_CHOICES = (2, 4, 8, 16)  # AIV_LANES = 2 owners per lane, so TP = 1 has no valid split

_DP_CHOICES = (1, 2, 4, 8, 16)

_TP_DEFAULT = CONFIG_TP

def _parse_int_argv(name, default=None):
    if name == "--tp":
        return 8
    if name == "--dp":
        return 1
    for i, tok in enumerate(sys.argv):
        if tok == name and i + 1 < len(sys.argv):
            return int(sys.argv[i + 1])
        if tok.startswith(f"{name}="):
            return int(tok.split("=", 1)[1])
    return default

TP_SIZE: int = _parse_int_argv("--tp") or _TP_DEFAULT

DP_SIZE: int = _parse_int_argv("--dp") or 1

VOCAB_PER_TP = VOCAB // TP_SIZE

AIV_LANES = 2

OWNER_PAIRS = TP_SIZE // AIV_LANES

OWNERS_PER_LANE = TP_SIZE // AIV_LANES

FUSED_LM_HEAD_CORES = 24

MAX_LOGIT_ROWS = DECODE_TOKENS

GROUP_LOGIT_ROWS = TP_SIZE * MAX_LOGIT_ROWS

SAMPLED_IDS_PAD = 8

FUSED_K_TILE = 256

FUSED_VOCAB_TILE = 256

HIDDEN_GATHER_TILE = 512

LOGITS_COMM_TILE = 2048

COMBINE_ROW_TILE = ((MAX_LOGIT_ROWS + 7) // 8) * 8

VOCAB_LAST_TILE = VOCAB_PER_TP % FUSED_VOCAB_TILE

VOCAB_TILES = VOCAB_PER_TP // FUSED_VOCAB_TILE + (1 if VOCAB_LAST_TILE != 0 else 0)

SHARDS_VOCAB = VOCAB_TILES * FUSED_VOCAB_TILE

MATMUL_GROUP_ROWS = ((GROUP_LOGIT_ROWS + 15) // 16) * 16

SHARD_ROWS = MATMUL_GROUP_ROWS // AIV_LANES

LOGITS_COMM_TAIL = VOCAB_PER_TP % LOGITS_COMM_TILE

N_LOGITS_COMM_TILES = VOCAB_PER_TP // LOGITS_COMM_TILE

LOGITS_COMM_BLOCKS = min(FUSED_LM_HEAD_CORES, N_LOGITS_COMM_TILES + (1 if LOGITS_COMM_TAIL != 0 else 0))

LOGITS_TAIL_BLOCK = N_LOGITS_COMM_TILES % LOGITS_COMM_BLOCKS

assert TP_SIZE in _TP_CHOICES, f"--tp must be one of {_TP_CHOICES} (got {TP_SIZE})"

assert DP_SIZE in _DP_CHOICES, f"--dp must be one of {_DP_CHOICES} (got {DP_SIZE})"

@pl.jit.inline(auto_scope=False)
def _lm_head_with_greedy_partials(
    hidden_states: pl.Tensor,
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
    greedy_scores: pl.Tensor[[LOGITS_COMM_BLOCKS * MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.FP32],
    greedy_indices: pl.Tensor[[LOGITS_COMM_BLOCKS * MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32],
):
    # Scratch is allocated just outside the scope that first writes it: a
    # create_tensor inside a pl.at yields a tile, not a GM tensor view.
    selected_hidden = pl.create_tensor([MAX_LOGIT_ROWS, D], dtype=pl.BF16)
    owner_hiddens = pl.create_tensor([MATMUL_GROUP_ROWS, D], dtype=pl.BF16)

    # Publish this card's logit rows into every group member's window slot: the
    # window holds one slot per group member and each card writes only its own,
    # `tp_rank * MAX_LOGIT_ROWS`. One block per logit row, one [1, D] put per peer.
    for row in pl.spmd(MAX_LOGIT_ROWS, name_hint="lm_head_dispatch_push", allow_early_resolve=True):
        hidden_rows = pl.tensor.dim(hidden_states, 0)
        source_row_raw = pl.read(logit_row_indices, [row])
        # Clamp so the load address is always inside hidden_states even if a
        # caller hands over a stale index; the -1 guard below decides whether
        # the row is actually used.
        safe_raw = pl.max(pl.min(source_row_raw, hidden_rows - 1), 0)
        # Full-width [1, D] tile: the block owns the whole row.
        selected_hidden[row : row + 1, :] = pl.full([1, D], dtype=pl.BF16, value=0.0)
        if source_row_raw >= 0:
            source_row = pl.cast(safe_raw, target_type=pl.INDEX)
            selected_hidden[row : row + 1, :] = hidden_states[source_row : source_row + 1, :]

        # Self-target rides the same put; put drains before the notify issues.
        for peer_tp in pl.range(TP_SIZE):
            pld.tensor.put(
                dst=hidden_window,
                peer=group_base + peer_tp,
                src=selected_hidden,
                dst_offsets=[tp_rank * MAX_LOGIT_ROWS + row, 0],
                src_offsets=[row, 0],
                shape=[1, D],
            )

        # Notify folded into the push: MAX_LOGIT_ROWS notifies per source per epoch.
        for peer_tp in pl.range(TP_SIZE):
            if peer_tp != tp_rank:
                pld.system.notify(
                    target=hidden_done,
                    peer=group_base + peer_tp,
                    offsets=[tp_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )

    # Anchor the blocking wait to the local hidden-state producer.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_dispatch_wait", allow_early_resolve=True) as _dwait_tid:
        _hidden_anchor = pl.read(hidden_states, [0, 0])
        for owner_tp in pl.range(TP_SIZE):
            if owner_tp != tp_rank:
                pld.system.wait(
                    signal=hidden_done,
                    offsets=[owner_tp, 0],
                    expected=pl.cast(done_epoch * MAX_LOGIT_ROWS, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # Window -> matmul operand: a local copy split over k-tiles. Keeps the matmul's
    # auto-dep on owner_hiddens.
    with pl.spmd(
        D // HIDDEN_GATHER_TILE, name_hint="lm_head_dispatch_gather", allow_early_resolve=True, deps=[_dwait_tid]
    ) as _dgather_tid:
        gkb = pl.tile.get_block_idx()
        gk0 = gkb * HIDDEN_GATHER_TILE
        if MATMUL_GROUP_ROWS == GROUP_LOGIT_ROWS:
            owner_hiddens[0:GROUP_LOGIT_ROWS, gk0 : gk0 + HIDDEN_GATHER_TILE] = hidden_window[:, gk0 : gk0 + HIDDEN_GATHER_TILE]
        else:
            owner_hiddens[:, gk0 : gk0 + HIDDEN_GATHER_TILE] = pl.full([MATMUL_GROUP_ROWS, HIDDEN_GATHER_TILE], dtype=pl.BF16, value=0.0)
            for owner in pl.range(TP_SIZE):
                src_row = owner * MAX_LOGIT_ROWS
                dst_lane = owner // OWNERS_PER_LANE
                dst_slot = owner % OWNERS_PER_LANE
                dst_row = dst_lane * SHARD_ROWS + dst_slot * MAX_LOGIT_ROWS
                owner_tile = hidden_window[src_row : src_row + MAX_LOGIT_ROWS, gk0 : gk0 + HIDDEN_GATHER_TILE]
                owner_hiddens[dst_row : dst_row + MAX_LOGIT_ROWS, gk0 : gk0 + HIDDEN_GATHER_TILE] = owner_tile

    # Mixed cube + comm kernel: the cube projects a vocab tile, pl.aiv_shard carries
    # that accumulator across the C->V edge, and pld.tensor.remote_store pushes each
    # owner's rows to that peer's window, so a tile goes on the wire while the cube
    # projects the next. The ragged last tile rides the same loop as a padded tile.
    logits_shards = pl.create_tensor([MATMUL_GROUP_ROWS, SHARDS_VOCAB], dtype=pl.FP32)
    with pl.spmd(
        FUSED_LM_HEAD_CORES,
        name_hint="lm_head_matmul_push", allow_early_resolve=True,
        optimizations=[pl.cross_core_slot(slot_num=2)],
    ) as _push_tid:
        lm_core = pl.tile.get_block_idx()
        vocab_base = tp_rank * VOCAB_PER_TP
        for mm_ob in pl.range(lm_core, VOCAB_TILES, FUSED_LM_HEAD_CORES):
            mm_o0 = mm_ob * FUSED_VOCAB_TILE
            # Real width of this tile: the full tile everywhere except the ragged
            # last one, where the weight rows run out early. The box stays static
            # so the accumulator keeps one shape for every iteration.
            mm_valid_n = pl.min(VOCAB_PER_TP - mm_o0, FUSED_VOCAB_TILE)
            # M is the full group extent -- one matmul per vocab tile.
            mm_acc = pl.create_tensor([MATMUL_GROUP_ROWS, FUSED_VOCAB_TILE], dtype=pl.FP32)
            for mm_kb in pl.pipeline(0, D // FUSED_K_TILE, stage=2):
                mm_k0 = mm_kb * FUSED_K_TILE
                mm_hidden_tile = owner_hiddens[:, mm_k0 : mm_k0 + FUSED_K_TILE]
                mm_weight_tile = pl.slice(
                    lm_head_weight,
                    [FUSED_VOCAB_TILE, FUSED_K_TILE],
                    [mm_o0, mm_k0],
                    valid_shape=[mm_valid_n, FUSED_K_TILE],
                )
                mm_acc = pl.matmul_acc(
                    mm_acc, mm_hidden_tile, mm_weight_tile, b_trans=True, init_cond=(mm_kb == 0)
                )

            # Columns must be fully valid across the C->V crossing: the boundary
            # transports the full box and rejects a runtime-valued column extent.
            # The ragged tile's padding is cut off by the push below.
            mm_acc = pl.set_validshape(mm_acc, MATMUL_GROUP_ROWS, FUSED_VOCAB_TILE)

            # WORKAROUND -- revert once a subview of a pl.aiv_shard result is legal
            # (filed against ptoas). The shard cannot be sliced, so it is stored whole
            # to GM and the per-owner pushes read slices back out.
            # Race-free by construction: UP_DOWN gives each lane a disjoint row band
            # that it both writes and reads, guarded by the SHARD_ROWS assert above.
            for aiv_id in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.UP_DOWN):
                mm_shard = pl.aiv_shard(mm_acc)
                lane_r0 = aiv_id * SHARD_ROWS
                logits_shards[
                    lane_r0 : lane_r0 + SHARD_ROWS, mm_o0 : mm_o0 + FUSED_VOCAB_TILE
                ] = mm_shard
                for owner_slot in pl.range(OWNERS_PER_LANE):
                    owner_tp = aiv_id * OWNERS_PER_LANE + owner_slot
                    src_r0 = aiv_id * SHARD_ROWS + owner_slot * MAX_LOGIT_ROWS
                    # valid_shape rather than a narrower box: the box has to stay
                    # static for the tile type, and a full-width push of the ragged
                    # tile would run past this rank's slice of the shared window
                    # into the next rank's. pl.min is inlined instead of reusing
                    # mm_valid_n because a named cube-side scalar cannot be
                    # materialized inside the AIV function.
                    pld.tensor.remote_store(
                        pl.slice(
                            logits_shards,
                            [MAX_LOGIT_ROWS, FUSED_VOCAB_TILE],
                            [src_r0, mm_o0],
                            valid_shape=[
                                MAX_LOGIT_ROWS,
                                pl.min(VOCAB_PER_TP - mm_o0, FUSED_VOCAB_TILE),
                            ],
                        ),
                        logits_window,
                        group_base + owner_tp,
                        [0, vocab_base + mm_o0],
                    )

        # Notify folded into the push: each block signals every peer after its own
        # stores, so a peer sees FUSED_LM_HEAD_CORES notifies per source per epoch.
        for aiv_id in pl.split_aiv(AIV_LANES, mode=pl.SplitMode.NONE):
            for notify_pair in pl.range(OWNER_PAIRS):
                owner_tp = notify_pair * AIV_LANES + aiv_id
                if owner_tp != tp_rank:
                    pld.system.notify(
                        target=logits_done,
                        peer=group_base + owner_tp,
                        offsets=[tp_rank, 0],
                        value=1,
                        op=pld.NotifyOp.AtomicAdd,
                    )

    # Keep the blocking wait behind the local logits push.
    with pl.at(
        level=pl.Level.CORE_GROUP, name_hint="lm_head_combine_wait", allow_early_resolve=True, deps=[_push_tid]
    ) as _cwait_tid:
        for src_tp in pl.range(TP_SIZE):
            if src_tp != tp_rank:
                pld.system.wait(
                    signal=logits_done,
                    offsets=[src_tp, 0],
                    expected=pl.cast(done_epoch * FUSED_LM_HEAD_CORES, pl.INT32),
                    cmp=pld.WaitCmp.Ge,
                )

    # Assemble full-vocabulary logits, same vocab-tile split. deps on _cwait_tid for
    # the peers' stores; our own tiles ride the local RAW edge on logits_window.
    with pl.spmd(
        LOGITS_COMM_BLOCKS, name_hint="lm_head_combine_gather", allow_early_resolve=True, deps=[_cwait_tid]
    ) as _gather_tid:
        gblk = pl.tile.get_block_idx()
        running_scores = pl.tile.full([COMBINE_ROW_TILE, SAMPLED_IDS_PAD], dtype=pl.FP32, value=-3.4028234663852886e38)
        running_indices = pl.tile.full([COMBINE_ROW_TILE, SAMPLED_IDS_PAD], dtype=pl.INT32, value=1073741824)
        for src_tp in pl.range(TP_SIZE):
            src_vocab_base = src_tp * VOCAB_PER_TP
            for ob in pl.range(gblk, N_LOGITS_COMM_TILES, LOGITS_COMM_BLOCKS):
                o0 = ob * LOGITS_COMM_TILE
                lo = src_vocab_base + o0
                scores = pl.load(logits_window, [0, lo], [COMBINE_ROW_TILE, LOGITS_COMM_TILE], valid_shape=[MAX_LOGIT_ROWS, LOGITS_COMM_TILE])
                pl.store(scores, [0, lo], logits)
                scores = pl.fillpad(scores, pad_value=pl.PadValue.zero)
                scores = pl.set_validshape(scores, COMBINE_ROW_TILE, LOGITS_COMM_TILE)
                reduce_tmp = pl.create_tile([COMBINE_ROW_TILE, LOGITS_COMM_TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                tile_maxima = pl.row_max(scores, reduce_tmp)
                tile_indices = pl.row_argmax(scores, reduce_tmp)
                score_zeros = pl.tile.full([COMBINE_ROW_TILE, SAMPLED_IDS_PAD], dtype=pl.FP32, value=0.0)
                index_zeros = pl.tile.full([COMBINE_ROW_TILE, SAMPLED_IDS_PAD], dtype=pl.INT32, value=0)
                candidate_scores = pl.row_expand_add(score_zeros, tile_maxima)
                candidate_indices = pl.add(pl.row_expand_add(index_zeros, tile_indices), pl.cast(lo, pl.INT32))
                newer = pl.cmp(candidate_scores, running_scores, cmp_type=4)
                select_tmp = pl.create_tile([COMBINE_ROW_TILE, SAMPLED_IDS_PAD], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
                running_indices = pl.sel(newer, candidate_indices, running_indices, select_tmp)
                running_scores = pl.maximum(running_scores, candidate_scores)

            if LOGITS_COMM_TAIL != 0:
                if gblk == LOGITS_TAIL_BLOCK:
                    tail_o0 = N_LOGITS_COMM_TILES * LOGITS_COMM_TILE
                    tl = src_vocab_base + tail_o0
                    tail_scores = pl.load(logits_window, [0, tl], [COMBINE_ROW_TILE, LOGITS_COMM_TAIL], valid_shape=[MAX_LOGIT_ROWS, LOGITS_COMM_TAIL])
                    pl.store(tail_scores, [0, tl], logits)
                    tail_scores = pl.fillpad(tail_scores, pad_value=pl.PadValue.zero)
                    tail_scores = pl.set_validshape(tail_scores, COMBINE_ROW_TILE, LOGITS_COMM_TAIL)
                    tail_reduce_tmp = pl.create_tile([COMBINE_ROW_TILE, LOGITS_COMM_TAIL], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
                    tail_tile_maxima = pl.row_max(tail_scores, tail_reduce_tmp)
                    tail_tile_indices = pl.row_argmax(tail_scores, tail_reduce_tmp)
                    tail_score_zeros = pl.tile.full([COMBINE_ROW_TILE, SAMPLED_IDS_PAD], dtype=pl.FP32, value=0.0)
                    tail_index_zeros = pl.tile.full([COMBINE_ROW_TILE, SAMPLED_IDS_PAD], dtype=pl.INT32, value=0)
                    tail_candidate_scores = pl.row_expand_add(tail_score_zeros, tail_tile_maxima)
                    tail_candidate_indices = pl.add(pl.row_expand_add(tail_index_zeros, tail_tile_indices), pl.cast(tl, pl.INT32))
                    tail_newer = pl.cmp(tail_candidate_scores, running_scores, cmp_type=4)
                    tail_select_tmp = pl.create_tile([COMBINE_ROW_TILE, SAMPLED_IDS_PAD], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
                    running_indices = pl.sel(tail_newer, tail_candidate_indices, running_indices, tail_select_tmp)
                    running_scores = pl.maximum(running_scores, tail_candidate_scores)

        partial_row = gblk * MAX_LOGIT_ROWS
        running_scores = pl.set_validshape(running_scores, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD)
        pl.store(running_scores, [partial_row, 0], greedy_scores)
        running_indices = pl.set_validshape(running_indices, MAX_LOGIT_ROWS, SAMPLED_IDS_PAD)
        pl.store(running_indices, [partial_row, 0], greedy_indices)

    # Every local wait has observed all current-round peer notifies before the
    # logits gather can complete. Clear only this rank's counters so a retained
    # CommDomain can safely reuse the fixed done_epoch on the next forward.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="lm_head_signal_clear") as s6_lm_cleanup_tid:
        _completion_anchor = pl.read(logits, [0, 0])
        zero = pl.cast(0, pl.INT32)
        for src_tp in pl.range(TP_SIZE):
            pl.write(hidden_done, [src_tp, 0], zero)
            pl.write(logits_done, [src_tp, 0], zero)
    return logits, s6_lm_cleanup_tid

@pl.jit.inline(auto_scope=False)
def lm_head_with_sampling(
    hidden_states: pl.Tensor[[T_DYN, D], pl.BF16],
    lm_head_weight: pl.Tensor[[VOCAB_PER_TP, D], pl.BF16],
    logit_row_indices: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_temperatures: pl.Tensor[[MAX_LOGIT_ROWS], pl.FP32],
    sampling_top_ks: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_seeds: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    sampling_positions: pl.Tensor[[MAX_LOGIT_ROWS], pl.INT32],
    logits: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32]],
    sampled_ids: pl.Out[pl.Tensor[[MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], pl.INT32]],
    hidden_window: pld.DistributedTensor[[GROUP_LOGIT_ROWS, D], pl.BF16],
    hidden_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    logits_window: pld.DistributedTensor[[MAX_LOGIT_ROWS, VOCAB], pl.FP32],
    logits_done: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
    group_base: pl.Scalar[pl.INT32],
    tp_rank: pl.Scalar[pl.INT32],
    done_epoch: pl.Scalar[pl.INT32],
):
    """Project logits and sample tokens in one opaque L2 entry."""
    greedy_scores = pl.create_tensor([LOGITS_COMM_BLOCKS * MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], dtype=pl.FP32)
    greedy_indices = pl.create_tensor([LOGITS_COMM_BLOCKS * MAX_LOGIT_ROWS, SAMPLED_IDS_PAD], dtype=pl.INT32)
    _s6_logits, s6_lm_cleanup_tid = _lm_head_with_greedy_partials(
        hidden_states,
        lm_head_weight,
        logit_row_indices,
        logits,
        hidden_window,
        hidden_done,
        logits_window,
        logits_done,
        group_base,
        tp_rank,
        done_epoch,
        greedy_scores, greedy_indices,
    )
    sampled_ids = sample_from_greedy_partials(
        logits,
        sampling_temperatures,
        sampling_top_ks,
        sampling_seeds,
        sampling_positions,
        sampled_ids,
        greedy_scores, greedy_indices,
    )
    return logits, sampled_ids, s6_lm_cleanup_tid
