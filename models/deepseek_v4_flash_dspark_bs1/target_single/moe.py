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

from .config import FLASH as M, MOE_TOKENS, TP

from .expert_routed import IDX_PAD, N_BANK, N_SLOTS_B, RECV_MAX, SHARED_EID, SH_SLOT

from .expert_routed_persistent_balanced import WORK_SLOTS_PAD, expert_routed_persistent_planned

from .gate import gate, ROUTE_ROW_PAD

from .hc_post import hc_post

from .hc_pre import hc_pre

T = MOE_TOKENS

D = M.hidden_size

TOPK = M.num_experts_per_tok

VOCAB = M.vocab_size

HC_MULT = M.hc_mult

MIX_HC = M.mix_hc

HC_DIM = M.hc_dim

MOE_INTER = M.moe_intermediate_size // TP  # this rank's slice of the intermediate dim

N_EXPERTS = M.n_routed_experts

N_ROUTES = T * TOPK

N_RANKS = TP

T_PAD = ((T + 15) // 16) * 16

REDUCE_LANE_ROWS = N_RANKS * T_PAD

REDUCE_WINDOW_ROWS = 2 * REDUCE_LANE_ROWS

REDUCE_D_TILE = 512  # D // this = 8 reduce blocks, each summing [T, 512] tiles

@pl.jit.inline
def clear_moe_signals(
    completion_anchor: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
):
    """Clear this rank's all-reduce counters after its final MoE completes."""
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_signal_clear", allow_early_resolve=True):
        # The final MoE output depends on this rank observing every peer's final
        # publish notify, so no peer can issue another one in this forward.
        _completion_anchor = pl.read(completion_anchor, [0, 0, 0])
        zero = pl.cast(0, pl.INT32)
        for src in pl.range(N_RANKS):
            pl.write(reduce_signal, [src, 0], zero)

@pl.jit.inline
def route_group(
    indices: pl.Tensor[[T, ROUTE_ROW_PAD], pl.INT32],
    weights: pl.Tensor[[T, ROUTE_ROW_PAD], pl.FP32],
    x_norm_i8: pl.Tensor[[T, D], pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1], pl.FP32],
    # compact per-slot outputs consumed by the routed expert / combine_local
    recv_x: pl.Tensor[[N_SLOTS_B, RECV_MAX, D], pl.INT8],
    recv_scale: pl.Tensor[[N_SLOTS_B, RECV_MAX], pl.FP32],
    recv_w: pl.Tensor[[N_SLOTS_B, RECV_MAX], pl.FP32],
    work_slot: pl.Tensor[[WORK_SLOTS_PAD], pl.INT32],
    work_rows: pl.Tensor[[WORK_SLOTS_PAD], pl.INT32],
    slot_expert: pl.Tensor[[N_SLOTS_B, IDX_PAD], pl.INT32],
    route_slot_row: pl.Tensor[[N_ROUTES, IDX_PAD], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """Group the T * TOPK routes by expert into one padded row tile per expert.

    Slot ``SH_SLOT`` is the shared expert: every active token, at weight 1.0.
    Routing fills at most ``T * TOPK == N_SLOTS`` slots, so it never claims it.
    """
    recv_x_flat = pl.reshape(recv_x, [N_SLOTS_B * RECV_MAX, D])
    expert_slot = pl.create_tensor([1, N_EXPERTS], dtype=pl.INT32, manual_dep=True)
    work_slot_2d = pl.reshape(work_slot, [1, WORK_SLOTS_PAD])
    work_rows_2d = pl.reshape(work_rows, [1, WORK_SLOTS_PAD])

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="route_table_init", allow_early_resolve=True) as init_tid:
        expert_slot[:, :] = pl.full([1, N_EXPERTS], dtype=pl.INT32, value=-1)
        work_slot_2d[:, :] = pl.full([1, WORK_SLOTS_PAD], dtype=pl.INT32, value=0)
        work_rows_2d[:, :] = pl.full([1, WORK_SLOTS_PAD], dtype=pl.INT32, value=0)
        slot_expert[:, :] = pl.full([N_SLOTS_B, IDX_PAD], dtype=pl.INT32, value=0)
        recv_scale[:, :] = pl.full([N_SLOTS_B, RECV_MAX], dtype=pl.FP32, value=0.0)
        recv_w[:, :] = pl.full([N_SLOTS_B, RECV_MAX], dtype=pl.FP32, value=0.0)
        route_slot_row[:, :] = pl.full([N_ROUTES, IDX_PAD], dtype=pl.INT32, value=0)
        # The shared expert's slot is static: same id every step.
        slot_expert[SH_SLOT : SH_SLOT + 1, :] = pl.full(
            [1, IDX_PAD], dtype=pl.INT32, value=SHARED_EID)

    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)

    # One core owns the slot table: adjacent scalar writes from several SPMD
    # cores can race through overlapping DMA units.
    with pl.spmd(1, name_hint="route_group_scalar", allow_early_resolve=True, deps=[init_tid]) as scalar_tid:
        scalar_core = pl.tile.get_block_idx()
        next_slot = pl.cast(0, pl.INDEX)
        for token in pl.range(scalar_core, active_tokens):
            token_scale = pl.read(x_norm_scale, [token, 0])
            for k in pl.range(TOPK):
                expert_id = pl.read(indices, [token, k])
                expert_col = pl.cast(expert_id, pl.INDEX)
                slot = pl.read(expert_slot, [0, expert_col])
                if slot < 0:
                    slot = pl.cast(next_slot, pl.INT32)
                    pl.write(expert_slot, [0, expert_col], slot)
                    pl.write(slot_expert, [next_slot, 0], expert_id)
                    pl.write(work_slot, [next_slot], slot)
                    next_slot = next_slot + 1
                slot_row = pl.cast(slot, pl.INDEX)
                count = pl.read(work_rows, [slot_row])
                count_col = pl.cast(count, pl.INDEX)
                pl.write(work_rows, [slot_row], pl.cast(count + 1, pl.INT32))
                pl.write(recv_scale, [slot_row, count_col], token_scale)
                pl.write(recv_w, [slot_row, count_col], pl.read(weights, [token, k]))
                route = token * TOPK + k
                pl.write(route_slot_row, [route, 0], pl.cast(slot * RECV_MAX + count, pl.INT32))
            # The shared expert takes this token ungated, at the same quant scale.
            pl.write(recv_scale, [SH_SLOT, token], token_scale)
            pl.write(recv_w, [SH_SLOT, token], pl.cast(1.0, pl.FP32))
        pl.write(work_slot, [next_slot], pl.cast(SH_SLOT, pl.INT32))
        pl.write(work_rows, [next_slot], pl.cast(active_tokens, pl.INT32))

    # One block per token -- it moves that token's TOPK routed rows -- plus one
    # for the shared expert. A 4 KB row is far too little to earn a block of its
    # own, and partitioning by token makes the source row the block index, so
    # only the destination needs a lookup.
    with pl.spmd(T + 1, name_hint="route_gather", allow_early_resolve=True, deps=[scalar_tid]) as _gather_tid:
        blk = pl.tile.get_block_idx()
        if blk < T:
            if blk < active_tokens:
                for k in pl.range(TOPK):
                    dst_row = pl.cast(pl.read(route_slot_row, [blk * TOPK + k, 0]), pl.INDEX)
                    recv_x_flat[dst_row : dst_row + 1, :] = x_norm_i8[blk : blk + 1, :]
        else:
            # The shared expert takes every token in order, so its rows are one
            # static contiguous tile -- no lookup, one copy. Rows past
            # active_tokens are never read by the planned expert work item.
            sh_base = pl.cast(SH_SLOT * RECV_MAX, pl.INDEX)
            recv_x_flat[sh_base : sh_base + T, :] = x_norm_i8[0:T, :]

@pl.jit.inline
def combine_local(
    recv_y: pl.Tensor[[N_SLOTS_B, RECV_MAX, D], pl.BF16],
    route_slot_row: pl.Tensor[[N_ROUTES, IDX_PAD], pl.INT32],
    ffn_partial: pl.Tensor[[T, D], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
):
    """ffn_partial[t] = recv_y[SH_SLOT, t] + Sigma_k recv_y[route_slot_row[t * TOPK + k]].

    The shared expert's row is just the first summand: it came out of the same
    kernel, in the same layout, so it needs no separate operand."""
    recv_y_flat = pl.reshape(recv_y, [N_SLOTS_B * RECV_MAX, D])

    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)

    with pl.spmd(T, name_hint="shared_routed", allow_early_resolve=True):
        t = pl.tile.get_block_idx()
        if t < active_tokens:
            sh_row = pl.cast(SH_SLOT * RECV_MAX + t, pl.INDEX)
            acc = pl.cast(recv_y_flat[sh_row : sh_row + 1, :], target_type=pl.FP32)
            for k in pl.range(TOPK):
                src_row = pl.cast(pl.read(route_slot_row, [t * TOPK + k, 0]), pl.INDEX)
                acc = pl.add(acc, pl.cast(recv_y_flat[src_row : src_row + 1, :], target_type=pl.FP32))
            ffn_partial[t : t + 1, :] = acc
        else:
            ffn_partial[t : t + 1, :] = pl.full([1, D], dtype=pl.FP32, value=0.0)

@pl.jit.inline
def all_reduce_ffn(
    ffn_partial: pl.Tensor[[T, D], pl.FP32],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
    reduce_window: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, D], pl.FP32],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    # 1-based MoE call id; reduce_signal is monotonic so waits use `>= moe_epoch`.
    moe_epoch: pl.Scalar[pl.INT32],
):
    """Sum every rank's intermediate-slice partial into a full-precision ffn_out."""
    lane_base = pl.cast(((moe_epoch - 1) % 2) * REDUCE_LANE_ROWS, pl.INDEX)
    my_row = lane_base + pl.cast(my_rank, pl.INDEX) * T_PAD

    with pl.spmd(N_RANKS, name_hint="moe_reduce_publish", allow_early_resolve=True) as publish_tid:
        peer = pl.tile.get_block_idx()
        pld.tensor.put(
            dst=reduce_window,
            peer=peer,
            src=ffn_partial,
            dst_offsets=[my_row, 0],
            src_offsets=[0, 0],
            shape=[T, D],
            chunk_rows=T,
            chunk_cols=D,
        )
        pld.system.notify(
            target=reduce_signal,
            peer=peer,
            offsets=[my_rank, 0],
            value=1,
            op=pld.NotifyOp.AtomicAdd,
        )

    # Split from the push so the notify rides the push scope's program order and
    # only the wait holds a core group.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="moe_reduce_barrier", allow_early_resolve=True, deps=[publish_tid]) as barrier_tid:
        for src in pl.range(N_RANKS):
            pld.system.wait(
                signal=reduce_signal,
                offsets=[src, 0],
                expected=moe_epoch,
                cmp=pld.WaitCmp.Ge,
            )

    # A rank's T rows are contiguous inside its band, so one block sums whole
    # [T, REDUCE_D_TILE] tiles -- eight of them, one per rank -- instead of
    # walking the band a single row at a time.
    with pl.spmd(D // REDUCE_D_TILE, name_hint="moe_reduce", allow_early_resolve=True, deps=[barrier_tid]) as _reduce_tid:
        d0 = pl.tile.get_block_idx() * REDUCE_D_TILE
        acc = pl.load(reduce_window, [lane_base, d0], [T, REDUCE_D_TILE])
        for src_rank in pl.range(1, N_RANKS):
            src_row = lane_base + src_rank * T_PAD
            acc = pl.add(acc, pl.load(reduce_window, [src_row, d0], [T, REDUCE_D_TILE]))
        reduced = pl.cast(acc, target_type=pl.BF16, mode="rint")
        pl.store(reduced, [0, d0], ffn_out)

@pl.jit.inline
def reduce_ffn_tp1(
    ffn_partial: pl.Tensor[[T, D], pl.FP32],
    ffn_out: pl.Tensor[[T, D], pl.BF16],
):
    """Single-rank path: the partial is already the full sum."""
    with pl.spmd(D // REDUCE_D_TILE, name_hint="moe_reduce", allow_early_resolve=True):
        d0 = pl.tile.get_block_idx() * REDUCE_D_TILE
        partial = ffn_partial[0:T, d0 : d0 + REDUCE_D_TILE]
        ffn_out[0:T, d0 : d0 + REDUCE_D_TILE] = pl.cast(partial, target_type=pl.BF16, mode="rint")

@pl.jit.inline(auto_scope=False)
def moe(
    # model inputs
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    hc_ffn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_ffn_scale: pl.Tensor[[3], pl.FP32],
    hc_ffn_base: pl.Tensor[[MIX_HC], pl.FP32],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    # Expert SHARED_EID of each bank is the shared expert (see route_group).
    routed_w1: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8],
    routed_w1_scale: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32],
    routed_w3: pl.Tensor[[N_BANK, MOE_INTER, D], pl.INT8],
    routed_w3_scale: pl.Tensor[[N_BANK, MOE_INTER], pl.FP32],
    routed_w2: pl.Tensor[[N_BANK, D, MOE_INTER], pl.INT8],
    routed_w2_scale: pl.Tensor[[N_BANK, D], pl.FP32],
    # final output
    x_next: pl.Out[pl.Tensor[[T, HC_MULT, D], pl.FP32]],
    # windows
    reduce_window: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, D], pl.FP32],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    # scalars last: runtime TaskArgs forbids a tensor arg after a scalar arg.
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    # 1-based MoE call id for the shared reduce window (distinct from layer_id).
    moe_epoch: pl.Scalar[pl.INT32],
) -> pl.Tensor[[T, HC_MULT, D], pl.FP32]:
    # Non-output intermediates allocate locally, in their producer's scope.
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post_ffn = pl.create_tensor([T, HC_MULT], dtype=pl.FP32, manual_dep=True)
    comb_ffn = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(
        x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base,
        x_mixed, post_ffn, comb_ffn,
    )

    x_norm_i8 = pl.create_tensor([T, D], dtype=pl.INT8)
    x_norm_scale = pl.create_tensor([T, 1], dtype=pl.FP32, manual_dep=True)
    indices = pl.create_tensor([T, ROUTE_ROW_PAD], dtype=pl.INT32)
    weights = pl.create_tensor([T, ROUTE_ROW_PAD], dtype=pl.FP32)
    gate(
        x_mixed, norm_w, gate_w, gate_bias,
        layer_id, num_tokens, tid2eid, input_ids,
        x_norm_i8, x_norm_scale, indices, weights,
    )

    recv_x = pl.create_tensor([N_SLOTS_B, RECV_MAX, D], dtype=pl.INT8)
    recv_scale = pl.create_tensor([N_SLOTS_B, RECV_MAX], dtype=pl.FP32)
    recv_w = pl.create_tensor([N_SLOTS_B, RECV_MAX], dtype=pl.FP32)
    work_slot = pl.create_tensor([WORK_SLOTS_PAD], dtype=pl.INT32)
    work_rows = pl.create_tensor([WORK_SLOTS_PAD], dtype=pl.INT32)
    slot_expert = pl.create_tensor([N_SLOTS_B, IDX_PAD], dtype=pl.INT32)
    route_slot_row = pl.create_tensor([N_ROUTES, IDX_PAD], dtype=pl.INT32)
    route_group(
        indices, weights, x_norm_i8, x_norm_scale,
        recv_x, recv_scale, recv_w, work_slot, work_rows, slot_expert, route_slot_row,
        num_tokens,
    )

    with pl.scope():
        recv_y = pl.create_tensor([N_SLOTS_B, RECV_MAX, D], dtype=pl.BF16)
        expert_routed_persistent_planned(
            recv_x, recv_scale, recv_w, work_slot, work_rows, slot_expert,
            routed_w1, routed_w1_scale, routed_w3, routed_w3_scale,
            routed_w2, routed_w2_scale,
            recv_y,
        )

        ffn_partial = pl.create_tensor([T, D], dtype=pl.FP32)
        combine_local(recv_y, route_slot_row, ffn_partial, num_tokens)

        ffn_out = pl.create_tensor([T, D], dtype=pl.BF16)
        if N_RANKS == 1:
            reduce_ffn_tp1(ffn_partial, ffn_out)
        else:
            all_reduce_ffn(ffn_partial, ffn_out, reduce_window, reduce_signal, my_rank, moe_epoch)

        hc_post(ffn_out, x_hc, post_ffn, comb_ffn, x_next)
    return x_next
