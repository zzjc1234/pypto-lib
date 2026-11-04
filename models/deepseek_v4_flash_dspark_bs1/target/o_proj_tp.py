# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys as _sys

import pypto.language as pl

import pypto.language.distributed as pld

from .config import FLASH as M, DECODE_BATCH, DECODE_SEQ, INT8_SCALE_MAX, INT8_AMAX_EPS, TP

B = DECODE_BATCH

S = DECODE_SEQ

T = B * S

D = M.hidden_size

H = M.num_attention_heads

HEAD_DIM = M.head_dim

O_LORA = M.o_lora_rank

O_GROUPS = M.o_groups

HEADS_PER_GROUP = H // O_GROUPS

O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

N_RANKS = TP

A_K_TILE = 256  # proj_a cube K frag

MM_T_TILE = 16

T_PAD = ((T + MM_T_TILE - 1) // MM_T_TILE) * MM_T_TILE

B_K_TILE = 256  # proj_b_mm cube K frag

def _parse_int_argv(name, default):
    for i, tok in enumerate(_sys.argv):
        if tok == name and i + 1 < len(_sys.argv):
            return int(_sys.argv[i + 1])
        if tok.startswith(f"{name}="):
            return int(tok.split("=", 1)[1])
    return default

QUANT_CHUNKS = _parse_int_argv("--quant-chunks", 1)

PUT_CHUNK_COLS = _parse_int_argv("--put-chunk-cols", D)

PUT_PIPELINE = any(t == "--put-pipeline" for t in _sys.argv)

QUANT_COL_TILE = O_LORA // QUANT_CHUNKS

assert O_LORA % QUANT_CHUNKS == 0, "--quant-chunks must divide O_LORA"

PROJ_A_MM_N_TILE = 64  # proj_a cube N frag; O_LORA // this = 16 blocks

PROJ_B_D_TILE = 256  # proj_b_mm D chunk per task; D // this = 16 blocks

PROJ_B_MM_N_TILE = 256

PRE_SYNC = any(t == "--pre-sync" for t in _sys.argv)

EVICT_L2 = any(t == "--evict-l2" for t in _sys.argv)

PRE_SYNC_PEERS = N_RANKS if PRE_SYNC else 0

REDUCE_LANE_ROWS = N_RANKS * T_PAD

REDUCE_WINDOW_ROWS = 2 * REDUCE_LANE_ROWS

SCALE_LANE_ROWS = N_RANKS * T

SCALE_WINDOW_ROWS = 2 * SCALE_LANE_ROWS

DEQUANT_N_TILE = 1024  # D // this = 4 dequant blocks

REDUCE_D_TILE = 1024  # D // this = 4 reduce blocks

QUANT_ROW_TILE = max(8, T)

assert O_GROUPS == N_RANKS, f"TP-by-group needs one group per rank: O_GROUPS={O_GROUPS}, TP={N_RANKS}"

@pl.jit.inline
def _quant(
    o_r_pad: pl.Tensor[[T_PAD, O_LORA], pl.FP32],
    o_r_i8_pad: pl.Tensor[[T_PAD, O_LORA], pl.INT8],
    act_scale_col: pl.Tensor[[T, 1], pl.FP32],
    pa_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """Row amax over the full O_LORA width, then an INT8 quantize of one chunk.

    Blocks of one spmd cannot read each other's partials, so a chunked amax would
    need a second task to combine them. Re-reading every chunk instead keeps the
    quantize a single task: `max` is associative, so each block lands on the same
    amax, and the extra reads cost less than the task edge and its GM round trip.
    """
    with pl.spmd(QUANT_CHUNKS, name_hint="quant", deps=[pa_tid], allow_early_resolve=True) as q_tid:
        qb = pl.tile.get_block_idx()
        q0 = qb * QUANT_COL_TILE
        g_amax = pl.full([1, QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for pc in pl.unroll(QUANT_CHUNKS):
            p0 = pc * QUANT_COL_TILE
            p_abs = pl.abs(o_r_pad[0:QUANT_ROW_TILE, p0 : p0 + QUANT_COL_TILE])
            g_amax = pl.maximum(g_amax, pl.reshape(pl.row_max(p_abs), [1, QUANT_ROW_TILE]))
        g_scale_num = pl.full([1, QUANT_ROW_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
        g_sq_row = pl.div(g_scale_num, g_amax)
        # Every block folds to the same scale; one block records it.
        if qb == 0:
            sc_row = pl.recip(g_sq_row)
            scale_carrier = pl.reshape(sc_row, [QUANT_ROW_TILE, 1])
            act_scale_col[0:T, 0:1] = pl.set_validshape(scale_carrier, T, 1)
        g_sq_col = pl.reshape(g_sq_row, [QUANT_ROW_TILE, 1])
        oc_q = o_r_pad[0:QUANT_ROW_TILE, q0 : q0 + QUANT_COL_TILE]
        oq_scaled = pl.row_expand_mul(oc_q, g_sq_col)
        oq_i32 = pl.cast(oq_scaled, target_type=pl.INT32, mode="rint")
        oq_half = pl.cast(oq_i32, target_type=pl.FP16, mode="round")
        oq_i8 = pl.cast(oq_half, target_type=pl.INT8, mode="trunc", saturation_mode="on")
        o_r_i8_pad[0:T, q0 : q0 + QUANT_COL_TILE] = pl.set_validshape(oq_i8, T, QUANT_COL_TILE)
    return q_tid

@pl.jit.inline
def _proj_chain(
    o_packed: pl.Tensor[[O_GROUPS * T, O_GROUP_IN], pl.BF16],
    wo_a_g: pl.Tensor[[1, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_g: pl.Tensor[[D, O_LORA], pl.INT8],
    o_r_pad: pl.Tensor[[T_PAD, O_LORA], pl.FP32],
    o_r_i8_pad: pl.Tensor[[T_PAD, O_LORA], pl.INT8],
    act_scale_col: pl.Tensor[[T, 1], pl.FP32],
    partial: pl.Tensor[[T_PAD, D], pl.INT32],
    row_base_o: pl.Scalar[pl.INDEX],
    upstream: pl.Scalar[pl.TASK_ID],
    pad_tid: pl.Scalar[pl.TASK_ID],
) -> pl.Scalar[pl.TASK_ID]:
    """One group's proj_a -> quant -> proj_b chain; returns proj_b's TaskId."""
    with pl.spmd(
        O_LORA // PROJ_A_MM_N_TILE, name_hint="proj_a_mm", deps=[upstream], allow_early_resolve=True
    ) as pa_tid:
        nf = pl.tile.get_block_idx()
        n0 = nf * PROJ_A_MM_N_TILE
        acc_a = pl.create_tensor([1, MM_T_TILE, PROJ_A_MM_N_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, O_GROUP_IN // A_K_TILE, stage=2):
            k0 = kb * A_K_TILE
            xa_k_chunk = pl.slice(
                o_packed, [MM_T_TILE, A_K_TILE], [row_base_o, k0], valid_shape=[T, A_K_TILE]
            )
            wa_k_chunk = wo_a_g[0:1, n0 : n0 + PROJ_A_MM_N_TILE, k0 : k0 + A_K_TILE]
            acc_a = pl.matmul_acc(acc_a, xa_k_chunk, wa_k_chunk, b_trans=True, init_cond=(kb == 0))
        # acc_a is 3D (wo_a_g keeps its group axis), which subscript-write cannot express.
        o_r_pad = pl.assemble(o_r_pad, acc_a, [0, n0])

    q_tid = _quant(o_r_pad, o_r_i8_pad, act_scale_col, pa_tid)

    with pl.spmd(
        D // PROJ_B_D_TILE,
        name_hint="proj_b_mm",
        deps=[q_tid, pad_tid],
        allow_early_resolve=True,
        optimizations=[pl.cross_core_slot(slot_num=2)],
    ) as pb_tid:
        dc = pl.tile.get_block_idx()
        d0 = dc * PROJ_B_D_TILE
        for nf in pl.range(PROJ_B_D_TILE // PROJ_B_MM_N_TILE):
            n0 = d0 + nf * PROJ_B_MM_N_TILE
            acc_b = pl.create_tensor([MM_T_TILE, PROJ_B_MM_N_TILE], dtype=pl.INT32)
            for kb in pl.pipeline(0, O_LORA // B_K_TILE, stage=2):
                k0 = kb * B_K_TILE
                acc_b = pl.matmul_acc(
                    acc_b,
                    o_r_i8_pad[:, k0 : k0 + B_K_TILE],
                    wo_b_g[n0 : n0 + PROJ_B_MM_N_TILE, k0 : k0 + B_K_TILE],
                    b_trans=True,
                    init_cond=(kb == 0),
                )
            partial[0:MM_T_TILE, n0 : n0 + PROJ_B_MM_N_TILE] = acc_b
    return pb_tid

@pl.jit.inline
def o_proj_tp_core(
    o_packed: pl.Tensor[[O_GROUPS * T, O_GROUP_IN], pl.BF16],
    upstream: pl.Scalar[pl.TASK_ID],
    wo_a_shard: pl.Tensor[[1, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_shard: pl.Tensor[[D, O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    attn_out: pl.Tensor[[T, D], pl.BF16],
    reduce_window: pld.DistributedTensor[[REDUCE_WINDOW_ROWS, D], pl.FP32],
    scale_window: pld.DistributedTensor[[SCALE_WINDOW_ROWS, 1], pl.FP32],
    reduce_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    sync_signal: pld.DistributedTensor[[N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    # 1-based call id; the signals are monotonic within a dispatch, so waits use
    # `>= oproj_epoch` and `clear_oproj_signals` resets them once at the end.
    oproj_epoch: pl.Scalar[pl.INT32],
):
    """Card `my_rank` runs group `my_rank`'s chain, then all-reduces its partial.

    The dequantized partial is published as FP32 rather than the INT32 accumulator
    because each group carries its own activation scale, which cannot factor out
    of the cross-rank sum. `wo_b_scale` is a shared per-channel factor and is
    applied once, after the sum, so the arithmetic matches a single card's g loop.
    """
    o_r_pad = pl.create_tensor([T_PAD, O_LORA], dtype=pl.FP32)
    o_r_i8_pad = pl.create_tensor([T_PAD, O_LORA], dtype=pl.INT8)
    act_scale_col = pl.create_tensor([T, 1], dtype=pl.FP32)
    partial = pl.create_tensor([T_PAD, D], dtype=pl.INT32)
    part_f32 = pl.create_tensor([T, D], dtype=pl.FP32)
    barrier_out = pl.array.create(1, pl.TASK_ID)

    lane_base = pl.cast(((oproj_epoch - 1) % 2) * REDUCE_LANE_ROWS, pl.INDEX)
    scale_lane_base = pl.cast(((oproj_epoch - 1) % 2) * SCALE_LANE_ROWS, pl.INDEX)
    my_row = lane_base + pl.cast(my_rank, pl.INDEX) * T_PAD
    my_scale_row = scale_lane_base + pl.cast(my_rank, pl.INDEX) * T
    row_base_o = pl.cast(my_rank, pl.INDEX) * T
    wo_b_scale_2d = pl.reshape(wo_b_scale, [1, D])

    with pl.manual_scope():
        # Optional leading barrier, anchored on the actual attention producer.
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="oproj_pre_sync", deps=[upstream]) as _sync_tid:
            _sync_anchor = pl.read(o_packed, [0, 0])
            sync_peers = PRE_SYNC_PEERS
            for peer in pl.range(sync_peers):
                pld.system.notify(
                    target=sync_signal,
                    peer=peer,
                    offsets=[my_rank, 0],
                    value=1,
                    op=pld.NotifyOp.AtomicAdd,
                )
            for src in pl.range(sync_peers):
                pld.system.wait(
                    signal=sync_signal,
                    offsets=[src, 0],
                    expected=oproj_epoch,
                    cmp=pld.WaitCmp.Ge,
                )

        # The padding rows carry no data and depend on nothing, so they are zeroed
        # off the proj_a -> quant -> proj_b chain rather than inside quant, where the
        # fill is the same [T_PAD - T, O_LORA] element count as the real quantize.
        with pl.at(
            level=pl.Level.CORE_GROUP, name_hint="quant_pad_zero", deps=[_sync_tid], allow_early_resolve=True
        ) as pad_tid:
            zero_half = pl.full([T_PAD - T, O_LORA], dtype=pl.FP16, value=0.0)
            zero_i8 = pl.cast(zero_half, target_type=pl.INT8, mode="trunc", saturation_mode="on")
            o_r_i8_pad[T:T_PAD, 0:O_LORA] = zero_i8

        pb_tid = _proj_chain(
            o_packed,
            wo_a_shard,
            wo_b_shard,
            o_r_pad,
            o_r_i8_pad,
            act_scale_col,
            partial,
            row_base_o,
            upstream,
            pad_tid,
        )

        # Dequantize before the wire, so what crosses it is addable. `wo_b_scale` is
        # per-channel and shared across ranks, so it distributes over the cross-rank
        # sum and rides this pass too -- the band then holds the finished value and
        # nothing is left to scale on the far side.
        with pl.spmd(
            D // DEQUANT_N_TILE, name_hint="oproj_dequant", deps=[pb_tid], allow_early_resolve=True
        ) as dq_tid:
            dq0 = pl.tile.get_block_idx() * DEQUANT_N_TILE
            p_f32 = pl.cast(partial[0:T, dq0 : dq0 + DEQUANT_N_TILE], target_type=pl.FP32, mode="none")
            if T < 8:
                for row in pl.range(T):
                    p_scale = pl.read(act_scale_col, [row, 0])
                    partial_row = p_f32[row : row + 1, :]
                    scaled_row = pl.mul(partial_row, p_scale)
                    channel_row = pl.col_expand_mul(scaled_row, wo_b_scale_2d[0:1, dq0 : dq0 + DEQUANT_N_TILE])
                    part_f32[row : row + 1, dq0 : dq0 + DEQUANT_N_TILE] = channel_row
            else:
                p_act = pl.row_expand_mul(p_f32, act_scale_col[0:T, 0:1])
                part_f32[0:T, dq0 : dq0 + DEQUANT_N_TILE] = pl.col_expand_mul(
                    p_act, wo_b_scale_2d[0:1, dq0 : dq0 + DEQUANT_N_TILE]
                )

        with pl.spmd(N_RANKS, name_hint="oproj_publish", deps=[dq_tid]) as publish_tid:
            peer = pl.tile.get_block_idx()
            pld.tensor.put(
                dst=reduce_window,
                peer=peer,
                src=part_f32,
                dst_offsets=[my_row, 0],
                src_offsets=[0, 0],
                shape=[T, D],
                chunk_rows=T,
                chunk_cols=PUT_CHUNK_COLS,
                pipeline=PUT_PIPELINE,
            )
            pld.system.notify(
                target=reduce_signal,
                peer=peer,
                offsets=[my_rank, 0],
                value=1,
                op=pld.NotifyOp.AtomicAdd,
            )

        # Split from the push so the notify rides the push scope's program order
        # and only the wait holds a core group.
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="oproj_barrier",
            deps=[pb_tid],
            allow_early_resolve=True,
        ) as barrier_tid:
            for src in pl.range(N_RANKS):
                pld.system.wait(
                    signal=reduce_signal,
                    offsets=[src, 0],
                    expected=oproj_epoch,
                    cmp=pld.WaitCmp.Ge,
                )

        # Carry the all-peer completion fence outside the manual scope.
        barrier_out[0] = barrier_tid

    # The reduce is the sole attn_out writer and must sit OUTSIDE the manual scope:
    # manual_scope suppresses auto-dep, so a consumer of attn_out in the caller
    # (hc_post, in the full layer) would carry no edge to this write and would read
    # the buffer while the reduce is still filling it. The barrier's TaskId rides an
    # Array[TASK_ID], the sanctioned way out of a scope that has closed.
    # One block per D tile carrying all T rows: [1, REDUCE_D_TILE] blocks make this
    # 64 tasks whose per-task fixed cost dominates 2 KB of loads.
    with pl.spmd(
        D // REDUCE_D_TILE, name_hint="oproj_reduce", deps=[barrier_out[0]], allow_early_resolve=True
    ) as _reduce_tid:
        d0 = pl.tile.get_block_idx() * REDUCE_D_TILE
        acc = pl.tile.full([T, REDUCE_D_TILE], dtype=pl.FP32, value=0.0)
        for source_rank in pl.range(N_RANKS):
            source_row = lane_base + source_rank * T_PAD
            partial_rank = pl.load(reduce_window, [source_row, d0], [T, REDUCE_D_TILE])
            acc = pl.add(acc, partial_rank)
        narrowed = pl.cast(acc, target_type=pl.BF16, mode="rint")
        pl.store(narrowed, [0, d0], attn_out)

    return attn_out
