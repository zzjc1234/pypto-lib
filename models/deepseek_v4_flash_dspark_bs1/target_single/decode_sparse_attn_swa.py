# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl
from ..target.fp32_numerics import rne_exp_nonpositive, rne_exp_signed

from .config import FLASH as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE

ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")

B = DECODE_BATCH

S = DECODE_SEQ

T = B * S

H = M.num_attention_heads

HEAD_DIM = M.head_dim

ROPE_DIM = M.qk_rope_head_dim

HALF_ROPE = ROPE_DIM // 2

NOPE_DIM = M.nope_head_dim

WIN = M.sliding_window

SOFTMAX_SCALE = M.softmax_scale

O_GROUPS = M.o_groups

HEADS_PER_GROUP = H // O_GROUPS

H_LOCAL = HEADS_PER_GROUP

O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

H_TILE = 8               # == H_LOCAL: one card's head slice is one tile

QK_M_TILE = 8            # qk_pv real M rows == one card's heads

QK_M_BOX = 16

ATTN_K_TILE = 128

ROPE_TILE = 16

ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE

SPARSE_BLOCKS = 1        # the SWA window fits one attention K tile

PADDED_TOPK = SPARSE_BLOCKS * ATTN_K_TILE

assert WIN == BLOCK_SIZE, "a window spans at most two paged blocks only while WIN == BLOCK_SIZE"

assert WIN == ATTN_K_TILE, f"SWA decode expects WIN ({WIN}) == ATTN_K_TILE ({ATTN_K_TILE})"

@pl.jit.inline
def prepare_swa_output_rope(
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    rope_cos_il: pl.Out[pl.Tensor[[T, ROPE_DIM], pl.FP32]],
    rope_sin_signed: pl.Out[pl.Tensor[[T, ROPE_DIM], pl.FP32]],
    rope_swap_idx: pl.Out[pl.Tensor[[H_TILE, ROPE_DIM], pl.INT32]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_cs", allow_early_resolve=True) as rope_tid:
        swap_ones = pl.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        swap_range_i32 = pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32)
        swap_range = pl.cast(swap_range_i32, target_type=pl.FP32)
        swap_col = pl.col_expand_mul(swap_ones, swap_range)
        swap_half = pl.mul(swap_col, 0.5)
        swap_dup_i32 = pl.cast(swap_half, target_type=pl.INT32, mode="trunc")
        swap_dup_f = pl.cast(swap_dup_i32, target_type=pl.FP32)
        swap_lane = pl.sub(swap_col, pl.mul(swap_dup_f, 2.0))
        swap_next = pl.add(swap_col, 1.0)
        swap_stride = pl.mul(swap_lane, 2.0)
        swap_idx_f = pl.sub(swap_next, swap_stride)
        rope_swap_idx[:, :] = pl.cast(swap_idx_f, target_type=pl.INT32)

        cs_ones = pl.full([T, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=1.0)
        cs_range_i32 = pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32)
        cs_range = pl.cast(cs_range_i32, target_type=pl.FP32)
        cs_col = pl.col_expand_mul(cs_ones, cs_range)
        cs_half = pl.mul(cs_col, 0.5)
        cs_dup_i32 = pl.cast(cs_half, target_type=pl.INT32, mode="trunc")
        cs_dup_f = pl.cast(cs_dup_i32, target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))
        cs_sign_base = pl.sub(pl.mul(cs_lane, 2.0), 1.0)
        cs_sign = pl.neg(cs_sign_base)
        for cp in pl.range(HALF_ROPE // ROPE_TILE):
            cp_r0 = cp * ROPE_TILE
            cp_c0 = 2 * cp_r0
            cs_cos = pl.cast(freqs_cos[0:T, cp_r0 : cp_r0 + ROPE_TILE], target_type=pl.FP32)
            cs_sin = pl.cast(freqs_sin[0:T, cp_r0 : cp_r0 + ROPE_TILE], target_type=pl.FP32)
            cs_cos_dup = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
            cs_sin_dup = pl.gather(cs_sin, dim=-1, index=cs_dup_idx)
            cs_sin_signed = pl.mul(cs_sin_dup, cs_sign)
            rope_cos_il[0:T, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = cs_cos_dup
            rope_sin_signed[0:T, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = cs_sin_signed

@pl.jit.inline
def sparse_attn_swa_packed(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    swa_lens: pl.Tensor[[T], pl.INT32],
    sparse_bias: pl.Tensor[[T, PADDED_TOPK], pl.FP32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    o_packed: pl.Tensor[[O_GROUPS * T, O_GROUP_IN], pl.BF16],
    my_rank: pl.Scalar[pl.INT32],
    rope_cos_il: pl.Tensor[[T, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[H_TILE, ROPE_DIM], pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Sparse decode attention over the sliding window, and inverse RoPE, up to the packed head output.

    Split out so the output projection can be swapped for the TP-by-group form
    without duplicating the attention body. Returns merge_norm's TaskId, which the
    projection depends on.
    """
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    ori_kv_flat = pl.reshape(ori_kv, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    # SWA metadata already lowered each logical window row to a physical cache
    # slot. Current decode tokens must be inserted into ori_kv by the caller
    # before this function runs; there is no MTP overlay path here.

    # qk_pv writes per-tile (mi, li, oi) to GM; merge_norm reads them back. Not
    # fused on a2a3: the PV output (Acc) -> online rescale (Vec) needs an
    # unsupported tmov, and a [H_TILE, HEAD_DIM] carry overflows the Vec buffer.
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    head_base = pl.cast(my_rank, pl.INDEX) * H_LOCAL
    o_packed_heads = pl.reshape(o_packed, [O_GROUPS * T * HEADS_PER_GROUP, HEAD_DIM])
    sparse_blk_mi = pl.create_tensor([T * QK_M_BOX * SPARSE_BLOCKS, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([T * QK_M_BOX * SPARSE_BLOCKS, 1], dtype=pl.FP32)
    sparse_blk_li_flat = pl.reshape(sparse_blk_li, [1, T * QK_M_BOX * SPARSE_BLOCKS])
    sparse_blk_oi = pl.create_tensor([T * QK_M_BOX * SPARSE_BLOCKS, HEAD_DIM], dtype=pl.FP32)

    with pl.spmd(T, name_hint="qk_pv", allow_early_resolve=True) as qk_tid:
        qk_t = pl.tile.get_block_idx()
        qk_token_base = qk_t * QK_M_BOX * SPARSE_BLOCKS
        # A window is WIN consecutive positions and WIN == BLOCK_SIZE, so it is the
        # tail of block P plus the head of block P+1. Assemble both runs into ONE
        # contiguous L1 operand with two bulk DMAs: valid_shape carries the runtime
        # split without touching the tile's allocation, so the cube still sees a
        # single ATTN_K_TILE-column matmul and the KV never round-trips through GM.
        qk_slot0 = pl.max(pl.read(swa_indices, [qk_t, 0]), 0)
        qk_r = qk_slot0 % BLOCK_SIZE
        qk_head = WIN - qk_r
        qk_blk1 = pl.max(pl.read(swa_indices, [qk_t, qk_head % WIN]), 0)
        qk_len = pl.min(pl.max(pl.read(swa_lens, [qk_t]), 0), WIN)
        qk_kv = pl.create_l1([ATTN_K_TILE, HEAD_DIM], pl.BF16)
        qk_src0 = pl.cast(qk_slot0, pl.INDEX)
        qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [0, 0], [qk_src0, 0],
                              [ATTN_K_TILE, HEAD_DIM], valid_shape=[qk_head, HEAD_DIM])
        if qk_r > 0:
            qk_src1 = pl.cast(qk_blk1, pl.INDEX)
            qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_head, 0], [qk_src1, 0],
                                  [ATTN_K_TILE, HEAD_DIM], valid_shape=[qk_r, HEAD_DIM])
        qk_tail = WIN - qk_len
        if qk_tail > 0:
            # Cold start: rows past the visible prefix are allocated but not yet
            # written. Replicate the window's oldest visible slot over them --
            # that row is written by definition, whereas a run starting anywhere
            # else in the pool is no more initialized than the rows it replaces.
            # sparse_bias masks their scores, but a non-finite value would reach
            # the QK matmul and propagate through row_max before it applied.
            for qk_fill in pl.range(qk_tail):
                qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_len + qk_fill, 0],
                                      [qk_src0, 0], [1, HEAD_DIM])
        for qk_sb in pl.unroll(SPARSE_BLOCKS):
            qk_s0 = qk_sb * ATTN_K_TILE
            qk_bias_row = sparse_bias[qk_t : qk_t + 1, qk_s0 : qk_s0 + ATTN_K_TILE]

            # One head batch per card now, so there is no second batch to reuse
            # the resident KV tile against.
            for qk_hb in pl.pipeline(H_LOCAL // QK_M_TILE, stage=2):
                qk_h0 = qk_hb * QK_M_TILE
                qk_head_row = qk_t * H + head_base + qk_h0
                qk_q_tile = pl.slice(
                    q_flat, [QK_M_BOX, HEAD_DIM], [qk_head_row, 0],
                    valid_shape=[QK_M_TILE, HEAD_DIM])
                qk_raw = pl.matmul(qk_q_tile, qk_kv, b_trans=True, out_dtype=pl.FP32)
                qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)
                qk_scores = pl.col_expand_add(qk_scaled, qk_bias_row)
                qk_mi = pl.row_max(qk_scores)
                # Invalid lanes (NEG_INF bias, zero kv rows) exp to ~0; all-invalid
                # blocks die in the merge alpha/beta -- no mask multiply needed.
                qk_delta = pl.create_tensor([QK_M_BOX, ATTN_K_TILE], dtype=pl.FP32)
                qk_delta = pl.row_expand_sub(qk_scores, qk_mi)
                qk_exp = rne_exp_nonpositive(qk_delta, rows=QK_M_BOX, cols=ATTN_K_TILE)
                qk_sum64 = pl.add(qk_exp[0:QK_M_BOX, 0:64], qk_exp[0:QK_M_BOX, 64:128])
                qk_sum32 = pl.add(qk_sum64[0:QK_M_BOX, 0:32], qk_sum64[0:QK_M_BOX, 32:64])
                qk_sum16 = pl.add(qk_sum32[0:QK_M_BOX, 0:16], qk_sum32[0:QK_M_BOX, 16:32])
                qk_sum8 = pl.add(qk_sum16[0:QK_M_BOX, 0:8], qk_sum16[0:QK_M_BOX, 8:16])
                qk_sum8_t_physical = pl.transpose(qk_sum8, 0, 1)
                qk_sum8_t = pl.slice(qk_sum8_t_physical, [8, QK_M_BOX], [0, 0], valid_shape=[8, QK_M_TILE])
                qk_sum4_t = pl.add(qk_sum8_t[0:4, 0:QK_M_BOX], qk_sum8_t[4:8, 0:QK_M_BOX])
                qk_sum2_t = pl.add(qk_sum4_t[0:2, 0:QK_M_BOX], qk_sum4_t[2:4, 0:QK_M_BOX])
                qk_sum1_t = pl.add(qk_sum2_t[0:1, 0:QK_M_BOX], qk_sum2_t[1:2, 0:QK_M_BOX])
                qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
                qk_oi = pl.matmul(qk_exp_bf16, qk_kv, out_dtype=pl.FP32)
                # One head tile per card, so the whole (valid) accumulator stores
                # straight out -- slicing it to 8 rows would ask the cube for a
                # partial 16x16 box, which has no address.
                qk_row = qk_token_base + qk_sb * QK_M_BOX
                sparse_blk_mi[qk_row : qk_row + QK_M_BOX, 0 : 1] = qk_mi
                sparse_blk_li_flat[0:1, qk_row : qk_row + QK_M_BOX] = qk_sum1_t
                sparse_blk_oi[qk_row : qk_row + QK_M_BOX, 0 : HEAD_DIM] = qk_oi

    # Flatten the one-block SWA merge over token/head tiles into a single
    # 32-block grid, which fits in one AIV wave and avoids eight group-grid
    # submissions. Each block writes two output-projection groups using the
    # same contiguous per-group stores.
    with pl.spmd(T * (H_LOCAL // H_TILE), name_hint="merge_norm", deps=[qk_tid],
                 allow_early_resolve=True) as merge_tid:
        m_idx = pl.tile.get_block_idx()
        m_t = m_idx // (H_LOCAL // H_TILE)
        m_h_idx = m_idx - m_t * (H_LOCAL // H_TILE)
        m_h0 = m_h_idx * H_TILE
        m_blk_base = m_t * QK_M_BOX * SPARSE_BLOCKS + m_h0
        m_mi = sparse_blk_mi[m_blk_base : m_blk_base + H_TILE, 0:1]
        m_li = sparse_blk_li[m_blk_base : m_blk_base + H_TILE, 0:1]
        m_oi = sparse_blk_oi[m_blk_base : m_blk_base + H_TILE, 0:HEAD_DIM]

        m_h_global = head_base + m_h0
        n_sink = pl.reshape(attn_sink[m_h_global : m_h_global + H_TILE], [H_TILE, 1])
        n_sink_delta = pl.create_tensor([H_TILE, 1], dtype=pl.FP32)
        n_sink_delta = pl.sub(n_sink, m_mi)
        n_sink_exp = rne_exp_signed(n_sink_delta, rows=H_TILE, cols=1, safe_zero=True, negate=False)
        n_denom = pl.add(m_li, n_sink_exp)
        n_div_seed = pl.row_expand_div(m_oi, n_denom)[0:H_TILE, 0:HEAD_DIM]
        n_div_left_v = m_oi[:, 0:0 + 256]
        n_div_left_estimate = n_div_seed[:, 0:0 + 256]
        n_div_left_zero = pl.sub(n_div_left_v, n_div_left_v)
        n_div_left_expanded = pl.row_expand_add(n_div_left_zero, n_denom)
        n_div_left_a_split = pl.mul(n_div_left_estimate, 4097.0)
        n_div_left_a_delta = pl.sub(n_div_left_a_split, n_div_left_estimate)
        n_div_left_a_hi = pl.sub(n_div_left_a_split, n_div_left_a_delta)
        n_div_left_a_lo = pl.sub(n_div_left_estimate, n_div_left_a_hi)
        n_div_left_b_split = pl.mul(n_div_left_expanded, 4097.0)
        n_div_left_b_delta = pl.sub(n_div_left_b_split, n_div_left_expanded)
        n_div_left_b_hi = pl.sub(n_div_left_b_split, n_div_left_b_delta)
        n_div_left_b_lo = pl.sub(n_div_left_expanded, n_div_left_b_hi)
        n_div_left_product = pl.mul(n_div_left_estimate, n_div_left_expanded)
        n_div_left_high_product = pl.mul(n_div_left_a_hi, n_div_left_b_hi)
        n_div_left_error0 = pl.sub(n_div_left_high_product, n_div_left_product)
        n_div_left_high_low = pl.mul(n_div_left_a_hi, n_div_left_b_lo)
        n_div_left_error1 = pl.add(n_div_left_error0, n_div_left_high_low)
        n_div_left_low_high = pl.mul(n_div_left_a_lo, n_div_left_b_hi)
        n_div_left_error2 = pl.add(n_div_left_error1, n_div_left_low_high)
        n_div_left_low_product = pl.mul(n_div_left_a_lo, n_div_left_b_lo)
        n_div_left_error = pl.add(n_div_left_error2, n_div_left_low_product)
        n_div_left_residual_hi = pl.sub(n_div_left_v, n_div_left_product)
        n_div_left_residual = pl.sub(n_div_left_residual_hi, n_div_left_error)
        n_div_left_correction = pl.div(n_div_left_residual, n_div_left_expanded)
        n_div_left_fixed = pl.add(n_div_left_estimate, n_div_left_correction)
        n_div_right_v = m_oi[:, 256:256 + 256]
        n_div_right_estimate = n_div_seed[:, 256:256 + 256]
        n_div_right_zero = pl.sub(n_div_right_v, n_div_right_v)
        n_div_right_expanded = pl.row_expand_add(n_div_right_zero, n_denom)
        n_div_right_a_split = pl.mul(n_div_right_estimate, 4097.0)
        n_div_right_a_delta = pl.sub(n_div_right_a_split, n_div_right_estimate)
        n_div_right_a_hi = pl.sub(n_div_right_a_split, n_div_right_a_delta)
        n_div_right_a_lo = pl.sub(n_div_right_estimate, n_div_right_a_hi)
        n_div_right_b_split = pl.mul(n_div_right_expanded, 4097.0)
        n_div_right_b_delta = pl.sub(n_div_right_b_split, n_div_right_expanded)
        n_div_right_b_hi = pl.sub(n_div_right_b_split, n_div_right_b_delta)
        n_div_right_b_lo = pl.sub(n_div_right_expanded, n_div_right_b_hi)
        n_div_right_product = pl.mul(n_div_right_estimate, n_div_right_expanded)
        n_div_right_high_product = pl.mul(n_div_right_a_hi, n_div_right_b_hi)
        n_div_right_error0 = pl.sub(n_div_right_high_product, n_div_right_product)
        n_div_right_high_low = pl.mul(n_div_right_a_hi, n_div_right_b_lo)
        n_div_right_error1 = pl.add(n_div_right_error0, n_div_right_high_low)
        n_div_right_low_high = pl.mul(n_div_right_a_lo, n_div_right_b_hi)
        n_div_right_error2 = pl.add(n_div_right_error1, n_div_right_low_high)
        n_div_right_low_product = pl.mul(n_div_right_a_lo, n_div_right_b_lo)
        n_div_right_error = pl.add(n_div_right_error2, n_div_right_low_product)
        n_div_right_residual_hi = pl.sub(n_div_right_v, n_div_right_product)
        n_div_right_residual = pl.sub(n_div_right_residual_hi, n_div_right_error)
        n_div_right_correction = pl.div(n_div_right_residual, n_div_right_expanded)
        n_div_right_fixed = pl.add(n_div_right_estimate, n_div_right_correction)
        n_full = pl.concat(n_div_left_fixed, n_div_right_fixed)
        n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")

        m_rope = n_full[:, NOPE_DIM:HEAD_DIM]
        m_swapped = pl.gather(m_rope, dim=-1, index=rope_swap_idx[:, :])
        m_cos_il = rope_cos_il[m_t : m_t + 1, 0:ROPE_DIM]
        m_sin_signed = rope_sin_signed[m_t : m_t + 1, 0:ROPE_DIM]
        m_rope_cos = pl.col_expand_mul(m_rope, m_cos_il)
        m_swap_sin = pl.col_expand_mul(m_swapped, m_sin_signed)
        m_rot = pl.add(m_rope_cos, m_swap_sin)
        n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode="rint")

        for m_sg in pl.unroll(H_TILE // HEADS_PER_GROUP):
            m_src_h0 = m_sg * HEADS_PER_GROUP
            n_pack_row = pl.cast(my_rank, pl.INDEX) * T + m_t
            n_dst_head = n_pack_row * HEADS_PER_GROUP
            o_packed_heads[n_dst_head : n_dst_head + HEADS_PER_GROUP, 0:NOPE_DIM] = n_bf16[
                m_src_h0 : m_src_h0 + HEADS_PER_GROUP, 0:NOPE_DIM
            ]
            o_packed_heads[n_dst_head : n_dst_head + HEADS_PER_GROUP, NOPE_DIM:HEAD_DIM] = n_rope_bf16[
                m_src_h0 : m_src_h0 + HEADS_PER_GROUP, 0:ROPE_DIM
            ]

    # ========================================================================
    # Back-to-back grouped output projection (manual scope, PER-GROUP INT8 quant).
    #
    # Per-GROUP amax localizes the quant reduction to each O_LORA group (vs the
    # per-ROW-amax form, where a full-8192-row reduction is a hard barrier between
    # proj_a and proj_b), so the three stages PIPELINE per group with qwen3-style
    # fine-grained deps: proj_b[*, g] waits only on quant[g], which waits only on
    # proj_a[g, *] -- so proj_b's cube for group g runs while proj_a/quant of later
    # groups are still in flight (a genuine proj_a<->proj_b back-to-back GEMM).
    #
    # manual_scope SUPPRESSES auto-dep, so every edge is explicit: proj_a[g]
    # reads only its o_packed slab -> deps=[merge_tid]; quant[g] deps on group
    # g's proj_a task; proj_b depends directly on quant[g] and writes a disjoint
    # group partial. proj_b_act combines those partials with their group row scales,
    # applies the per-channel weight scale, and is the consolidated attn_out writer.
    # ========================================================================

    return merge_tid
