# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl

from .config import FLASH as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, DECODE_CMP_BLOCK_NUM, KV_CMP_MAX_BLOCKS

ORI_BLOCK_NUM_DYN = pl.dynamic('NATIVE_S_ORI_BLOCK_NUM_DYN')

CMP_BLOCK_NUM_DYN = pl.dynamic('NATIVE_S_CMP_BLOCK_NUM_DYN')

B = DECODE_BATCH

S = DECODE_SEQ

T = B * S

H = M.num_attention_heads

HEAD_DIM = M.head_dim

ROPE_DIM = M.qk_rope_head_dim

HALF_ROPE = ROPE_DIM // 2

NOPE_DIM = M.nope_head_dim

WIN = M.sliding_window

MAX_SEQ_LEN = M.max_position_embeddings

SOFTMAX_SCALE = M.softmax_scale

O_GROUPS = M.o_groups

HEADS_PER_GROUP = H // O_GROUPS

H_LOCAL = HEADS_PER_GROUP

O_GROUP_IN = HEADS_PER_GROUP * HEAD_DIM

COMPRESS_RATIO = 128

CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE // COMPRESS_RATIO

CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS

CMP_BLOCK_NUM = DECODE_CMP_BLOCK_NUM

NUM_QK_CORES = 24

H_TILE = 8               # == H_LOCAL: one card's head slice is one tile

QK_M_TILE = 8            # qk_pv real M rows == one card's heads

QK_M_BOX = 16

ATTN_K_TILE = 128

ROPE_TILE = 16

ROPE_INTERLEAVE_TILE = 2 * ROPE_TILE

CMP_CAPACITY = MAX_SEQ_LEN // COMPRESS_RATIO

CMP_TOPK = ((CMP_CAPACITY + ATTN_K_TILE - 1) // ATTN_K_TILE) * ATTN_K_TILE

MAX_SUPPORTED_SEQ = CMP_TOPK * COMPRESS_RATIO

CMP_BLOCKS_PER_REQ = (CMP_TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE

TOPK = WIN + CMP_TOPK    # cache-first window slots + the ratio-128 compressed tail

SPARSE_BLOCKS = max(2, (TOPK + ATTN_K_TILE - 1) // ATTN_K_TILE)

assert CMP_BLOCKS_PER_REQ <= CMP_MAX_BLOCKS, (
    f"compressed block table ({CMP_MAX_BLOCKS} blocks) must index the whole "
    f"{CMP_TOPK}-slot tail; MAX_SUPPORTED_SEQ={MAX_SUPPORTED_SEQ}")

assert B * CMP_BLOCKS_PER_REQ <= CMP_BLOCK_NUM, (
    f"compressed KV pool ({CMP_BLOCK_NUM} blocks) must hold B={B} requests x "
    f"{CMP_BLOCKS_PER_REQ} blocks; MAX_SUPPORTED_SEQ={MAX_SUPPORTED_SEQ}")

assert WIN == ATTN_K_TILE, f"HCA window tile requires WIN ({WIN}) == ATTN_K_TILE ({ATTN_K_TILE})"

assert SPARSE_BLOCKS == 1 + CMP_TOPK // ATTN_K_TILE, (
    f"qk_pv assembles block 0 from the window and block k >= 1 from compressed "
    f"tail columns [(k - 1) * ATTN_K_TILE, k * ATTN_K_TILE), got {SPARSE_BLOCKS}")

assert WIN == BLOCK_SIZE, "a window spans at most two paged blocks only while WIN == BLOCK_SIZE"

@pl.jit.inline
def prepare_hca_output_rope(
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    rope_cos_il: pl.Out[pl.Tensor[[T, ROPE_DIM], pl.FP32]],
    rope_sin_signed: pl.Out[pl.Tensor[[T, ROPE_DIM], pl.FP32]],
    rope_swap_idx: pl.Out[pl.Tensor[[H_TILE, ROPE_DIM], pl.INT32]],
):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rope_swap", allow_early_resolve=True):
        sw_col = pl.col_expand_mul(
            pl.full([H_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32), target_type=pl.FP32))
        sw_dup_f = pl.cast(pl.cast(pl.mul(sw_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        sw_lane = pl.sub(sw_col, pl.mul(sw_dup_f, 2.0))                                           # j%2
        rope_swap_idx[0:H_TILE, 0:ROPE_DIM] = pl.cast(
            pl.sub(pl.add(sw_col, 1.0), pl.mul(sw_lane, 2.0)), target_type=pl.INT32)              # j^1
    for cp in pl.spmd(HALF_ROPE // ROPE_TILE, name_hint="rope_cs", allow_early_resolve=True):
        cp_r0 = cp * ROPE_TILE
        cp_c0 = 2 * cp_r0
        cs_col = pl.col_expand_mul(
            pl.full([T, ROPE_INTERLEAVE_TILE], dtype=pl.FP32, value=1.0),
            pl.cast(pl.arange(0, [1, ROPE_INTERLEAVE_TILE], dtype=pl.INT32), target_type=pl.FP32))
        cs_dup_f = pl.cast(pl.cast(pl.mul(cs_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        cs_dup_idx = pl.cast(cs_dup_f, target_type=pl.INT32)                                      # j>>1
        cs_lane = pl.sub(cs_col, pl.mul(cs_dup_f, 2.0))                                           # j%2
        cs_sign = pl.neg(pl.sub(pl.mul(cs_lane, 2.0), 1.0))                                       # [+1,-1,...] (conjugate)
        cs_cos = pl.cast(freqs_cos[0:T, cp_r0 : cp_r0 + ROPE_TILE], target_type=pl.FP32)
        cs_sin = pl.cast(freqs_sin[0:T, cp_r0 : cp_r0 + ROPE_TILE], target_type=pl.FP32)
        rope_cos_il[0:T, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = pl.gather(cs_cos, dim=-1, index=cs_dup_idx)
        rope_sin_signed[0:T, cp_c0 : cp_c0 + ROPE_INTERLEAVE_TILE] = pl.mul(
            pl.gather(cs_sin, dim=-1, index=cs_dup_idx), cs_sign)

@pl.jit.inline
def sparse_attn_hca_packed(
    q: pl.Tensor[[T, H, HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T], pl.INT32],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    cmp_seq_lens: pl.Tensor[[T], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    freqs_cos: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[T, ROPE_DIM], pl.BF16],
    o_packed: pl.Tensor[[O_GROUPS * T, O_GROUP_IN], pl.BF16],
    my_rank: pl.Scalar[pl.INT32],
    rope_cos_il: pl.Tensor[[T, ROPE_DIM], pl.FP32],
    rope_sin_signed: pl.Tensor[[T, ROPE_DIM], pl.FP32],
    rope_swap_idx: pl.Tensor[[H_TILE, ROPE_DIM], pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    """Sparse decode attention over the compressed + window cache, and inverse RoPE, up to the packed head output.

    Split out so the output projection can be swapped for the TP-by-group form
    without duplicating the attention body. Returns merge_norm's TaskId, which the
    projection depends on.
    """
    ori_block_num = pl.tensor.dim(ori_kv, 0)
    cmp_block_num = pl.tensor.dim(cmp_kv, 0)
    ori_kv_flat = pl.reshape(ori_kv, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    cmp_kv_flat = pl.reshape(cmp_kv, [cmp_block_num * CMP_STORAGE_BLOCK_SIZE, HEAD_DIM])
    cmp_run_bases = pl.create_tensor([B, CMP_TOPK // ATTN_K_TILE], dtype=pl.INT32)
    for plan_b in pl.spmd(B, name_hint="hca_page_runs", allow_early_resolve=True):
        plan_last_t = plan_b * S + S - 1
        plan_length = pl.cast(pl.read(cmp_seq_lens, [plan_last_t]), pl.INDEX)
        plan_blocks = (plan_length + ATTN_K_TILE - 1) // ATTN_K_TILE
        plan_cols_i32 = pl.arange(0, [1, ATTN_K_TILE], dtype=pl.INT32)
        plan_cols = pl.cast(plan_cols_i32, target_type=pl.FP32)
        for plan_cb in pl.range(plan_blocks):
            plan_c0 = plan_cb * ATTN_K_TILE
            plan_rows = pl.min(ATTN_K_TILE, plan_length - plan_c0)
            plan_first = pl.read(cmp_block_table, [plan_b, plan_c0])
            plan_pages_i32 = pl.slice(cmp_block_table, [1, ATTN_K_TILE], [plan_b, plan_c0], valid_shape=[1, plan_rows])
            plan_pages = pl.cast(plan_pages_i32, target_type=pl.FP32)
            plan_expected = pl.add(plan_cols, pl.cast(plan_first, pl.FP32))
            plan_delta = pl.abs(pl.sub(plan_pages, plan_expected))
            plan_delta_valid = pl.set_validshape(plan_delta, 1, plan_rows)
            plan_delta_padded = pl.fillpad(plan_delta_valid, pad_value=pl.PadValue.zero)
            plan_delta_rows = pl.col_expand(pl.full([8, ATTN_K_TILE], dtype=pl.FP32, value=0.0), plan_delta_padded)
            plan_error = pl.row_max(plan_delta_rows)
            plan_base = pl.cast(-1, pl.INT32)
            if pl.read(plan_error, [0, 0]) == 0.0:
                if plan_first >= 0:
                    if plan_first + plan_rows <= cmp_block_num * CMP_STORAGE_BLOCK_SIZE:
                        plan_base = plan_first
            pl.write(cmp_run_bases, [plan_b, plan_cb], plan_base)
    q_flat = pl.reshape(q, [T * H, HEAD_DIM])
    head_base = pl.cast(my_rank, pl.INDEX) * H_LOCAL
    sparse_blk_mi = pl.create_tensor([T * QK_M_BOX * SPARSE_BLOCKS, 1], dtype=pl.FP32)
    sparse_blk_li = pl.create_tensor([T * QK_M_BOX * SPARSE_BLOCKS, 1], dtype=pl.FP32)
    sparse_blk_oi = pl.create_tensor([T * QK_M_BOX * SPARSE_BLOCKS, HEAD_DIM], dtype=pl.FP32)

    with pl.spmd(NUM_QK_CORES, name_hint="qk_pv", allow_early_resolve=True) as qk_tid:
        qk_core = pl.tile.get_block_idx()
        for qk_b in pl.range(B):
            if SPARSE_BLOCKS == 2:
                qk_request_blocks = pl.cast(SPARSE_BLOCKS, pl.INDEX)
            else:
                qk_last_t = qk_b * S + S - 1
                qk_last_len = pl.cast(pl.read(cmp_seq_lens, [qk_last_t]), pl.INDEX)
                qk_request_blocks = 1 + (qk_last_len + ATTN_K_TILE - 1) // ATTN_K_TILE
            qk_lane_iters = pl.max((S * qk_request_blocks - qk_core + NUM_QK_CORES - 1) // NUM_QK_CORES, 0)
            for qk_it in pl.range(qk_lane_iters):
                qk_work = qk_core + qk_it * NUM_QK_CORES
                qk_t = qk_b * S + qk_work % S
                qk_sb = qk_work // S
                qk_cmp_len = pl.cast(pl.read(cmp_seq_lens, [qk_t]), pl.INDEX)
                qk_live_blocks = 1 + (qk_cmp_len + ATTN_K_TILE - 1) // ATTN_K_TILE
                qk_cmp_flag = pl.min(qk_sb, 1)
                qk_window_len = pl.cast(pl.read(window_swa_lens, [qk_t]), pl.INDEX)
                qk_cmp_rows = pl.min(ATTN_K_TILE, pl.max(qk_cmp_len - (qk_sb - 1) * ATTN_K_TILE, 0))
                qk_valid = pl.min(WIN, pl.max(qk_window_len, 0)) * (1 - qk_cmp_flag) + qk_cmp_rows * qk_cmp_flag
                if qk_sb < qk_live_blocks:
                    qk_token_base = qk_t * QK_M_BOX * SPARSE_BLOCKS
                    qk_kv = pl.create_l1([ATTN_K_TILE, HEAD_DIM], pl.BF16)
                    if qk_sb == 0:
                        qk_slot0 = pl.max(pl.read(window_swa_indices, [qk_t, 0]), 0)
                        qk_r = qk_slot0 % BLOCK_SIZE
                        qk_head = WIN - qk_r
                        qk_blk1 = pl.max(pl.read(window_swa_indices, [qk_t, qk_head % WIN]), 0)
                        qk_len = pl.min(pl.max(pl.read(window_swa_lens, [qk_t]), 0), WIN)
                        qk_src0 = pl.cast(qk_slot0, pl.INDEX)
                        qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [0, 0], [qk_src0, 0],
                                              [ATTN_K_TILE, HEAD_DIM], valid_shape=[qk_head, HEAD_DIM])
                        if qk_r > 0:
                            qk_src1 = pl.cast(qk_blk1, pl.INDEX)
                            qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_head, 0], [qk_src1, 0],
                                                  [ATTN_K_TILE, HEAD_DIM], valid_shape=[qk_r, HEAD_DIM])
                        qk_tail = WIN - qk_len
                        if qk_tail > 0:
                            # Fill the cold-start tail from the oldest visible row.
                            for qk_fill in pl.range(qk_tail):
                                qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_len + qk_fill, 0],
                                                      [qk_src0, 0], [1, HEAD_DIM])
                    else:
                        qk_col0 = (qk_sb - 1) * ATTN_K_TILE
                        qk_base_i32 = pl.read(cmp_run_bases, [qk_b, qk_sb - 1])
                        if qk_base_i32 >= 0:
                            qk_base = pl.cast(qk_base_i32, pl.INDEX)
                            qk_kv = pl.gather_row(qk_kv, cmp_kv_flat, [0, 0], [qk_base, 0], [ATTN_K_TILE, HEAD_DIM], valid_shape=[qk_valid, HEAD_DIM])
                            qk_pad_rows = ATTN_K_TILE - qk_valid
                            if qk_pad_rows > 0:
                                qk_pad_src0 = pl.cast(pl.read(window_swa_indices, [qk_t, 0]), pl.INDEX)
                                qk_pad_head = pl.min(qk_pad_rows, BLOCK_SIZE - qk_pad_src0 % BLOCK_SIZE)
                                qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_valid, 0], [qk_pad_src0, 0], [ATTN_K_TILE, HEAD_DIM], valid_shape=[qk_pad_head, HEAD_DIM])
                                if qk_pad_rows > qk_pad_head:
                                    qk_pad_src1 = pl.cast(pl.read(window_swa_indices, [qk_t, qk_pad_head]), pl.INDEX)
                                    qk_kv = pl.gather_row(qk_kv, ori_kv_flat, [qk_valid + qk_pad_head, 0], [qk_pad_src1, 0], [ATTN_K_TILE, HEAD_DIM], valid_shape=[qk_pad_rows - qk_pad_head, HEAD_DIM])
                        else:
                            qk_first_src = pl.cast(pl.read(cmp_block_table, [qk_b, qk_col0]), pl.INDEX)
                            for qk_cr in pl.range(ATTN_K_TILE):
                                qk_cs = qk_first_src
                                if qk_cr < qk_valid:
                                    qk_cs = pl.cast(pl.read(cmp_block_table, [qk_b, qk_col0 + qk_cr]), pl.INDEX)
                                qk_kv = pl.gather_row(qk_kv, cmp_kv_flat, [qk_cr, 0], [qk_cs, 0], [1, HEAD_DIM])
                    for qk_hb in pl.pipeline(H_LOCAL // QK_M_TILE, stage=2):
                        qk_h0 = qk_hb * QK_M_TILE
                        qk_head_row = qk_t * H + head_base + qk_h0
                        qk_q_tile = pl.slice(
                            q_flat, [QK_M_BOX, HEAD_DIM], [qk_head_row, 0],
                            valid_shape=[QK_M_TILE, HEAD_DIM])
                        qk_raw = pl.matmul(qk_q_tile, qk_kv, b_trans=True, out_dtype=pl.FP32)
                        qk_scaled = pl.mul(qk_raw, SOFTMAX_SCALE)
                        qk_scores_valid = pl.set_validshape(qk_scaled, QK_M_TILE, qk_valid)
                        qk_scores_padded = pl.fillpad(qk_scores_valid, pad_value=pl.PadValue.min)
                        qk_scores = pl.set_validshape(qk_scores_padded, QK_M_TILE, ATTN_K_TILE)
                        qk_mi = pl.row_max(qk_scores)
                        qk_exp = pl.exp(pl.row_expand_sub(qk_scores, qk_mi))
                        qk_li = pl.row_sum(qk_exp)
                        qk_exp_bf16 = pl.cast(qk_exp, target_type=pl.BF16, mode="rint")
                        qk_oi = pl.matmul(qk_exp_bf16, qk_kv, out_dtype=pl.FP32)
                        qk_row = qk_token_base + qk_sb * QK_M_BOX
                        sparse_blk_mi[qk_row : qk_row + QK_M_BOX, 0 : 1] = qk_mi
                        sparse_blk_li[qk_row : qk_row + QK_M_BOX, 0 : 1] = qk_li
                        sparse_blk_oi[qk_row : qk_row + QK_M_BOX, 0 : HEAD_DIM] = qk_oi

    # Online-softmax merge across sparse-K tiles, sink-norm, then fused inverse RoPE.
    # One spmd block per (token, head-tile) -- T*(H//H_TILE) blocks -- so the merge
    # fans out over that many AIVs instead of T blocks each running a serial head-tile
    # loop. The inverse-RoPE rotation + rope-column pack is fused in (was a separate
    # "rope" spmd reading an attn_rope_stage GM round-trip): the head-tile's fp32 rope
    # segment is rotated in UB and packed straight into o_packed's rope columns.
    # with-form spmd so the dispatch TaskId (merge_tid) can be an explicit dep of
    # the manual-scope proj_a tasks below (which read merge_norm's o_packed cols).
    with pl.spmd(T * (H_LOCAL // H_TILE), name_hint="merge_norm", allow_early_resolve=True) as merge_tid:
        m_idx = pl.tile.get_block_idx()
        m_t = m_idx // (H_LOCAL // H_TILE)
        m_h_idx = m_idx - m_t * (H_LOCAL // H_TILE)
        m_h0 = m_h_idx * H_TILE
        m_blk_base = m_idx * SPARSE_BLOCKS * QK_M_BOX
        m_mi = sparse_blk_mi[m_blk_base : m_blk_base + H_TILE, 0 : 1]
        m_li = sparse_blk_li[m_blk_base : m_blk_base + H_TILE, 0 : 1]
        m_oi = sparse_blk_oi[m_blk_base : m_blk_base + H_TILE, 0 : HEAD_DIM]

        # Merge visible KV tiles in their original order.
        m_cmp_len = pl.cast(pl.read(cmp_seq_lens, [m_t]), pl.INDEX)
        m_live_blocks = 1 + (m_cmp_len + ATTN_K_TILE - 1) // ATTN_K_TILE
        if SPARSE_BLOCKS == 2:
            if m_cmp_len > 0:
                m_row = m_blk_base + QK_M_BOX
                m_cur_mi = sparse_blk_mi[m_row : m_row + H_TILE, 0 : 1]
                m_cur_li = sparse_blk_li[m_row : m_row + H_TILE, 0 : 1]
                m_cur_oi = sparse_blk_oi[m_row : m_row + H_TILE, 0 : HEAD_DIM]
                m_mi_new = pl.maximum(m_mi, m_cur_mi)
                m_alpha = pl.exp(pl.sub(m_mi, m_mi_new))
                m_beta = pl.exp(pl.sub(m_cur_mi, m_mi_new))
                m_li = pl.add(pl.mul(m_alpha, m_li), pl.mul(m_beta, m_cur_li))
                m_oi = pl.add(pl.row_expand_mul(m_oi, m_alpha), pl.row_expand_mul(m_cur_oi, m_beta))
                m_mi = m_mi_new

        else:
            for m_sb in pl.pipeline(1, m_live_blocks, stage=2):
                m_row = m_blk_base + m_sb * QK_M_BOX
                m_cur_mi = sparse_blk_mi[m_row : m_row + H_TILE, 0 : 1]
                m_cur_li = sparse_blk_li[m_row : m_row + H_TILE, 0 : 1]
                m_cur_oi = sparse_blk_oi[m_row : m_row + H_TILE, 0 : HEAD_DIM]
                m_mi_new = pl.maximum(m_mi, m_cur_mi)
                m_alpha = pl.exp(pl.sub(m_mi, m_mi_new))
                m_beta = pl.exp(pl.sub(m_cur_mi, m_mi_new))
                m_li = pl.add(pl.mul(m_alpha, m_li), pl.mul(m_beta, m_cur_li))
                m_oi = pl.add(pl.row_expand_mul(m_oi, m_alpha), pl.row_expand_mul(m_cur_oi, m_beta))
                m_mi = m_mi_new

        m_h_global = head_base + m_h0
        n_sink_bias = pl.reshape(attn_sink[m_h_global : m_h_global + H_TILE], [H_TILE, 1])
        n_sink_tile = pl.add(pl.sub(m_mi, m_mi), n_sink_bias)
        n_denom = pl.add(m_li, pl.exp(pl.sub(n_sink_tile, m_mi)))
        n_full = pl.row_expand_div(m_oi, n_denom)[0 : H_TILE, 0 : HEAD_DIM]
        n_bf16 = pl.cast(n_full, target_type=pl.BF16, mode="rint")

        # Inverse RoPE on this head-tile's fp32 rope segment. cos_il / sign*sin are
        # head-invariant for token m_t, so col_expand them over the H_TILE head rows;
        # rope_swap_idx (j^1, prebuilt above) pairs the interleaved real/imag lanes.
        # Rounded to bf16 (golden also rounds inverse-RoPE to bf16) and packed into
        # o_packed's rope columns.
        m_rope = n_full[0 : H_TILE, NOPE_DIM : HEAD_DIM]
        m_cos_il = rope_cos_il[m_t : m_t + 1, 0 : ROPE_DIM]
        m_sin_signed = rope_sin_signed[m_t : m_t + 1, 0 : ROPE_DIM]
        m_swapped = pl.gather(m_rope, dim=-1, index=rope_swap_idx[0:H_TILE, 0:ROPE_DIM])
        m_rot = pl.add(pl.col_expand_mul(m_rope, m_cos_il), pl.col_expand_mul(m_swapped, m_sin_signed))
        n_rope_bf16 = pl.cast(m_rot, target_type=pl.BF16, mode="rint")
        n_full_bf16 = pl.concat(n_bf16[:, : NOPE_DIM], n_rope_bf16)

        for n_hi in pl.unroll(H_TILE):
            n_pack_row = pl.cast(my_rank, pl.INDEX) * T + m_t
            n_col = ((m_h0 + n_hi) % HEADS_PER_GROUP) * HEAD_DIM
            # one HEAD_DIM-wide store per head row instead of two: concat the nope and
            # inverse-RoPE halves on chip so o_packed takes a single contiguous write.
            o_packed[n_pack_row : n_pack_row + 1, n_col : n_col + HEAD_DIM] = n_full_bf16[n_hi : n_hi + 1, :]

    # Back-to-back grouped output projection: proj_a[g] -> quant[g] -> proj_b[g]
    # pipelines per group, because the PER-GROUP amax keeps the quant reduction
    # inside one O_LORA group instead of barriering the whole row. manual_scope
    # suppresses auto-dep, so every edge is explicit: proj_a waits on merge_norm,
    # quant[g] on proj_a[g], proj_b[g] on quant[g]. proj_b_act combines the group
    # partials and is the consolidated attn_out writer.

    return merge_tid
