# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from .qkv_proj_rope import prepare_qkv_rope_metadata

from .decode_sparse_attn_csa import prepare_csa_output_rope, H_TILE as CSA_ROPE_ROWS

from .decode_indexer import prepare_indexer_rope_indices, ROPE_ROW_TILE as IDX_ROPE_ROWS

import pypto.language as pl

import pypto.language.distributed as pld

from .config import FLASH as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, C4A_COMPRESSOR_BLOCK_SIZE, IDX_CACHE_MAX_BLOCKS, KV_CMP_MAX_BLOCKS

from .decode_compressor_ratio4 import compressor_ratio4

from .hc_post import hc_post

from .hc_pre import hc_pre

from .decode_indexer import indexer

from .qkv_proj_rope import qkv_proj_rope

from .rmsnorm import rms_norm

from .rope_interleave import rope_interleave

from .decode_sparse_attn_csa import TOKEN_SHARDED, sparse_attn_csa_head_packed, csa_q_all_to_all, sparse_attn_csa_packed

from . import decode_sparse_attn_csa as CSA

from . import o_proj_tp as OTP

from .o_proj_tp import o_proj_tp_core

B = DECODE_BATCH

S = DECODE_SEQ

T = B * S

D = M.hidden_size

N_EXPERTS = M.n_routed_experts

H = M.num_attention_heads

HEAD_DIM = M.head_dim

ROPE_HEAD_DIM = M.qk_rope_head_dim

HALF_ROPE = ROPE_HEAD_DIM // 2

Q_LORA = M.q_lora_rank

WIN = M.sliding_window

MAX_SEQ_LEN = M.max_position_embeddings

HC_MULT = M.hc_mult

MIX_HC = M.mix_hc

HC_DIM = M.hc_dim

IDX_N_HEADS = M.index_n_heads

IDX_HEAD_DIM = M.index_head_dim

INDEXER_SCORE_LEN = MAX_SEQ_LEN // 4

O_LORA = M.o_lora_rank

O_GROUPS = M.o_groups

O_GROUP_IN = H * HEAD_DIM // O_GROUPS

OPROJ_N_RANKS = OTP.N_RANKS

TOK_Q_ROWS = CSA.H

TOK_O_COLS = CSA.O_SHARD_COLS

OPROJ_REDUCE_ROWS = OTP.REDUCE_WINDOW_ROWS

OPROJ_SCALE_ROWS = OTP.SCALE_WINDOW_ROWS

WO_A_FLAT = O_LORA * O_GROUP_IN

WO_B_FLAT = D * O_LORA

assert O_GROUPS == OPROJ_N_RANKS, f"one group per card: O_GROUPS={O_GROUPS}, TP={OPROJ_N_RANKS}"

COMPRESS_RATIO = 4

CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE // COMPRESS_RATIO

OVERLAP = COMPRESS_RATIO == 4

COFF = 1 + int(OVERLAP)

MAIN_OUT_DIM = COFF * HEAD_DIM

MAIN_STATE_DIM = 2 * MAIN_OUT_DIM

MAIN_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE

MAIN_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + MAIN_STATE_BLOCK_SIZE - 1) // MAIN_STATE_BLOCK_SIZE

MAIN_STATE_BLOCK_NUM_DYN = pl.dynamic('NATIVE_S_CSA_STATE_BLOCK_NUM_DYN')

INNER_OUT_DIM = COFF * IDX_HEAD_DIM

INNER_STATE_DIM = 2 * INNER_OUT_DIM

INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE

INNER_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + INNER_STATE_BLOCK_SIZE - 1) // INNER_STATE_BLOCK_SIZE

INNER_STATE_BLOCK_NUM_DYN = pl.dynamic('NATIVE_S_INNER_STATE_BLOCK_NUM_DYN')

IDX_CACHE_BLOCK_NUM_DYN = pl.dynamic('NATIVE_S_IDX_CACHE_BLOCK_NUM_DYN')

ORI_BLOCK_NUM_DYN = pl.dynamic('NATIVE_S_ORI_BLOCK_NUM_DYN')

CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS

CMP_BLOCK_NUM_DYN = pl.dynamic('NATIVE_S_CMP_BLOCK_NUM_DYN')

CSA_WB_TOKEN_TILE = 8

@pl.jit.inline
def prepare_csa_metadata(
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    position_ids: pl.Tensor[[T], pl.INT32],
    rope_cos_t: pl.Out[pl.Tensor[[T, ROPE_HEAD_DIM], pl.BF16]],
    rope_sin_t: pl.Out[pl.Tensor[[T, ROPE_HEAD_DIM], pl.BF16]],
    step_cos_il: pl.Out[pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32]],
    step_sin_signed: pl.Out[pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32]],
    cmp_cos_il: pl.Out[pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32]],
    cmp_sin_signed: pl.Out[pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32]],
    q_rope_cos_il: pl.Out[pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32]],
    q_rope_sin_signed: pl.Out[pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32]],
    q_rope_swap_idx: pl.Out[pl.Tensor[[T, ROPE_HEAD_DIM], pl.INT32]],
    out_rope_cos_il: pl.Out[pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32]],
    out_rope_sin_signed: pl.Out[pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32]],
    out_rope_swap_idx: pl.Out[pl.Tensor[[CSA_ROPE_ROWS, ROPE_HEAD_DIM], pl.INT32]],
    idx_rope_swap_idx: pl.Out[pl.Tensor[[IDX_ROPE_ROWS, ROPE_HEAD_DIM], pl.INT32]],
):
    step_cos = pl.create_tensor([T, HALF_ROPE], dtype=pl.FP32)
    step_sin = pl.create_tensor([T, HALF_ROPE], dtype=pl.FP32)
    cmp_cos = pl.create_tensor([T, HALF_ROPE], dtype=pl.FP32)
    cmp_sin = pl.create_tensor([T, HALF_ROPE], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="csa_rope_step"):
        for token in pl.range(T):
            position = pl.read(position_ids, [token])
            row = pl.cast(position, pl.INDEX)
            rope_cos_t[token : token + 1, :] = freqs_cos[row : row + 1, :]
            rope_sin_t[token : token + 1, :] = freqs_sin[row : row + 1, :]
            step_cos[token : token + 1, :] = pl.cast(freqs_cos[row : row + 1, 0 : HALF_ROPE], pl.FP32)
            step_sin[token : token + 1, :] = pl.cast(freqs_sin[row : row + 1, 0 : HALF_ROPE], pl.FP32)
            boundary_position = 0
            if (position + 1) % COMPRESS_RATIO == 0:
                boundary_position = position - COMPRESS_RATIO + 1
            boundary_row = pl.cast(boundary_position, pl.INDEX)
            cmp_cos[token : token + 1, :] = pl.cast(freqs_cos[boundary_row : boundary_row + 1, 0 : HALF_ROPE], pl.FP32)
            cmp_sin[token : token + 1, :] = pl.cast(freqs_sin[boundary_row : boundary_row + 1, 0 : HALF_ROPE], pl.FP32)
    rope_interleave(step_cos, step_sin, step_cos_il, step_sin_signed)
    rope_interleave(cmp_cos, cmp_sin, cmp_cos_il, cmp_sin_signed)
    prepare_qkv_rope_metadata(rope_cos_t, rope_sin_t, q_rope_cos_il, q_rope_sin_signed, q_rope_swap_idx)
    prepare_csa_output_rope(rope_cos_t, rope_sin_t, out_rope_cos_il, out_rope_sin_signed, out_rope_swap_idx)
    prepare_indexer_rope_indices(idx_rope_swap_idx)

@pl.jit.inline
def attention_csa_packed(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos_t: pl.Tensor[[T, ROPE_HEAD_DIM], pl.BF16],
    rope_sin_t: pl.Tensor[[T, ROPE_HEAD_DIM], pl.BF16],
    step_cos_il: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    step_sin_signed: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    cmp_cos_il: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    cmp_sin_signed: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B, MAIN_STATE_MAX_BLOCKS], pl.INT32],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[B, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32],
    idx_block_table: pl.Tensor[[B, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a_w: pl.Tensor[[1, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_w: pl.Tensor[[D, O_LORA], pl.INT8],
    o_packed: pl.Tensor[[O_GROUPS * T, O_GROUP_IN], pl.BF16],
    post_t: pl.Tensor[[T, HC_MULT], pl.FP32],
    comb_t: pl.Tensor[[T, HC_MULT * HC_MULT], pl.FP32],
    q_window: pld.DistributedTensor[[TOK_Q_ROWS, HEAD_DIM], pl.BF16],
    q_signal: pld.DistributedTensor[[OPROJ_N_RANKS, 1], pl.INT32],
    o_window: pld.DistributedTensor[[T, TOK_O_COLS], pl.BF16],
    o_signal: pld.DistributedTensor[[OPROJ_N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    tok_epoch: pl.Scalar[pl.INT32],
    q_rope_cos_il: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    q_rope_sin_signed: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    q_rope_swap_idx: pl.Tensor[[T, ROPE_HEAD_DIM], pl.INT32],
    out_rope_cos_il: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    out_rope_sin_signed: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    out_rope_swap_idx: pl.Tensor[[CSA_ROPE_ROWS, ROPE_HEAD_DIM], pl.INT32],
    idx_rope_swap_idx: pl.Tensor[[IDX_ROPE_ROWS, ROPE_HEAD_DIM], pl.INT32],
) -> pl.Scalar[pl.TASK_ID]:
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post_t, comb_t)

    x_normed_t = pl.create_tensor([T, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed_t)
    # SDMA CMO L2 warm of this layer's attention weights, in deadline order.
    wq_a_flat = pl.reshape(wq_a, [D * Q_LORA])
    wkv_flat = pl.reshape(wkv, [D * HEAD_DIM])
    cmp_wkv_flat = pl.reshape(cmp_wkv, [MAIN_OUT_DIM * D])
    cmp_wgate_flat = pl.reshape(cmp_wgate, [MAIN_OUT_DIM * D])
    idx_wq_b_flat = pl.reshape(idx_wq_b, [Q_LORA * IDX_N_HEADS * IDX_HEAD_DIM])
    weights_proj_flat = pl.reshape(weights_proj, [D * IDX_N_HEADS])
    inner_wkv_flat = pl.reshape(inner_wkv, [INNER_OUT_DIM * D])
    inner_wgate_flat = pl.reshape(inner_wgate, [INNER_OUT_DIM * D])
    wo_a_flat = pl.reshape(wo_a_w, [WO_A_FLAT])
    wo_b_flat = pl.reshape(wo_b_w, [WO_B_FLAT])
    gate_w_flat = pl.reshape(gate_w, [N_EXPERTS * D])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefetch_attn_w", allow_early_resolve=True):
        _prefetch_anchor = pl.read(x_hc, [0, 0, 0])
        warm_ctx = pl.prefetch.make_context()
        pl.prefetch.async_prefetch(wq_a_flat, warm_ctx)
        pl.prefetch.async_prefetch(wkv_flat, warm_ctx)
        pl.prefetch.async_prefetch(cmp_wkv_flat, warm_ctx)
        pl.prefetch.async_prefetch(cmp_wgate_flat, warm_ctx)
        pl.prefetch.async_prefetch(idx_wq_b_flat, warm_ctx)
        pl.prefetch.async_prefetch(weights_proj_flat, warm_ctx)
        pl.prefetch.async_prefetch(inner_wkv_flat, warm_ctx)
        pl.prefetch.async_prefetch(inner_wgate_flat, warm_ctx)
        pl.prefetch.async_prefetch(wo_a_flat, warm_ctx)
        pl.prefetch.async_prefetch(wo_b_flat, warm_ctx)
        pl.prefetch.async_prefetch(gate_w_flat, warm_ctx)
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    kv_dep = pl.system.task_dummy(deps=[])
    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    qkv_proj_rope(
        x_normed_t, wq_a, wq_b, wq_b_scale, wkv,
        rope_cos_t, rope_sin_t, gamma_cq, gamma_ckv,
        q, kv, qr, qr_scale, kv_dep,
        pl.cast(my_rank, pl.INT32) * (H // O_GROUPS), q_rope_cos_il, q_rope_sin_signed, q_rope_swap_idx,
    )

    # Transpose q onto the token shard right here. Everything between this push
    # and qk_pv -- the KV writeback, the compressor, the whole indexer -- reads
    # no q, so the exchange flies under work the layer was doing anyway and only
    # its wait lands in front of the consumer.
    q_push_tid = kv_dep
    if TOKEN_SHARDED:
        q_push_tid = csa_q_all_to_all(q, q_window, q_signal, my_rank, tok_epoch)

    ori_block_num = pl.tensor.dim(kv_cache, 0)
    kv_cache_flat = pl.reshape(kv_cache, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    for wb_blk in pl.spmd((T + CSA_WB_TOKEN_TILE - 1) // CSA_WB_TOKEN_TILE, name_hint="csa_cache_writeback"):
        wb_t0 = wb_blk * CSA_WB_TOKEN_TILE
        for write_dt in pl.range(CSA_WB_TOKEN_TILE):
            write_t = wb_t0 + write_dt
            if write_t < T:
                write_row_i64 = pl.read(ori_slot_mapping, [write_t])
                if write_row_i64 >= 0:
                    write_row = pl.cast(write_row_i64, pl.INDEX)
                    kv_cache_flat[write_row : write_row + 1, 0 : HEAD_DIM] = kv[write_t : write_t + 1, 0 : HEAD_DIM]

    x_normed = pl.reshape(x_normed_t, [B, S, D])
    cmp_out = pl.create_tensor([B, S, HEAD_DIM], dtype=pl.FP32)
    position_ids_bsd = pl.reshape(position_ids, [B, S])
    cmp_slot_mapping_bsd = pl.reshape(cmp_slot_mapping, [B, S])
    idx_slot_mapping_bsd = pl.reshape(idx_slot_mapping, [B, S])
    state_slot_mapping_bsd = pl.reshape(state_slot_mapping, [B, S])
    inner_state_slot_mapping_bsd = pl.reshape(inner_state_slot_mapping, [B, S])
    compressor_ratio4(
        x_normed, cmp_out,
        compress_state, compress_state_block_table,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        cmp_cos_il, cmp_sin_signed, cmp_kv,
        position_ids_bsd, cmp_slot_mapping_bsd, state_slot_mapping_bsd,
        kv_dep,
    )

    idx_kv_unused = pl.create_tensor([B, S, IDX_HEAD_DIM], dtype=pl.FP32)
    idx_score_unused = pl.create_tensor([B, S, INDEXER_SCORE_LEN], dtype=pl.FP32)
    idx_topk_full = pl.create_tensor([B, S, INDEXER_SCORE_LEN], dtype=pl.INT32)
    indexer(
        x_normed, qr, qr_scale, idx_wq_b, idx_wq_b_scale,
        weights_proj, step_cos_il, step_sin_signed, cmp_cos_il, cmp_sin_signed, hadamard_idx,
        idx_kv_unused, inner_compress_state, inner_compress_state_block_table,
        inner_wkv, inner_wgate, inner_ape, inner_norm_w,
        idx_kv_cache, idx_kv_scale, idx_block_table,
        idx_score_unused, idx_topk_full,
        position_ids_bsd, idx_slot_mapping_bsd, inner_state_slot_mapping_bsd,
        kv_seq_lens, 0, late_dep,
        my_rank, idx_rope_swap_idx, pl.const(0, pl.INT32),
    )

    # sparse_attn_csa now folds the compressed-slot masking + valid-block flags in from
    # the raw indexer topk + position, so pass those directly.
    idx_topk_flat = pl.reshape(idx_topk_full, [T, INDEXER_SCORE_LEN])
    position_ids_t1 = pl.reshape(position_ids, [T, 1])

    if TOKEN_SHARDED:
        merge_tid = sparse_attn_csa_packed(
            q_window, q_signal, q_push_tid, kv_cache, window_swa_indices,
            cmp_kv, cmp_block_table, idx_topk_flat, position_ids_t1,
            attn_sink, rope_cos_t, rope_sin_t, o_packed,
            o_window, o_signal, my_rank, tok_epoch, out_rope_cos_il, out_rope_sin_signed, out_rope_swap_idx,
        )
    else:
        merge_tid = sparse_attn_csa_head_packed(
            q, kv_cache, window_swa_indices, cmp_kv, cmp_block_table, idx_topk_flat,
            position_ids_t1, attn_sink, o_packed, my_rank,
            out_rope_cos_il, out_rope_sin_signed, out_rope_swap_idx,
        )
    return merge_tid

@pl.jit.inline
def attention_csa(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos_t: pl.Tensor[[T, ROPE_HEAD_DIM], pl.BF16],
    rope_sin_t: pl.Tensor[[T, ROPE_HEAD_DIM], pl.BF16],
    step_cos_il: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    step_sin_signed: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    cmp_cos_il: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    cmp_sin_signed: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[[MAIN_STATE_BLOCK_NUM_DYN, MAIN_STATE_BLOCK_SIZE, MAIN_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B, MAIN_STATE_MAX_BLOCKS], pl.INT32],
    idx_wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    idx_wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    hadamard_idx: pl.Tensor[[IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[IDX_HEAD_DIM], pl.BF16],
    inner_compress_state: pl.Tensor[[INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32],
    inner_compress_state_block_table: pl.Tensor[[B, INNER_STATE_MAX_BLOCKS], pl.INT32],
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8],
    idx_kv_scale: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32],
    idx_block_table: pl.Tensor[[B, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    idx_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    attn_sink: pl.Tensor[[H], pl.FP32],
    wo_a_shard: pl.Tensor[[1, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_shard: pl.Tensor[[D, O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    reduce_window: pld.DistributedTensor[[OPROJ_REDUCE_ROWS, D], pl.FP32],
    scale_window: pld.DistributedTensor[[OPROJ_SCALE_ROWS, 1], pl.FP32],
    reduce_signal: pld.DistributedTensor[[OPROJ_N_RANKS, 1], pl.INT32],
    sync_signal: pld.DistributedTensor[[OPROJ_N_RANKS, 1], pl.INT32],
    q_window: pld.DistributedTensor[[TOK_Q_ROWS, HEAD_DIM], pl.BF16],
    q_signal: pld.DistributedTensor[[OPROJ_N_RANKS, 1], pl.INT32],
    o_window: pld.DistributedTensor[[T, TOK_O_COLS], pl.BF16],
    o_signal: pld.DistributedTensor[[OPROJ_N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    oproj_epoch: pl.Scalar[pl.INT32],
    # Counts CSA layers only. oproj's epoch is shared with the HCA layers and so
    # advances twice per loop; these windows are notified by CSA layers alone, and
    # a wait for an epoch nobody will reach hangs the card.
    tok_epoch: pl.Scalar[pl.INT32],
    q_rope_cos_il: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    q_rope_sin_signed: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    q_rope_swap_idx: pl.Tensor[[T, ROPE_HEAD_DIM], pl.INT32],
    out_rope_cos_il: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    out_rope_sin_signed: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    out_rope_swap_idx: pl.Tensor[[CSA_ROPE_ROWS, ROPE_HEAD_DIM], pl.INT32],
    idx_rope_swap_idx: pl.Tensor[[IDX_ROPE_ROWS, ROPE_HEAD_DIM], pl.INT32],
):
    """CSA layer whose attention is token-sharded and projection group-sharded.

    Everything ahead of the projection is replicated exactly as today -- the
    shard boundary is the group axis, and group `my_rank` is heads
    [8*my_rank, 8*my_rank+8), whose o_packed rows this card already produced.
    """
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)
    post_t = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    merge_tid = attention_csa_packed(
        x_hc, gate_w, hc_attn_fn, hc_attn_scale, hc_attn_base, attn_norm_w, wq_a, wq_b, wq_b_scale, wkv, gamma_cq, gamma_ckv, rope_cos_t, rope_sin_t, step_cos_il, step_sin_signed, cmp_cos_il, cmp_sin_signed, cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w, compress_state, compress_state_block_table, idx_wq_b, idx_wq_b_scale, weights_proj, hadamard_idx, inner_wkv, inner_wgate, inner_ape, inner_norm_w, inner_compress_state, inner_compress_state_block_table, kv_cache, cmp_kv, cmp_block_table, idx_kv_cache, idx_kv_scale, idx_block_table, ori_slot_mapping, window_swa_indices, window_swa_lens, cmp_slot_mapping, idx_slot_mapping, state_slot_mapping, inner_state_slot_mapping, position_ids, kv_seq_lens, attn_sink,
        wo_a_shard, wo_b_shard, o_packed, post_t, comb_t,
        q_window, q_signal, o_window, o_signal, my_rank, tok_epoch, q_rope_cos_il, q_rope_sin_signed, q_rope_swap_idx, out_rope_cos_il, out_rope_sin_signed, out_rope_swap_idx, idx_rope_swap_idx,
    )
    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    o_proj_tp_core(
        o_packed, merge_tid, wo_a_shard, wo_b_shard, wo_b_scale, attn_out,
        reduce_window, scale_window, reduce_signal, sync_signal, my_rank, oproj_epoch,
    )
    hc_post(attn_out, x_hc, post_t, comb_t, x_out)
    return x_out

@pl.jit.inline
def clear_csa_oproj_signals(
    completion_anchor: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    reduce_signal: pld.DistributedTensor[[OPROJ_N_RANKS, 1], pl.INT32],
    sync_signal: pld.DistributedTensor[[OPROJ_N_RANKS, 1], pl.INT32],
    q_signal: pld.DistributedTensor[[OPROJ_N_RANKS, 1], pl.INT32],
    o_signal: pld.DistributedTensor[[OPROJ_N_RANKS, 1], pl.INT32],
):
    """Clear this rank's o-projection counters after its last CSA layer.

    The projection's signals are monotonic within a dispatch so one window can
    carry every CSA layer; the caller resets them once, after the layer whose
    output proves every peer's final notify was observed.
    """
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="oproj_signal_clear") as s6_oproj_cleanup_tid:
        _completion_anchor = pl.read(completion_anchor, [0, 0, 0])
        zero = pl.cast(0, pl.INT32)
        for src in pl.range(OPROJ_N_RANKS):
            pl.write(reduce_signal, [src, 0], zero)
            pl.write(sync_signal, [src, 0], zero)
            pl.write(q_signal, [src, 0], zero)
            pl.write(o_signal, [src, 0], zero)
    return s6_oproj_cleanup_tid
