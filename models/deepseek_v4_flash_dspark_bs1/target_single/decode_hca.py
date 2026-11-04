# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from .qkv_proj_rope import prepare_qkv_rope_metadata

from .decode_sparse_attn_hca import prepare_hca_output_rope, H_TILE as HCA_ROPE_ROWS

import pypto.language as pl

import pypto.language.distributed as pld

from . import o_proj_tp as OTP

from .o_proj_tp import o_proj_tp_core

from .config import FLASH as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, C128_COMPRESSOR_BLOCK_SIZE, KV_CMP_MAX_BLOCKS

from .hc_pre import hc_pre

from .hc_post import hc_post

from .qkv_proj_rope import qkv_proj_rope

from .rmsnorm import rms_norm

from .rope_interleave import rope_interleave

from .decode_compressor_ratio128 import compressor_ratio128

from .decode_sparse_attn_hca import sparse_attn_hca_packed

B = DECODE_BATCH

S = DECODE_SEQ

T = B * S

D = M.hidden_size

N_EXPERTS = M.n_routed_experts

H = M.num_attention_heads

HEAD_DIM = M.head_dim

ROPE_HEAD_DIM = M.qk_rope_head_dim

Q_LORA = M.q_lora_rank

WIN = M.sliding_window

HC_MULT = M.hc_mult

MIX_HC = M.mix_hc

HC_DIM = M.hc_dim

MAX_SEQ_LEN = M.max_position_embeddings

O_LORA = M.o_lora_rank

O_GROUPS = M.o_groups

O_GROUP_IN = H * HEAD_DIM // O_GROUPS

COMPRESS_RATIO = 128  # HCA

CMP_STORAGE_BLOCK_SIZE = BLOCK_SIZE // COMPRESS_RATIO

OVERLAP = COMPRESS_RATIO == 4   # always False for HCA

COFF = 1 + int(OVERLAP)         # always 1 for HCA

MAIN_OUT_DIM = COFF * HEAD_DIM

ORI_BLOCK_NUM_DYN = pl.dynamic("ORI_BLOCK_NUM_DYN")

CMP_MAX_BLOCKS = KV_CMP_MAX_BLOCKS

CMP_BLOCK_NUM_DYN = pl.dynamic("CMP_BLOCK_NUM_DYN")

COMPRESS_STATE_BLOCK_SIZE = C128_COMPRESSOR_BLOCK_SIZE

COMPRESS_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + COMPRESS_STATE_BLOCK_SIZE - 1) // COMPRESS_STATE_BLOCK_SIZE

COMPRESS_STATE_BLOCK_NUM_DYN = pl.dynamic("HCA_STATE_BLOCK_NUM_DYN")

COMPRESS_STATE_DIM = 2 * MAIN_OUT_DIM

COMPRESS_TOPK = MAX_SEQ_LEN // COMPRESS_RATIO   # demo 32; flash 128 (= 16384/128); max compressed positions

HCA_TOPK_LIMIT = COMPRESS_TOPK

HCA_LENGTH_TOKEN_TILE = 8

HCA_WB_TOKEN_TILE = 8  # tokens per cache-writeback SPMD block

WO_A_FLAT = O_LORA * O_GROUP_IN

WO_B_FLAT = D * O_LORA

OPROJ_N_RANKS = OTP.N_RANKS

OPROJ_REDUCE_ROWS = OTP.REDUCE_WINDOW_ROWS

OPROJ_SCALE_ROWS = OTP.SCALE_WINDOW_ROWS

@pl.jit.inline
def prepare_hca_metadata(
    freqs_cos: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[MAX_SEQ_LEN, ROPE_HEAD_DIM], pl.BF16],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    rope_cos_t: pl.Out[pl.Tensor[[T, ROPE_HEAD_DIM], pl.BF16]],
    rope_sin_t: pl.Out[pl.Tensor[[T, ROPE_HEAD_DIM], pl.BF16]],
    cmp_cos_il: pl.Out[pl.Tensor[[B, ROPE_HEAD_DIM], pl.FP32]],
    cmp_sin_signed: pl.Out[pl.Tensor[[B, ROPE_HEAD_DIM], pl.FP32]],
    cmp_seq_lens: pl.Out[pl.Tensor[[T], pl.INT32]],
    q_rope_cos_il: pl.Out[pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32]],
    q_rope_sin_signed: pl.Out[pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32]],
    q_rope_swap_idx: pl.Out[pl.Tensor[[T, ROPE_HEAD_DIM], pl.INT32]],
    out_rope_cos_il: pl.Out[pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32]],
    out_rope_sin_signed: pl.Out[pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32]],
    out_rope_swap_idx: pl.Out[pl.Tensor[[HCA_ROPE_ROWS, ROPE_HEAD_DIM], pl.INT32]],
):
    cmp_cos = pl.create_tensor([B, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
    cmp_sin = pl.create_tensor([B, ROPE_HEAD_DIM // 2], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="hca_rope"):
        for b in pl.range(B):
            first_t = b * S
            first_pos_b = pl.read(position_ids, [first_t])
            cmp_offset_b = COMPRESS_RATIO - (first_pos_b % COMPRESS_RATIO)
            cmp_pos_b = pl.cast(first_pos_b + cmp_offset_b - COMPRESS_RATIO, pl.INDEX)
            cmp_cos_row = freqs_cos[cmp_pos_b : cmp_pos_b + 1, 0 : ROPE_HEAD_DIM // 2]
            cmp_sin_row = freqs_sin[cmp_pos_b : cmp_pos_b + 1, 0 : ROPE_HEAD_DIM // 2]
            cmp_cos[b : b + 1, 0 : ROPE_HEAD_DIM // 2] = pl.cast(cmp_cos_row, target_type=pl.FP32)
            cmp_sin[b : b + 1, 0 : ROPE_HEAD_DIM // 2] = pl.cast(cmp_sin_row, target_type=pl.FP32)
            for s in pl.range(S):
                t = b * S + s
                pos_b = pl.cast(pl.read(position_ids, [t]), pl.INDEX)
                step_cos_row = pl.cast(freqs_cos[pos_b : pos_b + 1, 0 : ROPE_HEAD_DIM], target_type=pl.FP32)
                step_sin_row = pl.cast(freqs_sin[pos_b : pos_b + 1, 0 : ROPE_HEAD_DIM], target_type=pl.FP32)
                rope_cos_t[t : t + 1, 0 : ROPE_HEAD_DIM] = pl.cast(step_cos_row, target_type=pl.BF16, mode="rint")
                rope_sin_t[t : t + 1, 0 : ROPE_HEAD_DIM] = pl.cast(step_sin_row, target_type=pl.BF16, mode="rint")

    rope_interleave(cmp_cos, cmp_sin, cmp_cos_il, cmp_sin_signed)

    for length_block in pl.spmd((T + HCA_LENGTH_TOKEN_TILE - 1) // HCA_LENGTH_TOKEN_TILE, name_hint="hca_lengths", allow_early_resolve=True):
        length_t0 = length_block * HCA_LENGTH_TOKEN_TILE
        for length_dt in pl.range(HCA_LENGTH_TOKEN_TILE):
            length_t = length_t0 + length_dt
            if length_t < T:
                length_b = length_t // S
                length_pos = pl.read(position_ids, [length_t])
                length_kv = pl.read(kv_seq_lens, [length_b])
                length_valid = pl.min(HCA_TOPK_LIMIT, pl.min((length_pos + 1) // COMPRESS_RATIO, length_kv // COMPRESS_RATIO))
                pl.write(cmp_seq_lens, [length_t], pl.cast(pl.max(length_valid, 0), pl.INT32))
    prepare_qkv_rope_metadata(rope_cos_t, rope_sin_t, q_rope_cos_il, q_rope_sin_signed, q_rope_swap_idx)
    prepare_hca_output_rope(rope_cos_t, rope_sin_t, out_rope_cos_il, out_rope_sin_signed, out_rope_swap_idx)

@pl.jit.inline
def attention_hca(
    x_hc: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    # hc_pre weights
    hc_attn_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_attn_scale: pl.Tensor[[3], pl.FP32],
    hc_attn_base: pl.Tensor[[MIX_HC], pl.FP32],
    # qkv_proj_rope weights
    attn_norm_w: pl.Tensor[[D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    rope_cos_t: pl.Tensor[[T, ROPE_HEAD_DIM], pl.BF16],
    rope_sin_t: pl.Tensor[[T, ROPE_HEAD_DIM], pl.BF16],
    cmp_cos_il: pl.Tensor[[B, ROPE_HEAD_DIM], pl.FP32],
    cmp_sin_signed: pl.Tensor[[B, ROPE_HEAD_DIM], pl.FP32],
    cmp_seq_lens: pl.Tensor[[T], pl.INT32],
    # main compressor (head_dim=HEAD_DIM, ratio=128, overlap=False)
    cmp_wkv: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_wgate: pl.Tensor[[MAIN_OUT_DIM, D], pl.BF16],
    cmp_ape: pl.Tensor[[COMPRESS_RATIO, MAIN_OUT_DIM], pl.FP32],
    cmp_norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    # KV cache split into ori (sliding window) and cmp (compressed) pools to match sparse_attn's contract.
    # cmp_kv is shared with the compressor: it writes the compressed row directly into this pool.
    kv_cache: pl.Tensor[[ORI_BLOCK_NUM_DYN, BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_kv: pl.Tensor[[CMP_BLOCK_NUM_DYN, CMP_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[B, CMP_MAX_BLOCKS], pl.INT32],
    ori_slot_mapping: pl.Tensor[[T], pl.INT64],
    window_swa_indices: pl.Tensor[[T, WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[T], pl.INT32],
    cmp_slot_mapping: pl.Tensor[[T], pl.INT64],
    state_slot_mapping: pl.Tensor[[T], pl.INT64],
    position_ids: pl.Tensor[[T], pl.INT32],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    # sparse_attn
    attn_sink: pl.Tensor[[H], pl.FP32],
    # o_proj, sharded one group per card
    wo_a_shard: pl.Tensor[[1, O_LORA, O_GROUP_IN], pl.BF16],
    wo_b_shard: pl.Tensor[[D, O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[D], pl.FP32],
    x_out: pl.Tensor[[T, HC_MULT, D], pl.FP32],
    reduce_window: pld.DistributedTensor[[OPROJ_REDUCE_ROWS, D], pl.FP32],
    scale_window: pld.DistributedTensor[[OPROJ_SCALE_ROWS, 1], pl.FP32],
    reduce_signal: pld.DistributedTensor[[OPROJ_N_RANKS, 1], pl.INT32],
    sync_signal: pld.DistributedTensor[[OPROJ_N_RANKS, 1], pl.INT32],
    my_rank: pl.Scalar[pl.INT32],
    oproj_epoch: pl.Scalar[pl.INT32],
    q_rope_cos_il: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    q_rope_sin_signed: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    q_rope_swap_idx: pl.Tensor[[T, ROPE_HEAD_DIM], pl.INT32],
    out_rope_cos_il: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    out_rope_sin_signed: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    out_rope_swap_idx: pl.Tensor[[HCA_ROPE_ROWS, ROPE_HEAD_DIM], pl.INT32],
):
    """HCA decode orchestration for compress_ratio=128."""
    x_mixed = pl.create_tensor([T, D], dtype=pl.BF16)
    post_t = pl.create_tensor([T, HC_MULT], dtype=pl.FP32)
    comb_t = pl.create_tensor([T, HC_MULT * HC_MULT], dtype=pl.FP32)
    hc_pre(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base, x_mixed, post_t, comb_t)

    x_normed = pl.create_tensor([T, D], dtype=pl.BF16)
    rms_tid = rms_norm(x_mixed, attn_norm_w, x_normed)
    # SDMA CMO L2 warm of this layer's attention weights, in deadline order.
    wq_a_flat = pl.reshape(wq_a, [D * Q_LORA])
    wkv_flat = pl.reshape(wkv, [D * HEAD_DIM])
    wo_a_flat = pl.reshape(wo_a_shard, [WO_A_FLAT])
    wo_b_flat = pl.reshape(wo_b_shard, [WO_B_FLAT])
    gate_w_flat = pl.reshape(gate_w, [N_EXPERTS * D])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="prefetch_attn_w", allow_early_resolve=True):
        _prefetch_anchor = pl.read(x_hc, [0, 0, 0])
        warm_ctx = pl.prefetch.make_context()
        pl.prefetch.async_prefetch(wq_a_flat, warm_ctx)
        pl.prefetch.async_prefetch(wkv_flat, warm_ctx)
        pl.prefetch.async_prefetch(wo_a_flat, warm_ctx)
        pl.prefetch.async_prefetch(wo_b_flat, warm_ctx)
        pl.prefetch.async_prefetch(gate_w_flat, warm_ctx)
    late_dep = pl.system.task_dummy(deps=[rms_tid])
    q = pl.create_tensor([T, H, HEAD_DIM], dtype=pl.BF16)
    kv = pl.create_tensor([T, HEAD_DIM], dtype=pl.BF16)
    qr = pl.create_tensor([T, Q_LORA], dtype=pl.INT8)        # unused on HCA path
    qr_scale = pl.create_tensor([T, 1], dtype=pl.FP32)
    qkv_proj_rope(
        x_normed, wq_a, wq_b, wq_b_scale, wkv,
        rope_cos_t, rope_sin_t, gamma_cq, gamma_ckv,
        q, kv, qr, qr_scale, rms_tid,
        pl.cast(my_rank, pl.INT32) * (H // O_GROUPS), q_rope_cos_il, q_rope_sin_signed, q_rope_swap_idx,
    )

    ori_block_num = pl.tensor.dim(kv_cache, 0)
    kv_cache_flat = pl.reshape(kv_cache, [ori_block_num * BLOCK_SIZE, HEAD_DIM])
    for wb_blk in pl.spmd(
        (T + HCA_WB_TOKEN_TILE - 1) // HCA_WB_TOKEN_TILE, name_hint="hca_cache_writeback", allow_early_resolve=True
    ):
        wb_t0 = wb_blk * HCA_WB_TOKEN_TILE
        for write_dt in pl.range(HCA_WB_TOKEN_TILE):
            write_t = wb_t0 + write_dt
            if write_t < T:
                write_row_i64 = pl.read(ori_slot_mapping, [write_t])
                if write_row_i64 >= 0:
                    write_row = pl.cast(write_row_i64, pl.INDEX)
                    kv_cache_flat[write_row : write_row + 1, 0 : HEAD_DIM] = kv[write_t : write_t + 1, 0 : HEAD_DIM]

    x_normed_bsd = pl.reshape(x_normed, [B, S, D])
    cmp_kv_proj = pl.create_tensor([B, S, HEAD_DIM], dtype=pl.FP32)
    position_ids_bsd = pl.reshape(position_ids, [B, S])
    cmp_slot_mapping_bsd = pl.reshape(cmp_slot_mapping, [B, S])
    state_slot_mapping_bsd = pl.reshape(state_slot_mapping, [B, S])
    compressor_ratio128(
        x_normed_bsd, cmp_kv_proj,
        compress_state, compress_state_block_table,
        cmp_wkv, cmp_wgate, cmp_ape, cmp_norm_w,
        cmp_cos_il, cmp_sin_signed, cmp_kv,
        position_ids_bsd, cmp_slot_mapping_bsd, state_slot_mapping_bsd,
        late_dep,
    )

    attn_out = pl.create_tensor([T, D], dtype=pl.BF16)
    o_packed = pl.create_tensor([O_GROUPS * T, O_GROUP_IN], dtype=pl.BF16)
    merge_tid = sparse_attn_hca_packed(
        q, kv_cache, window_swa_indices, window_swa_lens,
        cmp_kv, cmp_block_table, cmp_seq_lens,
        attn_sink, rope_cos_t, rope_sin_t, o_packed, my_rank, out_rope_cos_il, out_rope_sin_signed, out_rope_swap_idx,
    )
    o_proj_tp_core(
        o_packed, merge_tid, wo_a_shard, wo_b_shard, wo_b_scale, attn_out,
        reduce_window, scale_window, reduce_signal, sync_signal, my_rank, oproj_epoch,
    )

    hc_post(attn_out, x_hc, post_t, comb_t, x_out)
    return x_out
