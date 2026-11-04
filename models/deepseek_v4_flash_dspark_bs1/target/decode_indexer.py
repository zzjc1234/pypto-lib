# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl

from .config import FLASH as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, C4A_COMPRESSOR_BLOCK_SIZE, IDX_CACHE_MAX_BLOCKS, FP32_NEG_INF, INT8_SCALE_MAX, INT8_AMAX_EPS

from .decode_indexer_compressor import indexer_compressor

B = DECODE_BATCH

S = DECODE_SEQ

T = B * S

D = M.hidden_size

Q_LORA = M.q_lora_rank

ROPE_HEAD_DIM = M.qk_rope_head_dim

IDX_N_HEADS = M.index_n_heads

IDX_HEAD_DIM = M.index_head_dim

IDX_NOPE_HEAD_DIM = M.index_nope_head_dim

WEIGHTS_SCALE = M.index_weights_scale

MAX_SEQ_LEN = M.max_position_embeddings

COMPRESS_RATIO = 4  # the indexer only runs on ratio-4 layers

IDX_STORAGE_BLOCK_SIZE = BLOCK_SIZE // COMPRESS_RATIO

IDX_TOPK = M.index_topk

INNER_OVERLAP = COMPRESS_RATIO == 4

INNER_COFF = 1 + int(INNER_OVERLAP)

INNER_HEAD_DIM = IDX_HEAD_DIM

INNER_OUT_DIM = INNER_COFF * INNER_HEAD_DIM

INNER_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE

INNER_STATE_MAX_BLOCKS = (
    MAX_SEQ_LEN + INNER_STATE_BLOCK_SIZE - 1
) // INNER_STATE_BLOCK_SIZE

INNER_STATE_BLOCK_NUM_DYN = pl.dynamic('NATIVE_S_INNER_STATE_BLOCK_NUM_DYN')

INNER_STATE_DIM = 2 * INNER_OUT_DIM

IDX_KV_LEN = MAX_SEQ_LEN // COMPRESS_RATIO

IDX_CACHE_BLOCK_NUM_DYN = pl.dynamic('NATIVE_S_IDX_CACHE_BLOCK_NUM_DYN')

SCORE_LEN = IDX_KV_LEN

CACHE_TILE = 64

assert BLOCK_SIZE % CACHE_TILE == 0, (
    "CACHE_TILE must not cross a paged idx_kv_cache block"
)

REDUCE_TILE = BLOCK_SIZE // COMPRESS_RATIO

SCORE_PAGE_GROUP = 4

SCORE_GROUP_TILE = SCORE_PAGE_GROUP * REDUCE_TILE

SCORE_LANE_PAGES = SCORE_PAGE_GROUP // 2

SCORE_LANE_TILE = SCORE_LANE_PAGES * REDUCE_TILE

assert SCORE_PAGE_GROUP % 2 == 0, "an AIV lane must receive whole pages"

SCORE_SLOT_NUM = 2

REDUCE_NSPLIT = 16

IDX_N_RANKS = 8

TOKEN_SHARDED = T == IDX_N_RANKS

QUERY_SLOTS = 1 if TOKEN_SHARDED else T

Q_TILE = 256

Q_OUT_TILE = 1024

MM_N_TILE = 512

MM_ROW_TILE = 16

T_PAD = ((T + MM_ROW_TILE - 1) // MM_ROW_TILE) * MM_ROW_TILE

BATCH_QUERY_PROJECTION = T == 6 and not TOKEN_SHARDED

QR_ACC_ROWS = T_PAD if BATCH_QUERY_PROJECTION else QUERY_SLOTS * T_PAD

QR_ACC_ROW_STRIDE = 1 if BATCH_QUERY_PROJECTION else T_PAD

assert T_PAD == MM_ROW_TILE, (
    "weights_proj single-row-tile scope assumes decode T <= MM_ROW_TILE"
)

D_TILE = 512

WEIGHTS_OK = 4

WEIGHTS_K_SLICE = D // WEIGHTS_OK

assert WEIGHTS_K_SLICE % D_TILE == 0

QH_QUANT_TILE = 64

QH_MM_TILE = 64

QH_HEAD_DIM_TILE = 64

ROPE_ROW_TILE = 32

TOPK_HALF_LEN = SCORE_LEN // 2

TOPK_HALF_PAIR_OFFSET = 2 * TOPK_HALF_LEN

TOPK_PAIR_WIDTH = 2 * IDX_TOPK

assert SCORE_LEN == 2 * TOPK_HALF_LEN, (
    "decode indexer topk expects an even score length"
)

TOPK_SINGLE_ROW = SCORE_LEN == 4096

TOPK_LEAF_LEN = 8192

TOPK_RUN_LANES = TOPK_LEAF_LEN // 2

TOPK_LEAVES = max(1, SCORE_LEN // TOPK_LEAF_LEN)

TOPK_LEAF_BLOCKS = min(TOPK_LEAVES, REDUCE_NSPLIT)

TOPK_MERGE_FANIN = 3

TOPK_ARENA_ROWS = TOPK_LEAVES + TOPK_MERGE_FANIN

assert TOPK_SINGLE_ROW or SCORE_LEN % TOPK_LEAF_LEN == 0, (
    "decode indexer topk leaves must tile the score row"
)

assert not TOPK_SINGLE_ROW or TOPK_HALF_LEN == 2048, (
    "decode indexer 4096-value topk uses two 2048-value halves"
)

assert IDX_TOPK <= TOPK_HALF_LEN, (
    "per-half candidate list must cover the final topk width"
)

@pl.jit.inline
def prepare_indexer_rope_indices(
    rope_swap_idx_t: pl.Out[pl.Tensor[[ROPE_ROW_TILE, ROPE_HEAD_DIM], pl.INT32]],
):
    with pl.at(
        level=pl.Level.CORE_GROUP,
        name_hint="qr_rope_swap_idx",
        allow_early_resolve=True,
    ):
        sw_col = pl.col_expand_mul(
            pl.full([ROPE_ROW_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0),
            pl.cast(
                pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32
            ),
        )
        sw_dup_f = pl.cast(
            pl.cast(pl.mul(sw_col, 0.5), target_type=pl.INT32, mode="trunc"),
            target_type=pl.FP32,
        )
        sw_lane = pl.sub(sw_col, pl.mul(sw_dup_f, 2.0))  # j%2
        rope_swap_idx_t[0:ROPE_ROW_TILE, 0:ROPE_HEAD_DIM] = pl.cast(
            pl.sub(pl.add(sw_col, 1.0), pl.mul(sw_lane, 2.0)), target_type=pl.INT32
        )  # j^1

@pl.jit.inline
def indexer_all_visible(
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    materialize_scores: pl.Scalar[pl.INT32],
) -> pl.Scalar[pl.INT32]:
    all_visible = pl.cast(0, pl.INT32)
    if materialize_scores == 0:
        all_visible = pl.cast(1, pl.INT32)
        for query in pl.range(T):
            batch = query // S
            token = query - batch * S
            cache_len = pl.read(kv_seq_lens, [batch]) // COMPRESS_RATIO
            position = pl.read(position_ids, [batch, token])
            visible = pl.min(pl.min(cache_len, (position + 1) // COMPRESS_RATIO), SCORE_LEN)
            if visible > IDX_TOPK:
                all_visible = pl.cast(0, pl.INT32)
    return all_visible

@pl.jit.inline
def indexer(
    x: pl.Tensor[[B, S, D], pl.BF16],
    qr: pl.Tensor[[T, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T, 1], pl.FP32],
    wq_b: pl.Tensor[[Q_LORA, IDX_N_HEADS * IDX_HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[IDX_N_HEADS * IDX_HEAD_DIM], pl.FP32],
    weights_proj: pl.Tensor[[D, IDX_N_HEADS], pl.BF16],
    # Interleave-duplicated (j>>1) cos and sign-folded sin, built once by the caller:
    #   cos[j] = cos_half[j>>1];  sin[j] = sin_half[j>>1] * sign[j], sign = [-1,+1,...]
    cos: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    cmp_cos: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    cmp_sin: pl.Tensor[[T, ROPE_HEAD_DIM], pl.FP32],
    hadamard: pl.Tensor[
        [IDX_HEAD_DIM, IDX_HEAD_DIM], pl.BF16
    ],  # shared by q rotation and inner Compressor
    inner_kv: pl.Tensor[[B, S, INNER_HEAD_DIM], pl.FP32],
    inner_compress_state: pl.Tensor[
        [INNER_STATE_BLOCK_NUM_DYN, INNER_STATE_BLOCK_SIZE, INNER_STATE_DIM], pl.FP32
    ],
    inner_compress_state_block_table: pl.Tensor[[B, INNER_STATE_MAX_BLOCKS], pl.INT32],
    inner_wkv: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_wgate: pl.Tensor[[INNER_OUT_DIM, D], pl.BF16],
    inner_ape: pl.Tensor[[COMPRESS_RATIO, INNER_OUT_DIM], pl.FP32],
    inner_norm_w: pl.Tensor[[INNER_HEAD_DIM], pl.BF16],
    # C8 indexer cache: INT8 KV (quant-on-write) + per-position FP32 dequant scale; no bf16 cache.
    idx_kv_cache: pl.InOut[
        pl.Tensor[
            [IDX_CACHE_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, IDX_HEAD_DIM], pl.INT8
        ]
    ],
    idx_kv_scale: pl.InOut[
        pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32]
    ],
    idx_block_table: pl.Tensor[[B, IDX_CACHE_MAX_BLOCKS], pl.INT32],
    score: pl.Tensor[[B, S, SCORE_LEN], pl.FP32],
    topk_idxs: pl.Tensor[[B, S, SCORE_LEN], pl.INT32],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    idx_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    kv_seq_lens: pl.Tensor[[B], pl.INT32],
    offset: pl.Scalar[pl.INT32],
    late_dep: pl.Scalar[pl.TASK_ID],
    my_rank: pl.Scalar[pl.INT32],
    rope_swap_idx_t: pl.Tensor[[ROPE_ROW_TILE, ROPE_HEAD_DIM], pl.INT32],
    materialize_scores: pl.Scalar[pl.INT32],
):
    all_visible = indexer_all_visible(kv_seq_lens, position_ids, materialize_scores)
    if all_visible != 0:
        indexer_compressor(
            x,
            inner_kv,
            inner_compress_state,
            inner_compress_state_block_table,
            inner_wkv,
            inner_wgate,
            inner_ape,
            inner_norm_w,
            cmp_cos,
            cmp_sin,
            hadamard,
            idx_kv_cache,
            idx_kv_scale,
            position_ids,
            idx_slot_mapping,
            inner_state_slot_mapping,
            late_dep,
        )

        topk_idxs_flat_fast = pl.reshape(topk_idxs, [T, SCORE_LEN])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="all_visible_init", allow_early_resolve=True) as init_tid:
            if TOPK_SINGLE_ROW:
                topk_idxs_flat_fast[0:T, :] = pl.full([T, SCORE_LEN], dtype=pl.INT32, value=-1)
            else:
                topk_idxs_flat_fast[0:T, 0:IDX_TOPK] = pl.full([T, IDX_TOPK], dtype=pl.INT32, value=-1)
        for query_slot in pl.spmd(QUERY_SLOTS, name_hint="all_visible_indices", deps=[init_tid], allow_early_resolve=True):
            query_token = query_slot
            if TOKEN_SHARDED:
                query_token = pl.cast(my_rank, pl.INDEX)
            query_batch = query_token // S
            query_column = query_token - query_batch * S
            query_cache_len = pl.read(kv_seq_lens, [query_batch]) // COMPRESS_RATIO
            query_position = pl.read(position_ids, [query_batch, query_column])
            query_visible = pl.min(pl.min(query_cache_len, (query_position + 1) // COMPRESS_RATIO), SCORE_LEN)
            if query_visible > 0:
                indices = pl.arange(0, [1, IDX_TOPK], dtype=pl.INT32)
                valid_indices = pl.set_validshape(indices, 1, query_visible)
                offset_i32 = pl.cast(offset, pl.INT32)
                topk_idxs_flat_fast[query_token : query_token + 1, 0:IDX_TOPK] = pl.add(valid_indices, offset_i32)
    else:
        qr_acc_pad = pl.create_tensor([QR_ACC_ROWS, IDX_N_HEADS * IDX_HEAD_DIM], dtype=pl.INT32)
        if BATCH_QUERY_PROJECTION:
            for ot in pl.spmd(IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE, name_hint='idx_qr_proj_matmul', allow_early_resolve=True):
                o_base = ot * Q_OUT_TILE
                for ns in pl.range(0, Q_OUT_TILE, MM_N_TILE):
                    qr_acc = pl.create_tensor([MM_ROW_TILE, MM_N_TILE], dtype=pl.INT32)
                    for kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
                        q0 = kb * Q_TILE
                        qr_tile = pl.slice(qr, [T_PAD, Q_TILE], [0, q0], valid_shape=[T, Q_TILE])
                        wq_tile = wq_b[q0:q0 + Q_TILE, o_base + ns:o_base + ns + MM_N_TILE]
                        qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile, init_cond=kb == 0)
                    qr_acc_pad[0:T_PAD, o_base + ns:o_base + ns + MM_N_TILE] = qr_acc
        else:
            for ot in pl.spmd(
                QUERY_SLOTS * (IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE),
                name_hint="idx_qr_proj_matmul",
                allow_early_resolve=True,
            ):
                qproj_slot = ot // (IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE)
                qproj_token = qproj_slot
                if TOKEN_SHARDED:
                    qproj_token = pl.cast(my_rank, pl.INDEX)
                qproj_column = ot % (IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE)
                o_base = qproj_column * Q_OUT_TILE
                for ns in pl.range(0, Q_OUT_TILE, MM_N_TILE):
                    qr_acc = pl.create_tensor([MM_ROW_TILE, MM_N_TILE], dtype=pl.INT32)
                    for kb in pl.pipeline(0, Q_LORA // Q_TILE, stage=2):
                        q0 = kb * Q_TILE
                        qr_tile = pl.slice(
                            qr, [T_PAD, Q_TILE], [qproj_token, q0], valid_shape=[1, Q_TILE]
                        )
                        wq_tile = wq_b[q0 : q0 + Q_TILE, o_base + ns : o_base + ns + MM_N_TILE]
                        qr_acc = pl.matmul_acc(qr_acc, qr_tile, wq_tile, init_cond=(kb == 0))
                    qr_acc_pad[qproj_slot * T_PAD : qproj_slot * T_PAD + T_PAD, o_base + ns : o_base + ns + MM_N_TILE] = qr_acc
        qr_bf16 = pl.create_tensor([QUERY_SLOTS * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.BF16)

        for ot in pl.spmd(
            QUERY_SLOTS * (IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE),
            name_hint="idx_qr_dequant_rope",
            allow_early_resolve=True,
        ):
            qdeq_slot = ot // (IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE)
            qdeq_token = qdeq_slot
            if TOKEN_SHARDED:
                qdeq_token = pl.cast(my_rank, pl.INDEX)
            qdeq_column = ot % (IDX_N_HEADS * IDX_HEAD_DIM // Q_OUT_TILE)
            o_base = qdeq_column * Q_OUT_TILE
            o0 = qdeq_slot * IDX_N_HEADS + qdeq_column * (Q_OUT_TILE // IDX_HEAD_DIM)
            wq_scale = pl.reshape(wq_b_scale[o_base : o_base + Q_OUT_TILE], [1, Q_OUT_TILE])
            acc_fp32 = pl.cast(
                qr_acc_pad[qdeq_slot * QR_ACC_ROW_STRIDE : qdeq_slot * QR_ACC_ROW_STRIDE + 1, o_base : o_base + Q_OUT_TILE],
                target_type=pl.FP32,
                mode="none",
            )
            query_scale = pl.read(qr_scale, [qdeq_token, 0])
            qr_scaled = pl.mul(acc_fp32, query_scale)
            qr_dequant = pl.col_expand_mul(qr_scaled, wq_scale)
            qr_dequant_flat = pl.reshape(
                qr_dequant, [Q_OUT_TILE // IDX_HEAD_DIM, IDX_HEAD_DIM]
            )
            batch_idx = qdeq_token // S
            rope_swap_idx = rope_swap_idx_t[0 : Q_OUT_TILE // IDX_HEAD_DIM, 0:ROPE_HEAD_DIM]
            cos_row = cos[qdeq_token : qdeq_token + 1, 0:ROPE_HEAD_DIM]
            sin_row = sin[qdeq_token : qdeq_token + 1, 0:ROPE_HEAD_DIM]
            qr_nope_slice = qr_dequant_flat[:, 0:IDX_NOPE_HEAD_DIM]
            qr_rope_slice = qr_dequant_flat[:, IDX_NOPE_HEAD_DIM:IDX_HEAD_DIM]
            qr_swapped = pl.gather(qr_rope_slice, dim=-1, index=rope_swap_idx)
            rope_direct = pl.col_expand_mul(qr_rope_slice, cos_row)
            rope_swapped = pl.col_expand_mul(qr_swapped, sin_row)
            rope_rot = pl.add(rope_direct, rope_swapped)
            qr_nope_bf16 = pl.cast(qr_nope_slice, target_type=pl.BF16, mode="rint")
            qr_rope_bf16 = pl.cast(rope_rot, target_type=pl.BF16, mode="rint")
            qr_vec = pl.concat(qr_nope_bf16, qr_rope_bf16)
            qr_bf16[o0 : o0 + Q_OUT_TILE // IDX_HEAD_DIM, :] = qr_vec

        # cube-only scope: q @ hadamard lands in GM, keeping the vector amax/quant below
        # in its own scope so the two run as separate cube and vector tasks.
        qh_acc_gm = pl.create_tensor([QUERY_SLOTS * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.FP32)
        for idx in pl.spmd(
            QUERY_SLOTS * (IDX_N_HEADS // QH_MM_TILE),
            name_hint="qr_hadamard_matmul",
            allow_early_resolve=True,
        ):
            o0 = idx * QH_MM_TILE
            qh_acc = pl.matmul(
                qr_bf16[o0 : o0 + QH_MM_TILE, :], hadamard, out_dtype=pl.FP32
            )
            qh_acc_gm[o0 : o0 + QH_MM_TILE, :] = qh_acc

        qr_hadamard_i8 = pl.create_tensor([QUERY_SLOTS * IDX_N_HEADS, IDX_HEAD_DIM], dtype=pl.INT8)
        qr_hadamard_scale_dq = pl.create_tensor([QUERY_SLOTS * IDX_N_HEADS, 1], dtype=pl.FP32)
        for idx in pl.spmd(
            QUERY_SLOTS * (IDX_N_HEADS // QH_QUANT_TILE),
            name_hint="qr_hadamard_quant",
            allow_early_resolve=True,
        ):
            o0 = idx * QH_QUANT_TILE
            qh_amax = pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
            for h0 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
                qh_a_f32 = qh_acc_gm[o0 : o0 + QH_QUANT_TILE, h0 : h0 + QH_HEAD_DIM_TILE]
                qh_a_abs = pl.abs(qh_a_f32)
                qh_a_max = pl.reshape(pl.row_max(qh_a_abs), [1, QH_QUANT_TILE])
                qh_amax = pl.maximum(qh_amax, qh_a_max)
            qh_scale_quant_row = pl.div(
                pl.full([1, QH_QUANT_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), qh_amax
            )
            qh_scale_dq = pl.reshape(pl.recip(qh_scale_quant_row), [QH_QUANT_TILE, 1])
            qr_hadamard_scale_dq[o0 : o0 + QH_QUANT_TILE, :] = qh_scale_dq
            qh_scale_quant = pl.reshape(qh_scale_quant_row, [QH_QUANT_TILE, 1])
            for h1 in pl.range(0, IDX_HEAD_DIM, QH_HEAD_DIM_TILE):
                qh_q_f32 = qh_acc_gm[o0 : o0 + QH_QUANT_TILE, h1 : h1 + QH_HEAD_DIM_TILE]
                qh_q_scaled = pl.row_expand_mul(qh_q_f32, qh_scale_quant)
                qh_q_i32 = pl.cast(qh_q_scaled, target_type=pl.INT32, mode="rint")
                qh_q_half = pl.cast(qh_q_i32, target_type=pl.FP16, mode="round")
                qh_i8 = pl.cast(
                    qh_q_half, target_type=pl.INT8, mode="trunc", saturation_mode="on"
                )
                qr_hadamard_i8[o0 : o0 + QH_QUANT_TILE, h1 : h1 + QH_HEAD_DIM_TILE] = qh_i8

        x_flat = pl.reshape(x, [T, D])
        weights = pl.create_tensor([T_PAD, IDX_N_HEADS], dtype=pl.FP32)
        weights_partial = pl.create_tensor(
            [WEIGHTS_OK * MM_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32
        )
        # Deferred behind the caller's rms_norm dummy barrier: qkv's qr_proj_matmul is the
        # critical path and must win the cores when rms_norm retires.
        with pl.spmd(WEIGHTS_OK, name_hint="weights_proj", deps=[late_dep]):
            kb = pl.tile.get_block_idx()
            k_base = kb * WEIGHTS_K_SLICE
            weights_acc = pl.create_tensor([MM_ROW_TILE, IDX_N_HEADS], dtype=pl.FP32)
            for db in pl.range(WEIGHTS_K_SLICE // D_TILE):
                d0 = k_base + db * D_TILE
                x_tile = pl.slice(
                    x_flat,
                    [MM_ROW_TILE, D_TILE],
                    [0, d0],
                    valid_shape=[pl.min(MM_ROW_TILE, T), D_TILE],
                )
                weights_proj_tile = weights_proj[d0 : d0 + D_TILE, :]
                weights_acc = pl.matmul_acc(
                    weights_acc, x_tile, weights_proj_tile, init_cond=(db == 0)
                )
            weights_partial[kb * MM_ROW_TILE : kb * MM_ROW_TILE + MM_ROW_TILE, :] = (
                weights_acc
            )

        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="weights_proj_reduce",
            allow_early_resolve=True,
        ):
            w_sum = weights_partial[0:MM_ROW_TILE, :]
            for kb in pl.unroll(1, WEIGHTS_OK):
                w_sum = pl.add(
                    w_sum,
                    weights_partial[kb * MM_ROW_TILE : kb * MM_ROW_TILE + MM_ROW_TILE, :],
                )
            weights[0:MM_ROW_TILE, :] = pl.mul(w_sum, WEIGHTS_SCALE)

        indexer_compressor(
            x,
            inner_kv,
            inner_compress_state,
            inner_compress_state_block_table,
            inner_wkv,
            inner_wgate,
            inner_ape,
            inner_norm_w,
            cmp_cos,
            cmp_sin,
            hadamard,
            idx_kv_cache,
            idx_kv_scale,
            position_ids,
            idx_slot_mapping,
            inner_state_slot_mapping,
            late_dep,
        )

        idx_block_num = pl.tensor.dim(idx_kv_cache, 0)
        kv_cache_i8_flat = pl.reshape(
            idx_kv_cache, [idx_block_num * IDX_STORAGE_BLOCK_SIZE, IDX_HEAD_DIM]
        )
        kv_scale_flat = pl.reshape(
            idx_kv_scale, [idx_block_num * IDX_STORAGE_BLOCK_SIZE, 1]
        )
        idx_block_table_flat = pl.reshape(idx_block_table, [B * IDX_CACHE_MAX_BLOCKS])
        score_flat = pl.reshape(score, [T, SCORE_LEN])
        # (token, head) row views: one 512B row load per chunk instead of 64 x 4B rows.
        qh_scale_flat = pl.reshape(qr_hadamard_scale_dq, [1, QUERY_SLOTS * IDX_N_HEADS])
        weights_flat = pl.reshape(weights, [1, T_PAD * IDX_N_HEADS])

        # S8 scores one rank-owned token; other widths score every real query locally.
        for split in pl.spmd(
            QUERY_SLOTS * REDUCE_NSPLIT,
            name_hint="score",
            allow_early_resolve=True,
            optimizations=[pl.cross_core_slot(slot_num=SCORE_SLOT_NUM)],
        ):
            score_slot = split // REDUCE_NSPLIT
            score_split = split % REDUCE_NSPLIT
            t0 = score_slot
            if TOKEN_SHARDED:
                t0 = pl.cast(my_rank, pl.INDEX)
            b = t0 // S
            s0 = t0 - b * S
            clen_b = pl.read(kv_seq_lens, [b]) // COMPRESS_RATIO
            pos_k = pl.read(position_ids, [b, s0])
            visible = pl.min(pl.min(clen_b, (pos_k + 1) // COMPRESS_RATIO), SCORE_LEN)
            cblk_g = (visible + REDUCE_TILE - 1) // REDUCE_TILE
            lane_iters = pl.max(cblk_g - score_split + REDUCE_NSPLIT - 1, 0) // REDUCE_NSPLIT
            group_iters = (lane_iters + SCORE_PAGE_GROUP - 1) // SCORE_PAGE_GROUP
            c0 = t0 * IDX_N_HEADS
            score_head0 = score_slot * IDX_N_HEADS
            qr_chunk = pl.slice(qr_hadamard_i8, [IDX_N_HEADS, IDX_HEAD_DIM], [score_head0, 0])
            qh_scale_c = pl.slice(qh_scale_flat, [1, IDX_N_HEADS], [0, score_head0])
            weights_c = pl.slice(weights_flat, [1, IDX_N_HEADS], [0, c0])
            for g_local in pl.pipeline(0, group_iters, stage=2):
                page0 = g_local * SCORE_PAGE_GROUP
                kv_group = pl.create_l1([SCORE_GROUP_TILE, IDX_HEAD_DIM], pl.INT8)
                for pg in pl.unroll(SCORE_PAGE_GROUP):
                    page_idx = pl.min(page0 + pg, lane_iters - 1)
                    cb = score_split + page_idx * REDUCE_NSPLIT
                    idx_blk_id = pl.cast(
                        pl.read(idx_block_table_flat, [b * IDX_CACHE_MAX_BLOCKS + cb]),
                        pl.INDEX,
                    )
                    kv0 = idx_blk_id * IDX_STORAGE_BLOCK_SIZE
                    kv_group = pl.gather_row(
                        kv_group,
                        kv_cache_i8_flat,
                        [pg * REDUCE_TILE, 0],
                        [kv0, 0],
                        [REDUCE_TILE, IDX_HEAD_DIM],
                    )
                score_acc_red = pl.matmul(
                    kv_group, qr_chunk, out_dtype=pl.INT32, b_trans=True
                )
                for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
                    score_acc_shard = pl.aiv_shard(score_acc_red)
                    score_tile_red = pl.cast(
                        score_acc_shard, target_type=pl.FP32, mode="none"
                    )
                    score_tile_red = pl.col_expand_mul(score_tile_red, qh_scale_c)
                    relu_score_red = pl.maximum(score_tile_red, 0.0)
                    weighted_score_red = pl.col_expand_mul(relu_score_red, weights_c)
                    weighted_score_line = pl.reshape(pl.row_sum(weighted_score_red), [1, SCORE_LANE_TILE])
                    lane_page0 = page0 + pl.cast(aiv_id * SCORE_LANE_PAGES, pl.INDEX)
                    for lp in pl.unroll(SCORE_LANE_PAGES):
                        if lane_page0 + lp < lane_iters:
                            cb_lane = score_split + (lane_page0 + lp) * REDUCE_NSPLIT
                            lane_blk_id = pl.cast(
                                pl.read(idx_block_table_flat, [b * IDX_CACHE_MAX_BLOCKS + cb_lane]),
                                pl.INDEX,
                            )
                            kv0_lane = lane_blk_id * IDX_STORAGE_BLOCK_SIZE
                            kv_dq_page = kv_scale_flat[kv0_lane : kv0_lane + REDUCE_TILE, :]
                            r0 = lp * REDUCE_TILE
                            kv_dq_line = pl.reshape(kv_dq_page, [1, REDUCE_TILE])
                            page_score_s = pl.mul(weighted_score_line[:, r0 : r0 + REDUCE_TILE], kv_dq_line)
                            cache0 = cb_lane * REDUCE_TILE
                            score_flat[t0 : t0 + 1, cache0 : cache0 + REDUCE_TILE] = page_score_s

        topk_idxs_flat = pl.reshape(topk_idxs, [T, SCORE_LEN])

        # Top-k consumes only the query rows scored on this rank. The S8 token path
        # keeps other rows invalid; head-sharded widths produce all real query rows.
        if TOPK_SINGLE_ROW:
            with pl.at(
                level=pl.Level.CORE_GROUP, name_hint="topk_init", allow_early_resolve=True
            ) as topk_init_tid:
                topk_idxs_flat[0:T, :] = pl.full([T, SCORE_LEN], dtype=pl.INT32, value=-1)

            for _tk in pl.spmd(
                QUERY_SLOTS, name_hint="topk", allow_early_resolve=True, deps=[topk_init_tid]
            ):
                t = _tk
                if TOKEN_SHARDED:
                    t = pl.cast(my_rank, pl.INDEX)
                batch_idx = t // S
                token_s = t - batch_idx * S
                cache_len_b = pl.read(kv_seq_lens, [batch_idx]) // COMPRESS_RATIO
                pos_t = pl.read(position_ids, [batch_idx, token_s])
                visible_len_t = pl.min(
                    pl.min(cache_len_b, (pos_t + 1) // COMPRESS_RATIO), SCORE_LEN
                )
                if visible_len_t > 0:
                    offset_i32 = pl.cast(offset, target_type=pl.INT32)
                    if visible_len_t <= IDX_TOPK:
                        topk_visible = pl.arange(0, [1, IDX_TOPK], dtype=pl.INT32)
                        topk_valid = pl.set_validshape(topk_visible, 1, visible_len_t)
                        topk_offset = pl.add(topk_valid, offset_i32)
                        topk_idxs_flat[t:t + 1, 0:IDX_TOPK] = topk_offset
                    else:
                        score_full_raw = score_flat[t : t + 1, 0:SCORE_LEN]
                        score_full = pl.fillpad(
                            pl.set_validshape(score_full_raw, 1, visible_len_t),
                            pad_value=pl.PadValue.min,
                        )
                        score_full = pl.maximum(score_full, FP32_NEG_INF)
                        idx_init = pl.arange(0, [1, SCORE_LEN], dtype=pl.UINT32)
                        sorted_full = pl.sort32(score_full, idx_init)
                        sorted_full = pl.mrgsort(sorted_full, block_len=64)
                        sorted_full = pl.mrgsort(sorted_full, block_len=256)
                        sorted_full = pl.mrgsort(sorted_full, block_len=1024)

                        # After the 1024 merge, the 4096-score row is two sorted 2048-score
                        # runs. sort32/mrgsort keeps score/index pairs interleaved, so the
                        # second 2048-score run starts at pair-lane offset 2 * 2048.
                        half0_candidates = sorted_full[:, 0:TOPK_PAIR_WIDTH]
                        half1_candidates = sorted_full[
                            :, TOPK_HALF_PAIR_OFFSET : TOPK_HALF_PAIR_OFFSET + TOPK_PAIR_WIDTH
                        ]
                        merged_candidates = pl.mrgsort(half0_candidates, half1_candidates)
                        topk_pairs = merged_candidates[:, 0:TOPK_PAIR_WIDTH]
                        topk_idxs_tile = pl.gather(
                            topk_pairs,
                            mask_pattern=pl.tile.MaskPattern.P1010,
                            output_dtype=pl.INT32,
                        )
                        valid_topk = pl.min(IDX_TOPK, visible_len_t)
                        topk_idxs_valid = pl.set_validshape(topk_idxs_tile, 1, valid_topk)
                        topk_idxs_flat[t : t + 1, 0:IDX_TOPK] = pl.add(topk_idxs_valid, offset_i32)
        else:
            topk_leaf_pairs = pl.create_tensor([QUERY_SLOTS * TOPK_ARENA_ROWS, TOPK_PAIR_WIDTH], dtype=pl.FP32)
            with pl.at(
                level=pl.Level.CORE_GROUP, name_hint="topk_init", allow_early_resolve=True
            ) as topk_init_tid:
                topk_idxs_flat[0:T, 0:IDX_TOPK] = pl.full([T, IDX_TOPK], dtype=pl.INT32, value=-1)

            for leaf_block in pl.spmd(
                QUERY_SLOTS * TOPK_LEAF_BLOCKS, name_hint="topk_leaf", allow_early_resolve=True
            ):
                leaf_slot = leaf_block // TOPK_LEAF_BLOCKS
                leaf_lane = leaf_block % TOPK_LEAF_BLOCKS
                leaf_arena0 = leaf_slot * TOPK_ARENA_ROWS
                t = leaf_slot
                if TOKEN_SHARDED:
                    t = pl.cast(my_rank, pl.INDEX)
                batch_idx = t // S
                token_s = t - batch_idx * S
                cache_len_b = pl.read(kv_seq_lens, [batch_idx]) // COMPRESS_RATIO
                pos_t = pl.read(position_ids, [batch_idx, token_s])
                visible_len_t = pl.min(
                    pl.min(cache_len_b, (pos_t + 1) // COMPRESS_RATIO), SCORE_LEN
                )
                leaf_count = (visible_len_t + TOPK_LEAF_LEN - 1) // TOPK_LEAF_LEN
                for leaf in pl.range(leaf_lane, leaf_count, TOPK_LEAF_BLOCKS):
                    leaf_begin = leaf * TOPK_LEAF_LEN
                    leaf_valid = pl.min(TOPK_LEAF_LEN, visible_len_t - leaf_begin)
                    leaf_raw = score_flat[t : t + 1, leaf_begin : leaf_begin + TOPK_LEAF_LEN]
                    leaf_scores = pl.fillpad(
                        pl.set_validshape(leaf_raw, 1, leaf_valid),
                        pad_value=pl.PadValue.min,
                    )
                    leaf_scores = pl.maximum(leaf_scores, FP32_NEG_INF)
                    leaf_idx = pl.reinterpret_view(
                        pl.add(
                            pl.arange(0, [1, TOPK_LEAF_LEN], dtype=pl.INT32),
                            pl.cast(leaf_begin, target_type=pl.INT32),
                        ),
                        pl.UINT32,
                    )
                    leaf_sorted = pl.sort32(leaf_scores, leaf_idx)
                    leaf_sorted = pl.mrgsort(leaf_sorted, block_len=64)
                    leaf_sorted = pl.mrgsort(leaf_sorted, block_len=256)
                    leaf_sorted = pl.mrgsort(leaf_sorted, block_len=1024)
                    # Four sorted 2048-score runs; each run's top IDX_TOPK pairs lead it.
                    leaf_merged = pl.mrgsort(
                        leaf_sorted[:, 0:TOPK_PAIR_WIDTH],
                        leaf_sorted[:, TOPK_RUN_LANES : TOPK_RUN_LANES + TOPK_PAIR_WIDTH],
                        leaf_sorted[:, 2 * TOPK_RUN_LANES : 2 * TOPK_RUN_LANES + TOPK_PAIR_WIDTH],
                        leaf_sorted[:, 3 * TOPK_RUN_LANES : 3 * TOPK_RUN_LANES + TOPK_PAIR_WIDTH],
                    )
                    topk_leaf_pairs[leaf_arena0 + leaf : leaf_arena0 + leaf + 1, 0:TOPK_PAIR_WIDTH] = leaf_merged[:, 0:TOPK_PAIR_WIDTH]

            for _tk in pl.spmd(
                QUERY_SLOTS, name_hint="topk_merge", allow_early_resolve=True, deps=[topk_init_tid]
            ):
                merge_arena0 = _tk * TOPK_ARENA_ROWS
                t = _tk
                if TOKEN_SHARDED:
                    t = pl.cast(my_rank, pl.INDEX)
                batch_idx = t // S
                token_s = t - batch_idx * S
                cache_len_b = pl.read(kv_seq_lens, [batch_idx]) // COMPRESS_RATIO
                pos_t = pl.read(position_ids, [batch_idx, token_s])
                visible_len_t = pl.min(
                    pl.min(cache_len_b, (pos_t + 1) // COMPRESS_RATIO), SCORE_LEN
                )
                if visible_len_t > 0:
                    offset_i32 = pl.cast(offset, target_type=pl.INT32)
                    if visible_len_t <= IDX_TOPK:
                        merge_visible = pl.arange(0, [1, IDX_TOPK], dtype=pl.INT32)
                        merge_valid = pl.set_validshape(merge_visible, 1, visible_len_t)
                        merge_offset = pl.add(merge_valid, offset_i32)
                        topk_idxs_flat[t:t + 1, 0:IDX_TOPK] = merge_offset
                    else:
                        leaf_count = (visible_len_t + TOPK_LEAF_LEN - 1) // TOPK_LEAF_LEN
                        merge_steps = (leaf_count + TOPK_MERGE_FANIN - 2) // TOPK_MERGE_FANIN
                        # The last step's missing inputs are most-negative lists.
                        pad_pairs = pl.full([1, TOPK_PAIR_WIDTH], dtype=pl.FP32, value=FP32_NEG_INF)
                        for pad_row in pl.range(leaf_count, 1 + merge_steps * TOPK_MERGE_FANIN):
                            topk_leaf_pairs[merge_arena0 + pad_row : merge_arena0 + pad_row + 1, 0:TOPK_PAIR_WIDTH] = pad_pairs
                        for step in pl.range(merge_steps):
                            row0 = 1 + step * TOPK_MERGE_FANIN
                            step_merged = pl.mrgsort(
                                topk_leaf_pairs[merge_arena0 : merge_arena0 + 1, 0:TOPK_PAIR_WIDTH],
                                topk_leaf_pairs[merge_arena0 + row0 : merge_arena0 + row0 + 1, 0:TOPK_PAIR_WIDTH],
                                topk_leaf_pairs[merge_arena0 + row0 + 1 : merge_arena0 + row0 + 2, 0:TOPK_PAIR_WIDTH],
                                topk_leaf_pairs[merge_arena0 + row0 + 2 : merge_arena0 + row0 + 3, 0:TOPK_PAIR_WIDTH],
                            )
                            topk_leaf_pairs[merge_arena0 : merge_arena0 + 1, 0:TOPK_PAIR_WIDTH] = step_merged[:, 0:TOPK_PAIR_WIDTH]
                        final_pairs = topk_leaf_pairs[merge_arena0 : merge_arena0 + 1, 0:TOPK_PAIR_WIDTH]
                        topk_idxs_tile = pl.gather(
                            final_pairs,
                            mask_pattern=pl.tile.MaskPattern.P1010,
                            output_dtype=pl.INT32,
                        )
                        valid_topk = pl.min(IDX_TOPK, visible_len_t)
                        topk_idxs_valid = pl.set_validshape(topk_idxs_tile, 1, valid_topk)
                        topk_idxs_flat[t : t + 1, 0:IDX_TOPK] = pl.add(topk_idxs_valid, offset_i32)

    return score, topk_idxs
