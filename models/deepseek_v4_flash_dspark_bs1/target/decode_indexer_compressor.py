# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl

from .config import FLASH as M, DECODE_BATCH, DECODE_SEQ, BLOCK_SIZE, C4A_COMPRESSOR_BLOCK_SIZE, FP32_NEG_INF, INT8_SCALE_MAX, INT8_AMAX_EPS

B = DECODE_BATCH

S = DECODE_SEQ

EPS = M.rms_norm_eps

D = M.hidden_size

HEAD_DIM = M.index_head_dim

HEAD_DIM_INV = 1.0 / HEAD_DIM

ROPE_HEAD_DIM = M.qk_rope_head_dim

NOPE_HEAD_DIM = M.index_nope_head_dim

MAX_SEQ_LEN = M.max_position_embeddings

COMPRESS_RATIO = 4

IDX_STORAGE_BLOCK_SIZE = BLOCK_SIZE // COMPRESS_RATIO

OVERLAP = COMPRESS_RATIO == 4

COFF = 1 + int(OVERLAP)

OUT_DIM = COFF * HEAD_DIM

STATE_LEN = COFF * COMPRESS_RATIO

COMPRESS_STATE_BLOCK_SIZE = C4A_COMPRESSOR_BLOCK_SIZE

COMPRESS_STATE_MAX_BLOCKS = (MAX_SEQ_LEN + COMPRESS_STATE_BLOCK_SIZE - 1) // COMPRESS_STATE_BLOCK_SIZE

COMPRESS_STATE_DIM = 2 * OUT_DIM

IDX_CACHE_BLOCK_NUM_DYN = pl.dynamic('NATIVE_S_IDX_CACHE_BLOCK_NUM_DYN')

COMPRESS_STATE_BLOCK_NUM_DYN = pl.dynamic('NATIVE_S_INNER_STATE_BLOCK_NUM_DYN')

K_TILE = 512

OUT_TILE = 64

PROJ_OUT_TILE = 32  # kv_score_proj N-tile

assert PROJ_OUT_TILE % 16 == 0, "cube tile cols must be a multiple of 16"

MM_B_TILE = 16

BS_PAD = ((B * S + MM_B_TILE - 1) // MM_B_TILE) * MM_B_TILE

HEAD_TILE = 64

RMS_PAD_TILE = 16  # padded token rows for the Hadamard matmul

RMS_PAD_ROWS = BS_PAD  # one padded row per input token

@pl.jit.inline
def indexer_compressor(
    x: pl.Tensor[[B, S, D], pl.BF16],
    kv: pl.Tensor[[B, S, HEAD_DIM], pl.FP32],
    compress_state: pl.Tensor[[COMPRESS_STATE_BLOCK_NUM_DYN, COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM], pl.FP32],
    compress_state_block_table: pl.Tensor[[B, COMPRESS_STATE_MAX_BLOCKS], pl.INT32],
    wkv: pl.Tensor[[OUT_DIM, D], pl.BF16],
    wgate: pl.Tensor[[OUT_DIM, D], pl.BF16],
    ape: pl.Tensor[[COMPRESS_RATIO, OUT_DIM], pl.FP32],
    norm_w: pl.Tensor[[HEAD_DIM], pl.BF16],
    # Interleave-duplicated (j>>1) cos and sign-folded sin, built once by the caller:
    #   cos[j] = cos_half[j>>1];  sin[j] = sin_half[j>>1] * sign[j], sign = [-1,+1,...]
    cos: pl.Tensor[[B * S, ROPE_HEAD_DIM], pl.FP32],
    sin: pl.Tensor[[B * S, ROPE_HEAD_DIM], pl.FP32],
    hadamard: pl.Tensor[[HEAD_DIM, HEAD_DIM], pl.BF16],
    idx_kv_cache: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, HEAD_DIM], pl.INT8],
    idx_kv_scale: pl.Tensor[[IDX_CACHE_BLOCK_NUM_DYN, IDX_STORAGE_BLOCK_SIZE, 1, 1], pl.FP32],
    position_ids: pl.Tensor[[B, S], pl.INT32],
    idx_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    inner_state_slot_mapping: pl.Tensor[[B, S], pl.INT64],
    late_dep: pl.Scalar[pl.TASK_ID],
):
    x_flat = pl.reshape(x, [B * S, D])
    kv_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    score_proj_pad = pl.create_tensor([BS_PAD, OUT_DIM], dtype=pl.FP32)
    compress_state_block_num = pl.tensor.dim(compress_state, 0)
    idx_block_num = pl.tensor.dim(idx_kv_cache, 0)
    compress_state_flat = pl.reshape(compress_state, [compress_state_block_num * COMPRESS_STATE_BLOCK_SIZE, COMPRESS_STATE_DIM])
    kv_flat = pl.reshape(kv, [B * S, HEAD_DIM])
    idx_kv_cache_flat = pl.reshape(idx_kv_cache, [idx_block_num * IDX_STORAGE_BLOCK_SIZE, HEAD_DIM])
    idx_kv_scale_flat = pl.reshape(idx_kv_scale, [idx_block_num * IDX_STORAGE_BLOCK_SIZE, 1])

    # Deferred behind the caller's rms_norm dummy barrier: qkv's qr_proj_matmul is the
    # critical path and must win the cores when rms_norm retires.
    with pl.spmd(
        BS_PAD * OUT_DIM // (MM_B_TILE * PROJ_OUT_TILE), name_hint="kv_score_proj", deps=[late_dep]
    ):
        idx = pl.tile.get_block_idx()
        global_row0 = (idx // (OUT_DIM // PROJ_OUT_TILE)) * MM_B_TILE
        o0 = (idx % (OUT_DIM // PROJ_OUT_TILE)) * PROJ_OUT_TILE
        kv_acc = pl.create_tensor([MM_B_TILE, PROJ_OUT_TILE], dtype=pl.FP32)
        score_acc = pl.create_tensor([MM_B_TILE, PROJ_OUT_TILE], dtype=pl.FP32)
        for kb in pl.pipeline(0, D // K_TILE, stage=2):
            k0 = kb * K_TILE
            x_rows = pl.min(MM_B_TILE, B * S - global_row0)
            x_tile = pl.slice(x_flat, [MM_B_TILE, K_TILE], [global_row0, k0], valid_shape=[x_rows, K_TILE])
            # Weights stored transposed [OUT_DIM, D] and consumed via b_trans=True so the
            # GM->L1 load is a DN2ZN (each [PROJ_OUT_TILE, K_TILE] row is K-contiguous = long
            # bursts) instead of ND2NZ on [K_TILE, PROJ_OUT_TILE] (K strided = many short
            # bursts). Mirrors the main compressor (decode_compressor_ratio4); the strided
            # ND2NZ form here was ~2x slower on this matmul (43us -> ~20us per task).
            wkv_tile = wkv[o0 : o0 + PROJ_OUT_TILE, k0 : k0 + K_TILE]
            wgate_tile = wgate[o0 : o0 + PROJ_OUT_TILE, k0 : k0 + K_TILE]
            kv_acc = pl.matmul_acc(kv_acc, x_tile, wkv_tile, b_trans=True, init_cond=(k0 == 0))
            score_acc = pl.matmul_acc(score_acc, x_tile, wgate_tile, b_trans=True, init_cond=(k0 == 0))

        kv_proj_pad[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + PROJ_OUT_TILE] = kv_acc
        score_proj_pad[global_row0 : global_row0 + MM_B_TILE, o0 : o0 + PROJ_OUT_TILE] = score_acc

    # Pool each boundary from immutable history plus this invocation's projections.
    pooled_kv = pl.create_tensor([BS_PAD, HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="scatter_softmax_pool", allow_early_resolve=True) as pool_tid:
        pooled_kv[:, :] = pl.full([BS_PAD, HEAD_DIM], dtype=pl.FP32, value=0.0)
        for c_idx in pl.range(B):
            first_pos_b = pl.read(position_ids, [c_idx, 0])
            for s_idx in pl.range(S):
                token = c_idx * S + s_idx
                token_pos = pl.read(position_ids, [c_idx, s_idx])
                if (token_pos + 1) % COMPRESS_RATIO == 0:
                    window_start = token_pos - STATE_LEN + 1
                    for h0 in pl.range(0, HEAD_DIM, HEAD_TILE):
                        last_ape = pl.cast(token_pos % COMPRESS_RATIO, pl.INDEX)
                        last_score = score_proj_pad[token : token + 1, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_TILE]
                        last_bias = ape[last_ape : last_ape + 1, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_TILE]
                        mi = pl.add(last_score, last_bias)
                        initial_delta = pl.sub(mi, mi)
                        li = pl.exp(initial_delta)
                        oi = kv_proj_pad[token : token + 1, HEAD_DIM + h0 : HEAD_DIM + h0 + HEAD_TILE]
                        for state_idx in pl.range(STATE_LEN - 1):
                            logical_pos = window_start + state_idx
                            state_half = 0
                            if state_idx >= COMPRESS_RATIO:
                                state_half = HEAD_DIM
                            value = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=0.0)
                            score = pl.full([1, HEAD_TILE], dtype=pl.FP32, value=FP32_NEG_INF)
                            if logical_pos >= 0 and logical_pos < first_pos_b:
                                page = logical_pos // COMPRESS_STATE_BLOCK_SIZE
                                physical = pl.read(compress_state_block_table, [c_idx, page])
                                if physical >= 0:
                                    state_row = physical * COMPRESS_STATE_BLOCK_SIZE + logical_pos % COMPRESS_STATE_BLOCK_SIZE
                                    col = state_half + h0
                                    value = compress_state_flat[state_row : state_row + 1, col : col + HEAD_TILE]
                                    score_col = OUT_DIM + col
                                    score = compress_state_flat[state_row : state_row + 1, score_col : score_col + HEAD_TILE]
                            if logical_pos >= first_pos_b and logical_pos <= token_pos:
                                overlay = c_idx * S + logical_pos - first_pos_b
                                ape_row = pl.cast(logical_pos % COMPRESS_RATIO, pl.INDEX)
                                col = state_half + h0
                                value = kv_proj_pad[overlay : overlay + 1, col : col + HEAD_TILE]
                                score_base = score_proj_pad[overlay : overlay + 1, col : col + HEAD_TILE]
                                score_bias = ape[ape_row : ape_row + 1, col : col + HEAD_TILE]
                                score = pl.add(score_base, score_bias)
                            mi_next = pl.maximum(mi, score)
                            alpha_delta = pl.sub(mi, mi_next)
                            beta_delta = pl.sub(score, mi_next)
                            alpha = pl.exp(alpha_delta)
                            beta = pl.exp(beta_delta)
                            li_scaled = pl.mul(alpha, li)
                            li = pl.add(li_scaled, beta)
                            oi_scaled = pl.mul(oi, alpha)
                            value_scaled = pl.mul(value, beta)
                            oi = pl.add(oi_scaled, value_scaled)
                            mi = mi_next
                        pooled_kv[token : token + 1, h0 : h0 + HEAD_TILE] = pl.div(oi, li)

    # History must remain unchanged until every boundary has finished pooling.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="compress_state_commit", deps=[pool_tid]):
        for c_idx in pl.range(B):
            for s_idx in pl.range(S):
                commit_state_row_i64 = pl.read(inner_state_slot_mapping, [c_idx, s_idx])
                if commit_state_row_i64 >= 0:
                    commit_state_row = pl.cast(commit_state_row_i64, pl.INDEX)
                    token = c_idx * S + s_idx
                    token_pos = pl.read(position_ids, [c_idx, s_idx])
                    commit_ape_row = pl.cast(token_pos % COMPRESS_RATIO, pl.INDEX)
                    compress_state_flat[commit_state_row : commit_state_row + 1, 0 : OUT_DIM] = kv_proj_pad[token : token + 1, :]
                    commit_score_base = score_proj_pad[token : token + 1, :]
                    commit_score_bias = ape[commit_ape_row : commit_ape_row + 1, :]
                    compress_state_flat[commit_state_row : commit_state_row + 1, OUT_DIM : COMPRESS_STATE_DIM] = pl.add(commit_score_base, commit_score_bias)

    normed_kv = pl.create_tensor([RMS_PAD_ROWS, HEAD_DIM], dtype=pl.BF16)
    norm_w_2d = pl.reshape(norm_w, [1, HEAD_DIM])
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="rmsnorm_rope", deps=[pool_tid]):
        # single 16-row block: B real rows at rows 0..B-1, rows B..15 are pad.
        # cos/sin arrive interleave-duplicated and sign-folded, so these land at the
        # full ROPE_HEAD_DIM width and feed the rotation with no in-scope dup-gather.
        cos_b = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        sin_b = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=0.0)
        cos_b[0:B * S, 0 : ROPE_HEAD_DIM] = cos[0:B * S, 0 : ROPE_HEAD_DIM]
        sin_b[0:B * S, 0 : ROPE_HEAD_DIM] = sin[0:B * S, 0 : ROPE_HEAD_DIM]
        partial_sq = pl.full([1, RMS_PAD_TILE], dtype=pl.FP32, value=0.0)
        for k0 in pl.pipeline(0, HEAD_DIM, HEAD_TILE, stage=2):
            kv_rms_chunk = pooled_kv[0 : RMS_PAD_TILE, k0 : k0 + HEAD_TILE]
            kv_rms_sq = pl.mul(kv_rms_chunk, kv_rms_chunk)
            kv_rms_rowsum = pl.reshape(pl.row_sum(kv_rms_sq), [1, RMS_PAD_TILE])
            partial_sq = pl.add(partial_sq, kv_rms_rowsum)

        variance = pl.reshape(pl.add(pl.mul(partial_sq, HEAD_DIM_INV), EPS), [RMS_PAD_TILE, 1])
        inv_rms = pl.recip(pl.sqrt(variance))
        for k0 in pl.pipeline(0, NOPE_HEAD_DIM, HEAD_TILE, stage=2):
            kv_norm_chunk = pooled_kv[0 : RMS_PAD_TILE, k0 : k0 + HEAD_TILE]
            gamma = pl.cast(norm_w_2d[:, k0 : k0 + HEAD_TILE], pl.FP32)
            normed_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_norm_chunk, inv_rms), gamma)
            normed_kv[0 : RMS_PAD_TILE, k0 : k0 + HEAD_TILE] = pl.cast(
                normed_chunk,
                target_type=pl.BF16,
                mode="rint",
            )

        kv_rope_norm = pooled_kv[0 : RMS_PAD_TILE, NOPE_HEAD_DIM : HEAD_DIM]
        gamma_rope = pl.cast(norm_w_2d[:, NOPE_HEAD_DIM : HEAD_DIM], pl.FP32)
        # A3 interleaved swap-gather (same form as kv_rms_norm_rope in qkv_proj_rope),
        # replacing the de-interleave gather + rotate + re-interleave scatter. gamma+inv_rms
        # are folded into rope_normed BEFORE the swap, so the swapped lane n[j^1] correctly
        # carries gamma[j^1]; inv_rms is per-row so it commutes. Only swap_idx (j^1) is built
        # in-kernel -- it permutes data, so no table can hold it; the interleaved cos and
        # sign-folded sin come in ready to use. normed_kv is BF16 -> cast on write.
        #   out[j] = n[j]*cos_il[j] + n[j^1]*sin_il_signed[j]
        rope_normed = pl.col_expand_mul(pl.row_expand_mul(kv_rope_norm, inv_rms), gamma_rope)
        rope_ones = pl.full([RMS_PAD_TILE, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        rope_col = pl.col_expand_mul(rope_ones, pl.cast(pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32), target_type=pl.FP32))
        rope_dup_f = pl.cast(pl.cast(pl.mul(rope_col, 0.5), target_type=pl.INT32, mode="trunc"), target_type=pl.FP32)
        rope_lane = pl.sub(rope_col, pl.mul(rope_dup_f, 2.0))                                          # j%2
        rope_swap_idx = pl.cast(pl.sub(pl.add(rope_col, 1.0), pl.mul(rope_lane, 2.0)), target_type=pl.INT32)  # j^1
        swapped = pl.gather(rope_normed, dim=-1, index=rope_swap_idx)
        rope_rot = pl.add(pl.mul(rope_normed, cos_b), pl.mul(swapped, sin_b))
        normed_kv[0 : RMS_PAD_TILE, NOPE_HEAD_DIM : HEAD_DIM] = pl.cast(
            rope_rot,
            target_type=pl.BF16,
            mode="rint",
        )

    kv_final = pl.create_tensor([RMS_PAD_ROWS, HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_hadamard"):
        kv_proj_tile = normed_kv[0 : RMS_PAD_TILE, 0 : HEAD_DIM]
        for o0 in pl.range(0, HEAD_DIM, OUT_TILE):
            hadamard_tile = hadamard[0 : HEAD_DIM, o0 : o0 + OUT_TILE]
            kv_hadamard_acc = pl.matmul(kv_proj_tile, hadamard_tile, out_dtype=pl.FP32)
            kv_final[0 : RMS_PAD_TILE, o0 : o0 + OUT_TILE] = kv_hadamard_acc

    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_and_cache_write", allow_early_resolve=True):
        # C8 quant-on-write: per-row INT8 quant of the block (M=RMS_PAD_TILE keeps tiles 32B-aligned;
        # quantize the bf16-rounded value to match golden)
        kv_blk_f32 = pl.cast(
            pl.cast(kv_final[0 : RMS_PAD_TILE, 0 : HEAD_DIM], target_type=pl.BF16, mode="rint"),
            target_type=pl.FP32)
        # amax = max(|x|); abs-based (max(row_max, -row_min) is wrong on signed KV)
        kv_amax = pl.reshape(pl.row_max(pl.abs(kv_blk_f32)), [1, RMS_PAD_TILE])
        kv_amax = pl.maximum(kv_amax, INT8_AMAX_EPS)
        kv_scale_q_row = pl.div(pl.full([1, RMS_PAD_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX), kv_amax)
        kv_scale_dq_col = pl.reshape(pl.recip(kv_scale_q_row), [RMS_PAD_TILE, 1])
        kv_scale_q_col = pl.reshape(kv_scale_q_row, [RMS_PAD_TILE, 1])
        kv_scaled = pl.row_expand_mul(kv_blk_f32, kv_scale_q_col)
        kv_i32 = pl.cast(kv_scaled, target_type=pl.INT32, mode="rint")
        kv_half = pl.cast(kv_i32, target_type=pl.FP16, mode="round")
        kv_i8_blk = pl.cast(kv_half, target_type=pl.INT8, mode="trunc")
        for token in pl.range(B * S):
            c_idx = token // S
            s_idx = token - c_idx * S
            position = pl.read(position_ids, [c_idx, s_idx])
            cache_row_i64 = pl.read(idx_slot_mapping, [c_idx, s_idx])
            if (position + 1) % COMPRESS_RATIO == 0 and cache_row_i64 >= 0:
                cache_row = pl.cast(cache_row_i64, pl.INDEX)
                kv_flat[token : token + 1, :] = kv_final[token : token + 1, :]
                idx_kv_cache_flat[cache_row : cache_row + 1, :] = kv_i8_blk[token : token + 1, :]
                pl.write(idx_kv_scale_flat, [cache_row, 0], pl.read(kv_scale_dq_col, [token, 0]))

    return kv
