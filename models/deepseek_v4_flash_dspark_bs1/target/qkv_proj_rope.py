# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl

from .config import FLASH as M, DECODE_BATCH, DECODE_SEQ, INT8_SCALE_MAX, INT8_AMAX_EPS

T_DYN = pl.dynamic('NATIVE_S_T_DYN')  # T = B * S

D = M.hidden_size

H = M.num_attention_heads

Q_HEADS_LOCAL = H // M.o_groups

HEAD_DIM = M.head_dim

ROPE_DIM = M.qk_rope_head_dim

NOPE_DIM = M.nope_head_dim

Q_LORA = M.q_lora_rank

EPS = M.rms_norm_eps

Q_PROJ_TILE = 128       # qproj K-tile (Q_LORA reduction); 8 slices -> deep stage=2 pipeline, double-buffered Mat fits

QPROJ_MM_N_TILE = 512    # qproj output-column tile

Q_LORA_TILE = 256       # qr rms-norm / quant N granularity (decoupled from qr_proj matmul)

KV_TILE = 64            # kv rms-norm / rope / NOPE N granularity (decoupled from kv_proj matmul)

QUANT_TILE = 256

T_TILE = 8

MATMUL_T_TILE = 16

QR_M_TILE = MATMUL_T_TILE  # qr_proj token (M) tile; cube rows must be a 16-row boxed tile

QR_N_TILE = 128         # qr_proj Q_LORA (N) per matmul

QR_K_TILE = 256         # qr_proj D (K) reduction tile    | divides QR_K_SLICE

QR_OK = 1               # qr_proj split-K factor

QR_K_SLICE = D // QR_OK # qr_proj K columns per split

KV_M_TILE = MATMUL_T_TILE  # kv_proj token (M) tile; decode pads from 8 real rows to 16

KV_N_TILE = 128         # kv_proj HEAD_DIM (N) per matmul

KV_K_TILE = 256         # kv_proj D (K) reduction tile    | divides KV_K_SLICE

KV_OK = 2               # kv_proj split-K factor

KV_K_SLICE = D // KV_OK # kv_proj K columns per split

QPROJ_M_TILE = MATMUL_T_TILE  # qproj token (M) tile; decode pads from 8 real rows to 16

KV_RMS_T_TILE = 8       # kv rms-norm + rope fused token (T) tile

Q_ROPE_T_TILE = 8

Q_ROPE_H_TILE = 4       # heads per fused qproj dequant/rms/rope task; cos/sin build amortizes over them

assert H % Q_ROPE_H_TILE == 0

assert DECODE_BATCH * DECODE_SEQ <= MATMUL_T_TILE

assert Q_LORA % QR_N_TILE == 0 and D % QR_OK == 0 and QR_K_SLICE % QR_K_TILE == 0

assert HEAD_DIM % KV_N_TILE == 0 and D % KV_OK == 0 and KV_K_SLICE % KV_K_TILE == 0

assert (Q_HEADS_LOCAL * HEAD_DIM) % QPROJ_MM_N_TILE == 0

assert Q_LORA % Q_PROJ_TILE == 0 and QPROJ_MM_N_TILE * QPROJ_M_TILE * 4 <= 128 * 1024  # L0C Acc cap




from .fp32_numerics import newton_sqrt_seed, midpoint_sqrt_seed

@pl.jit.inline(auto_scope=False)
def _q_rms_newton_sqrt_seed(
    var: pl.Tensor[[1, 8], pl.FP32],
    raw_sqrt: pl.Tensor[[1, 8], pl.FP32],
) -> pl.Tensor[[1, 8], pl.FP32]:
    result = newton_sqrt_seed(var, raw_sqrt, rows=1, cols=8)
    return result


@pl.jit.inline(auto_scope=False)
def _q_rms_midpoint_sqrt_seed(
    var: pl.Tensor[[1, 8], pl.FP32],
    newton: pl.Tensor[[1, 8], pl.FP32],
) -> pl.Tensor[[1, 8], pl.FP32]:
    result = midpoint_sqrt_seed(var, newton, rows=1, cols=8)
    return result


@pl.jit.inline(auto_scope=False)
def _q_rms_reciprocal_eft(
    correct_sqrt: pl.Tensor[[1, 8], pl.FP32],
    factor: pl.constexpr,
) -> pl.Tensor[[1, 8], pl.FP32]:
    result = correct_sqrt
    ones_row = pl.full([1, 8], dtype=pl.FP32, value=1.0)
    ef_work = pl.mul(correct_sqrt, factor)
    ef_estimate = pl.div(ones_row, ef_work)
    ef_left_split = pl.mul(ef_estimate, 4097.0)
    ef_left_delta = pl.sub(ef_left_split, ef_estimate)
    ef_left_hi = pl.sub(ef_left_split, ef_left_delta)
    ef_left_lo = pl.sub(ef_estimate, ef_left_hi)
    ef_right_split = pl.mul(ef_work, 4097.0)
    ef_right_delta = pl.sub(ef_right_split, ef_work)
    ef_right_hi = pl.sub(ef_right_split, ef_right_delta)
    ef_right_lo = pl.sub(ef_work, ef_right_hi)
    ef_product = pl.mul(ef_estimate, ef_work)
    ef_hi_product = pl.mul(ef_left_hi, ef_right_hi)
    ef_error = pl.sub(ef_hi_product, ef_product)
    ef_hi_lo = pl.mul(ef_left_hi, ef_right_lo)
    ef_error = pl.add(ef_error, ef_hi_lo)
    ef_lo_hi = pl.mul(ef_left_lo, ef_right_hi)
    ef_error = pl.add(ef_error, ef_lo_hi)
    ef_lo_product = pl.mul(ef_left_lo, ef_right_lo)
    ef_error = pl.add(ef_error, ef_lo_product)
    ef_residual_hi = pl.sub(ones_row, ef_product)
    ef_residual = pl.sub(ef_residual_hi, ef_error)
    ef_correction = pl.div(ef_residual, ef_work)
    ef_corrected_work = pl.add(ef_estimate, ef_correction)
    result = pl.mul(ef_corrected_work, factor)
    return result


@pl.jit.inline
def prepare_qkv_rope_metadata(
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    q_rope_cos_il: pl.Out[pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32]],
    q_rope_sin_signed: pl.Out[pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32]],
    q_rope_swap_idx: pl.Out[pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32]],
):
    t_dim = pl.tensor.dim(rope_cos, 0)
    rope_cos_view = pl.reshape(rope_cos, [t_dim, ROPE_DIM])
    rope_sin_view = pl.reshape(rope_sin, [t_dim, ROPE_DIM])
    for qrp_idx in pl.spmd((t_dim + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE, name_hint="q_rope_prepare", allow_early_resolve=True):
        qrp_t0 = qrp_idx * Q_ROPE_T_TILE
        qrp_rows = pl.min(Q_ROPE_T_TILE, t_dim - qrp_t0)
        qrp_ones = pl.full([Q_ROPE_T_TILE, ROPE_DIM], dtype=pl.FP32, value=1.0)
        qrp_idx_i32 = pl.arange(0, [1, ROPE_DIM], dtype=pl.INT32)
        qrp_idx_fp32 = pl.cast(qrp_idx_i32, target_type=pl.FP32)
        qrp_col = pl.col_expand_mul(qrp_ones, qrp_idx_fp32)
        qrp_half = pl.mul(qrp_col, 0.5)
        qrp_dup_i32 = pl.cast(qrp_half, target_type=pl.INT32, mode="trunc")
        qrp_dup_f = pl.cast(qrp_dup_i32, target_type=pl.FP32)
        qrp_dup_idx = pl.cast(qrp_dup_f, target_type=pl.INT32)
        qrp_lane = pl.sub(qrp_col, pl.mul(qrp_dup_f, 2.0))
        qrp_next_col = pl.add(qrp_col, 1.0)
        qrp_lane_offset = pl.mul(qrp_lane, 2.0)
        qrp_swap_f = pl.sub(qrp_next_col, qrp_lane_offset)
        qrp_swap_idx = pl.cast(qrp_swap_f, target_type=pl.INT32)
        qrp_sign = pl.sub(pl.mul(qrp_lane, 2.0), 1.0)
        qrp_cos_rows = pl.slice(rope_cos_view, [Q_ROPE_T_TILE, ROPE_DIM], [qrp_t0, 0], valid_shape=[qrp_rows, ROPE_DIM])
        qrp_sin_rows = pl.slice(rope_sin_view, [Q_ROPE_T_TILE, ROPE_DIM], [qrp_t0, 0], valid_shape=[qrp_rows, ROPE_DIM])
        qrp_cos = pl.cast(qrp_cos_rows, target_type=pl.FP32)
        qrp_sin = pl.cast(qrp_sin_rows, target_type=pl.FP32)
        qrp_cos_il = pl.gather(qrp_cos, dim=-1, index=qrp_dup_idx)
        qrp_sin_il = pl.gather(qrp_sin, dim=-1, index=qrp_dup_idx)
        qrp_sin_signed = pl.mul(qrp_sin_il, qrp_sign)
        q_rope_cos_il[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :] = pl.set_validshape(qrp_cos_il, qrp_rows, ROPE_DIM)
        q_rope_sin_signed[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :] = pl.set_validshape(qrp_sin_signed, qrp_rows, ROPE_DIM)
        q_rope_swap_idx[qrp_t0 : qrp_t0 + Q_ROPE_T_TILE, :] = pl.set_validshape(qrp_swap_idx, qrp_rows, ROPE_DIM)

@pl.jit.inline
def qkv_proj_rope(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wq_a: pl.Tensor[[D, Q_LORA], pl.BF16],
    wq_b: pl.Tensor[[Q_LORA, H * HEAD_DIM], pl.INT8],
    wq_b_scale: pl.Tensor[[H * HEAD_DIM], pl.FP32],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    gamma_cq: pl.Tensor[[Q_LORA], pl.BF16],
    gamma_ckv: pl.Tensor[[HEAD_DIM], pl.BF16],
    q: pl.Tensor[[T_DYN, H, HEAD_DIM], pl.BF16],
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    qr: pl.Tensor[[T_DYN, Q_LORA], pl.INT8],
    qr_scale: pl.Tensor[[T_DYN, 1], pl.FP32],
    late_dep: pl.Scalar[pl.TASK_ID],
    q_head_base: pl.Scalar[pl.INT32],
    q_rope_cos_il: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    q_rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    q_rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
):
    t_dim = pl.tensor.dim(x, 0)
    x_view = pl.reshape(x, [t_dim, D])
    kv_view = pl.reshape(kv, [t_dim, HEAD_DIM])
    qr_view = pl.reshape(qr, [t_dim, Q_LORA])
    qr_scale_view = pl.reshape(qr_scale, [t_dim, 1])
    t_matmul = pl.max(t_dim, MATMUL_T_TILE)

    # Split-K qr_proj (M=t_dim, K=D=4096, N=Q_LORA=1024). QR_N_TILE=128 gives
    # eight N-groups; QR_OK=2 expands them to 16 cube blocks and atomic-adds the
    # K partials into a zero-seeded output. Auto-dep on qr_fp32 orders the seed
    # before every atomic RMW. Seeded on-core (not create_tensor init_value=0):
    # AICPU init serializes on the scheduler/orchestration path and roughly
    # doubles the decode orch window, whereas the on-core memset overlaps.
    qr_fp32 = pl.create_tensor([t_matmul, Q_LORA], dtype=pl.FP32)
    qr_i8_matmul = pl.create_tensor([t_matmul, Q_LORA], dtype=pl.INT8)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="qr_proj_seed", allow_early_resolve=True):
        for tc in pl.range(t_matmul // QR_M_TILE):
            ts0 = tc * QR_M_TILE
            for nb in pl.range(Q_LORA // QR_N_TILE):
                nseed0 = nb * QR_N_TILE
                qr_fp32[ts0 : ts0 + QR_M_TILE, nseed0 : nseed0 + QR_N_TILE] = pl.full(
                    [QR_M_TILE, QR_N_TILE], dtype=pl.FP32, value=0.0
                )
    for qbg_idx in pl.spmd((Q_LORA // QR_N_TILE) * QR_OK, name_hint="qr_proj_matmul", allow_early_resolve=True):
        q_a_col0 = (qbg_idx // QR_OK) * QR_N_TILE
        qr_k_base = (qbg_idx % QR_OK) * QR_K_SLICE
        for tc in pl.range(t_matmul // QR_M_TILE):
            t0 = tc * QR_M_TILE
            q_acc = pl.create_tensor([QR_M_TILE, QR_N_TILE], dtype=pl.FP32)
            for db in pl.pipeline(QR_K_SLICE // QR_K_TILE, stage=2):
                qr_d0 = qr_k_base + db * QR_K_TILE
                qr_rows = pl.min(QR_M_TILE, t_dim - t0)
                q_x_chunk_bf16 = pl.slice(x_view, [QR_M_TILE, QR_K_TILE], [t0, qr_d0], valid_shape=[qr_rows, QR_K_TILE])
                w_chunk = wq_a[qr_d0 : qr_d0 + QR_K_TILE, q_a_col0 : q_a_col0 + QR_N_TILE]
                q_acc = pl.matmul_acc(q_acc, q_x_chunk_bf16, w_chunk, init_cond=(db == 0))
            qr_fp32 = pl.assemble(qr_fp32, q_acc, [t0, q_a_col0], atomic=pl.AtomicType.Add)

    # Two passes per block: pass 1 computes amax; pass 2 recomputes norm and quantizes.
    for tg_idx in pl.spmd((t_dim + T_TILE - 1) // T_TILE, name_hint="qr_rms_norm_quant", allow_early_resolve=True):
        tg = tg_idx * T_TILE
        qr_rows_out = pl.min(T_TILE, t_dim - tg)
        qr_sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        qr_amax_g = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for qr_rms_qb in pl.pipeline(Q_LORA // Q_LORA_TILE, stage=2):
            qr_rms_col0 = qr_rms_qb * Q_LORA_TILE
            qr_rms_chunk = qr_fp32[tg : tg + T_TILE, qr_rms_col0 : qr_rms_col0 + Q_LORA_TILE]
            # Fixed half-stride FP32 tree, shared with the independent reference.
            qr_sq_values = pl.mul(qr_rms_chunk, qr_rms_chunk)
            qr_sq128 = pl.add(qr_sq_values[:, :128], qr_sq_values[:, 128:256])
            qr_sq64 = pl.add(qr_sq128[:, :64], qr_sq128[:, 64:128])
            qr_sq32 = pl.add(qr_sq64[:, :32], qr_sq64[:, 32:64])
            qr_sq16 = pl.add(qr_sq32[:, :16], qr_sq32[:, 16:32])
            qr_sq8 = pl.add(qr_sq16[:, :8], qr_sq16[:, 8:16])
            # Keep eight FP32 columns (32 bytes) while reducing the last lanes.
            qr_sq8_t = pl.transpose(qr_sq8, 0, 1)
            qr_sq4_t = pl.add(qr_sq8_t[:4, :], qr_sq8_t[4:8, :])
            qr_sq2_t = pl.add(qr_sq4_t[:2, :], qr_sq4_t[2:4, :])
            qr_sq1_t = pl.add(qr_sq2_t[:1, :], qr_sq2_t[1:2, :])
            qr_sq_sum = pl.add(qr_sq_sum, qr_sq1_t)
        qr_scale_shape_input = pl.slice(qr_fp32, [1, T_TILE], [tg, 0])
        qr_var = pl.reshape(qr_scale_shape_input, [1, T_TILE])
        qr_var = pl.add(pl.mul(qr_sq_sum, 1.0 / Q_LORA), EPS)
        qr_raw_sqrt = qr_var
        qr_raw_sqrt = pl.sqrt(qr_var)
        qr_sqrt_newton = _q_rms_newton_sqrt_seed(qr_var, qr_raw_sqrt)
        qr_sqrt_fixed = _q_rms_midpoint_sqrt_seed(qr_var, qr_sqrt_newton)
        qr_inv_rms = _q_rms_reciprocal_eft(qr_sqrt_fixed, factor=5.421010862427522e-20)
        qr_inv_rms_t = pl.reshape(qr_inv_rms, [T_TILE, 1])
        # Match the owning reference: normalize, multiply gamma, THEN find amax.
        # Hoisting the inverse past the maximum changes FP32 rounding boundaries.
        qr_amax_normed = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_AMAX_EPS)
        for qr_amax_qb in pl.pipeline(Q_LORA // Q_LORA_TILE, stage=2):
            qr_amax_col0 = qr_amax_qb * Q_LORA_TILE
            qr_amax_chunk = qr_fp32[tg : tg + T_TILE, qr_amax_col0 : qr_amax_col0 + Q_LORA_TILE]
            qr_amax_gamma_raw = pl.cast(gamma_cq[qr_amax_col0 : qr_amax_col0 + Q_LORA_TILE], target_type=pl.FP32)
            qr_amax_gamma = pl.reshape(qr_amax_gamma_raw, [1, Q_LORA_TILE])
            qr_amax_scaled = pl.row_expand_mul(qr_amax_chunk, qr_inv_rms_t)
            qr_amax_output = pl.col_expand_mul(qr_amax_scaled, qr_amax_gamma)
            qr_amax_chunk_max = pl.reshape(pl.row_max(pl.abs(qr_amax_output)), [1, T_TILE])
            qr_amax_normed = pl.maximum(qr_amax_normed, qr_amax_chunk_max)
        qr_tile_amax = qr_amax_normed

        qd_qr_numerator = pl.full([1, T_TILE], dtype=pl.FP32, value=INT8_SCALE_MAX)
        qd_qr_work = pl.mul(qr_tile_amax, 5.421010862427522e-20)
        qd_qr_estimate = pl.div(qd_qr_numerator, qd_qr_work)
        qd_qr_left_split = pl.mul(qd_qr_estimate, 4097.0)
        qd_qr_left_delta = pl.sub(qd_qr_left_split, qd_qr_estimate)
        qd_qr_left_hi = pl.sub(qd_qr_left_split, qd_qr_left_delta)
        qd_qr_left_lo = pl.sub(qd_qr_estimate, qd_qr_left_hi)
        qd_qr_right_split = pl.mul(qd_qr_work, 4097.0)
        qd_qr_right_delta = pl.sub(qd_qr_right_split, qd_qr_work)
        qd_qr_right_hi = pl.sub(qd_qr_right_split, qd_qr_right_delta)
        qd_qr_right_lo = pl.sub(qd_qr_work, qd_qr_right_hi)
        qd_qr_product = pl.mul(qd_qr_estimate, qd_qr_work)
        qd_qr_hi_product = pl.mul(qd_qr_left_hi, qd_qr_right_hi)
        qd_qr_error = pl.sub(qd_qr_hi_product, qd_qr_product)
        qd_qr_hi_lo = pl.mul(qd_qr_left_hi, qd_qr_right_lo)
        qd_qr_error = pl.add(qd_qr_error, qd_qr_hi_lo)
        qd_qr_lo_hi = pl.mul(qd_qr_left_lo, qd_qr_right_hi)
        qd_qr_error = pl.add(qd_qr_error, qd_qr_lo_hi)
        qd_qr_lo_product = pl.mul(qd_qr_left_lo, qd_qr_right_lo)
        qd_qr_error = pl.add(qd_qr_error, qd_qr_lo_product)
        qd_qr_residual_hi = pl.sub(qd_qr_numerator, qd_qr_product)
        qd_qr_residual = pl.sub(qd_qr_residual_hi, qd_qr_error)
        qd_qr_correction = pl.div(qd_qr_residual, qd_qr_work)
        qd_qr_corrected_work = pl.add(qd_qr_estimate, qd_qr_correction)
        qr_scale_quant_row = pl.reshape(qr_scale_shape_input, [1, T_TILE])
        qr_scale_quant_row = pl.mul(qd_qr_corrected_work, 5.421010862427522e-20)
        qr_scale_quant_t = pl.reshape(qr_scale_quant_row, [T_TILE, 1])
        qr_tile_scale_dq = pl.reshape(_q_rms_reciprocal_eft(qr_scale_quant_row, factor=65536.0), [T_TILE, 1])
        qr_scale_view[tg : tg + T_TILE, :] = pl.set_validshape(qr_tile_scale_dq, qr_rows_out, 1)

        for qa in pl.pipeline(0, Q_LORA, QUANT_TILE, stage=2):
            qr_chunk = qr_fp32[tg : tg + T_TILE, qa : qa + QUANT_TILE]
            gamma_q_cast = pl.cast(gamma_cq[qa : qa + QUANT_TILE], target_type=pl.FP32)
            gamma_q_chunk = pl.reshape(gamma_q_cast, [1, QUANT_TILE])
            qr_q_normed = pl.col_expand_mul(pl.row_expand_mul(qr_chunk, qr_inv_rms_t), gamma_q_chunk)
            qr_q_scaled = pl.row_expand_mul(qr_q_normed, qr_scale_quant_t)
            qr_q_i32 = pl.cast(qr_q_scaled, target_type=pl.INT32, mode="rint")
            qr_q_half = pl.cast(qr_q_i32, target_type=pl.FP16, mode="round")
            qr_q_i8 = pl.cast(qr_q_half, target_type=pl.INT8, mode="trunc", saturation_mode="on")
            qr_view[tg : tg + T_TILE, qa : qa + QUANT_TILE] = pl.set_validshape(qr_q_i8, qr_rows_out, QUANT_TILE)
            qr_i8_matmul[tg : tg + T_TILE, qa : qa + QUANT_TILE] = qr_q_i8

    # UN-MIXED qproj: keep the pure-matmul scope (cube, INT32 -> GM) separate from
    # downstream vector work. This lets the scheduler defer q dequant until AIV is free
    # instead of pinning it next to qproj and competing with the critical qr_proj AIV work.
    q_proj_i32 = pl.create_tensor([t_matmul, H * HEAD_DIM], dtype=pl.INT32)
    # One output-column fragment per task.
    q_col_base = pl.cast(q_head_base, pl.INDEX) * HEAD_DIM
    for qproj_n_idx in pl.spmd((Q_HEADS_LOCAL * HEAD_DIM) // QPROJ_MM_N_TILE, name_hint="qproj_matmul", allow_early_resolve=True):
        w_col0 = q_col_base + qproj_n_idx * QPROJ_MM_N_TILE
        for tc in pl.range(t_matmul // QPROJ_M_TILE):
            t0 = tc * QPROJ_M_TILE
            col_acc = pl.create_tensor([QPROJ_M_TILE, QPROJ_MM_N_TILE], dtype=pl.INT32)
            for qb in pl.pipeline(0, Q_LORA // Q_PROJ_TILE, stage=2):
                qr_proj_col0 = qb * Q_PROJ_TILE
                qproj_rows = pl.min(QPROJ_M_TILE, t_dim - t0)
                qr_i8_chunk = pl.slice(qr_i8_matmul, [QPROJ_M_TILE, Q_PROJ_TILE], [t0, qr_proj_col0], valid_shape=[qproj_rows, Q_PROJ_TILE])
                wq_chunk = wq_b[qr_proj_col0 : qr_proj_col0 + Q_PROJ_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE]
                col_acc = pl.matmul_acc(col_acc, qr_i8_chunk, wq_chunk, init_cond=(qb == 0))
            q_proj_i32[t0 : t0 + QPROJ_M_TILE, w_col0 : w_col0 + QPROJ_MM_N_TILE] = col_acc

    # Fuse qproj dequant, per-head RMSNorm, NOPE writeback, and interleaved RoPE.
    # A full [token, head] tile fits in Vec UB, so dequantize each head once and
    # retain it across the RMS reduction instead of rereading/recomputing NOPE.
    # RoPE: out[j] = inv_rms * (x[j] * cos[j] + x[j^1] * sign[j] * sin[j]).
    q_flat = pl.reshape(q, [t_dim, H * HEAD_DIM])
    for hg_idx in pl.spmd(Q_HEADS_LOCAL // Q_ROPE_H_TILE, name_hint="qproj_dequant_rms_nope_rope", allow_early_resolve=True):
        hg = pl.cast(q_head_base, pl.INDEX) + hg_idx * Q_ROPE_H_TILE
        for tg_idx in pl.range((t_dim + Q_ROPE_T_TILE - 1) // Q_ROPE_T_TILE):
            tg = tg_idx * Q_ROPE_T_TILE
            q_rows_out = pl.min(Q_ROPE_T_TILE, t_dim - tg)
            qr_scale_dq_t = pl.slice(qr_scale_view, [Q_ROPE_T_TILE, 1], [tg, 0], valid_shape=[q_rows_out, 1])
            qr_scale_dq_t = pl.set_validshape(qr_scale_dq_t, Q_ROPE_T_TILE, 1)
            q_cos_il = pl.slice(q_rope_cos_il, [Q_ROPE_T_TILE, ROPE_DIM], [tg, 0], valid_shape=[q_rows_out, ROPE_DIM])
            q_cos_il = pl.set_validshape(q_cos_il, Q_ROPE_T_TILE, ROPE_DIM)
            q_sin_signed = pl.slice(q_rope_sin_signed, [Q_ROPE_T_TILE, ROPE_DIM], [tg, 0], valid_shape=[q_rows_out, ROPE_DIM])
            q_sin_signed = pl.set_validshape(q_sin_signed, Q_ROPE_T_TILE, ROPE_DIM)
            q_swap_idx = pl.slice(q_rope_swap_idx, [Q_ROPE_T_TILE, ROPE_DIM], [tg, 0], valid_shape=[q_rows_out, ROPE_DIM])
            q_swap_idx = pl.fillpad(q_swap_idx, pad_value=pl.PadValue.zero)
            q_swap_idx = pl.set_validshape(q_swap_idx, Q_ROPE_T_TILE, ROPE_DIM)
            # Pipeline adjacent heads so the next head's GM reads overlap the
            # current head's vector RMS/rotation work, as in Qwen's decode loop.
            for h_inner in pl.pipeline(Q_ROPE_H_TILE, stage=2):
                h = hg + h_inner
                h0 = h * HEAD_DIM
                q_head_acc = pl.slice(q_proj_i32, [Q_ROPE_T_TILE, HEAD_DIM], [tg, h0], valid_shape=[q_rows_out, HEAD_DIM])
                q_head_acc = pl.set_validshape(q_head_acc, Q_ROPE_T_TILE, HEAD_DIM)
                q_head_scale = pl.reshape(wq_b_scale[h0 : h0 + HEAD_DIM], [1, HEAD_DIM])
                q_head_acc_fp32 = pl.cast(q_head_acc, target_type=pl.FP32, mode="none")
                q_head_row_scaled = pl.row_expand_mul(q_head_acc_fp32, qr_scale_dq_t)
                q_head_dq = pl.col_expand_mul(q_head_row_scaled, q_head_scale)
                q_head_sq = pl.mul(q_head_dq, q_head_dq)
                q_rms_sq_transposed = pl.transpose(q_head_sq, 0, 1)
                q_rms_left_256 = q_rms_sq_transposed[0:256, :]
                q_rms_right_256 = q_rms_sq_transposed[256:512, :]
                q_rms_sum_256 = pl.add(q_rms_left_256, q_rms_right_256)
                q_rms_left_128 = q_rms_sum_256[0:128, :]
                q_rms_right_128 = q_rms_sum_256[128:256, :]
                q_rms_sum_128 = pl.add(q_rms_left_128, q_rms_right_128)
                q_rms_left_64 = q_rms_sum_128[0:64, :]
                q_rms_right_64 = q_rms_sum_128[64:128, :]
                q_rms_sum_64 = pl.add(q_rms_left_64, q_rms_right_64)
                q_rms_left_32 = q_rms_sum_64[0:32, :]
                q_rms_right_32 = q_rms_sum_64[32:64, :]
                q_rms_sum_32 = pl.add(q_rms_left_32, q_rms_right_32)
                q_rms_left_16 = q_rms_sum_32[0:16, :]
                q_rms_right_16 = q_rms_sum_32[16:32, :]
                q_rms_sum_16 = pl.add(q_rms_left_16, q_rms_right_16)
                q_rms_left_8 = q_rms_sum_16[0:8, :]
                q_rms_right_8 = q_rms_sum_16[8:16, :]
                q_rms_sum_8 = pl.add(q_rms_left_8, q_rms_right_8)
                q_rms_left_4 = q_rms_sum_8[0:4, :]
                q_rms_right_4 = q_rms_sum_8[4:8, :]
                q_rms_sum_4 = pl.add(q_rms_left_4, q_rms_right_4)
                q_rms_left_2 = q_rms_sum_4[0:2, :]
                q_rms_right_2 = q_rms_sum_4[2:4, :]
                q_rms_sum_2 = pl.add(q_rms_left_2, q_rms_right_2)
                q_rms_left_1 = q_rms_sum_2[0:1, :]
                q_rms_right_1 = q_rms_sum_2[1:2, :]
                q_head_sq_row = pl.add(q_rms_left_1, q_rms_right_1)
                q_head_sq_sum = q_head_sq_row
                q_head_sq_mean = pl.mul(q_head_sq_sum, 1.0 / HEAD_DIM)
                q_rms_shape_carrier = pl.slice(qr_scale_view, [Q_ROPE_T_TILE, 1], [tg, 0], valid_shape=[q_rows_out, 1])
                q_head_var = pl.reshape(q_rms_shape_carrier, [1, Q_ROPE_T_TILE])
                q_head_var = pl.add(q_head_sq_mean, EPS)
                q_rms_raw_sqrt = q_head_var
                q_rms_raw_sqrt = pl.sqrt(q_head_var)
                q_rms_newton = _q_rms_newton_sqrt_seed(q_head_var, q_rms_raw_sqrt)
                q_rms_fixed_sqrt = _q_rms_midpoint_sqrt_seed(q_head_var, q_rms_newton)
                q_head_inv_rms = _q_rms_reciprocal_eft(q_rms_fixed_sqrt, factor=5.421010862427522e-20)
                q_head_inv_rms_t = pl.reshape(q_head_inv_rms, [Q_ROPE_T_TILE, 1])

                q_nope_normed = pl.row_expand_mul(q_head_dq[:, 0:NOPE_DIM], q_head_inv_rms_t)
                q_nope_bf16 = pl.cast(q_nope_normed, target_type=pl.BF16, mode="rint")
                q_flat[tg : tg + Q_ROPE_T_TILE, h0 : h0 + NOPE_DIM] = pl.set_validshape(q_nope_bf16, q_rows_out, NOPE_DIM)

                # RoPE writeback on columns [h0+NOPE_DIM:h0+HEAD_DIM). Fold inv_rms in
                # BEFORE the rotation (normalize-then-rotate), matching the kv path. This
                # is mathematically equivalent to rotating then normalizing — inv_rms is a
                # per-row scalar so inv_rms*(a*cos+b*sin) == (a*inv_rms)*cos+(b*inv_rms)*sin
                # — but keeps the rotation intermediates small. Rotating the raw (large)
                # dequantized values first produced large intermediates that lost precision
                # on Ascend950 (A5), corrupting the query RoPE region; normalizing first
                # avoids that without changing the result on A2A3.
                q_rope_chunk_raw = q_head_dq[:, NOPE_DIM:HEAD_DIM]
                q_rope_chunk = pl.row_expand_mul(q_rope_chunk_raw, q_head_inv_rms_t)
                q_rope_swapped = pl.gather(q_rope_chunk, dim=-1, index=q_swap_idx)
                q_rope_base = pl.mul(q_rope_chunk, q_cos_il)
                q_rope_delta = pl.mul(q_rope_swapped, q_sin_signed)
                q_rope_rot = pl.add(q_rope_base, q_rope_delta)
                q_rope_bf16 = pl.cast(q_rope_rot, target_type=pl.BF16, mode="rint")
                q_flat[tg : tg + Q_ROPE_T_TILE, h0 + NOPE_DIM : h0 + NOPE_DIM + ROPE_DIM] = pl.set_validshape(q_rope_bf16, q_rows_out, ROPE_DIM)

    # Split-K kv_proj uses four 128-column N-groups and KV_OK=2, again producing
    # 16 cube blocks. KV is off the critical path, so more K splits only add atomic
    # contention without shortening decode.
    kv_fp32 = pl.create_tensor([t_matmul, HEAD_DIM], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_proj_seed", allow_early_resolve=True):
        for tc in pl.range(t_matmul // KV_M_TILE):
            kts0 = tc * KV_M_TILE
            for nb in pl.range(HEAD_DIM // KV_N_TILE):
                kvseed0 = nb * KV_N_TILE
                kv_fp32[kts0 : kts0 + KV_M_TILE, kvseed0 : kvseed0 + KV_N_TILE] = pl.full(
                    [KV_M_TILE, KV_N_TILE], dtype=pl.FP32, value=0.0
                )
    with pl.spmd(
        (HEAD_DIM // KV_N_TILE) * KV_OK,
        name_hint="kv_proj_matmul",
        allow_early_resolve=True,
        deps=[late_dep],
    ) as _kv_proj_tid:
        kbg = pl.tile.get_block_idx()
        kv_col0 = (kbg // KV_OK) * KV_N_TILE
        kv_k_base = (kbg % KV_OK) * KV_K_SLICE
        for tc in pl.range(t_matmul // KV_M_TILE):
            t0 = tc * KV_M_TILE
            kv_acc = pl.create_tensor([KV_M_TILE, KV_N_TILE], dtype=pl.FP32)
            for db in pl.pipeline(KV_K_SLICE // KV_K_TILE, stage=2):
                d0 = kv_k_base + db * KV_K_TILE
                kv_rows = pl.min(KV_M_TILE, t_dim - t0)
                kv_x_chunk_bf16 = pl.slice(x_view, [KV_M_TILE, KV_K_TILE], [t0, d0], valid_shape=[kv_rows, KV_K_TILE])
                wkv_chunk = wkv[d0 : d0 + KV_K_TILE, kv_col0 : kv_col0 + KV_N_TILE]
                kv_acc = pl.matmul_acc(kv_acc, kv_x_chunk_bf16, wkv_chunk, init_cond=(db == 0))
            kv_fp32 = pl.assemble(kv_fp32, kv_acc, [t0, kv_col0], atomic=pl.AtomicType.Add)

    # Fused KV RMSNorm + interleaved (CANN A3) RoPE. One spmd task per [KV_RMS_T_TILE, HEAD_DIM]
    # row block computes the per-row inv_rms once (pass 1) and consumes it locally for
    # BOTH the NOPE writeback and the rope rotation -- so inv_rms no longer round-trips
    # through GM (the old kv_inv_rms_tensor) and the two passes collapse into a single
    # dispatch. NOPE columns [0:NOPE_DIM) and rope columns [NOPE_DIM:HEAD_DIM) are
    # disjoint, so each task writes a clean, conflict-free row block of kv. Vec UB stays
    # well under the 192 KB cap (chunks are at most [KV_RMS_T_TILE, KV_TILE] fp32).
    for tg_idx in pl.spmd((t_dim + KV_RMS_T_TILE - 1) // KV_RMS_T_TILE, name_hint="kv_rms_norm_rope"):
        tg = tg_idx * KV_RMS_T_TILE
        kv_rows_out = pl.min(KV_RMS_T_TILE, t_dim - tg)
        # Pass 1: per-row sum of squares over the full HEAD_DIM -> inv_rms.
        # Fixed cascade order matching the owning FP32 mean reference.
        # Four 8-lane accumulators, then left-to-right vector and lane folds.
        kv_all_rows = kv_fp32[tg : tg + KV_RMS_T_TILE, 0:HEAD_DIM]
        kv_squared_rows = pl.mul(kv_all_rows, kv_all_rows)
        kv_cascade = pl.full([KV_RMS_T_TILE, 32], dtype=pl.FP32, value=0.0)
        for kv_step in pl.range(16):
            kv_start = kv_step * 32
            kv_cascade = pl.add(kv_cascade, kv_squared_rows[:, kv_start:kv_start + 32])
        kv_cascade_columns = pl.transpose(kv_cascade, 0, 1)
        kv_cascade_01 = pl.add(kv_cascade_columns[0:8, :], kv_cascade_columns[8:16, :])
        kv_cascade_012 = pl.add(kv_cascade_01, kv_cascade_columns[16:24, :])
        kv_cascade_lanes = pl.add(kv_cascade_012, kv_cascade_columns[24:32, :])
        kv_sq_sum = pl.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=0.0)
        for kv_lane in pl.range(8):
            kv_sq_sum = pl.add(kv_sq_sum, kv_cascade_lanes[kv_lane:kv_lane + 1, :])
        kv_rms_variance = pl.full([1, KV_RMS_T_TILE], dtype=pl.FP32, value=0.0)
        kv_rms_variance = pl.add(pl.mul(kv_sq_sum, 1.0 / HEAD_DIM), EPS)
        kv_rms_original = pl.rsqrt(kv_rms_variance, high_precision=True)
        kv_rms_work = pl.minimum(kv_rms_variance, 1099511627776.0)
        kv_rms_raw_sqrt = kv_rms_work
        kv_rms_raw_sqrt = pl.sqrt(kv_rms_work)
        kvn_result = kv_rms_work
        kvn_work = pl.mul(kv_rms_work, 0.25)
        kvn_z = pl.mul(kv_rms_raw_sqrt, 0.5)
        kvn_split = pl.mul(kvn_z, 4097.0)
        kvn_delta = pl.sub(kvn_split, kvn_z)
        kvn_hi = pl.sub(kvn_split, kvn_delta)
        kvn_lo = pl.sub(kvn_z, kvn_hi)
        kvn_product = pl.mul(kvn_z, kvn_z)
        kvn_hi_product = pl.mul(kvn_hi, kvn_hi)
        kvn_error = pl.sub(kvn_hi_product, kvn_product)
        kvn_cross = pl.mul(kvn_hi, kvn_lo)
        kvn_error = pl.add(kvn_error, kvn_cross)
        kvn_error = pl.add(kvn_error, kvn_cross)
        kvn_lo_product = pl.mul(kvn_lo, kvn_lo)
        kvn_error = pl.add(kvn_error, kvn_lo_product)
        kvn_residual_high = pl.sub(kvn_work, kvn_product)
        kvn_residual = pl.sub(kvn_residual_high, kvn_error)
        kvn_denominator = pl.mul(kvn_z, 2.0)
        kvn_step = pl.div(kvn_residual, kvn_denominator)
        kvn_adjusted = pl.add(kvn_z, kvn_step)
        kvn_result = pl.mul(kvn_adjusted, 2.0)
        kv_rms_newton = kvn_result
        kvm_result = kv_rms_newton
        kvm_neighbor_delta = pl.mul(kv_rms_newton, 2.0 ** (-24) + 2.0 ** (-47))
        kvm_lower = pl.sub(kv_rms_newton, kvm_neighbor_delta)
        kvm_upper = pl.add(kv_rms_newton, kvm_neighbor_delta)
        kvm_z = pl.mul(kv_rms_newton, 0.5)
        kvm_lo_gap_raw = pl.sub(kv_rms_newton, kvm_lower)
        kvm_lo_gap = pl.mul(kvm_lo_gap_raw, 0.5)
        kvm_hi_gap_raw = pl.sub(kvm_upper, kv_rms_newton)
        kvm_hi_gap = pl.mul(kvm_hi_gap_raw, 0.5)
        kvm_split = pl.mul(kvm_z, 4097.0)
        kvm_split_delta = pl.sub(kvm_split, kvm_z)
        kvm_hi = pl.sub(kvm_split, kvm_split_delta)
        kvm_lo = pl.sub(kvm_z, kvm_hi)
        kvm_product = pl.mul(kvm_z, kvm_z)
        kvm_cross = pl.mul(kvm_hi, kvm_lo)
        kvm_hi_product = pl.mul(kvm_hi, kvm_hi)
        kvm_error = pl.sub(kvm_hi_product, kvm_product)
        kvm_error = pl.add(kvm_error, kvm_cross)
        kvm_error = pl.add(kvm_error, kvm_cross)
        kvm_lo_product = pl.mul(kvm_lo, kvm_lo)
        kvm_error = pl.add(kvm_error, kvm_lo_product)
        kvm_quarter_var = pl.mul(kv_rms_work, 0.25)
        kvm_residual_input = pl.sub(kvm_quarter_var, kvm_product)
        kvm_negative_error = pl.neg(kvm_error)
        kvm_rh = pl.add(kvm_residual_input, kvm_negative_error)
        kvm_residual_z = pl.sub(kvm_rh, kvm_residual_input)
        kvm_residual_sum_minus_z = pl.sub(kvm_rh, kvm_residual_z)
        kvm_residual_a_error = pl.sub(kvm_residual_input, kvm_residual_sum_minus_z)
        kvm_residual_b_error = pl.sub(kvm_negative_error, kvm_residual_z)
        kvm_rl = pl.add(kvm_residual_a_error, kvm_residual_b_error)
        kvm_z_lo_gap = pl.mul(kvm_z, kvm_lo_gap)
        kvm_l1 = pl.add(kvm_rh, kvm_z_lo_gap)
        kvm_lower_first_z = pl.sub(kvm_l1, kvm_rh)
        kvm_lower_first_sum_minus_z = pl.sub(kvm_l1, kvm_lower_first_z)
        kvm_lower_first_a_error = pl.sub(kvm_rh, kvm_lower_first_sum_minus_z)
        kvm_lower_first_b_error = pl.sub(kvm_z_lo_gap, kvm_lower_first_z)
        kvm_e1 = pl.add(kvm_lower_first_a_error, kvm_lower_first_b_error)
        kvm_negative_lo_gap = pl.neg(kvm_lo_gap)
        kvm_negative_lo_square = pl.mul(kvm_negative_lo_gap, kvm_lo_gap)
        kvm_negative_lo_quarter_square = pl.mul(kvm_negative_lo_square, 0.25)
        kvm_l2 = pl.add(kvm_l1, kvm_negative_lo_quarter_square)
        kvm_lower_second_z = pl.sub(kvm_l2, kvm_l1)
        kvm_lower_second_sum_minus_z = pl.sub(kvm_l2, kvm_lower_second_z)
        kvm_lower_second_a_error = pl.sub(kvm_l1, kvm_lower_second_sum_minus_z)
        kvm_lower_second_b_error = pl.sub(kvm_negative_lo_quarter_square, kvm_lower_second_z)
        kvm_e2 = pl.add(kvm_lower_second_a_error, kvm_lower_second_b_error)
        kvm_lower_errors_first = pl.add(kvm_e1, kvm_rl)
        kvm_lower_errors = pl.add(kvm_lower_errors_first, kvm_e2)
        kvm_dlo = pl.add(kvm_l2, kvm_lower_errors)
        kvm_negative_z = pl.neg(kvm_z)
        kvm_negative_z_hi_gap = pl.mul(kvm_negative_z, kvm_hi_gap)
        kvm_h1 = pl.add(kvm_rh, kvm_negative_z_hi_gap)
        kvm_upper_first_z = pl.sub(kvm_h1, kvm_rh)
        kvm_upper_first_sum_minus_z = pl.sub(kvm_h1, kvm_upper_first_z)
        kvm_upper_first_a_error = pl.sub(kvm_rh, kvm_upper_first_sum_minus_z)
        kvm_upper_first_b_error = pl.sub(kvm_negative_z_hi_gap, kvm_upper_first_z)
        kvm_f1 = pl.add(kvm_upper_first_a_error, kvm_upper_first_b_error)
        kvm_negative_hi_gap = pl.neg(kvm_hi_gap)
        kvm_negative_hi_square = pl.mul(kvm_negative_hi_gap, kvm_hi_gap)
        kvm_negative_hi_quarter_square = pl.mul(kvm_negative_hi_square, 0.25)
        kvm_h2 = pl.add(kvm_h1, kvm_negative_hi_quarter_square)
        kvm_upper_second_z = pl.sub(kvm_h2, kvm_h1)
        kvm_upper_second_sum_minus_z = pl.sub(kvm_h2, kvm_upper_second_z)
        kvm_upper_second_a_error = pl.sub(kvm_h1, kvm_upper_second_sum_minus_z)
        kvm_upper_second_b_error = pl.sub(kvm_negative_hi_quarter_square, kvm_upper_second_z)
        kvm_f2 = pl.add(kvm_upper_second_a_error, kvm_upper_second_b_error)
        kvm_upper_errors_first = pl.add(kvm_f1, kvm_rl)
        kvm_upper_errors = pl.add(kvm_upper_errors_first, kvm_f2)
        kvm_dhi = pl.add(kvm_h2, kvm_upper_errors)
        kvm_negative_dlo = pl.neg(kvm_dlo)
        kvm_down_denominator = pl.mul(kvm_z, kvm_lo_gap)
        kvm_down_ratio = pl.div(kvm_negative_dlo, kvm_down_denominator)
        kvm_down_scaled = pl.mul(kvm_down_ratio, 2.0 ** 40)
        kvm_down_bounded = pl.minimum(kvm_down_scaled, 1.0)
        kvm_down_flag = pl.maximum(kvm_down_bounded, 0.0)
        kvm_up_denominator = pl.mul(kvm_z, kvm_hi_gap)
        kvm_up_ratio = pl.div(kvm_dhi, kvm_up_denominator)
        kvm_up_scaled = pl.mul(kvm_up_ratio, 2.0 ** 40)
        kvm_up_bounded = pl.minimum(kvm_up_scaled, 1.0)
        kvm_up_flag = pl.maximum(kvm_up_bounded, 0.0)
        kvm_lower_difference = pl.sub(kvm_lower, kv_rms_newton)
        kvm_down_adjustment = pl.mul(kvm_down_flag, kvm_lower_difference)
        kvm_first_result = pl.add(kv_rms_newton, kvm_down_adjustment)
        kvm_upper_difference = pl.sub(kvm_upper, kv_rms_newton)
        kvm_up_adjustment = pl.mul(kvm_up_flag, kvm_upper_difference)
        kvm_result = pl.add(kvm_first_result, kvm_up_adjustment)
        kv_rms_correct_sqrt = kvm_result
        kvr_result = kv_rms_correct_sqrt
        kvr_ones_row = pl.full([1, 8], dtype=pl.FP32, value=1.0)
        kvr_ef_work = pl.mul(kv_rms_correct_sqrt, 5.421010862427522e-20)
        kvr_ef_estimate = pl.div(kvr_ones_row, kvr_ef_work)
        kvr_ef_left_split = pl.mul(kvr_ef_estimate, 4097.0)
        kvr_ef_left_delta = pl.sub(kvr_ef_left_split, kvr_ef_estimate)
        kvr_ef_left_hi = pl.sub(kvr_ef_left_split, kvr_ef_left_delta)
        kvr_ef_left_lo = pl.sub(kvr_ef_estimate, kvr_ef_left_hi)
        kvr_ef_right_split = pl.mul(kvr_ef_work, 4097.0)
        kvr_ef_right_delta = pl.sub(kvr_ef_right_split, kvr_ef_work)
        kvr_ef_right_hi = pl.sub(kvr_ef_right_split, kvr_ef_right_delta)
        kvr_ef_right_lo = pl.sub(kvr_ef_work, kvr_ef_right_hi)
        kvr_ef_product = pl.mul(kvr_ef_estimate, kvr_ef_work)
        kvr_ef_hi_product = pl.mul(kvr_ef_left_hi, kvr_ef_right_hi)
        kvr_ef_error = pl.sub(kvr_ef_hi_product, kvr_ef_product)
        kvr_ef_hi_lo = pl.mul(kvr_ef_left_hi, kvr_ef_right_lo)
        kvr_ef_error = pl.add(kvr_ef_error, kvr_ef_hi_lo)
        kvr_ef_lo_hi = pl.mul(kvr_ef_left_lo, kvr_ef_right_hi)
        kvr_ef_error = pl.add(kvr_ef_error, kvr_ef_lo_hi)
        kvr_ef_lo_product = pl.mul(kvr_ef_left_lo, kvr_ef_right_lo)
        kvr_ef_error = pl.add(kvr_ef_error, kvr_ef_lo_product)
        kvr_ef_residual_hi = pl.sub(kvr_ones_row, kvr_ef_product)
        kvr_ef_residual = pl.sub(kvr_ef_residual_hi, kvr_ef_error)
        kvr_ef_correction = pl.div(kvr_ef_residual, kvr_ef_work)
        kvr_ef_corrected_work = pl.add(kvr_ef_estimate, kvr_ef_correction)
        kvr_result = pl.mul(kvr_ef_corrected_work, 5.421010862427522e-20)
        kv_rms_correct_inv = kvr_result
        kv_rms_upper = pl.add(pl.sub(kv_rms_work, kv_rms_work), 1099511758848.0)
        kv_rms_inside = pl.mul(pl.sub(kv_rms_upper, kv_rms_variance), 2.0 ** -17)
        kv_rms_inside = pl.minimum(pl.maximum(kv_rms_inside, 0.0), 1.0)
        kv_rms_other = pl.sub(pl.add(pl.sub(kv_rms_work, kv_rms_work), 1.0), kv_rms_inside)
        kv_rms_preferred = pl.mul(kv_rms_correct_inv, kv_rms_inside)
        kv_rms_fallback = pl.mul(kv_rms_original, kv_rms_other)
        kv_inv_rms = pl.add(kv_rms_preferred, kv_rms_fallback)
        kv_inv_rms_t = pl.reshape(kv_inv_rms, [KV_RMS_T_TILE, 1])

        # NOPE writeback: rms-normalize columns [0:NOPE_DIM) with per-column gamma.
        for nb in pl.pipeline(NOPE_DIM // KV_TILE, stage=2):
            n0 = nb * KV_TILE
            kv_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE]
            gamma_kv_cast = pl.cast(gamma_ckv[n0 : n0 + KV_TILE], target_type=pl.FP32)
            gamma_kv_chunk = pl.reshape(gamma_kv_cast, [1, KV_TILE])
            kv_normed = pl.col_expand_mul(pl.row_expand_mul(kv_chunk, kv_inv_rms_t), gamma_kv_chunk)
            kv_normed_bf16 = pl.cast(kv_normed, target_type=pl.BF16, mode="rint")
            kv_view[tg : tg + KV_RMS_T_TILE, n0 : n0 + KV_TILE] = pl.set_validshape(kv_normed_bf16, kv_rows_out, KV_TILE)

        # RoPE writeback on columns [NOPE_DIM:HEAD_DIM), interleaved (CANN A3) swap-gather
        # (same form as qproj_dequant_rms_nope_rope). inv_rms (per-row, the same factor used
        # for NOPE above) and gamma (per-column, full ROPE_DIM) are folded into
        # kv_rope_norm_chunk BEFORE the swap so the swapped lane n[j^1] carries gamma[j^1]
        # (gamma does NOT commute with the rotation; inv_rms does).
        #   out[j] = n[j]*cos_il[j] + n[j^1]*sin_il_signed[j]
        #
        # q_rope_prepare above already built cos_il / sign-folded sin / swap_idx over the
        # full [t_dim, ROPE_DIM] token rows from the same rope_cos_view -- slice them here
        # instead of re-running the arange chain and re-gathering the same two tables.
        # Values are bit-identical: same source rows, same j>>1 index, and folding the
        # +/-1 sign into sin only flips a sign bit, so (n[j^1]*sign)*sin == n[j^1]*(sin*sign).
        gamma_rope_cast = pl.cast(gamma_ckv[NOPE_DIM : NOPE_DIM + ROPE_DIM], target_type=pl.FP32)
        gamma_rope = pl.reshape(gamma_rope_cast, [1, ROPE_DIM])
        kv_rope_chunk = kv_fp32[tg : tg + KV_RMS_T_TILE, NOPE_DIM : NOPE_DIM + ROPE_DIM]
        kv_rope_norm_chunk = pl.col_expand_mul(pl.row_expand_mul(kv_rope_chunk, kv_inv_rms_t), gamma_rope)
        kv_cos_il = pl.slice(q_rope_cos_il, [KV_RMS_T_TILE, ROPE_DIM], [tg, 0], valid_shape=[kv_rows_out, ROPE_DIM])
        kv_sin_signed = pl.slice(q_rope_sin_signed, [KV_RMS_T_TILE, ROPE_DIM], [tg, 0], valid_shape=[kv_rows_out, ROPE_DIM])
        # Initialize indices for all internal tile rows.
        kv_swap_idx = pl.slice(q_rope_swap_idx, [KV_RMS_T_TILE, ROPE_DIM], [tg, 0], valid_shape=[kv_rows_out, ROPE_DIM])
        kv_swap_idx = pl.fillpad(kv_swap_idx, pad_value=pl.PadValue.zero)
        kv_swap_idx = pl.set_validshape(kv_swap_idx, KV_RMS_T_TILE, ROPE_DIM)
        kv_swapped = pl.gather(kv_rope_norm_chunk, dim=-1, index=kv_swap_idx)
        kv_rope_rot = pl.add(pl.mul(kv_rope_norm_chunk, kv_cos_il), pl.mul(kv_swapped, kv_sin_signed))
        kv_rope_i16 = pl.cast(kv_rope_rot, target_type=pl.BF16, mode="rint")
        kv_view[tg : tg + KV_RMS_T_TILE, NOPE_DIM : NOPE_DIM + ROPE_DIM] = pl.set_validshape(kv_rope_i16, kv_rows_out, ROPE_DIM)

    return q
