# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl

from .config import FLASH as M

T_DYN = pl.dynamic('NATIVE_S_T_DYN')  # T = B * S

D = M.hidden_size

HC_MULT = M.hc_mult

HC_DIM = M.hc_dim

EPS = M.rms_norm_eps

HC_EPS = M.hc_eps

HC_DIM_INV = 1.0 / HC_DIM

HC_PAD = 16

HC_VPAD = 8

T_TILE = 8

LINEAR_T_TILE = 16

RMS_K_TILE = 512

LINEAR_K_TILE = 256

D_TILE = 512

D_SPMD = 512

LINEAR_OK = 16

RMS_OK = 16

PARTIAL_ROWS_TILE = LINEAR_OK * LINEAR_T_TILE

@pl.jit.inline
def hc_head(
    x_hc: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_head_fn: pl.Tensor[[HC_MULT, HC_DIM], pl.FP32],
    hc_head_scale: pl.Tensor[[1], pl.FP32],
    hc_head_base: pl.Tensor[[HC_MULT], pl.FP32],
    y: pl.Tensor[[T_DYN, D], pl.BF16],
):
    t_dim = pl.tensor.dim(x_hc, 0)
    t_linear = ((t_dim + LINEAR_T_TILE - 1) // LINEAR_T_TILE) * LINEAR_T_TILE
    t_vector = ((t_dim + T_TILE - 1) // T_TILE) * T_TILE
    x_flat = pl.reshape(x_hc, [t_dim, HC_DIM])
    y_flat = pl.reshape(y, [t_dim, D])
    # rms: split-K sum-of-squares, fanned over (token-tile x K-slice)
    sq_part = pl.create_tensor([RMS_OK, t_vector], dtype=pl.FP32)
    for task in pl.spmd(((t_dim + T_TILE - 1) // T_TILE) * RMS_OK, name_hint="hc_head_rms", allow_early_resolve=True):
        t0 = (task // RMS_OK) * T_TILE
        rms_rows = pl.min(T_TILE, t_dim - t0)
        ok = task % RMS_OK
        k_base = ok * (HC_DIM // RMS_OK)
        sq_sum = pl.full([1, T_TILE], dtype=pl.FP32, value=0.0)
        for kb in pl.pipeline(HC_DIM // RMS_OK // RMS_K_TILE, stage=4):
            k0 = k_base + kb * RMS_K_TILE
            x_rms = pl.slice(x_flat, [T_TILE, RMS_K_TILE], [t0, k0], valid_shape=[rms_rows, RMS_K_TILE])
            sq_col = pl.row_sum(pl.mul(x_rms, x_rms))
            sq_sum = pl.add(sq_sum, pl.reshape(sq_col, [1, T_TILE]))
        sq_part = pl.assemble(sq_part, sq_sum, [ok, t0])

    # M-major FP32 partials: [M block, K split, row, HC column].
    partial_rows = LINEAR_OK * t_linear
    mixes_partials = pl.create_tensor([partial_rows, HC_PAD], dtype=pl.FP32)
    linear_tasks = (t_linear // LINEAR_T_TILE) * LINEAR_OK
    with pl.spmd(linear_tasks, name_hint="hc_head_partial", allow_early_resolve=True) as linear_tid:
        task = pl.tile.get_block_idx()
        t0 = (task // LINEAR_OK) * LINEAR_T_TILE
        linear_rows = pl.min(LINEAR_T_TILE, t_dim - t0)
        k_base = (task % LINEAR_OK) * (HC_DIM // LINEAR_OK)
        acc = pl.create_tensor([LINEAR_T_TILE, HC_PAD], dtype=pl.FP32)
        for kb in pl.pipeline(0, HC_DIM // LINEAR_OK // LINEAR_K_TILE, stage=2):
            k0 = k_base + kb * LINEAR_K_TILE
            x_lin = pl.slice(x_flat, [LINEAR_T_TILE, LINEAR_K_TILE], [t0, k0], valid_shape=[linear_rows, LINEAR_K_TILE])
            w = pl.slice(hc_head_fn, [HC_PAD, LINEAR_K_TILE], [0, k0], valid_shape=[HC_MULT, LINEAR_K_TILE])
            acc = pl.matmul_acc(acc, x_lin, w, b_trans=True, init_cond=(kb == 0))
        partial_t0 = task * LINEAR_T_TILE
        partial = pl.set_validshape(acc, LINEAR_T_TILE, HC_PAD)
        mixes_partials[partial_t0 : partial_t0 + LINEAR_T_TILE, 0:HC_PAD] = partial

    # One 16 KiB load per M block, then positive-zero + K slots 0..15.
    mixes_raw = pl.create_tensor([t_linear, HC_PAD], dtype=pl.FP32)
    bulk_tasks = t_linear // LINEAR_T_TILE
    with pl.spmd(bulk_tasks, name_hint="hc_head_ordered_bulk", deps=[linear_tid], allow_early_resolve=True) as bulk_tid:
        block = pl.tile.get_block_idx()
        partial_t0 = block * LINEAR_OK * LINEAR_T_TILE
        partials = pl.load(mixes_partials, [partial_t0, 0], [PARTIAL_ROWS_TILE, HC_PAD], target_memory=pl.MemorySpace.Vec)
        total = pl.tile.full([LINEAR_T_TILE, HC_PAD], dtype=pl.FP32, value=0.0)
        for split in pl.range(LINEAR_OK):
            split_t0 = split * LINEAR_T_TILE
            next_partial = pl.tile.slice(partials, [LINEAR_T_TILE, HC_PAD], [split_t0, 0])
            total = pl.add(total, next_partial)
        output_t0 = block * LINEAR_T_TILE
        pl.store(total, [output_t0, 0], mixes_raw)

    # reduce: gate + hc mix, fanned over (token-tile x D-slice). The rsqrt/sigmoid gate is
    # recomputed per task instead of being published by its own scope.
    for blk in pl.spmd(
        ((t_dim + T_TILE - 1) // T_TILE) * (D // D_SPMD),
        name_hint="hc_head_reduce", deps=[bulk_tid], allow_early_resolve=True
    ):
        t0 = (blk // (D // D_SPMD)) * T_TILE
        d_base = (blk % (D // D_SPMD)) * D_SPMD
        output_rows = pl.min(T_TILE, t_dim - t0)
        scale = pl.read(hc_head_scale, [0])
        base = pl.reshape(pl.slice(hc_head_base, [HC_VPAD], [0], valid_shape=[HC_MULT]), [1, HC_VPAD])
        # sq_part is [RMS_OK, T], so the per-token total is a column sum; transposing the
        # slab makes it one row_sum landing in the [T_TILE, 1] shape row_expand_mul wants
        ssq_slab = pl.set_validshape(sq_part[0:RMS_OK, t0 : t0 + T_TILE], RMS_OK, T_TILE)
        ssq = pl.row_sum(pl.transpose(ssq_slab, axis1=0, axis2=1))
        inv_row = pl.rsqrt(pl.add(pl.mul(ssq, HC_DIM_INV), EPS), high_precision=True)
        inv_col = pl.reshape(inv_row, [T_TILE, 1])
        mix = mixes_raw[t0 : t0 + T_TILE, 0:HC_VPAD]
        scaled = pl.mul(pl.row_expand_mul(mix, inv_col), scale)
        logits = pl.add(scaled, pl.col_expand(scaled, base))
        pre_val = pl.add(pl.recip(pl.add(pl.exp(pl.neg(logits)), 1.0)), HC_EPS)
        # pin the valid shape: the gate carries valid=?x? from the dynamic-offset slice
        pre_val = pl.set_validshape(pre_val, T_TILE, HC_VPAD)
        pre_tile_t = pl.transpose(pre_val, axis1=0, axis2=1)
        pre0 = pl.reshape(pre_tile_t[0:1, 0:T_TILE], [T_TILE, 1])
        pre0 = pl.set_validshape(pre0, output_rows, 1)
        pre1 = pl.reshape(pre_tile_t[1:2, 0:T_TILE], [T_TILE, 1])
        pre1 = pl.set_validshape(pre1, output_rows, 1)
        pre2 = pl.reshape(pre_tile_t[2:3, 0:T_TILE], [T_TILE, 1])
        pre2 = pl.set_validshape(pre2, output_rows, 1)
        pre3 = pl.reshape(pre_tile_t[3:4, 0:T_TILE], [T_TILE, 1])
        pre3 = pl.set_validshape(pre3, output_rows, 1)
        for db in pl.pipeline(D_SPMD // D_TILE, stage=2):
            d0 = d_base + db * D_TILE
            x_h0 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 0 * D + d0], valid_shape=[output_rows, D_TILE])
            x_h1 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 1 * D + d0], valid_shape=[output_rows, D_TILE])
            x_h2 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 2 * D + d0], valid_shape=[output_rows, D_TILE])
            x_h3 = pl.slice(x_flat, [T_TILE, D_TILE], [t0, 3 * D + d0], valid_shape=[output_rows, D_TILE])
            y01 = pl.add(pl.row_expand_mul(x_h0, pre0), pl.row_expand_mul(x_h1, pre1))
            y23 = pl.add(pl.row_expand_mul(x_h2, pre2), pl.row_expand_mul(x_h3, pre3))
            y_tile = pl.add(y01, y23)
            y_bf16 = pl.cast(y_tile, target_type=pl.BF16, mode="rint")
            y_flat[t0 : t0 + T_TILE, d0 : d0 + D_TILE] = pl.set_validshape(y_bf16, output_rows, D_TILE)

    return y
