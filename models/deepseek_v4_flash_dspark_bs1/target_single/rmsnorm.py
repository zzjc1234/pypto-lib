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

T_DYN = pl.dynamic("T_DYN")  # T = B * S

D = M.hidden_size

EPS = M.rms_norm_eps

REDUCE_ROW_TILE = 16

@pl.jit.inline
def rms_norm(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    x_normed: pl.Tensor[[T_DYN, D], pl.BF16],
):
    t_dim = pl.tensor.dim(x, 0)
    norm_w_view = pl.reshape(norm_w, [1, D])
    with pl.spmd(t_dim, name_hint="rms_norm", allow_early_resolve=True) as rms_tid:
        tok = pl.tile.get_block_idx()
        x_row_bf16 = pl.tile.load(x, [tok, 0], [1, D])
        w_row_bf16 = pl.tile.load(norm_w_view, [0, 0], [1, D])
        x_row = pl.cast(x_row_bf16, target_type=pl.FP32)
        w_row = pl.cast(w_row_bf16, target_type=pl.FP32)
        x_sq = pl.mul(x_row, x_row)
        sq_rows = pl.reshape(x_sq, [REDUCE_ROW_TILE, D // REDUCE_ROW_TILE])
        partial_tmp = pl.create_tile([REDUCE_ROW_TILE, D // REDUCE_ROW_TILE], dtype=pl.FP32)
        partial = pl.row_sum(sq_rows, partial_tmp)
        reduce_tile = pl.create_tile([REDUCE_ROW_TILE, REDUCE_ROW_TILE], dtype=pl.FP32)
        reduce_tile[0:1, :] = pl.reshape(partial, [1, REDUCE_ROW_TILE])
        reduce_tile = pl.set_validshape(reduce_tile, 1, REDUCE_ROW_TILE)
        sum_tmp = pl.create_tile([REDUCE_ROW_TILE, REDUCE_ROW_TILE], dtype=pl.FP32)
        sum_tile = pl.row_sum(reduce_tile, sum_tmp)
        sum_row = pl.reshape(sum_tile, [1, REDUCE_ROW_TILE])
        x_sq_sum = pl.set_validshape(sum_row, 1, 1)
        x_mean_sq = pl.mul(x_sq_sum, 1.0 / D)
        x_mean_eps = pl.add(x_mean_sq, EPS)
        rsqrt_tmp = pl.create_tile([1, REDUCE_ROW_TILE], dtype=pl.FP32)
        x_inv_rms = pl.tile.rsqrt(x_mean_eps, rsqrt_tmp)
        inv_rms_scalar = pl.tile.read(x_inv_rms, [0, 0])
        x_scaled = pl.mul(x_row, inv_rms_scalar)
        x_weighted = pl.mul(x_scaled, w_row)
        normed_bf16 = pl.cast(x_weighted, target_type=pl.BF16, mode="rint")
        pl.tile.store(normed_bf16, [tok, 0], x_normed, shapes=[1, D])

    return rms_tid
