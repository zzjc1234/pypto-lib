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

T_TILE = 4

@pl.jit.inline
def hc_post(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    residual: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
    y: pl.Out[pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32]],
):
    t_dim = pl.tensor.dim(x, 0)

    residual_flat = pl.reshape(residual, [t_dim, HC_DIM])
    y_flat = pl.reshape(y, [t_dim, HC_DIM])

    for block in pl.spmd(((t_dim + T_TILE - 1) // T_TILE) * HC_MULT, name_hint="hc_post", allow_early_resolve=True):
        token_block = block // HC_MULT
        out_h = block % HC_MULT
        t0 = token_block * T_TILE
        token_end = pl.min(t0 + T_TILE, t_dim)
        for t in pl.pipeline(t0, token_end, stage=2):
            post_w = pl.read(post, [t, out_h])
            x_row = pl.cast(x[t : t + 1, 0:D], target_type=pl.FP32)
            y_row = pl.mul(x_row, post_w)
            for in_h in pl.pipeline(HC_MULT, stage=4):
                comb_w = pl.read(comb, [t, in_h * HC_MULT + out_h])
                res_d = in_h * D
                # residual is already FP32 (hc stream is FP32 end-to-end): no cast, read straight.
                res_row = residual_flat[t : t + 1, res_d : res_d + D]
                weighted = pl.mul(res_row, comb_w)
                y_row = pl.add(y_row, weighted)
            # y is FP32 (feeds the next layer's hc_pre directly): no BF16 output cast.
            y_flat[t : t + 1, out_h * D : out_h * D + D] = y_row
    return y
