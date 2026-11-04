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

ROPE_ROWS_DYN = pl.dynamic("LOWLAT_ROPE_INTERLEAVE_ROWS_DYN")

ROPE_HEAD_DIM = M.qk_rope_head_dim

HALF_ROPE = ROPE_HEAD_DIM // 2

@pl.jit.inline
def rope_interleave(
    cos_half: pl.Tensor[[ROPE_ROWS_DYN, HALF_ROPE], pl.FP32],
    sin_half: pl.Tensor[[ROPE_ROWS_DYN, HALF_ROPE], pl.FP32],
    cos_il: pl.Tensor[[ROPE_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
    sin_signed: pl.Tensor[[ROPE_ROWS_DYN, ROPE_HEAD_DIM], pl.FP32],
):
    """Expand half-width cos/sin rows to the interleaved, sign-folded rope layout."""
    rows = pl.tensor.dim(cos_half, 0)
    for row in pl.spmd(rows, name_hint="rope_interleave", allow_early_resolve=True):
        il_ones = pl.full([1, ROPE_HEAD_DIM], dtype=pl.FP32, value=1.0)
        il_columns = pl.arange(0, [1, ROPE_HEAD_DIM], dtype=pl.INT32)
        il_columns_f32 = pl.cast(il_columns, target_type=pl.FP32)
        il_col = pl.col_expand_mul(il_ones, il_columns_f32)
        il_half_col = pl.mul(il_col, 0.5)
        il_dup_idx = pl.cast(il_half_col, target_type=pl.INT32, mode="trunc")
        il_dup_f = pl.cast(il_dup_idx, target_type=pl.FP32)
        il_even_col = pl.mul(il_dup_f, 2.0)
        il_lane = pl.sub(il_col, il_even_col)
        il_double_lane = pl.mul(il_lane, 2.0)
        il_sign = pl.sub(il_double_lane, 1.0)
        cos_row = cos_half[row : row + 1, 0:HALF_ROPE]
        sin_row = sin_half[row : row + 1, 0:HALF_ROPE]
        cos_expanded = pl.gather(cos_row, dim=-1, index=il_dup_idx)
        sin_expanded = pl.gather(sin_row, dim=-1, index=il_dup_idx)
        cos_il[row : row + 1, 0:ROPE_HEAD_DIM] = cos_expanded
        sin_signed[row : row + 1, 0:ROPE_HEAD_DIM] = pl.mul(sin_expanded, il_sign)
    return cos_il, sin_signed
