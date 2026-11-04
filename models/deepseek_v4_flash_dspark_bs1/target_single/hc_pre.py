# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hyper-connection pre-mix with bounded FP32 normalization and maps."""

"""Single-query shape adapter for the shared hyper-connection pre-mix."""
import pypto.language as pl
from ..target.hc_pre import D, HC_MULT, HC_DIM, MIX_HC, hc_pre as shared_hc_pre
from .config import DECODE_BATCH, DECODE_SEQ

T_DYN = pl.dynamic("T_DYN")
if DECODE_BATCH != 1 or DECODE_SEQ != 1:
    raise ValueError("The single-query target requires batch one and one query")


@pl.jit.inline(auto_scope=False)
def hc_pre(
    x: pl.Tensor[[T_DYN, HC_MULT, D], pl.FP32],
    hc_fn: pl.Tensor[[MIX_HC, HC_DIM], pl.FP32],
    hc_scale: pl.Tensor[[3], pl.FP32],
    hc_base: pl.Tensor[[MIX_HC], pl.FP32],
    x_mixed: pl.Tensor[[T_DYN, D], pl.BF16],
    post: pl.Tensor[[T_DYN, HC_MULT], pl.FP32],
    comb: pl.Tensor[[T_DYN, HC_MULT * HC_MULT], pl.FP32],
):
    x_single = pl.reshape(x, [1, HC_MULT, D])
    mixed_single = pl.reshape(x_mixed, [1, D])
    post_single = pl.reshape(post, [1, HC_MULT])
    comb_single = pl.reshape(comb, [1, HC_MULT * HC_MULT])
    shared_hc_pre(x_single, hc_fn, hc_scale, hc_base, mixed_single, post_single, comb_single)
    return x_mixed
