# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Hyper-connection pre-mix with bounded FP32 normalization and maps."""

"""Single-query shape adapters for the shared QKV projection and RoPE."""
import pypto.language as pl
from ..target.qkv_proj_rope import D, H, HEAD_DIM, Q_LORA, ROPE_DIM
from ..target.qkv_proj_rope import prepare_qkv_rope_metadata as shared_prepare
from ..target.qkv_proj_rope import qkv_proj_rope as shared_projection
from .config import DECODE_BATCH, DECODE_SEQ

T_DYN = pl.dynamic("T_DYN")
if DECODE_BATCH != 1 or DECODE_SEQ != 1:
    raise ValueError("The single-query target requires batch one and one query")


@pl.jit.inline(auto_scope=False)
def prepare_qkv_rope_metadata(
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM], pl.BF16],
    q_rope_cos_il: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    q_rope_sin_signed: pl.Tensor[[T_DYN, ROPE_DIM], pl.FP32],
    q_rope_swap_idx: pl.Tensor[[T_DYN, ROPE_DIM], pl.INT32],
):
    cos_single = pl.reshape(rope_cos, [1, ROPE_DIM])
    sin_single = pl.reshape(rope_sin, [1, ROPE_DIM])
    cos_il_single = pl.reshape(q_rope_cos_il, [1, ROPE_DIM])
    sin_signed_single = pl.reshape(q_rope_sin_signed, [1, ROPE_DIM])
    swap_single = pl.reshape(q_rope_swap_idx, [1, ROPE_DIM])
    shared_prepare(cos_single, sin_single, cos_il_single, sin_signed_single, swap_single)


@pl.jit.inline(auto_scope=False)
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
    x_single = pl.reshape(x, [1, D])
    cos_single = pl.reshape(rope_cos, [1, ROPE_DIM])
    sin_single = pl.reshape(rope_sin, [1, ROPE_DIM])
    q_single = pl.reshape(q, [1, H, HEAD_DIM])
    kv_single = pl.reshape(kv, [1, HEAD_DIM])
    qr_single = pl.reshape(qr, [1, Q_LORA])
    scale_single = pl.reshape(qr_scale, [1, 1])
    cos_il_single = pl.reshape(q_rope_cos_il, [1, ROPE_DIM])
    sin_signed_single = pl.reshape(q_rope_sin_signed, [1, ROPE_DIM])
    swap_single = pl.reshape(q_rope_swap_idx, [1, ROPE_DIM])
    shared_projection(
        x_single, wq_a, wq_b, wq_b_scale, wkv, cos_single, sin_single,
        gamma_cq, gamma_ckv, q_single, kv_single, qr_single, scale_single,
        late_dep, q_head_base, cos_il_single, sin_signed_single, swap_single,
    )
    return q
