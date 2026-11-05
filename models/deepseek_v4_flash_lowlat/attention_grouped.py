# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Standalone grouped attention compositions; packed kernels stay shared."""

import pypto.language as pl

if __package__:
    from . import decode_sparse_attn_swa as SWA
    from . import decode_sparse_attn_hca as HCA
    from .o_proj_grouped import o_proj_grouped
else:
    import decode_sparse_attn_swa as SWA
    import decode_sparse_attn_hca as HCA
    from o_proj_grouped import o_proj_grouped


@pl.jit.inline
def sparse_attn_swa(
    q: pl.Tensor[[SWA.T, SWA.H, SWA.HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[SWA.ORI_BLOCK_NUM_DYN, SWA.BLOCK_SIZE, 1, SWA.HEAD_DIM], pl.BF16],
    swa_indices: pl.Tensor[[SWA.T, SWA.WIN], pl.INT32],
    swa_lens: pl.Tensor[[SWA.T], pl.INT32],
    sparse_bias: pl.Tensor[[SWA.T, SWA.PADDED_TOPK], pl.FP32],
    attn_sink: pl.Tensor[[SWA.H], pl.FP32],
    freqs_cos: pl.Tensor[[SWA.T, SWA.ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[SWA.T, SWA.ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[SWA.O_GROUPS, SWA.O_LORA, SWA.O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[SWA.D, SWA.O_GROUPS * SWA.O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[SWA.D], pl.FP32],
    attn_out: pl.Tensor[[SWA.T, SWA.D], pl.BF16],
):
    """Sparse decode attention over the sliding window, and inverse RoPE, then the replicated grouped o_proj.

    The model path stops at `sparse_attn_swa_packed` and hands `o_packed` to the TP-by-group
    projection in `o_proj_tp`; this replicated form stays as the single-card unit
    entry below, whose subject is the attention, not the projection.
    """
    o_packed = pl.create_tensor([SWA.O_GROUPS * SWA.T, SWA.O_GROUP_IN], dtype=pl.BF16)
    rope_cos_il = pl.create_tensor([SWA.T, SWA.ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([SWA.T, SWA.ROPE_DIM], dtype=pl.FP32)
    rope_swap_idx = pl.create_tensor([SWA.H_TILE, SWA.ROPE_DIM], dtype=pl.INT32)
    SWA.prepare_swa_output_rope(freqs_cos, freqs_sin, rope_cos_il, rope_sin_signed, rope_swap_idx)
    merge_tids = pl.array.create(SWA.O_GROUPS, pl.TASK_ID)
    for g in pl.range(SWA.O_GROUPS):
        g_tid = SWA.sparse_attn_swa_packed(
            q,
            ori_kv,
            swa_indices,
            swa_lens,
            sparse_bias,
            attn_sink,
            freqs_cos,
            freqs_sin,
            o_packed,
            pl.cast(g, pl.INT32),
            rope_cos_il,
            rope_sin_signed,
            rope_swap_idx,
        )
        merge_tids[g] = g_tid
    merge_tid = pl.system.task_dummy(deps=[merge_tids])
    o_proj_grouped(o_packed, merge_tid, wo_a, wo_b, wo_b_scale, attn_out)
    return attn_out


@pl.jit.inline
def sparse_attn_hca(
    q: pl.Tensor[[HCA.T, HCA.H, HCA.HEAD_DIM], pl.BF16],
    ori_kv: pl.Tensor[[HCA.ORI_BLOCK_NUM_DYN, HCA.BLOCK_SIZE, 1, HCA.HEAD_DIM], pl.BF16],
    window_swa_indices: pl.Tensor[[HCA.T, HCA.WIN], pl.INT32],
    window_swa_lens: pl.Tensor[[HCA.T], pl.INT32],
    cmp_kv: pl.Tensor[[HCA.CMP_BLOCK_NUM_DYN, HCA.CMP_STORAGE_BLOCK_SIZE, 1, HCA.HEAD_DIM], pl.BF16],
    cmp_block_table: pl.Tensor[[HCA.B, HCA.CMP_MAX_BLOCKS], pl.INT32],
    cmp_sparse_indices: pl.Tensor[[HCA.T, HCA.CMP_TOPK], pl.INT32],
    attn_sink: pl.Tensor[[HCA.H], pl.FP32],
    freqs_cos: pl.Tensor[[HCA.T, HCA.ROPE_DIM], pl.BF16],
    freqs_sin: pl.Tensor[[HCA.T, HCA.ROPE_DIM], pl.BF16],
    wo_a: pl.Tensor[[HCA.O_GROUPS, HCA.O_LORA, HCA.O_GROUP_IN], pl.BF16],
    wo_b: pl.Tensor[[HCA.D, HCA.O_GROUPS * HCA.O_LORA], pl.INT8],
    wo_b_scale: pl.Tensor[[HCA.D], pl.FP32],
    attn_out: pl.Tensor[[HCA.T, HCA.D], pl.BF16],
):
    """Sparse decode attention over the compressed + window cache, and inverse RoPE, then the replicated grouped o_proj.

    The model path stops at `sparse_attn_hca_packed` and hands `o_packed` to the TP-by-group
    projection in `o_proj_tp`; this replicated form stays as the single-card unit
    entry below, whose subject is the attention, not the projection.
    """
    o_packed = pl.create_tensor([HCA.O_GROUPS * HCA.T, HCA.O_GROUP_IN], dtype=pl.BF16)
    rope_cos_il = pl.create_tensor([HCA.T, HCA.ROPE_DIM], dtype=pl.FP32)
    rope_sin_signed = pl.create_tensor([HCA.T, HCA.ROPE_DIM], dtype=pl.FP32)
    rope_swap_idx = pl.create_tensor([HCA.H_TILE, HCA.ROPE_DIM], dtype=pl.INT32)
    HCA.prepare_hca_output_rope(freqs_cos, freqs_sin, rope_cos_il, rope_sin_signed, rope_swap_idx)
    cmp_seq_lens = pl.create_tensor([HCA.T], dtype=pl.INT32)
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="hca_fixture_lengths", allow_early_resolve=True):
        for length_t in pl.range(HCA.T):
            length_count = pl.cast(0, pl.INT32)
            for length_k in pl.range(HCA.CMP_TOPK):
                if pl.read(cmp_sparse_indices, [length_t, length_k]) >= 0:
                    length_count = length_count + pl.cast(1, pl.INT32)
            pl.write(cmp_seq_lens, [length_t], length_count)
    merge_tids = pl.array.create(HCA.O_GROUPS, pl.TASK_ID)
    for g in pl.range(HCA.O_GROUPS):
        g_tid = HCA.sparse_attn_hca_packed(
            q,
            ori_kv,
            window_swa_indices,
            window_swa_lens,
            cmp_kv,
            cmp_block_table,
            cmp_seq_lens,
            attn_sink,
            freqs_cos,
            freqs_sin,
            o_packed,
            pl.cast(g, pl.INT32),
            rope_cos_il,
            rope_sin_signed,
            rope_swap_idx,
        )
        merge_tids[g] = g_tid
    merge_tid = pl.system.task_dummy(deps=[merge_tids])
    o_proj_grouped(o_packed, merge_tid, wo_a, wo_b, wo_b_scale, attn_out)
    return attn_out
