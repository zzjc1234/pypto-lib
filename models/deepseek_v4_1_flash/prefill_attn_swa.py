# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Packed-prefill sliding-window attention for encoder layers 0 and 1."""


import pypto.language as pl
import pypto.language.distributed as pld
import torch

from utils import AttentionGoldenResult, run_swa_validation
from o_proj import (
    make_prefill_tp_output_all_reduce,
    prefill_tp_output_all_reduce as legacy_prefill_tp_output_all_reduce,
)
from config import (
    PREFILL_MAX_TOKENS as LEGACY_PREFILL_MAX_TOKENS,
    WINDOW_CACHE_GROUP,
    FLASH as M,
    ORI_BLOCKS_DYN,
    T_DYN,
    TP_SIZE,
)

from sparse_attention import (
    SWA_RUN_DESCRIPTOR_WIDTH as RUN_DESCRIPTOR_WIDTH,
    SWA_RUN_PARTS as RUN_PARTS,
    analyze_index_runs,
    attend_shared_cache,
    decode_shared_cache_after_publish,
    decode_shared_cache_e8m0_after_publish,
    golden_swa_attention,
    make_swa_program,
)
from o_proj import prefill_o_proj
from qkv_proj_rope import (
    prefill_kv_proj_rope,
    prefill_q_proj_qr,
    prefill_q_proj_rope,
)


# model config
LOCAL_H = M.num_attention_heads // TP_SIZE
LOCAL_O_GROUPS = M.o_groups // TP_SIZE
LOCAL_O_WIDTH = M.o_groups // TP_SIZE * M.o_lora_rank
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank
O_LORA = M.o_lora_rank
O_GROUP_IN = M.num_attention_heads * M.head_dim // M.o_groups


# tiling
QUERY_TILE = 128
WORKER_TILE = 64
E8M0_PUBLISH_WORKER_TILE = 64

# Communication capacity.
PREFILL_MAX_TOKENS = 8192

prefill_tp_output_all_reduce = make_prefill_tp_output_all_reduce(PREFILL_MAX_TOKENS)


@pl.jit.inline(auto_scope=False)
def prefill_publish_window_e8m0(
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.FP8E8M0],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    """Publish E4M3 payloads and E8M0 scales and return completion."""
    cache_rows = pl.tensor.dim(cache, 0) * 128
    flat = pl.reshape(cache, [cache_rows, HEAD_DIM])
    scale_flat = pl.reshape(scales, [cache_rows, HEAD_DIM // 32])
    with pl.spmd(E8M0_PUBLISH_WORKER_TILE, name_hint="prefill_swa_e8m0_cache_publish", deps=[cache_ready]) as publish_ready:
        worker = pl.tile.get_block_idx()
        for token in pl.range(worker, num_tokens, E8M0_PUBLISH_WORKER_TILE):
            slot_i64 = pl.read(slots, [token])
            if slot_i64 >= 0:
                slot = pl.cast(slot_i64, pl.INDEX)
                source = pl.slice(kv, [1, HEAD_DIM * 2], [token, 0], valid_shape=[1, HEAD_DIM])
                source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 1, HEAD_DIM * 2)
                value = pl.reshape(pl.cast(source, pl.FP32), [HEAD_DIM // 16, 32])
                amax = pl.maximum(pl.row_max(pl.abs(value)), 1e-4)
                raw = pl.mul(amax, 1.0 / 448.0)
                bits = pl.reinterpret_view(raw, pl.INT32)
                exponent = pl.shrs(pl.add(bits, 8388607), 23)
                scale = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
                payload = pl.cast(pl.row_expand_div(value, scale), pl.FP8E4M3FN, mode="rint")
                flat[slot:slot + 1, :] = pl.set_validshape(pl.reshape(payload, [1, HEAD_DIM * 2]), 1, HEAD_DIM)
                signed_exponent = pl.sub(exponent, pl.mul(pl.shrs(exponent, 7), 256))
                codes = pl.cast(signed_exponent, pl.INT8)
                encoded = pl.reinterpret_view(pl.reinterpret_view(codes, pl.UINT8), pl.FP8E8M0)
                encoded_row = pl.reshape(encoded, [1, HEAD_DIM // 16])
                scale_flat[slot:slot + 1, :] = pl.set_validshape(encoded_row, 1, HEAD_DIM // 32)
    return publish_ready



@pl.jit.inline
def prefill_publish_window(
    kv: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    slots: pl.Tensor[[T_DYN], pl.INT64],
    cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
    scales: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // 32], pl.BF16],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    blocks = pl.tensor.dim(cache, 0)
    cache_rows = blocks * 128
    flat = pl.reshape(cache, [cache_rows, HEAD_DIM])
    scale_flat = pl.reshape(scales, [cache_rows, HEAD_DIM // 32])
    with pl.spmd(WORKER_TILE, name_hint="prefill_swa_cache_publish", deps=[cache_ready]) as publish_tid:
        worker = pl.tile.get_block_idx()
        for t in pl.range(worker, num_tokens, WORKER_TILE):
            slot_i64 = pl.read(slots, [t])
            if slot_i64 >= 0:
                slot = pl.cast(slot_i64, pl.INDEX)
                source = pl.slice(kv, [1, HEAD_DIM * 2], [t, 0], valid_shape=[1, HEAD_DIM])
                source = pl.set_validshape(pl.fillpad(source, pad_value=pl.PadValue.zero), 1, HEAD_DIM * 2)
                value = pl.reshape(pl.cast(source, pl.FP32), [HEAD_DIM // 16, 32])
                amax = pl.maximum(pl.row_max(pl.abs(value)), 1e-4)
                raw = pl.mul(amax, 1.0 / 448.0)
                bits = pl.reinterpret_view(raw, pl.INT32)
                exponent = pl.shrs(pl.add(bits, 8388607), 23)
                scale = pl.reinterpret_view(pl.shls(exponent, 23), pl.FP32)
                payload = pl.cast(pl.row_expand_div(value, scale), pl.FP8E4M3FN, mode="rint")
                flat[slot:slot + 1, :] = pl.set_validshape(pl.reshape(payload, [1, HEAD_DIM * 2]), 1, HEAD_DIM)
                encoded = pl.cast(scale, pl.BF16, mode="rint")
                encoded_row = pl.reshape(encoded, [1, HEAD_DIM // 16])
                scale_flat[slot:slot + 1, :] = pl.set_validshape(encoded_row, 1, HEAD_DIM // 32)
    return publish_tid






def golden_prefill_attn_swa(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    attn_sink: torch.Tensor,
    wo_a: torch.Tensor,
    wo_b: torch.Tensor,
    wo_b_scale: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    window_slots: torch.Tensor,
    window_indices: torch.Tensor,
    window_cache: torch.Tensor,
    window_cache_scale: torch.Tensor,
) -> AttentionGoldenResult:
    return golden_swa_attention(
        x, wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
        wkv, wkv_scale, kv_norm_weight, attn_sink,
        wo_a, wo_b, wo_b_scale, rope_cos, rope_sin,
        window_slots, window_indices, window_cache, window_cache_scale,
    )


def make_prefill_attn_swa_partial(
    scale_dtype=pl.BF16, publish_fn=prefill_publish_window, decode_fn=decode_shared_cache_after_publish,
):
    """Specialize cache publication and decoding; retain the shared query pipeline."""
    @pl.jit.inline(auto_scope=False)
    def prefill_attn_swa_partial(
        x: pl.Tensor[[T_DYN, D], pl.BF16],
        wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[T_DYN], pl.INT64],
        window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
        window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
        window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], scale_dtype],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        partial: pl.Tensor[[T_DYN, D], pl.FP32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Write FP32 head-TP partials for packed causal SWA; write slots must be unique."""
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="swa_prefill_previous_epoch", allow_early_resolve=False) as cache_ready:
            for peer in pl.range(TP_SIZE):
                previous_epoch = (attention_epoch - 1) * 2
                pld.system.wait(output_arrived, offsets=[peer, 0], expected=previous_epoch, cmp=pld.WaitCmp.Ge)
        tokens = pl.tensor.dim(x, 0)
        kv_projection = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        # Gate every projection tile before scratch storage is reused by a new epoch.
        with pl.spmd(WORKER_TILE, name_hint="prefill_swa_begin", deps=[cache_ready]):
            worker = pl.tile.get_block_idx()
            for row in pl.range(worker, num_tokens, WORKER_TILE):
                kv_projection[row : row + 1, :] = pl.full([1, HEAD_DIM], dtype=pl.BF16, value=0.0)
        kv = pl.create_tensor([tokens, HEAD_DIM], dtype=pl.BF16)
        prefill_kv_proj_rope(x, wkv, wkv_scale, kv_norm_weight, rope_cos, rope_sin, kv_projection, kv, num_tokens)
        chunk_done = publish_fn(kv, window_slots, window_cache, window_cache_scale, num_tokens, cache_ready)
        cache_blocks = pl.tensor.dim(window_cache, 0)
        cache_rows = cache_blocks * 128
        decoded_cache = pl.create_tensor([cache_blocks, 128, 1, HEAD_DIM], dtype=pl.BF16)
        zero_row = pl.create_tensor([32, HEAD_DIM], dtype=pl.BF16)
        # Decode once for the entire packed prompt. The explicit publication
        # dependency protects the hoisted reader independently of query staging.
        decoded_ready = decode_fn(window_cache, window_cache_scale, decoded_cache, zero_row, chunk_done)

        chunk_x = pl.create_tensor([QUERY_TILE, D], dtype=pl.BF16)
        chunk_cos = pl.create_tensor([QUERY_TILE, ROPE_DIM // 2], dtype=pl.FP32)
        chunk_sin = pl.create_tensor([QUERY_TILE, ROPE_DIM // 2], dtype=pl.FP32)
        chunk_indices = pl.create_tensor([QUERY_TILE, 128], dtype=pl.INT32)
        qr = pl.create_tensor([QUERY_TILE, Q_LORA], dtype=pl.BF16)
        q = pl.create_tensor([QUERY_TILE, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
        descriptors = pl.create_tensor([QUERY_TILE, RUN_PARTS, RUN_DESCRIPTOR_WIDTH], dtype=pl.INT32)
        attended = pl.create_tensor([QUERY_TILE, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
        chunk_partial = pl.create_tensor([QUERY_TILE, D], dtype=pl.FP32)

        for start in pl.range(0, num_tokens, QUERY_TILE):
            active = pl.min(QUERY_TILE, num_tokens - start)
            # The prior chunk must finish reading every reusable scratch buffer.
            with pl.spmd(WORKER_TILE, name_hint="prefill_swa_stage", deps=[chunk_done]) as stage_tid:
                worker = pl.tile.get_block_idx()
                for row in pl.range(worker, active, WORKER_TILE):
                    source_row = start + row
                    for col in pl.range(0, D, 512):
                        chunk_x[row:row + 1, col:col + 512] = x[source_row:source_row + 1, col:col + 512]
                    chunk_cos[row:row + 1, :] = rope_cos[source_row:source_row + 1, :]
                    chunk_sin[row:row + 1, :] = rope_sin[source_row:source_row + 1, :]
                    chunk_indices[row:row + 1, :] = window_indices[source_row:source_row + 1, :]
            prefill_q_proj_qr(chunk_x, wq_a, wq_a_scale, q_norm_weight, qr, active)
            prefill_q_proj_rope(qr, wq_b, wq_b_scale, chunk_cos, chunk_sin, q, active)
            index_ready = analyze_index_runs(chunk_indices, descriptors, active, cache_rows)
            attend_shared_cache(
                q, decoded_cache, zero_row,
                descriptors, chunk_indices, attn_sink,
                attended, active, decoded_ready, index_ready,
            )
            prefill_o_proj(attended, wo_a, wo_b, wo_b_scale, chunk_cos, chunk_sin, chunk_partial, active)
            with pl.spmd(WORKER_TILE, name_hint="prefill_swa_collect") as collect_tid:
                worker = pl.tile.get_block_idx()
                for row in pl.range(worker, active, WORKER_TILE):
                    for col in pl.range(0, D, 512):
                        partial[start + row:start + row + 1, col:col + 512] = chunk_partial[row:row + 1, col:col + 512]
            chunk_done = collect_tid
        return partial

    return prefill_attn_swa_partial


prefill_attn_swa_partial = make_prefill_attn_swa_partial()
prefill_attn_swa_partial_e8m0 = make_prefill_attn_swa_partial(
    pl.FP8E8M0, prefill_publish_window_e8m0, decode_shared_cache_e8m0_after_publish,
)



def make_prefill_attn_swa(
    scale_dtype=pl.BF16, partial_fn=prefill_attn_swa_partial,
    capacity=PREFILL_MAX_TOKENS, reducer=prefill_tp_output_all_reduce,
):
    """Keep standalone SWA and legacy mixed-mode transport capacities distinct."""
    @pl.jit.inline(auto_scope=False)
    def prefill_attn_swa(
        x: pl.Tensor[[T_DYN, D], pl.BF16],
        wq_a: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
        wq_a_scale: pl.Tensor[[D // 32, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
        q_norm_weight: pl.Tensor[[Q_LORA], pl.BF16],
        wq_b: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
        wq_b_scale: pl.Tensor[[Q_LORA // 32, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
        wkv_scale: pl.Tensor[[D // 32, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
        kv_norm_weight: pl.Tensor[[HEAD_DIM], pl.BF16],
        attn_sink: pl.Tensor[[LOCAL_H], pl.FP32],
        wo_a: pl.Tensor[[LOCAL_O_GROUPS, O_LORA, O_GROUP_IN], pl.BF16],
        wo_b: pl.Tensor[[LOCAL_O_WIDTH, D], pl.FP8E4M3FN],
        wo_b_scale: pl.Tensor[[LOCAL_O_WIDTH // 32, D], pl.FP8E8M0, pl.MX_B_NN],
        rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
        window_slots: pl.Tensor[[T_DYN], pl.INT64],
        window_indices: pl.Tensor[[T_DYN, 128], pl.INT32],
        window_cache: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM], pl.FP8E4M3FN],
        window_cache_scale: pl.Tensor[[ORI_BLOCKS_DYN, 128, 1, HEAD_DIM // WINDOW_CACHE_GROUP], scale_dtype],
        output_window: pld.DistributedTensor[[capacity, D], pl.FP32],
        output_arrived: pld.DistributedTensor[[TP_SIZE, 1], pl.INT32],
        output: pl.Tensor[[T_DYN, D], pl.BF16],
        group_base: pl.Scalar[pl.INT32],
        tp_rank: pl.Scalar[pl.INT32],
        num_tokens: pl.Scalar[pl.INT32],
        attention_epoch: pl.Scalar[pl.INT32],
    ):
        """Return replicated SWA output for the standalone attention interface."""
        tokens = pl.tensor.dim(x, 0)
        partial = pl.create_tensor([tokens, D], dtype=pl.FP32)
        partial_fn(
            x,
            wq_a, wq_a_scale, q_norm_weight, wq_b, wq_b_scale,
            wkv, wkv_scale, kv_norm_weight, attn_sink,
            wo_a, wo_b, wo_b_scale,
            rope_cos, rope_sin,
            window_slots, window_indices, window_cache, window_cache_scale,
            output_arrived, partial, num_tokens, attention_epoch,
        )
        reducer(partial, output_window, output_arrived, output, group_base, tp_rank, num_tokens, attention_epoch)
        return output

    return prefill_attn_swa


prefill_attn_swa = make_prefill_attn_swa()
prefill_attn_swa_e8m0 = make_prefill_attn_swa(
    pl.FP8E8M0, prefill_attn_swa_partial_e8m0, LEGACY_PREFILL_MAX_TOKENS, legacy_prefill_tp_output_all_reduce,
)



__all__ = [
    "golden_prefill_attn_swa", "prefill_attn_swa", "prefill_attn_swa_partial",
    "prefill_attn_swa_e8m0", "prefill_attn_swa_partial_e8m0",
]


def validate(argv=None):
    """Validate the Prefill SWA leaf operator on A5."""
    return run_swa_validation(
        prefill_attn_swa, 'prefill', argv,
        capacity=PREFILL_MAX_TOKENS, default_tokens=257, default_requests=4,
        cases=('mixed', 'prefix', 'shuffle', 'masked', 'zero', 'sink'),
        program_builder=make_swa_program,
    )


def main():
    """Run local validation and return a failing exit status on precision errors."""
    result = validate()
    if not result.passed:
        raise SystemExit(result.error or 1)


if __name__ == "__main__":
    main()
