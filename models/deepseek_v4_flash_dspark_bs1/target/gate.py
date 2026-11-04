# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pypto.language as pl

from .config import FLASH as M, MOE_TOKENS, FP32_NEG_INF, INT8_SCALE_MAX, INT8_AMAX_EPS

T = MOE_TOKENS

D = M.hidden_size

NORM_EPS = M.rms_norm_eps

N_EXPERTS = M.n_routed_experts

TOPK = M.num_experts_per_tok

ROUTE_SCALE = M.routed_scaling_factor

VOCAB = M.vocab_size

N_HASH_LAYERS = M.num_hash_layers

T_TILE = 8

GATE_T_TILE = 1         # one token per routing block

ROUTE_ROW_PAD = 16     # 64-byte routing row stride

GATE_M_TILE = 16        # cube M-tile: matmul rows must be a multiple of 16 (fractal)

GATE_N_TILE = 16        # expert columns per gate spmd block

assert N_EXPERTS % GATE_N_TILE == 0

T_PAD = ((T + GATE_M_TILE - 1) // GATE_M_TILE) * GATE_M_TILE

ROW_PAD = 8

FFN_REDUCE_TILE = D // ROW_PAD

assert D % ROW_PAD == 0

GATE_D_TILE = 2048

assert D % GATE_D_TILE == 0, "gate K-loop must cover D"

QUANT_TILE = 256

SCORE_PAD = 256         # padded expert row for sort32 + mrgsort

TOPK_PAD = 8            # TOPK padded to 32B-aligned width

SORT_PAD = TOPK_PAD * 2 # (val, idx) interleaved slice width

assert TOPK <= TOPK_PAD

@pl.jit.inline
def gate(
    x_mixed: pl.Tensor[[T, D], pl.BF16],
    norm_w: pl.Tensor[[D], pl.BF16],
    gate_w: pl.Tensor[[N_EXPERTS, D], pl.FP32],
    gate_bias: pl.Tensor[[N_EXPERTS], pl.FP32],
    layer_id: pl.Scalar[pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    tid2eid: pl.Tensor[[VOCAB, TOPK], pl.INT32],
    input_ids: pl.Tensor[[T], pl.INT64],
    x_norm_i8: pl.Tensor[[T, D], pl.INT8],
    x_norm_scale: pl.Tensor[[T, 1], pl.FP32],
    indices: pl.Tensor[[T, ROUTE_ROW_PAD], pl.INT32],
    weights: pl.Tensor[[T, ROUTE_ROW_PAD], pl.FP32],
):
    # Routing outputs and x_norm_scale are valid only for active tokens.
    # Deferred RMSNorm (qwen3-style): store xg = x*gamma (NOT *inv_rms), because
    # the per-token positive scalar inv_rms factors out of everything downstream:
    #   - gate logits: inv_rms * (xg @ gate_w.T)  -> applied as a [16,1] row-scale
    #   - int8 quant : symmetric per-token quant of xg CANCELS inv_rms exactly;
    #                  inv_rms rides only x_norm_scale (= inv_rms * amax(xg)/127).
    # This lets sq_sum (needs raw x) and xg share ONE pass over x_mixed instead of
    # a sqsum pass followed by a separate normalize pass.
    xg_buf = pl.create_tensor([T_PAD, D], dtype=pl.FP32)
    inv_rms_buf = pl.create_tensor([T_PAD, 1], dtype=pl.FP32)
    # per-token int8 quant scale (= INT8_SCALE_MAX / amax(xg)), computed in ffn_norm
    # and consumed by x_norm_quant so quant skips its own amax pass.
    xn_scale_buf = pl.create_tensor([T_PAD, 1], dtype=pl.FP32)
    active_tokens = pl.cast(num_tokens, pl.INDEX)
    if active_tokens < 0:
        active_tokens = pl.cast(0, pl.INDEX)
    if active_tokens > T:
        active_tokens = pl.cast(T, pl.INDEX)
    active_gate_tiles = (active_tokens + GATE_M_TILE - 1) // GATE_M_TILE
    active_gate_tokens = active_gate_tiles * GATE_M_TILE
    if active_gate_tokens > T:
        active_gate_tokens = pl.cast(T, pl.INDEX)

    # One token per core with two-level full-row reductions.
    norm_w_2d = pl.reshape(norm_w, [1, D])
    if active_gate_tokens > 0:
        for tok in pl.spmd(active_gate_tokens, name_hint="ffn_norm", allow_early_resolve=True):
            rms_x = pl.cast(pl.tile.load(x_mixed, [tok, 0], [1, D]), pl.FP32)
            rms_w = pl.cast(pl.tile.load(norm_w_2d, [0, 0], [1, D]), pl.FP32)
            xg = pl.mul(rms_x, rms_w)
            pl.tile.store(xg, [tok, 0], xg_buf, shapes=[1, D])

            sq_rows = pl.reshape(pl.mul(rms_x, rms_x), [ROW_PAD, FFN_REDUCE_TILE])
            sq_partial_tmp = pl.create_tile([ROW_PAD, FFN_REDUCE_TILE], dtype=pl.FP32)
            sq_partial = pl.row_sum(sq_rows, sq_partial_tmp)
            sq_reduce = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
            sq_reduce[0:1, :] = pl.reshape(sq_partial, [1, ROW_PAD])
            sq_reduce = pl.set_validshape(sq_reduce, 1, ROW_PAD)
            sq_sum_tmp = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
            sq_sum = pl.row_sum(sq_reduce, sq_sum_tmp)
            sq_sum = pl.set_validshape(pl.reshape(sq_sum, [1, ROW_PAD]), 1, 1)
            inv_rms = pl.recip(pl.sqrt(pl.add(pl.mul(sq_sum, 1.0 / D), NORM_EPS)))
            pl.tile.store(inv_rms, [tok, 0], inv_rms_buf, shapes=[1, 1])

            xg_abs_rows = pl.reshape(pl.abs(xg), [ROW_PAD, FFN_REDUCE_TILE])
            amax_partial_tmp = pl.create_tile([ROW_PAD, FFN_REDUCE_TILE], dtype=pl.FP32)
            amax_partial = pl.row_max(xg_abs_rows, amax_partial_tmp)
            amax_reduce = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
            amax_reduce[0:1, :] = pl.reshape(amax_partial, [1, ROW_PAD])
            amax_reduce = pl.set_validshape(amax_reduce, 1, ROW_PAD)
            amax_tmp = pl.create_tile([ROW_PAD, ROW_PAD], dtype=pl.FP32)
            xg_amax = pl.row_max(amax_reduce, amax_tmp)
            xg_amax = pl.set_validshape(pl.reshape(xg_amax, [1, ROW_PAD]), 1, 1)
            xg_amax = pl.maximum(xg_amax, INT8_AMAX_EPS)
            # quant scale = INT8_SCALE_MAX / amax(xg); dequant scale rides inv_rms.
            scale_max = pl.tile.full([1, ROW_PAD], dtype=pl.FP32, value=INT8_SCALE_MAX)
            scale_max = pl.set_validshape(scale_max, 1, 1)
            xg_sq = pl.div(scale_max, xg_amax)
            xg_dequant_scale = pl.mul(xg_amax, 1.0 / INT8_SCALE_MAX)
            x_norm_dequant_scale = pl.mul(xg_dequant_scale, inv_rms)
            pl.tile.store(x_norm_dequant_scale, [tok, 0], x_norm_scale, shapes=[1, 1])
            pl.tile.store(xg_sq, [tok, 0], xn_scale_buf, shapes=[1, 1])

    seed_dummy = pl.system.task_dummy(deps=[])

    # Per-token symmetric INT8 quant of xg: scale precomputed in ffn_norm. inv_rms
    # cancels here (symmetric quant is invariant to a positive per-token scalar),
    # so x_norm_i8 = quant(xg). Early resolution lets the shared-expert chain
    # pre-stage before dispatch_push.
    for t0 in pl.parallel(0, active_gate_tokens, T_TILE):
        with pl.at(
            level=pl.Level.CORE_GROUP,
            name_hint="x_norm_quant",
            deps=[seed_dummy],
            allow_early_resolve=True,
        ):
            quant_rows = pl.min(T_TILE, active_gate_tokens - t0)
            xn_sq_col = pl.slice(xn_scale_buf, [T_TILE, 1], [t0, 0], valid_shape=[quant_rows, 1])
            for xq_b_k in pl.pipeline(0, D, QUANT_TILE, stage=2):
                xn_input = pl.slice(xg_buf, [T_TILE, QUANT_TILE], [t0, xq_b_k], valid_shape=[quant_rows, QUANT_TILE])
                xn_q_scaled = pl.row_expand_mul(xn_input, xn_sq_col)
                xn_q_i32 = pl.cast(xn_q_scaled, pl.INT32, mode="rint")
                xn_q_half = pl.cast(xn_q_i32, pl.FP16, mode="round")
                xn_q_i8 = pl.cast(xn_q_half, pl.INT8, mode="trunc", saturation_mode="on")
                xn_q_valid = pl.set_validshape(xn_q_i8, quant_rows, QUANT_TILE)
                x_norm_i8[t0 : t0 + T_TILE, xq_b_k : xq_b_k + QUANT_TILE] = xn_q_valid

    gate_logits_buf = pl.create_tensor([T_PAD, SCORE_PAD], dtype=pl.FP32)
    GATE_N_BLOCKS = N_EXPERTS // GATE_N_TILE
    if active_gate_tiles > 0:
        for gb_idx in pl.spmd(active_gate_tiles * GATE_N_BLOCKS, name_hint="gate_mm", allow_early_resolve=True):
            tg = gb_idx // GATE_N_BLOCKS
            nb = gb_idx % GATE_N_BLOCKS
            t1 = tg * GATE_M_TILE
            n0 = nb * GATE_N_TILE
            gate_logits_tile = pl.create_tensor([GATE_M_TILE, GATE_N_TILE], dtype=pl.FP32)
            for kb in pl.pipeline(0, D // GATE_D_TILE, stage=2):
                gd_kd = kb * GATE_D_TILE
                gd_x = xg_buf[t1 : t1 + GATE_M_TILE, gd_kd : gd_kd + GATE_D_TILE]
                gd_w = gate_w[n0 : n0 + GATE_N_TILE, gd_kd : gd_kd + GATE_D_TILE]
                gate_logits_tile = pl.matmul_acc(gate_logits_tile, gd_x, gd_w, b_trans=True, init_cond=(kb == 0))
            gate_logits_buf[t1 : t1 + GATE_M_TILE, n0 : n0 + GATE_N_TILE] = gate_logits_tile

    active_route_tiles = (active_tokens + GATE_T_TILE - 1) // GATE_T_TILE
    # Hash layers index via tid2eid[input_ids]; score layers sort+gather.
    if active_route_tiles > 0:
        if layer_id < N_HASH_LAYERS:
            for th_idx in pl.spmd(active_route_tiles, name_hint="route_hash", allow_early_resolve=True):
                t1 = th_idx * GATE_T_TILE
                route_logits = gate_logits_buf[t1 : t1 + GATE_T_TILE, 0:N_EXPERTS]
                route_inv_rms = pl.read(inv_rms_buf, [t1, 0])
                route_logits = pl.mul(route_logits, route_inv_rms)
                gp_relu = pl.maximum(route_logits, 0.0)
                gp_abs = pl.abs(route_logits)
                gp_softplus_log = pl.add(gp_relu, pl.log(pl.add(pl.exp(pl.neg(gp_abs)), 1.0)))
                gp_neg_floor_mask = pl.minimum(pl.maximum(pl.sub(pl.neg(route_logits), 10.0), 0.0), 1.0)
                gp_neg_floor = pl.mul(gp_neg_floor_mask, pl.exp(pl.minimum(route_logits, 0.0)))
                gp_softplus = pl.maximum(gp_softplus_log, gp_neg_floor)
                gp_score = pl.sqrt(gp_softplus)
                # Load the selected expert row into an aligned UB tile.
                hs_token = pl.cast(pl.read(input_ids, [t1]), pl.INDEX)
                hs_idx_tile = pl.slice(tid2eid, [GATE_T_TILE, TOPK_PAD], [hs_token, 0], valid_shape=[GATE_T_TILE, TOPK])
                # Normalize only the selected scores.
                gather_all = pl.gather(gp_score, dim=-1, index=hs_idx_tile)
                hs_reduce = pl.create_tensor([ROW_PAD, TOPK_PAD], dtype=pl.FP32)
                hs_reduce[0:GATE_T_TILE, :] = gather_all
                hs_reduce = pl.set_validshape(hs_reduce, GATE_T_TILE, TOPK)
                hs_denom = pl.row_sum(hs_reduce)
                hs_weight_mat = pl.mul(pl.row_expand_div(hs_reduce, hs_denom), ROUTE_SCALE)
                hs_indices_valid = pl.set_validshape(hs_idx_tile, GATE_T_TILE, TOPK)
                indices = pl.assemble(indices, hs_indices_valid, [t1, 0])
                weights = pl.assemble(weights, hs_weight_mat, [t1, 0])
        else:
            for ts_idx in pl.spmd(active_route_tiles, name_hint="route_sort", allow_early_resolve=True):
                t1 = ts_idx * GATE_T_TILE
                route_logits = gate_logits_buf[t1 : t1 + GATE_T_TILE, 0:N_EXPERTS]
                route_inv_rms = pl.read(inv_rms_buf, [t1, 0])
                route_logits = pl.mul(route_logits, route_inv_rms)
                gp_relu = pl.maximum(route_logits, 0.0)
                gp_abs = pl.abs(route_logits)
                gp_softplus_log = pl.add(gp_relu, pl.log(pl.add(pl.exp(pl.neg(gp_abs)), 1.0)))
                gp_neg_floor_mask = pl.minimum(pl.maximum(pl.sub(pl.neg(route_logits), 10.0), 0.0), 1.0)
                gp_neg_floor = pl.mul(gp_neg_floor_mask, pl.exp(pl.minimum(route_logits, 0.0)))
                gp_softplus = pl.maximum(gp_softplus_log, gp_neg_floor)
                gp_score = pl.sqrt(gp_softplus)
                gp_bias_row = pl.reshape(gate_bias, [1, N_EXPERTS])
                gp_biased = pl.create_tensor([GATE_T_TILE, SCORE_PAD], dtype=pl.FP32)
                gp_biased[:, :] = pl.full([GATE_T_TILE, SCORE_PAD], dtype=pl.FP32, value=FP32_NEG_INF)
                gp_biased[:, 0:N_EXPERTS] = pl.add(gp_score, gp_bias_row)
                # topk_idx_tile stays Tensor (created here, not a pl.full Tile) so
                # the batched pl.gather below accepts it — Tile-against-Tensor src
                # is rejected.
                topk_idx_tile = pl.create_tensor([GATE_T_TILE, TOPK_PAD], dtype=pl.INT32)
                # ptoas pto.tmrgsort requires src rows == 1; sort path iterates
                # row-by-row. sort32: [1,256]→[1,512] (8 runs of 64). mrgsort
                # format1 4-way: 8→2 runs of 256. format2 2-way: 2→1 run of 512.
                for sr_tt in pl.range(GATE_T_TILE):
                    sr_row = gp_biased[sr_tt : sr_tt + 1, :]
                    sr_idx_init = pl.arange(0, [1, SCORE_PAD], dtype=pl.UINT32)
                    sr_sorted = pl.sort32(sr_row, sr_idx_init)
                    sr_sorted = pl.mrgsort(sr_sorted, block_len=64)
                    sr_sorted = pl.mrgsort(sr_sorted[:, 0:256], sr_sorted[:, 256:512])
                    sr_pairs = sr_sorted[:, 0:SORT_PAD]
                    sr_i = pl.gather(sr_pairs, mask_pattern=pl.tile.MaskPattern.P1010, output_dtype=pl.INT32)
                    topk_idx_tile[sr_tt : sr_tt + 1, :] = sr_i
                # Normalize only the selected scores.
                gather_all = pl.gather(gp_score, dim=-1, index=topk_idx_tile)
                nm_reduce = pl.create_tensor([ROW_PAD, TOPK_PAD], dtype=pl.FP32)
                nm_reduce[0:GATE_T_TILE, :] = gather_all
                nm_reduce = pl.set_validshape(nm_reduce, GATE_T_TILE, TOPK)
                nm_denom = pl.row_sum(nm_reduce)
                nm_weight_mat = pl.mul(pl.row_expand_div(nm_reduce, nm_denom), ROUTE_SCALE)
                nm_indices_valid = pl.set_validshape(topk_idx_tile, GATE_T_TILE, TOPK)
                indices = pl.assemble(indices, nm_indices_valid, [t1, 0])
                weights = pl.assemble(weights, nm_weight_mat, [t1, 0])

    return weights
