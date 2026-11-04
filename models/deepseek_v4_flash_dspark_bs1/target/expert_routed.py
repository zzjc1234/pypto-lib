# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from .config import FLASH as M, DECODE_BATCH, DECODE_SEQ, TP

B = DECODE_BATCH

S = DECODE_SEQ

T = B * S

D = M.hidden_size

TOPK = M.num_experts_per_tok

MOE_INTER = M.moe_intermediate_size // TP  # this rank's slice of the intermediate dim

SWIGLU_LIMIT = M.swiglu_limit

N_EXPERTS = M.n_routed_experts

RECV_TILE = 16

N_SLOTS = min(N_EXPERTS, T * TOPK)

RECV_MAX = RECV_TILE

IDX_PAD = 8  # INT32 tile width: a vector fill needs a 32-byte row

SHARED_EID = N_EXPERTS

N_BANK = N_EXPERTS + 1

SH_SLOT = N_SLOTS

N_SLOTS_B = N_SLOTS + 1

assert T <= RECV_MAX, f"the shared-expert slot holds all {T} tokens in one tile of {RECV_MAX} rows"
