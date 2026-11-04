# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

def cpu_commit(draft_ids, sampled_ids, features, request_state, ring):
    """Exact greedy prefix/bonus and post-commit terminal cap from run_loop.py."""
    import torch
    epoch, anchor, remaining, previous_last = request_state.tolist()
    proposal = draft_ids.tolist()
    sampled = sampled_ids[:, 0].tolist()
    accepted = 0
    for actual, expected in zip(proposal, sampled):
        if actual != expected:
            break
        accepted += 1
    algorithmic = [*proposal[:accepted], sampled[accepted]]
    advance = accepted + 1
    actual = algorithmic[:remaining]
    # Commit all algorithmic input features even on the truncated final round.
    for offset in range(advance):
        ring[(anchor + 1 + offset) % 128].copy_(features[offset])
    control = torch.tensor([epoch, 1, anchor, accepted, advance, len(actual),
        anchor + advance, sampled[accepted], actual[-1] if actual else previous_last,
        remaining - len(actual), 1, anchor + 1, advance, int(remaining == len(actual)), 0, 0], dtype=torch.int64)
    emitted = torch.full((6,), -1, dtype=torch.int64)
    emitted[:len(actual)] = torch.tensor(actual, dtype=torch.int64)
    return control, emitted

