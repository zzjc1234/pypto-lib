# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

def check_packets(initial_state, packets, *, minimum_two=12):
    """Validate real control/emitted packets read after both device cycles."""
    if len(initial_state) != 4 or any(type(v) is not int for v in initial_state):
        raise ValueError("Initial state must be four real INT64 values")
    if len(packets) not in (1, 2):
        raise ValueError("Each submitted batch must contain one or two real cycles")
    from .commit_control import CommitControl
    if minimum_two not in (7, 12):
        raise ValueError('Only proven two-cycle thresholds7 or12 are allowed')
    state = list(initial_state)
    if state[2] <= 0:
        raise ValueError("Do not submit any cycle after the request terminates")
    if len(packets) == 2 and state[2] < minimum_two:
        raise ValueError("Two-cycle batch violates the explicitly selected proven minimum remaining budget")
    emitted = []
    advances = []
    states = [list(state)]
    for packet in packets:
        if state[2] == 0:
            raise ValueError("A second device cycle ran after request termination")
        control = CommitControl.parse(packet["control"])
        values = packet["emitted"]
        if len(values) != 6 or any(type(v) is not int for v in values):
            raise ValueError("Each emitted buffer is exactly INT64[6]")
        if control.epoch != state[0] or control.previous_anchor != state[1]:
            raise ValueError("Cycle did not consume its actual predecessor state")
        if control.actual_emitted != min(control.advance, state[2]):
            raise ValueError("Actual output count disagrees with the remaining budget")
        if control.remaining_after != state[2] - control.actual_emitted:
            raise ValueError("Remaining output budget is inconsistent")
        active = values[:control.actual_emitted]
        if any(not 0 <= v < 129280 for v in active):
            raise ValueError("Invalid actual token ID")
        if any(v != -1 for v in values[control.actual_emitted:]):
            raise ValueError("Unused output slots must preserve the original -1 contract")
        if not active or active[-1] != control.last_sampled:
            raise ValueError("Last emitted token differs from the next anchor token")
        emitted.extend(active)
        advances.append(control.advance)
        state = [control.epoch + 1, control.next_anchor,
                 control.remaining_after, control.last_sampled]
        states.append(list(state))
    return {"actual_tokens": len(emitted), "outputs": emitted,
            "advances": advances, "states": states,
            "final_state": state, "whole_model_qualified": False}


def check_continuous_chain(initial_state, packet_batches, expected_new_tokens, *, minimum_two=12):
    """Count actual tokens and validate every state transition of a full request."""
    state = list(initial_state)
    outputs = []
    advances = []
    widths = []
    for packets in packet_batches:
        result = check_packets(state, packets, minimum_two=minimum_two)
        state = result["final_state"]
        outputs.extend(result["outputs"])
        advances.extend(result["advances"])
        widths.append(len(packets))
    if len(outputs) != expected_new_tokens or state[2] != 0:
        raise ValueError("Continuous request did not emit its actual complete output budget")
    return {"actual_tokens": len(outputs), "outputs": outputs,
            "advances": advances, "batch_cycle_counts": widths,
            "final_state": state, "whole_model_qualified": False}


def check_commit_oracle(initial_state, packets, captures, initial_ring, actual_ring):
    """Compare both commits and the entire final ring with the existing oracle.

    Each capture must contain this execution's real proposal, sampled IDs and
    target features. Independent output buffers keep cycle zero alive until
    cycle one completes; no CPU read between cycles is needed.
    """
    import torch

    checked = check_packets(initial_state, packets)
    if len(captures) != len(packets):
        raise ValueError("Need every actual cycle's functional capture")
    from .commit_oracle import cpu_commit
    expected_ring = initial_ring.clone()
    controls = []
    emitted = []
    for index, (packet, capture) in enumerate(zip(packets, captures, strict=True)):
        proposal = capture["proposal"]
        if not isinstance(proposal, torch.Tensor):
            proposal = torch.tensor(proposal, dtype=torch.int32)
        expected_control, expected_emitted = cpu_commit(
            proposal, capture["sampled_ids"], capture["target_features"],
            torch.tensor(checked["states"][index], dtype=torch.int64), expected_ring)
        controls.append(packet["control"] == expected_control.tolist())
        emitted.append(packet["emitted"] == expected_emitted.tolist())
    ring_equal = torch.equal(actual_ring.view(torch.uint8),
                             expected_ring.view(torch.uint8))
    checks = {"control_exact": controls, "emitted_exact": emitted,
              "all136_history_rows_bitwise": bool(ring_equal)}
    if not all(controls) or not all(emitted) or not ring_equal:
        raise RuntimeError(f"Two-cycle device commits fail the original oracle: {checks}")
    return checks

