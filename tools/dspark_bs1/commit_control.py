# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Original commit invariants with the proved dynamic-bank anchor domain [0, ...]."""
from dataclasses import dataclass

@dataclass(frozen=True)
class CommitControl:
    epoch: int
    status: int
    previous_anchor: int
    accepted: int
    advance: int
    actual_emitted: int
    next_anchor: int
    next_token: int
    last_sampled: int
    remaining_after: int
    ring_commit_complete: int
    feature_start: int
    feature_rows: int
    terminated: int
    reserved14: int = 0
    reserved15: int = 0

    @classmethod
    def parse(cls, values):
        if len(values) != 16:
            raise ValueError('Device control is exactly INT64[16]')
        if any(type(v) is not int for v in values):
            raise ValueError('Control values must be actual Python integers')
        value = cls(*values)
        value.validate()
        return value

    def validate(self):
        if any(type(v) is not int for v in vars(self).values()):
            raise ValueError('Control fields must be actual integers')
        if self.epoch <= 0 or self.status != 1 or self.ring_commit_complete != 1:
            raise ValueError('Only a completed, positive-epoch device commit is consumable')
        if not 0 <= self.accepted <= 5 or self.advance != self.accepted + 1:
            raise ValueError('Invalid K5 acceptance/advance')
        if self.previous_anchor < 0 or self.next_anchor != self.previous_anchor + self.advance:
            raise ValueError('Invalid committed anchor transition')
        if self.feature_start != self.previous_anchor + 1 or self.feature_rows != self.advance:
            raise ValueError('Ring commitment must cover exactly the verified accepted feature rows')
        if not 1 <= self.actual_emitted <= self.advance or self.remaining_after < 0:
            raise ValueError('Invalid actual token accounting')
        if self.terminated not in (0, 1) or bool(self.terminated) != (self.remaining_after == 0):
            raise ValueError('Termination must match the output budget')
        if not self.terminated and (self.actual_emitted != self.advance or self.next_token != self.last_sampled):
            raise ValueError('A continuing request must emit the complete accepted advance')
        if not all(0 <= v < 129280 for v in (self.next_token, self.last_sampled)):
            raise ValueError('Invalid sampled token')
        if self.reserved14 or self.reserved15:
            raise ValueError('Reserved control fields must remain zero')
        return self

