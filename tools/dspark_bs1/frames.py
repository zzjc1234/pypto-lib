# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Two real HOST packet frames with caller-owned, validated device roots."""

from dataclasses import dataclass

import torch

from .tensors import base_name, matches

HOST_OUTPUTS = {'s_control': (143, (8, 2, 16)), 's_emitted': (144, (8, 2, 6))}


def allocate_host_frames():
    """Call before Worker construction so both output frames predate its fork."""
    return tuple({name: torch.full(shape, -777, dtype=torch.int64).share_memory_()
                  for name, (_, shape) in HOST_OUTPUTS.items()} for _ in range(2))


@dataclass
class Frame:
    arguments: tuple
    hosts: dict
    handle: object = None
    cycles: int = 0
    published: bool = False
    packet: object = None
    consumed: bool = False


class PublishedHandle:
    def __init__(self, handle, frame):
        self.handle, self.frame = handle, frame
        self.value = None

    @property
    def done(self):
        return self.handle.done

    def result(self, *args, **kwargs):
        if not self.frame.published:
            self.value = self.handle.result(*args, **kwargs)
            self.frame.published = True
        return self.value


class NativeFrames:
    def __init__(self, worker, programs, metadata, common, outputs, hosts, config):
        if set(programs) != {1, 2} or metadata[1]['params'] != metadata[2]['params']:
            raise ValueError('Actual one/two-cycle programs must have the same public ABI')
        params = metadata[1]['params']
        if len(params) != 155 or len(outputs) != 2 or len(hosts) != 2:
            raise ValueError('Public155 and exactly two output frames are required')
        self.worker, self.programs, self.config = worker, programs, config
        self.frames = []
        for frame in range(2):
            arguments = []
            for index, row in enumerate(params):
                name = base_name(row)
                if name in HOST_OUTPUTS:
                    slot, shape = HOST_OUTPUTS[name]
                    value = hosts[frame][name]
                    if index != slot or row['direction'] != 'Out' or tuple(row['shape']) != shape:
                        raise ValueError('The control/emitted HOST output contract changed')
                    if value.dtype != torch.int64 or not value.is_shared() or not value.is_contiguous():
                        raise ValueError('HOST outputs require exact prefork shared INT64 roots')
                else:
                    value = outputs[frame][name] if row['direction'] == 'Out' else common[name]
                if not matches(tuple(value.shape), row['shape']):
                    raise ValueError('Actual argument shape differs from the model: ' + name)
                arguments.append(value)
            for program in programs.values():
                infos = worker._states[program]['param_infos']
                if len(infos) != len(arguments):
                    raise ValueError('Registered public arity changed')
                for info, value in zip(infos, arguments, strict=True):
                    worker._validate_prepared_dispatch_arg(info, value)
            self.frames.append(Frame(tuple(arguments), hosts[frame]))
        seen = set()
        for index, row in enumerate(params):
            values = [frame.arguments[index] for frame in self.frames]
            if row['direction'] != 'Out':
                if values[0] is not values[1]:
                    raise ValueError('Persistent/static inputs must be the same actual objects')
                continue
            for value in values:
                for shard in getattr(value, 'shards', ()):
                    owner = getattr(shard, 'buffer', None)
                    if owner is None or id(owner) in seen:
                        raise ValueError('DEVICE frame outputs require distinct owning Buffers')
                    seen.add(id(owner))

    def _frame(self, frame):
        if type(frame) is not int or frame not in (0, 1):
            raise ValueError('Frame index must be zero or one')
        return self.frames[frame]

    def submit(self, frame, cycles):
        state = self._frame(frame)
        if type(cycles) is not int or cycles not in self.programs or state.handle is not None:
            raise ValueError('A consumed frame and an actual one/two-cycle program are required')
        handle = self.worker.submit(self.programs[cycles], *state.arguments, config=self.config)
        state.cycles = cycles
        state.handle = PublishedHandle(handle, state)
        return state.handle

    def capture(self, frame, cycles):
        state = self._frame(frame)
        if not state.published or state.handle is None or state.cycles != cycles:
            raise RuntimeError('Successful whole-parent result is required before HOST capture')
        if state.packet is None:
            state.packet = {'cycles': cycles,
                            'control': state.hosts['s_control'][:, :cycles].tolist(),
                            'emitted': state.hosts['s_emitted'][:, :cycles].tolist()}
        return state.packet

    def consume(self, frame):
        state = self._frame(frame)
        if state.packet is None or not state.published:
            raise RuntimeError('Capture and caller validation must precede packet consumption')
        state.consumed = True

    def release(self, frame):
        state = self._frame(frame)
        if not state.published or not state.consumed:
            raise RuntimeError('A frame remains owned through validation and delivery/discard')
        state.handle = state.packet = None
        state.cycles = 0
        state.published = state.consumed = False

    def idle(self):
        return all(frame.handle is None for frame in self.frames)
