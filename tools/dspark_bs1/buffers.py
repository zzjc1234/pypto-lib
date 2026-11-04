# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Plan exact worker-owned model buffers from actual generated metadata."""

import math

import torch

from .frames import HOST_OUTPUTS
from .tensors import DTYPES, base_name, dtype_name, matches, upload

CACHES = ('kv_cache', 'hca_cmp_kv', 'csa_cmp_kv', 'idx_kv_cache', 'idx_kv_scale',
          'hca_compress_state', 'csa_compress_state', 'csa_inner_compress_state')
STEP_FIELDS = {'input_ids', 'position_ids', 'kv_seq_lens', 'sampling_positions',
               'logit_row_indices', 'sampling_temperatures', 'sampling_top_ks',
               'sampling_seeds', 'num_tokens_per_owner'}


class BufferPlan:
    def __init__(self, native_metadata, single_metadata, bundle):
        self.bundle = bundle
        self.native_rows = native_metadata['params']
        self.single_rows = single_metadata['params']
        self.specs = {}
        self.native_keys = {}
        self.single_keys = {}
        self.broadcast_host_keys = set()
        self.step_host_keys = set()
        self.frame_keys = [{}, {}]
        for row in self.native_rows:
            name = base_name(row)
            if name in HOST_OUTPUTS:
                continue
            if row['direction'] == 'Out':
                for frame in range(2):
                    key = f'frame{frame}:{name}'
                    self.specs[key] = bundle.spec(name, row)
                    self.frame_keys[frame][name] = key
            else:
                self.specs[name] = bundle.spec(name, row)
                self.native_keys[name] = name
        for row in self.single_rows:
            name = base_name(row)
            shared = 's_' + name
            key = 'single:' + name
            if row['direction'] != 'Out' and name not in STEP_FIELDS and shared in self.native_keys:
                spec = self.specs[shared]
                if not matches(spec['shape'], row['shape']) or dtype_name(row['dtype']) != spec['dtype']:
                    raise ValueError('Single/native shared input geometry changed: ' + name)
                key = shared
            elif name in STEP_FIELDS and key not in bundle.tensors:
                if any(d <= 0 for d in row['shape']):
                    raise ValueError('Runtime metadata shape must be concrete')
                fill = 1 if name in ('num_tokens_per_owner', 'sampling_seeds') else 0
                self.specs[key] = dict(shape=row['shape'], dtype=dtype_name(row['dtype']), fill=fill)
            else:
                self.specs[key] = bundle.spec(key, row)
            if len(row['shape']) == 1:
                if (name != 'num_tokens_per_owner' or row['shape'] != [8]
                        or row['direction'] != 'In' or dtype_name(row['dtype']) != 'int32'):
                    raise ValueError('Unsupported unsliced single-program input: ' + name)
                # The generated HOST entry passes this entire vector to every
                # rank. Preserve normal public HOST-to-device input staging.
                self.broadcast_host_keys.add(key)
            if name in STEP_FIELDS:
                if row['direction'] != 'In':
                    raise ValueError('Single-step metadata must remain INPUT: ' + name)
                self.step_host_keys.add(key)
            self.single_keys[name] = key
        self.planned_bytes = [0] * 8
        for spec in self.specs.values():
            dtype = DTYPES[spec['dtype']]
            size = math.prod(spec['shape'][1:]) * torch.empty((), dtype=dtype).element_size()
            for rank in range(spec['shape'][0]):
                self.planned_bytes[rank] += size
        for key in self.broadcast_host_keys:
            spec = self.specs[key]
            element = torch.empty((), dtype=DTYPES[spec['dtype']]).element_size()
            # Replace the invalid scalar-shard budget with the full vector's
            # temporary input storage on each of the eight consumers.
            for rank in range(8):
                self.planned_bytes[rank] += (math.prod(spec['shape']) - 1) * element
        if any(len(spec['shape']) < 2 for key, spec in self.specs.items()
               if key not in self.broadcast_host_keys):
            raise ValueError('Device allocations require nonempty per-rank shapes')
        # These roots must exist before the Worker forks and remain alive
        # through its entire lifetime, just like the packet HOST frames.
        self.step_host_inputs = {}
        for key in self.step_host_keys:
            spec = self.specs[key]
            if 'fill' in spec:
                host = torch.full(tuple(spec['shape']), spec['fill'], dtype=DTYPES[spec['dtype']])
            else:
                host = torch.stack([bundle.shard(spec, rank) for rank in range(spec['shape'][0])])
            self.step_host_inputs[key] = host.contiguous().share_memory_()

    def verify_files(self):
        """Map actual payload headers before a lease or Worker is constructed."""
        for spec in self.specs.values():
            if 'fill' not in spec:
                for rank in range(spec['shape'][0]):
                    self.bundle.shard(spec, rank)
        self.bundle.recheck()

    def allocate(self, worker):
        from pypto.runtime import StackedDeviceTensor

        roots = {}
        # Allocate necessary target weights on all eight ranks first. Bulk
        # uploads follow successful allocations; no placeholder HBM is reserved.
        order = sorted(self.specs, key=lambda key: (
            not (key.startswith('s_') and 'fill' not in self.specs[key]),
            -math.prod(self.specs[key]['shape']), key))
        for key in order:
            spec = self.specs[key]
            shape = tuple(spec['shape'])
            dtype = DTYPES[spec['dtype']]
            if key in self.step_host_keys:
                roots[key] = self.step_host_inputs[key]
                continue
            shards = [worker.alloc_tensor(shape[1:], dtype, worker_id=rank)
                      for rank in range(shape[0])]
            roots[key] = StackedDeviceTensor(shards, shape, tuple(range(shape[0])))
        for key in order:
            spec = self.specs[key]
            if key in self.step_host_keys:
                continue
            for rank, shard in enumerate(roots[key].shards):
                upload(worker, shard, self.bundle.shard(spec, rank), rank=rank, fill=spec.get('fill', 0))
        self.bundle.recheck()
        common = {name: roots[key] for name, key in self.native_keys.items()}
        frames = tuple({name: roots[key] for name, key in fields.items()} for fields in self.frame_keys)
        single = {name: roots[key] for name, key in self.single_keys.items()}
        return roots, common, frames, single


def set_single_step(worker, arguments, position, token):
    values = {'position_ids': position, 'sampling_positions': position,
              'input_ids': token, 'kv_seq_lens': position + 1}
    for name, value in values.items():
        root = arguments[name]
        if not isinstance(root, torch.Tensor) or not root.is_shared():
            raise TypeError('Single-step metadata must use its prefork shared HOST input')
        root.fill_(value)


def public_arguments(metadata, named):
    return tuple(named[base_name(row)] for row in metadata['params'])
