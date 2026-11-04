# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Declarative prepared-checkpoint tensors and worker-owned device allocations.

Prepared files are data, never executable source or prebuilt model binaries.
NZ inputs must already have their declared native storage order; the uploader
does not silently transpose or requantize checkpoint weights.
"""

import hashlib
import json
import math
from pathlib import Path

import torch

DTYPES = {name: getattr(torch, name) for name in ['float32', 'float16', 'bfloat16', 'int8', 'uint8', 'int32', 'int64']}
ALIASES = {'fp32': 'float32', 'f32': 'float32', 'f16': 'float16', 'bf16': 'bfloat16',
           'i8': 'int8', 'u8': 'uint8', 'i32': 'int32', 'i64': 'int64'}
CHUNK_BYTES = 64 * 1024**2


def base_name(row):
    return row['name'].split('__ssa', 1)[0]


def matches(shape, expected):
    return len(shape) == len(expected) and all(want == -1 or got == want for got, want in zip(shape, expected, strict=True))


def dtype_name(name):
    name = name.lower().removeprefix('torch.')
    return ALIASES.get(name, name)


class TensorBundle:
    def __init__(self, manifest):
        self.path = Path(manifest).resolve()
        self.root = self.path.parent
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get('schema') != 1:
            raise ValueError('Prepared tensor manifest schema 1 is required')
        self.tensors = self.manifest['tensors']
        self.files = {}
        self.proof = dict(manifest_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(), files={},
                          loaded_as_data_only=True, implicit_quantization=False)

    def spec(self, key, row):
        spec = self.tensors.get(key)
        if spec is None:
            if row['direction'] != 'Out' or any(type(d) is not int or d <= 0 for d in row['shape']):
                raise ValueError('Prepared input is missing: ' + key)
            spec = dict(shape=row['shape'], dtype=row['dtype'], fill=0)
        spec = dict(spec, dtype=dtype_name(spec['dtype']))
        if not matches(spec['shape'], row['shape']) or spec['dtype'] != dtype_name(row['dtype']):
            raise ValueError('Prepared tensor differs from actual model metadata: ' + key)
        if spec['dtype'] not in DTYPES or spec['shape'][0] not in (4, 8):
            raise ValueError('Unexpected batch-one rank geometry: ' + key)
        if 'shards' in spec and len(spec['shards']) != spec['shape'][0]:
            raise ValueError('Every logical rank needs one actual tensor shard: ' + key)
        return spec

    def shard(self, spec, rank):
        shape = tuple(spec['shape'][1:])
        dtype = DTYPES[spec['dtype']]
        if 'fill' in spec:
            return None
        ref = spec['shards'][rank]
        path = (self.root / ref['file']).resolve()
        stat = path.stat()
        facts = [stat.st_size, stat.st_mtime_ns]
        expected = ref.get('stat')
        if expected is not None and facts != expected:
            raise ValueError('Prepared weight file changed: ' + ref['file'])
        key = str(path)
        if key not in self.files:
            if ref.get('format', 'torch') == 'raw':
                element = torch.empty((), dtype=dtype).element_size()
                if stat.st_size % element:
                    raise ValueError('Raw tensor length is not divisible by its actual dtype')
                self.files[key] = torch.from_file(str(path), shared=False,
                                                  size=stat.st_size // element, dtype=dtype)
            elif ref.get('format', 'torch') == 'torch':
                self.files[key] = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
            else:
                raise ValueError('Unknown prepared tensor file format')
            self.proof['files'][ref['file']] = dict(stat=facts, integrity='stat', declared_sha256=ref.get('sha256'))
        value = self.files[key]
        for component in ref.get('key', []):
            value = value[component]
        if not isinstance(value, torch.Tensor) or value.dtype != dtype:
            raise TypeError('Prepared payload has the wrong actual dtype')
        if not value.is_contiguous():
            raise ValueError('Prepared storage must be contiguous; export its native layout explicitly')
        start = ref.get('offset_elements', 0)
        if type(start) is not int or start < 0:
            raise ValueError('Invalid prepared storage offset')
        count = math.prod(shape)
        flat = value.view(-1)
        if start + count > flat.numel():
            raise ValueError('Prepared shard exceeds its actual storage')
        return flat[start:start + count].view(shape)

    def recheck(self):
        for name, proof in self.proof['files'].items():
            stat = (self.root / name).resolve().stat()
            if [stat.st_size, stat.st_mtime_ns] != proof['stat']:
                raise ValueError('Prepared checkpoint changed during execution: ' + name)


def allocate(worker, spec, bundle):
    from pypto.runtime import StackedDeviceTensor

    dtype = DTYPES[spec['dtype']]
    shape = tuple(spec['shape'])
    shards = []
    for rank in range(shape[0]):
        shard = worker.alloc_tensor(shape[1:], dtype, worker_id=rank)
        shards.append(shard)
        host = bundle.shard(spec, rank)
        upload(worker, shard, host, rank=rank, fill=spec.get('fill', 0))
    return StackedDeviceTensor(shards, shape, tuple(range(shape[0])))


def upload(worker, shard, host, *, rank, fill=0):
    count = math.prod(shard.shape)
    element = torch.empty((), dtype=shard.dtype).element_size()
    chunk = min(count, CHUNK_BYTES // element)
    source = host.view(-1) if host is not None else torch.full((chunk,), fill, dtype=shard.dtype)
    if host is not None and (host.dtype != shard.dtype or tuple(host.shape) != tuple(shard.shape)):
        raise ValueError('Upload changes the actual tensor shape or dtype')
    for start in range(0, count, chunk):
        size = min(chunk, count - start)
        current = source[start:start + size] if host is not None else source[:size]
        worker.copy_to(shard.data_ptr, current.data_ptr(), size * element,
                       dst_offset=start * element, worker_id=rank)


def read_shard(worker, root, rank=0):
    shard = root.shards[rank]
    host = torch.empty(shard.shape, dtype=shard.dtype)
    worker.copy_from(host.data_ptr(), shard.data_ptr, host.numel() * host.element_size(), worker_id=rank)
    return host
