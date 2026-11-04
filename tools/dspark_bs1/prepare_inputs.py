# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Import prepared checkpoint manifests and construct metadata, never model outputs.

The source formats contain typed raw checkpoint shards. Weight bytes are reused
in place; this tool does not quantize weights, sample tokens or create a golden.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from .geometry import build_d_banks, target_layout
from .rope import token_local_rope
from .tensors import base_name, dtype_name, matches


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    native = {base_name(row): row for row in read(args.native_metadata)['params']}
    single = {base_name(row): row for row in read(args.single_metadata)['params']}
    if len(native) != 155 or len(single) != 83:
        raise ValueError('Actual generated public155 and single-query83 metadata required')
    records = read(args.requests)
    if len(records) != 8 or len({(r['dataset'], r['question_id']) for r in records}) != 8:
        raise ValueError('Eight distinct real benchmark records required')
    for record in records:
        if not record['prompt_ids'] or record.get('truncated') or len(record['prompt_ids']) != record['prompt_tokens']:
            raise ValueError('Do not truncate, pad or split benchmark prompts')
    max_prompt = max(len(r['prompt_ids']) for r in records)
    layout = target_layout(max_prompt + 2048 - 1)
    specs, source_manifests = {}, []

    def reference(root, item, *, offset=0):
        path = (root / item['file']).resolve()
        stat = path.stat()
        return dict(file=os.path.relpath(path, output), format='raw',
                    offset_elements=offset, stat=[stat.st_size, stat.st_mtime_ns],
                    sha256=item.get('sha256'))

    for manifest in args.target_manifest:
        data = read(manifest)
        if data.get('complete') is not True or data.get('generated_fixture') is not False:
            raise ValueError('Complete real-checkpoint tensor manifests required')
        source_manifests.append(dict(file=os.path.relpath(manifest.resolve(), output), sha256=sha(manifest)))
        for entry in data['entries']:
            name = entry['name']
            if name not in single or single[name]['direction'] == 'Out':
                continue
            row = single[name]
            target_name = 's_' + name
            key = target_name if target_name in native and matches(row['shape'], native[target_name]['shape']) else 'single:' + name
            if entry.get('binding_kind') == 'global_host_tensor_full_L3_argument_shape':
                item = entry.get('tensor', entry)
                local = math.prod(item['shape'][1:])
                refs = [reference(manifest.parent, item, offset=rank * local) for rank in range(8)]
                shape = item['shape']
            else:
                values = entry['rank_refs']
                if len(values) != 8 or [r['rank'] for r in values] != list(range(8)):
                    raise ValueError('Target shards must retain their real rank order')
                refs = [reference(manifest.parent, item) for item in values]
                shape = [8, *values[0]['shape']]
            specs[key] = dict(shape=shape, dtype=dtype_name(row['dtype']), shards=refs)

    for manifest in args.draft_manifest:
        data = read(manifest)
        source_manifests.append(dict(file=os.path.relpath(manifest.resolve(), output), sha256=sha(manifest)))
        for entry in data['entries']:
            names = [prefix + entry['name'] for prefix in ('d_', 'm_') if prefix + entry['name'] in native]
            if len(names) != 1:
                raise ValueError('Ambiguous or unknown D/M weight: ' + entry['name'])
            name = names[0]
            values = entry['rank_refs']
            if len(values) != 4 or [r['rank'] for r in values] != list(range(4)):
                raise ValueError('D/M shards must retain their real four-rank order')
            specs[name] = dict(shape=[4, *values[0]['shape']], dtype=dtype_name(native[name]['dtype']),
                               shards=[reference(manifest.parent, item) for item in values])

    def tensor(name, value):
        value = value.contiguous()
        path = output / (name.replace(':', '-') + '.raw')
        path.write_bytes(value.view(torch.uint8).numpy().tobytes())
        stat = path.stat()
        local = math.prod(value.shape[1:])
        specs[name] = dict(shape=list(value.shape), dtype=str(value.dtype).removeprefix('torch.'),
                           shards=[dict(file=path.name, format='raw', offset_elements=rank * local,
                                        stat=[stat.st_size, stat.st_mtime_ns]) for rank in range(value.shape[0])])

    def fill(name, value=0, shape=None):
        row = native[name]
        shape = shape or row['shape']
        if any(d <= 0 for d in shape):
            raise ValueError('A concrete initializer geometry is required: ' + name)
        specs[name] = dict(shape=shape, dtype=dtype_name(row['dtype']), fill=value)

    for name, pool in layout['cache_pools'].items():
        key = 's_' + name
        shape = [8, *pool['shape']]
        if 'compress_state' in name:
            local = torch.zeros(pool['shape'], dtype=torch.float32)
            local[..., local.shape[-1] // 2:] = -3.4028234663852886e38
            tensor(key, local.unsqueeze(0).expand(8, *local.shape).contiguous())
        else:
            fill(key, shape=shape)
    for name, values in layout['tables'].items():
        tensor('s_' + name, torch.tensor(values, dtype=torch.int32).view(1, 1, -1).repeat(8, 1, 1))
    tensor('s_block_counts', torch.tensor(layout['block_counts'], dtype=torch.int32).view(1, 1, 7).repeat(8, 1, 1))
    for name, count in [('s_num_tokens_per_owner', 6), ('m_num_sampled', 1), ('m_next_prefill_tokens', 0)]:
        fill(name, count)
    for name in ['s_history_ring', 's_request_state', 'd_kv_caches', 's_bridge_markov_ids']:
        fill(name)
    for name in ['s_sampling_temperatures', 's_sampling_top_ks', 's_sampling_seeds']:
        fill(name, 1 if name.endswith('top_ks') else 0)
    tensor('s_logit_row_indices', torch.arange(6, dtype=torch.int32).view(1, 6).repeat(8, 1))
    rows = torch.full(native['m_logit_row_indices']['shape'], -1, dtype=torch.int32)
    rows[:, :5] = torch.arange(5, dtype=torch.int32)
    tensor('m_logit_row_indices', rows)

    from models.deepseek_v4_flash_dspark_bs1.drafter import dspark_drafter as draft
    table = torch.empty((4, 3, 1, 32768), dtype=torch.int32)
    logical = torch.arange(32768, dtype=torch.int32)
    for owner in range(4):
        for layer in range(3):
            table[owner, layer, 0] = owner * 8 * 6 + (logical + 7 * layer) % 6
    tensor('d_block_tables', table)
    positions = torch.arange(16384, dtype=torch.int32)
    cos, sin = token_local_rope(draft.M, 0, positions, max_seq_len=16384, dtype=torch.bfloat16)
    banks = build_d_banks(torch, list(range(max_prompt + 2048 + 1)), table, cos, sin, shared=False)
    for name, value in banks.items():
        tensor('d_' + name, value)
    missing = [name for name, row in native.items() if row['direction'] != 'Out' and name not in specs]
    if missing:
        raise ValueError('Prepared native inputs remain missing: ' + ', '.join(missing))
    generation = read(args.generation_config)
    eos = generation['eos_token_id']
    eos = [eos] if type(eos) is int else eos
    result = dict(schema=1, tensors=specs, requests_file=os.path.relpath(args.requests.resolve(), output),
                  requests_sha256=sha(args.requests), eos_token_ids=eos,
                  max_new_tokens=2048, temperature=0, batch_size=1, concurrency=1,
                  checkpoint_manifests=source_manifests, layout=layout,
                  target_prefix_policy='live single-query model execution; no saved continuation',
                  model_hidden_or_golden_outputs_created=False)
    path = output / 'tensor-manifest.json'
    path.write_text(json.dumps(result, indent=2) + '\n')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-metadata', type=Path, required=True)
    parser.add_argument('--single-metadata', type=Path, required=True)
    parser.add_argument('--target-manifest', type=Path, action='append', required=True)
    parser.add_argument('--draft-manifest', type=Path, action='append', required=True)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--generation-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(prepare(args), flush=True)


if __name__ == '__main__':
    main()
