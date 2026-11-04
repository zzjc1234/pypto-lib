# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Generate checkpoint-sized batch-one programs without allocating a device.

Run from the repository root as a module. Native compilation and request
execution are subsequent stages, not side effects of this command.
"""

import argparse
from contextlib import nullcontext
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

from . import native_source


def artifact_directory(compiled):
    """Return the directory owned by the public compiled-program result."""
    return Path(compiled.output_dir).resolve()


def source_manifest():
    root = Path(__file__).resolve().parent
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.suffix in ('.py', '.cpp', '.inc')}


def compile_program(profile, *, cache_root, device_ids):
    """Compile one immutable source profile; device IDs describe topology only."""
    if profile not in ('single', 'one', 'two', 'rsag'):
        raise ValueError('Unknown request-pipeline source profile')
    if len(device_ids) != 8 or len(set(device_ids)) != 8 or any(type(d) is not int or d < 0 for d in device_ids):
        raise ValueError('Eight distinct logical device IDs are required')
    from pypto.backend import BackendType
    from pypto.ir.distributed_compiled_program import DistributedConfig
    from pypto.jit import CacheConfig, configure_cache
    from pypto.runtime import ExecutionMode, RunConfig

    configure_cache(CacheConfig(enabled=True, root=Path(cache_root), readonly=False))
    before = source_manifest()
    saved_argv = sys.argv
    try:
        # The two namespaces have independently frozen import-time deployment
        # shapes. The drafter/Markov packages retain their own TP4 profile.
        sys.argv = ['dspark-bs1-build', '--tp', '8', '--dp', '1',
                    '--spec-tokens', '0' if profile == 'single' else '5',
                    '--max-seq-len', '16384', '--moe-banks', '43']
        if profile == 'single':
            module = importlib.import_module('.target_single.decode_fwd', __package__)
            function = module.l3_decode_fwd
            kwargs = {}
            selected_source = None
        else:
            module = importlib.import_module(
                '.native_rsag_pipeline' if profile == 'rsag' else '.native_pipeline', __package__)
            function = module.native_cycles
            selected_source = native_source.fingerprint()
            kwargs = dict(cycle_count=1 if profile == 'one' else 2,
                          native_w2_fingerprint=selected_source)
    finally:
        sys.argv = saved_argv
    config = RunConfig(
        backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
        device_id=device_ids[0], codegen_only=True,
        distributed_config=DistributedConfig(device_ids=device_ids, aicpu_thread_num=4),
    )
    context = native_source.use_native_w2(selected_source) if selected_source else nullcontext()
    with context:
        compiled = function.compile(config=config, **kwargs)
    directory = artifact_directory(compiled)
    metadata = json.loads((directory / 'distributed_meta.json').read_text())
    if len(metadata['params']) != (83 if profile == 'single' else 155):
        raise RuntimeError('The actual public parameter contract changed')
    if selected_source:
        # A validated GENERATED cache hit does not re-enter the generation hook.
        # Validate its source choice instead of mistaking the hit for new work.
        selection = json.loads((directory / 'report/native-w2-source-selection.json').read_text())
        if selection['source_sha256'] != selected_source or len(selection['selected']) != 10:
            raise RuntimeError('Cached program has a different native implementation')
        for item in selection['selected']:
            original = directory / 'report/original-native-source' / item['source']
            actual = (directory / item['source']).read_text()
            expected = native_source.replace_cube_source(original.read_text(), selected_source)
            if actual != expected or hashlib.sha256(actual.encode()).hexdigest() != item['selected_source_sha256']:
                raise RuntimeError('Selected native source differs from the reviewed graft')
    if before != source_manifest():
        raise RuntimeError('Pipeline sources changed during compilation')
    return compiled, dict(
        status='GENERATED_NOT_NATIVE_OR_DEVICE_TESTED', profile=profile,
        artifact_dir=str(directory), cache=str(Path(cache_root).resolve()),
        metadata_sha256=hashlib.sha256((directory / 'distributed_meta.json').read_bytes()).hexdigest(),
        public_parameters=len(metadata['params']), sources=before,
        native_w2_source_sha256=selected_source, old_binaries_loaded=False,
        model_weights_loaded=False, Worker=0, NPU=0, native=0,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['single', 'one', 'two', 'rsag'], required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--devices', default='0,1,2,3,4,5,6,7')
    args = parser.parse_args()
    if args.report.exists():
        parser.error('The report already exists; preserve it and choose a new run directory')
    devices = [int(v) for v in args.devices.split(',')]
    started = time.monotonic()
    _, report = compile_program(args.profile, cache_root=args.cache, device_ids=devices)
    report['generation_seconds'] = time.monotonic() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'sources'}), flush=True)


if __name__ == '__main__':
    main()
