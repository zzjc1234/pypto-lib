# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build and explicitly load source-verified native packages without a device.

This adapter uses PyPTO's low-level prebuilt API. Its package is deliberately
separate from the persistent JIT cache and never impersonates an official READY
artifact. Compiling/importing a package does not establish model correctness.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def build_package(generated_report, output, *, jobs=32):
    from models.deepseek_v4_flash_dspark_bs1.build import source_manifest
    from pypto.jit._artifact_manifest import BuildKind
    from pypto.runtime._prebuilt import prepare_prebuilt, read_prebuilt
    from pypto.runtime import device_runner

    report = read(generated_report)
    if not report['status'].startswith(('GENERATED_', 'PASS_ACTUAL_PUBLIC_PR_GENERATED_')):
        raise ValueError('Successful actual source generation is required')
    if source_manifest() != report['sources']:
        raise ValueError('Model source differs from the generated program')
    source = Path(report['artifact_dir']).resolve()
    if sha(source / 'distributed_meta.json') != report['metadata_sha256']:
        raise ValueError('Generated public metadata changed')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve the existing native build instead of overwriting it')
    shutil.copytree(source, output)
    # An immutable cache inventory belongs to its original source directory,
    # not this explicit mutable native-build copy.
    (output / 'artifact_manifest.json').unlink(missing_ok=True)
    started = time.monotonic()
    if type(jobs) is not int or not 1 <= jobs <= 64:
        raise ValueError('Native compilation requires between one and 64 jobs')
    executor = device_runner.ThreadPoolExecutor

    def bounded_executor(max_workers=None, *args, **kwargs):
        return executor(min(jobs, max_workers or jobs), *args, **kwargs)

    device_runner.ThreadPoolExecutor = bounded_executor
    try:
        prepare_prebuilt(output, 'a2a3', BuildKind.DISTRIBUTED)
    finally:
        device_runner.ThreadPoolExecutor = executor
    actual = read_prebuilt(output, 'a2a3', BuildKind.DISTRIBUTED)
    expected = {'l2_decode_fwd'} if report['profile'] == 'single' else {'l2_full_cycles', 'l2_s6_cycles_only'}
    if set(actual) != expected:
        raise RuntimeError('Native package role contract changed')
    inventory = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    manifest = dict(
        schema=1, status='NATIVE_BUILT_NOT_MODEL_TESTED', profile=report['profile'],
        public_parameters=report['public_parameters'], sources=report['sources'],
        metadata_sha256=report['metadata_sha256'], files=inventory,
        generated_report_sha256=sha(generated_report), native_seconds=time.monotonic() - started,
        compiler_owned_binaries=True, old_binary_reuse=False, Worker=0, NPU=0,
        official_JIT_READY=False, numerical_qualification=False, performance_qualification=False,
    )
    (output / 'native-package.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def validate_package(directory):
    from models.deepseek_v4_flash_dspark_bs1.build import source_manifest

    directory = Path(directory).resolve()
    manifest = read(directory / 'native-package.json')
    if manifest['status'] != 'NATIVE_BUILT_NOT_MODEL_TESTED' or manifest['official_JIT_READY']:
        raise ValueError('Explicit source-native package required')
    if (directory / 'artifact_manifest.json').exists():
        raise ValueError('A private native package must not impersonate a JIT cache artifact')
    if source_manifest() != manifest['sources']:
        raise ValueError('Native package was built from different model source')
    for name, digest in manifest['files'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or sha(directory / path) != digest:
            raise ValueError('Native package file changed: ' + name)
    return manifest


@contextmanager
def native_programs(packages, *, device_ids=None):
    """Retain actual callables while leaving all normal submit validators intact."""
    from pypto.ir.distributed_compiled_program import DistributedCompiledProgram
    from pypto.ir.distributed_compiled_program import DistributedConfig
    from pypto.jit._artifact_manifest import BuildKind
    from pypto.runtime import ExecutionMode
    from pypto.runtime import distributed_runner as runner
    from pypto.runtime._prebuilt import load_prebuilt

    original = runner._assemble_chip_callables
    selected = []
    programs = {}
    try:
        for label, path in packages.items():
            directory = Path(path).resolve()
            manifest = validate_package(directory)
            actual = load_prebuilt(directory, 'a2a3', BuildKind.DISTRIBUTED)
            chips = runner._ChipCallables()
            for name, (callable_, runtime, config) in actual.items():
                if runtime != 'tensormap_and_ringbuffer' or not config.get('enable_sdma'):
                    raise ValueError('The measured TMR/SDMA runtime contract changed')
                chips.add(name, callable_, config)
            override = {} if device_ids is None else dict(
                distributed_config=DistributedConfig(device_ids=list(device_ids), aicpu_thread_num=4))
            program = DistributedCompiledProgram.from_dir(directory, execution_mode=ExecutionMode.ONBOARD, **override)
            if len(program._param_infos) != manifest['public_parameters']:
                raise ValueError('Actual native parameter count changed')
            if vars(program).get('_artifact_runtime') is not None:
                raise ValueError('Explicit package unexpectedly carries a cache runtime')
            entry, _ = runner._load_orch_entry(directory)
            if not callable(entry):
                raise ValueError('Generated HOST entry is missing')
            programs[label] = program
            selected.append((program, chips, 'tensormap_and_ringbuffer', True))

        def assemble(compiled):
            for program, *result in selected:
                if compiled is program:
                    return tuple(result)
            return original(compiled)

        runner._assemble_chip_callables = assemble
        yield programs
    finally:
        if runner._assemble_chip_callables is not original:
            if 'assemble' not in locals() or runner._assemble_chip_callables is not assemble:
                raise RuntimeError('Native assembly scope was changed by another owner')
            runner._assemble_chip_callables = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=32)
    args = parser.parse_args()
    result = build_package(args.generated_report, args.output, jobs=args.jobs)
    print(json.dumps({k: v for k, v in result.items() if k not in ['sources', 'files']}), flush=True)


if __name__ == '__main__':
    main()
