# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Prepare, then execute eight distinct complete requests from source-built programs."""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback

import torch

from .buffers import BufferPlan
from .execution import Requests, atomic_json
from .frames import allocate_host_frames
from .native_package import native_programs, validate_package
from .resources import actual_free, claim, communication_bytes, owned_children
from .runtime import early_aic_control_mapping, omit_optional_sdma_warmup, worker_factory
from .tensors import TensorBundle


def packages_from_args(values):
    result = {}
    for value in values:
        name, path = value.split('=', 1)
        if name in result:
            raise ValueError('Duplicate program profile')
        result[name] = Path(path).resolve()
    if set(result) != {'one', 'two', 'rsag', 'single'}:
        raise ValueError('The four actual source-native packages are required')
    return result


def tools_manifest():
    root = Path(__file__).resolve().parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')}


def prepare(args):
    packages = packages_from_args(args.package)
    manifests = {name: validate_package(path) for name, path in packages.items()}
    with native_programs(packages):
        pass
    metadata = {name: json.loads((path / 'distributed_meta.json').read_text()) for name, path in packages.items()}
    if not metadata['one']['params'] == metadata['two']['params'] == metadata['rsag']['params']:
        raise ValueError('Actual native one/two/RSAG public ABIs must agree')
    bundle = TensorBundle(args.tensor_manifest)
    records_path = (bundle.root / bundle.manifest['requests_file']).resolve()
    if hashlib.sha256(records_path.read_bytes()).hexdigest() != bundle.manifest['requests_sha256']:
        raise ValueError('Benchmark requests changed')
    records = json.loads(records_path.read_text())
    if len(records) != 8 or len({(q['dataset'], q['question_id']) for q in records}) != 8:
        raise ValueError('Eight real independent request records are required')
    for question in records:
        if (not question['prompt_ids'] or len(question['prompt_ids']) != question['prompt_tokens']
                or question.get('truncated') or question.get('padded')
                or any(type(t) is not int or not 0 <= t < 129280 for t in question['prompt_ids'])):
            raise ValueError('Invalid, truncated or substituted request prompt')
    plan = BufferPlan(metadata['two'], metadata['single'], bundle)
    plan.verify_files()
    comm = communication_bytes(packages)
    planned = [a + b for a, b in zip(plan.planned_bytes, comm, strict=True)]
    with worker_factory() as depths:
        pass
    from simpler_setup import RuntimeBuilder
    with omit_optional_sdma_warmup(enabled=args.omit_optional_warmup) as optional:
        RuntimeBuilder('a2a3').get_binaries('tensormap_and_ringbuffer', build=False)
    ready = dict(status='CPU_READY_NOT_DEVICE_TESTED', packages={name: str(path) for name, path in packages.items()},
                 package_manifest_sha256={name: hashlib.sha256((path / 'native-package.json').read_bytes()).hexdigest()
                                          for name, path in packages.items()},
                 metadata_sha256={name: manifests[name]['metadata_sha256'] for name in packages},
                 tensor_manifest_sha256=bundle.proof['manifest_sha256'], tool_sources=tools_manifest(),
                 planned_tensor_bytes_per_rank=plan.planned_bytes, actual_declared_comm_bytes_per_rank=comm,
                 planned_device_bytes_per_rank=planned, records=8, prompt_lengths=[q['prompt_tokens'] for q in records],
                 max_new_tokens=2048, temperature=0, batch_size=1, concurrency=1, drafts=5,
                 depth_adapter=depths, optional_warmup=optional, input_proof=bundle.proof,
                 model_numerical_qualification=False, Worker=0, NPU=0)
    return ready, packages, metadata, bundle, records, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--package', action='append', required=True, help='profile=source-native-package-directory')
    parser.add_argument('--tensor-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-ready', type=Path)
    parser.add_argument('--lease-file', type=Path)
    parser.add_argument('--execution-lock', type=Path)
    parser.add_argument('--wait-seconds', type=int, default=90000)
    parser.add_argument('--early-aic-control-map', action='store_true')
    parser.add_argument('--omit-optional-warmup', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    ready, packages, metadata, bundle, records, plan = prepare(args)
    if args.prepare:
        atomic_json(args.output / 'cpu-ready.json', ready)
        print(json.dumps({k: ready[k] for k in ['status', 'records', 'prompt_lengths', 'planned_device_bytes_per_rank']}))
        return
    if args.cpu_ready is None or args.lease_file is None or args.execution_lock is None:
        parser.error('Execution requires its actual CPU-ready receipt and shared ownership locks')
    if not os.environ.get('DSPARK_EXPERIMENT_TOKEN'):
        parser.error('Run through the owned-process supervisor so the marker predates child creation')
    old = json.loads(args.cpu_ready.read_text())
    for key in ['packages', 'package_manifest_sha256', 'metadata_sha256', 'tensor_manifest_sha256',
                'tool_sources', 'planned_device_bytes_per_rank', 'prompt_lengths']:
        if old[key] != ready[key]:
            raise ValueError('Frozen CPU admission changed: ' + key)
    host_frames = allocate_host_frames()
    report = dict(status='WAITING_RESOURCES_READY', CPU_ready_sha256=hashlib.sha256(args.cpu_ready.read_bytes()).hexdigest(),
                  code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                     cwd=Path(__file__).resolve().parents[2], text=True).strip(),
                  automatic_model_retries=0, Worker_closed=False, NPU_started=False)
    atomic_json(args.output / 'execution.json', report)
    from pypto.backend import BackendType
    from pypto.ir.distributed_compiled_program import DistributedConfig
    from pypto.runtime import ExecutionMode, RunConfig
    from pypto.runtime.distributed_runner import DistributedWorker

    worker = runner = None
    try:
        with native_programs(packages) as programs:
            with claim(ready['planned_device_bytes_per_rank'], args.lease_file,
                       wait_seconds=args.wait_seconds, execution_lock=args.execution_lock) as lease:
                atomic_json(args.output / 'owned-device-claim.json', lease)
                devices = lease['device_ids']
                for program in programs.values():
                    program._distributed_config = replace(program._distributed_config, device_ids=list(devices))
                config = RunConfig(
                    backend_type=BackendType.Ascend910B, execution_mode=ExecutionMode.ONBOARD,
                    device_id=devices[0], distributed_config=DistributedConfig(device_ids=devices, aicpu_thread_num=4),
                    ring_heap=[2**30] * 4, ring_dep_pool=16384, ring_task_window=16384,
                    enable_pmu=0, enable_scope_stats=False, enable_dep_gen=False,
                    enable_chip_swimlane=0, enable_dump_args=0,
                )
                started = time.monotonic()
                def device_deadline(*unused):
                    raise TimeoutError('The bounded device performance window expired')
                signal.signal(signal.SIGALRM, device_deadline)
                signal.setitimer(signal.ITIMER_REAL, 1140)
                try:
                    with worker_factory() as depth, omit_optional_sdma_warmup(
                        enabled=args.omit_optional_warmup) as optional, early_aic_control_mapping(
                        args.output / 'early-map', enabled=args.early_aic_control_map):
                        worker = DistributedWorker(list(programs.values()), config=config, persistent=True,
                                                   reset_persistent_windows=False,
                                                   inherited_host_tensors=[v for frame in host_frames for v in frame.values()]
                                                       + list(plan.step_host_inputs.values()),
                                                   startup_timeout_s=90)
                    report.update(NPU_started=True, status='ALLOCATING_REAL_MODEL_INPUTS',
                                  initialization_seconds=time.monotonic() - started,
                                  depth_adapter=depth, optional_warmup=optional)
                    atomic_json(args.output / 'execution.json', report)
                    actual_free(worker, devices, ready['planned_device_bytes_per_rank'])
                    upload_start = time.monotonic()
                    roots, common, outputs, single = plan.allocate(worker)
                    report['allocation_upload_seconds'] = time.monotonic() - upload_start
                    report['actual_resident_roots'] = len(roots)
                    actual_free(worker, devices)
                    runner = Requests(worker, programs, metadata, common, outputs, single, host_frames,
                                      plan, records, bundle.manifest['eos_token_ids'], config,
                                      args.output, device_started=started)
                    report['result'] = runner.run()
                    report['status'] = 'MEASUREMENT_COMPLETE_NUMERICAL_CLOSEOUT_PENDING'
                finally:
                    if runner is not None:
                        runner.save()
                    if worker is not None:
                        worker.close()
                        report['Worker_closed'] = True
                    if runner is not None:
                        runner.single_entry.close()
                        report['single_entry_reuse'] = runner.single_entry.report()
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    report['owned_children_before_release'] = owned_children()
                    atomic_json(args.output / 'execution.json', report)
        report['lease_absent_after_close'] = not args.lease_file.exists()
        atomic_json(args.output / 'execution.json', report)
    except BaseException as error:
        report.update(status='FAILED_PRESERVED_NO_RETRY', error=repr(error), traceback=traceback.format_exc())
        if runner is not None:
            for record in runner.requests:
                if record['status'] == 'RUNNING':
                    record.update(status='FAILED_PRESERVED', error=repr(error))
            runner.save()
            runner.export()
        atomic_json(args.output / 'execution.json', report)
        raise


if __name__ == '__main__':
    main()
