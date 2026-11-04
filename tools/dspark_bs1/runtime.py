# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Add public Worker depth options to the original distributed constructor."""

import ast
import __future__
from contextlib import contextmanager
from dataclasses import replace
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading


WORKER_DEPTHS = {"pipeline_depth": 2, "launch_depth": 1}
_CONSTRUCTION_LOCK = threading.Lock()


@contextmanager
def omit_optional_sdma_warmup(*, enabled=False):
    """Explicitly omit only the optional init ELF, keeping real SDMA enabled."""
    if not enabled:
        yield dict(enabled=False)
        return
    from simpler_setup.runtime_builder import RuntimeBuilder, RuntimeBinaries

    original = RuntimeBuilder.get_binaries
    report = dict(enabled=True, calls=0, enable_sdma_changed=False)

    def selected(builder, name, build=False, *, build_shared=True, profiling_config=None):
        if name != 'tensormap_and_ringbuffer' or builder.platform != 'a2a3':
            return original(builder, name, build, build_shared=build_shared, profiling_config=profiling_config)
        if build:
            raise RuntimeError('Device setup must use already prepared runtime binaries')
        value = original(builder, name, False, build_shared=build_shared, profiling_config=profiling_config)
        if not isinstance(value, RuntimeBinaries):
            raise TypeError('Actual RuntimeBinaries required')
        report['calls'] += 1
        return replace(value, sdma_warmup_path=None)

    RuntimeBuilder.get_binaries = selected
    try:
        yield report
    finally:
        RuntimeBuilder.get_binaries = original
        report['restored'] = RuntimeBuilder.get_binaries is original


@contextmanager
def worker_factory():
    """Use only while constructing DistributedWorker; restore its original hook."""
    if WORKER_DEPTHS != {"pipeline_depth": 2, "launch_depth": 1}:
        raise ValueError("This adapter supports only pipeline_depth2/launch_depth1")
    from pypto.runtime import distributed_runner

    with _CONSTRUCTION_LOCK:
        original = distributed_runner._construct_worker
        source = inspect.getsource(original)
        module = ast.parse(source)
        calls = [node for node in ast.walk(module) if isinstance(node, ast.Return)
                 and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                 and node.value.func.id == "Worker"]
        if len(calls) != 1 or len(calls[0].value.keywords) != 1:
            raise RuntimeError("Original public Worker construction changed")
        keyword = calls[0].value.keywords[0]
        if keyword.arg is not None or not isinstance(keyword.value, ast.Name) or keyword.value.id != "worker_config":
            raise RuntimeError("Original worker_config forwarding changed")
        calls[0].value.keywords.extend(ast.keyword(arg=name, value=ast.Constant(value=value)) for name, value in WORKER_DEPTHS.items())
        ast.fix_missing_locations(module)
        namespace = dict(original.__globals__)
        exec(compile(module, str(Path(__file__).resolve()) + ":public_worker_depths", "exec",
                     flags=__future__.annotations.compiler_flag, dont_inherit=True), namespace)
        adapted = functools.update_wrapper(namespace[original.__name__], original)
        receipt = dict(pipeline_depth=2, launch_depth=1,
                       original_factory_file=inspect.getsourcefile(original),
                       original_factory_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                       adapter_file=str(Path(__file__).resolve()),
                       adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       modification="Only two public Worker keyword arguments; original body/placement/runtime/SDMA/timeout validation unchanged",
                       no_private_run_or_worker_ids=True)
        distributed_runner._construct_worker = adapted
        try:
            yield receipt
        finally:
            distributed_runner._construct_worker = original


@contextmanager
def early_aic_control_mapping(output, *, enabled=False):
    """Optionally perform the genuine CANN mapping before SDK child init.

    This is a scoped initialization-order adapter for the tested 910C setup.
    It does not invent addresses, change permissions, reset another process's
    device, or bypass the normal SDK register-table path. A driver rejection
    fails the initialization; it is not silently retried on another device.
    """
    if not enabled:
        yield
        return
    import ctypes
    import simpler.task_interface as task_interface

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    original = task_interface.ChipWorker.init

    def initialize(self, device_id, *args, **kwargs):
        acl = ctypes.CDLL('libascendcl.so', mode=ctypes.RTLD_GLOBAL)
        acl.aclInit.argtypes = [ctypes.c_void_p]
        acl.aclrtSetDevice.argtypes = [ctypes.c_int]
        init = int(acl.aclInit(None))
        if init not in (0, 100002):
            raise RuntimeError(f'ACL init failed before child startup: {init}')
        selected = int(acl.aclrtSetDevice(int(device_id)))
        if selected:
            raise RuntimeError(f'ACL device selection failed: {selected}')
        hal = ctypes.CDLL('libascend_hal.so')
        physical = ctypes.c_uint()
        mapping = hal.drvDeviceGetPhyIdByIndex
        mapping.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
        mapping.restype = ctypes.c_int
        mapped = int(mapping(int(device_id), ctypes.byref(physical)))
        if mapped:
            raise RuntimeError(f'Actual logical/physical device lookup failed: {mapped}')

        class Input(ctypes.Structure):
            _fields_ = [('addr_type', ctypes.c_uint), ('device', ctypes.c_uint)]

        class Output(ctypes.Structure):
            _fields_ = [('ptr', ctypes.c_ulonglong), ('length', ctypes.c_ulonglong)]

        control = hal.halMemCtl
        control.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
                            ctypes.c_void_p, ctypes.c_void_p]
        control.restype = ctypes.c_int
        request, response = Input(2, physical.value), Output()
        status = int(control(0, ctypes.byref(request), ctypes.sizeof(request), ctypes.byref(response), None))
        record = dict(pid=os.getpid(), logical_device=int(device_id), physical_device=physical.value,
                      acl_init=init, acl_set_device=selected, driver_rc=status,
                      address=response.ptr, length=response.length,
                      phase='genuine ACL/HAL before original SDK initialization')
        path = output / f'child-{device_id}-{os.getpid()}-early-map.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(path)
        if status or not response.ptr or not response.length:
            raise RuntimeError(f'AIC control mapping failed: driver_rc={status}')
        return original(self, device_id, *args, **kwargs)

    task_interface.ChipWorker.init = initialize
    try:
        yield
    finally:
        task_interface.ChipWorker.init = original
