# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Experimental reuse of fixed-input host argument packs, for serial A/B only.

Public prepared-argument validation and _submit_chip are unchanged. Each hit
re-enters the original retained-domain provider; descriptors are reused only
while the worker, inputs, metadata, callables and domain objects are identical.
Tensor contents may change in place. This helper must not be used with
concurrent/in-flight reuse of the same TaskArgs.
"""
from contextlib import ExitStack, contextmanager
import threading


class PreparedHostEntry:
    def __init__(self, entry, torch_module, device_types=()):
        self.entry = entry
        self.torch = torch_module
        self.device_types = device_types
        self.enabled = True
        self.cache = {}
        self.hits = 0
        self.builds = 0

    def fingerprint(self, value):
        if isinstance(value, self.torch.Tensor):
            # frames.bind creates fresh Python views of the same pre-fork
            # shared storage. Retain the storage object as the identity key:
            # identical views share a pack, different storage never does.
            return ('host', value.untyped_storage(), value.data_ptr(), tuple(value.shape),
                    tuple(value.stride()), value.dtype, value.storage_offset())
        if isinstance(value, (list, tuple)):
            return (type(value), tuple(self.fingerprint(v) for v in value))
        if isinstance(value, (int, float, str, bool, type(None))):
            return (type(value), value)
        # DeviceTensor/StackedDeviceTensor are frozen; keep references in the
        # cache, so ids cannot be recycled while a cached key is reachable.
        if isinstance(value, self.device_types):
            return (type(value), id(value))
        raise TypeError(f'Unsupported mutable prepared input: {type(value).__name__}')

    def __call__(self, orch, args, config, *, tensors, callables, sub_ids,
                 _keep, world_size, _domain_provider=None):
        kwargs = dict(tensors=tensors, callables=callables, sub_ids=sub_ids,
                      _keep=_keep, world_size=world_size,
                      _domain_provider=_domain_provider)
        if not self.enabled:
            return self.entry(orch, args, config, **kwargs)
        if _domain_provider is None:
            raise RuntimeError('Prepared-host experiment requires retained domains')
        key = (id(orch._worker), world_size,
               tuple((k, self.fingerprint(v)) for k, v in sorted(tensors.items())),
               tuple((k, id(v)) for k, v in sorted(callables.items())),
               tuple((k, id(v)) for k, v in sorted(sub_ids.items())))
        cached = self.cache.get(key)
        submit = self.entry.__globals__['_submit_chip']
        if cached is not None:
            with ExitStack() as stack:
                same_domains = all(stack.enter_context(_domain_provider(**spec)) is domain
                                   for spec, domain in cached['domains'])
                if same_domains:
                    _keep.extend(cached['keep'])
                    for callable_id, task_args, device in cached['submissions']:
                        submit(orch, callable_id, task_args, config, device)
                    self.hits += 1
                    return
            # A changed retained domain invalidates all old wire descriptors.
            del self.cache[key]
        domains, submissions = [], []
        keep_start = len(_keep)
        retained_inputs = dict(tensors)

        @contextmanager
        def record_domain(**spec):
            with _domain_provider(**spec) as domain:
                domains.append((spec, domain))
                yield domain

        def record_submit(current_orch, callable_id, task_args, current_config, device):
            submissions.append((callable_id, task_args, device))
            return submit(current_orch, callable_id, task_args, current_config, device)

        self.entry.__globals__['_submit_chip'] = record_submit
        try:
            self.entry(orch, args, config, **(kwargs | {'_domain_provider': record_domain}))
        finally:
            self.entry.__globals__['_submit_chip'] = submit
        self.cache[key] = dict(domains=domains, submissions=submissions,
                              keep=tuple(_keep[keep_start:]), inputs=retained_inputs,
                              callables=dict(callables), worker=orch._worker)
        self.builds += 1


class SerialSingleEntry:
    """Reuse only the single-token program, with result-bound serial ownership."""

    def __init__(self, worker, program):
        from pypto.runtime import DeviceTensor, StackedDeviceTensor
        import torch

        if len(program._param_infos) != 83:
            raise ValueError('Only the actual 83-parameter single-token program is supported')
        self.worker, self.program = worker, program
        self.state = worker._states[program]
        self.original = self.state['entry_fn']
        self.prepared = PreparedHostEntry(self.original, torch, (DeviceTensor, StackedDeviceTensor))
        self.active = self.failed = self.closed = False
        self._serial_gate = threading.Lock()
        self.state['entry_fn'] = self.prepared

    def submit_result(self, arguments, *, config):
        if not self._serial_gate.acquire(blocking=False):
            raise RuntimeError('Single-token descriptor reuse requires a completed successful predecessor')
        try:
            if self.failed or self.closed:
                raise RuntimeError('Single-token descriptor reuse is closed after failure or teardown')
            self.active = True
            try:
                return self.worker.submit(self.program, *arguments, config=config).result()
            except BaseException:
                self.failed = True
                raise
        finally:
            self.active = False
            self._serial_gate.release()

    def close(self):
        if not self._serial_gate.acquire(blocking=False):
            raise RuntimeError('Do not clear descriptor keepalives before the outstanding result')
        try:
            if self.closed:
                return
            if self.state['entry_fn'] is not self.prepared:
                raise RuntimeError('The selected HOST entry was replaced by another owner')
            self.state['entry_fn'] = self.original
            self.prepared.cache.clear()
            self.closed = True
        finally:
            self._serial_gate.release()

    def report(self):
        return dict(hits=self.prepared.hits, builds=self.prepared.builds,
                    closed=self.closed, failed=self.failed,
                    public_submit_and_child_validation_unchanged=True,
                    scope='serial 83-parameter single-token program only')
