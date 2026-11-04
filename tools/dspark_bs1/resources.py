# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Read actual dual-chip placement and claim only a ready experiment's devices."""

import ast
from collections import defaultdict
from contextlib import contextmanager
import ctypes
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import uuid

from .smi import bind_logical, parse_smi

GiB = 2**30


def integer(node, names=None):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.Name) and node.id in (names or {}):
        return names[node.id]
    if isinstance(node, ast.BinOp):
        left, right = integer(node.left, names), integer(node.right, names)
        for kind, operation in [(ast.Add, lambda: left + right), (ast.Sub, lambda: left - right),
                                (ast.Mult, lambda: left * right), (ast.FloorDiv, lambda: left // right)]:
            if isinstance(node.op, kind):
                return operation()
    raise ValueError('Communication-budget expression is not a static integer')


def communication_bytes(packages):
    result = [0] * 8
    for path in packages.values():
        metadata = json.loads((Path(path) / 'distributed_meta.json').read_text())
        count = len(metadata['distributed_config']['device_ids'])
        if count != 8:
            raise ValueError('Communication topology must match the actual eight-rank metadata')
        names = {'world_size': count}
        tree = ast.parse((Path(path) / 'orchestration/host_orch.py').read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            values = {key.arg: key.value for key in node.keywords}
            if 'workers' not in values or 'window_size' not in values:
                continue
            worker_nodes = values['workers']
            if not isinstance(worker_nodes, ast.List):
                raise ValueError('Actual domain workers must be a static list')
            workers = []
            for item in worker_nodes.elts:
                if isinstance(item, ast.Starred) and isinstance(item.value, ast.Call) and isinstance(item.value.func, ast.Name) and item.value.func.id == 'range':
                    limits = [integer(value, names) for value in item.value.args]
                    workers.extend(range(*limits))
                else:
                    workers.append(integer(item, names))
            size = integer(values['window_size'], names)
            if size <= 0 or any(type(rank) is not int or not 0 <= rank < 8 for rank in workers):
                raise ValueError('Invalid actual communication domain')
            for rank in workers:
                result[rank] += size
    return result


def mapping():
    library = ctypes.CDLL('libascend_hal.so')
    forward = library.halGetPhyDevIdByLogicDevId
    reverse = library.halGetLogicDevIdByPhyDevId
    for function in [forward, reverse]:
        function.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)]
        function.restype = ctypes.c_int
    rows = []
    for logical in range(16):
        physical, returned = ctypes.c_uint(), ctypes.c_uint()
        rc = int(forward(logical, ctypes.byref(physical)))
        back = int(reverse(physical.value, ctypes.byref(returned))) if rc == 0 else -1
        if rc or back or returned.value != logical:
            raise RuntimeError('Actual HAL device mapping failed')
        rows.append(dict(logical_device_id=logical, smi_physical_id=physical.value,
                         rc=rc, reverse_rc=back, roundtrip_logical=returned.value))
    return dict(mapping=rows)


def snapshot():
    if os.environ.get('ASCEND_RT_VISIBLE_DEVICES'):
        raise RuntimeError('Unverified device remapping is not admitted')
    raw = subprocess.check_output(['npu-smi', 'info'], text=True, stderr=subprocess.STDOUT, timeout=10)
    return dict(hostname=socket.gethostname(), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                epoch=time.time(), devices=bind_logical(parse_smi(raw), mapping()), raw=raw)


def select(snapshot_, planned):
    if len(planned) != 8 or any(type(size) is not int or size <= 0 for size in planned):
        raise ValueError('Eight actual rank budgets required')
    boards = defaultdict(list)
    for row in snapshot_['devices'].values():
        boards[row['npu_id']].append(row)
    pairs = [sorted(rows, key=lambda x: x['chip_id']) for _, rows in sorted(boards.items())
             if len(rows) == 2 and {r['chip_id'] for r in rows} == {0, 1}]
    for selected in itertools.combinations(pairs, 4):
        rows = [row for pair in selected for row in pair]
        if all(row['idle'] and row['health'] == 'OK' and row['free_bytes'] > size + 12 * GiB
               for row, size in zip(rows, planned, strict=True)):
            return rows
    raise RuntimeError('No four complete two-chip boards satisfy the ready experiment capacity')


def topology(rows):
    raw = subprocess.check_output(['npu-smi', 'info', '-t', 'topo'], text=True,
                                  stderr=subprocess.STDOUT, timeout=10)
    table = {}
    for line in raw.splitlines():
        tokens = line.split()
        if tokens and re.fullmatch(r'Phy-ID\d+', tokens[0]) and len(tokens) == 17:
            table[int(tokens[0][6:])] = tokens[1:]
    if set(table) != set(range(16)):
        raise RuntimeError('Actual sixteen-chip topology is incomplete')
    for i in range(0, 8, 2):
        x, y = rows[i]['smi_physical_id'], rows[i + 1]['smi_physical_id']
        if table[x][y] != 'SIO' or table[y][x] != 'SIO':
            raise RuntimeError('Selected physical board is not a verified symmetric SIO pair')
    return raw


def starttime(pid):
    text = (Path('/proc') / str(pid) / 'stat').read_text()
    return int(text.rsplit(')', 1)[1].split()[19])


def owned_children():
    marker = os.environ.get('DSPARK_EXPERIMENT_TOKEN')
    if not marker:
        raise RuntimeError('An owned process marker is required before Worker construction')
    expected = ('DSPARK_EXPERIMENT_TOKEN=' + marker).encode()
    found = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            if path.stat().st_uid == os.getuid() and expected in (path / 'environ').read_bytes().split(b'\0'):
                found.append(dict(pid=int(path.name), starttime=starttime(int(path.name))))
        except (OSError, ValueError):
            continue
    return found


@contextmanager
def claim(planned, lease_file, *, wait_seconds=90000, execution_lock=None):
    """Wait without HBM allocation; claim immediately before actual Worker setup."""
    lease_file = Path(lease_file).resolve()
    if not os.environ.get('DSPARK_EXPERIMENT_TOKEN'):
        raise RuntimeError('The owned-process supervisor must set its marker before execution')
    lease_file.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + wait_seconds
    lock = lease_file.with_name('device-owner.lock').open('a+')
    lane = Path(execution_lock).open('a+') if execution_lock is not None else None
    record = None
    while record is None:
        try:
            if lane is not None:
                fcntl.flock(lane, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if lease_file.exists():
                raise RuntimeError('An existing device lease is owned elsewhere')
            current = snapshot()
            rows = select(current, planned)
            links = topology(rows)
            record = dict(owner_pid=os.getpid(), owner_starttime=starttime(os.getpid()),
                          hostname=current['hostname'], boot_id=current['boot_id'], token=uuid.uuid4().hex,
                          purpose='DSPARK_BS1_COMPLETE_REQUEST_REPRODUCTION',
                          experiment_token_sha256=hashlib.sha256(os.environ['DSPARK_EXPERIMENT_TOKEN'].encode()).hexdigest(),
                          device_ids=[r['logical_device_id'] for r in rows],
                          physical_boards=[r['npu_id'] for r in rows], chips=[r['chip_id'] for r in rows],
                          capacity=current, topology=links, created_epoch=time.time())
            temporary = lease_file.with_name(lease_file.name + '.' + uuid.uuid4().hex + '.tmp')
            try:
                temporary.write_text(json.dumps(record, indent=2) + '\n')
                os.link(temporary, lease_file)
            finally:
                temporary.unlink(missing_ok=True)
        except (RuntimeError, BlockingIOError, FileExistsError):
            record = None
            fcntl.flock(lock, fcntl.LOCK_UN)
            if lane is not None:
                fcntl.flock(lane, fcntl.LOCK_UN)
            if time.monotonic() >= deadline:
                lock.close()
                if lane is not None:
                    lane.close()
                raise TimeoutError('No suitable physical board group became available')
            time.sleep(min(10, max(0, deadline - time.monotonic())))
    try:
        yield record
    finally:
        remaining = owned_children()
        if not remaining and lease_file.exists() and json.loads(lease_file.read_text()) == record:
            lease_file.unlink()
        elif remaining:
            lease_file.with_name('release-pending.json').write_text(json.dumps(
                dict(owner_pid=os.getpid(), owner_starttime=record['owner_starttime'],
                     remaining=remaining, lease_removed=False), indent=2) + '\n')
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        if lane is not None:
            fcntl.flock(lane, fcntl.LOCK_UN)
            lane.close()


def actual_free(worker, devices, planned=None):
    current = snapshot()
    for rank, device in enumerate(devices):
        row = current['devices'][device]
        info = worker.device_memory_info(rank)
        free = min(row['free_bytes'], info[0] if isinstance(info, tuple) else info.free_bytes)
        required = 2 * GiB + (planned[rank] if planned is not None else 0)
        if row['health'] != 'OK' or free <= required:
            raise RuntimeError('Actual post-Worker device capacity no longer satisfies the experiment')
    return current
