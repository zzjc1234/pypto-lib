# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run a bounded task and clean up only processes inheriting its unique marker."""
import argparse
import json
import os
from pathlib import Path
import signal
import hashlib
import socket
import fcntl
import subprocess
import time
import uuid


def members(marker):
    result = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            if p.stat().st_uid == os.getuid() and marker in (p / 'environ').read_bytes().split(b'\0'):
                result.append(int(p.name))
        except OSError:
            pass
    return result


p = argparse.ArgumentParser()
p.add_argument('--timeout', type=int, required=True)
p.add_argument('--record', type=Path, required=True)
p.add_argument('--lease-file', type=Path)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
assert a.command and a.timeout > 0 and not a.record.exists()
token = 'DSPARK_EXPERIMENT_TOKEN=' + str(uuid.uuid4())
env = dict(os.environ)
env['DSPARK_EXPERIMENT_TOKEN'] = token.split('=', 1)[1]
process = subprocess.Popen(a.command, env=env, start_new_session=True)
process_stat = (Path('/proc') / str(process.pid) / 'stat').read_text()
process_start = int(process_stat.rsplit(')', 1)[1].split()[19])
timed_out = False
rc = 1
lease_copy = None
try:
    overall_deadline = time.monotonic() + a.timeout
    device_deadline = None
    while process.poll() is None:
        if a.lease_file is not None and a.lease_file.exists():
            candidate = json.loads(a.lease_file.read_text())
            if (candidate.get('owner_pid') == process.pid and candidate.get('owner_starttime') == process_start
                    and candidate.get('experiment_token_sha256') == hashlib.sha256(env['DSPARK_EXPERIMENT_TOKEN'].encode()).hexdigest()):
                lease_copy = candidate
                if device_deadline is None:
                    device_deadline = time.monotonic() + max(0, 1200 - (time.time() - candidate['created_epoch']))
        if time.monotonic() >= min(overall_deadline, device_deadline or overall_deadline):
            raise subprocess.TimeoutExpired(a.command, a.timeout)
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    rc = process.returncode
except subprocess.TimeoutExpired:
    timed_out = True
    rc = 124
finally:
    found = members(token.encode())
    for pid in found:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    if found:
        time.sleep(2)
    remaining = members(token.encode())
    for pid in remaining:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.wait(timeout=10)
    time.sleep(.2)
    leftover = members(token.encode())
    removed = False
    if not leftover and a.lease_file is not None:
        with a.lease_file.with_name('device-owner.lock').open('a+') as lease_lock:
            fcntl.flock(lease_lock, fcntl.LOCK_EX)
            if a.lease_file.exists():
                current = json.loads(a.lease_file.read_text())
                valid = (current.get('owner_pid') == process.pid and current.get('owner_starttime') == process_start
                         and current.get('hostname') == socket.gethostname()
                         and current.get('boot_id') == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                         and current.get('experiment_token_sha256') == hashlib.sha256(env['DSPARK_EXPERIMENT_TOKEN'].encode()).hexdigest())
                if valid and (lease_copy is None or current == lease_copy):
                    lease_copy = current
                    a.lease_file.unlink()
                    removed = True
    a.record.write_text(json.dumps({'pid': process.pid, 'starttime': process_start, 'returncode': rc,
                                    'timed_out': timed_out, 'owned_processes_cleaned': found,
                                    'owned_processes_remaining': leftover, 'other_processes_touched': False,
                                    'lease_removed_after_owned_cleanup': removed,
                                    'exact_lease_identity': lease_copy}, indent=2) + '\n')
raise SystemExit(rc if not leftover else 125)
