# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Actual complete-request B1 execution, preserving decode-wall accounting."""

import csv
import json
import statistics
import time

import torch

from models.deepseek_v4_flash_lowlat.stream_controller import run_request

from .buffers import CACHES, public_arguments, set_single_step
from .checks import check_commit_oracle, check_packets
from .frames import NativeFrames
from .tensors import read_shard, upload
from .prepared import SerialSingleEntry


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class Requests:
    def __init__(self, worker, programs, metadata, common, outputs, single, hosts,
                 plan, records, eos, config, output, *, device_started):
        self.worker, self.programs, self.metadata = worker, programs, metadata
        self.common, self.outputs, self.single, self.plan = common, outputs, single, plan
        self.records, self.eos, self.config, self.output = records, set(eos), config, output
        self.device_started = device_started
        self.epoch = 1
        self.prefix_snapshot = {}
        self.requests = []
        self.fixed = []
        self.warmup = []
        self.cases = []
        self.adapters = {
            name: NativeFrames(worker, {1: programs['one'], 2: programs[key]},
                               {1: metadata['one'], 2: metadata[key]}, common, outputs, hosts, config)
            for name, key in [('oldStreaming', 'two'), ('newStreaming', 'rsag')]
        }
        self.single_arguments = public_arguments(metadata['single'], single)
        self.single_entry = SerialSingleEntry(worker, programs['single'])

    def save(self):
        atomic_json(self.output / 'result.json', dict(requests=self.requests, cases=self.cases,
                    fixed=self.fixed, warmup=self.warmup, epoch=self.epoch,
                    single_entry_reuse=self.single_entry.report(),
                    protocol=dict(batch_size=1, concurrency=1, drafts=5, max_new_tokens=2048,
                                  temperature=0, prequeue_threshold=24, two_cycle_threshold=7),
                    numerical_qualification=False, performance_reproduction='PENDING_FINAL_CLOSEOUT'))

    def require_idle(self):
        if not all(adapter.idle() for adapter in self.adapters.values()):
            raise RuntimeError('Full state/cache operations require all accepted parents to be consumed')

    def reset(self, prefix):
        self.require_idle()
        for name in CACHES:
            root = self.common['s_' + name]
            spec = self.plan.specs['s_' + name]
            for rank, shard in enumerate(root.shards):
                host = self.prefix_snapshot[name][rank] if prefix else self.plan.bundle.shard(spec, rank)
                upload(self.worker, shard, host, rank=rank, fill=spec.get('fill', 0))
        for rank, shard in enumerate(self.common['d_kv_caches'].shards):
            upload(self.worker, shard, None, rank=rank)
        for row in self.metadata['single']['params']:
            if row['direction'] == 'Out':
                root = self.single[row['name'].split('__ssa')[0]]
                for rank, shard in enumerate(root.shards):
                    upload(self.worker, shard, None, rank=rank)

    def single_step(self, position, token, *, features):
        set_single_step(self.worker, self.single, position, token)
        self.single_entry.submit_result(self.single_arguments, config=self.config)
        ids = read_shard(self.worker, self.single['sampled_ids'])
        hidden = read_shard(self.worker, self.single['dspark_target_hidden_out']) if features else None
        return int(ids[0, 0]), None if hidden is None else hidden[0].clone()

    def state_upload(self, state):
        value = torch.tensor(state, dtype=torch.int64)
        for rank, shard in enumerate(self.common['s_request_state'].shards):
            self.worker.copy_to(shard.data_ptr, value.data_ptr(), 32, worker_id=rank)

    def final_state(self, expected):
        self.require_idle()
        actual = [read_shard(self.worker, self.common['s_request_state'], rank).tolist() for rank in range(8)]
        if not all(value == expected for value in actual):
            raise RuntimeError('The actual final state differs across the eight target ranks')
        return actual

    def seed_history(self, hidden, prefix_length):
        first = max(0, prefix_length - 128)
        data = torch.stack([hidden[position] for position in range(first, prefix_length)]).contiguous()
        root = self.common['s_history_ring'].shards[0]
        physical = first % 128
        head = min(len(data), 128 - physical)
        self.worker.copy_to(root.data_ptr, data.data_ptr(), head * 24576,
                            dst_offset=physical * 24576, worker_id=0)
        if head < len(data):
            self.worker.copy_to(root.data_ptr, data.data_ptr(), (len(data) - head) * 24576,
                                src_offset=head * 24576, worker_id=0)

    def fixed_run(self, hidden, length, first, mode, cycles, frame, label, functional):
        self.reset(True)
        self.seed_history(hidden, length)
        initial = [self.epoch, length - 1, 2047, first]
        self.state_upload(initial)
        before = read_shard(self.worker, self.common['s_history_ring']) if functional else None
        adapter = self.adapters[mode]
        adapter.submit(frame, cycles).result()
        raw = adapter.capture(frame, cycles)
        packets = [{'control': raw['control'][0][i], 'emitted': raw['emitted'][0][i]} for i in range(cycles)]
        captures = []
        if functional:
            values = {name: read_shard(self.worker, self.outputs[frame][name]) for name in (
                'm_draft_token_ids', 'm_confidence_probs', 's_sampled_ids', 's_dspark_target_hidden_out')}
            for bank in range(cycles):
                captures.append(dict(proposal=values['m_draft_token_ids'][bank, 0].clone(),
                                     confidence_probs=values['m_confidence_probs'][bank, 0].clone(),
                                     sampled_ids=values['s_sampled_ids'][bank].clone(),
                                     target_features=values['s_dspark_target_hidden_out'][bank].clone()))
            after = read_shard(self.worker, self.common['s_history_ring'])
            path = self.output / (label + '.pt')
            temporary = path.with_suffix('.tmp')
            torch.save(dict(captures=captures, packets=packets, initial_state=initial,
                            initial_ring=before, actual_ring=after), temporary)
            temporary.replace(path)
        record = dict(label=label, mode=mode, frame=frame, cycles=cycles, initial_state=initial, raw=raw)
        (self.fixed if functional else self.warmup).append(record)
        self.save()
        checked = check_packets(initial, packets, minimum_two=7)
        if functional:
            if not all(bool(torch.isfinite(value.float()).all()) for c in captures for value in c.values()):
                raise RuntimeError('Nonfinite fixed output; capture was saved before the gate')
            record['commit_oracle'] = check_commit_oracle(initial, packets, captures, before, after)
            record['capture'] = str(path)
        adapter.consume(frame)
        adapter.release(frame)
        record['final_state_all8'] = self.final_state(checked['final_state'])
        self.epoch = checked['final_state'][0]
        self.save()

    def native_request(self, question, mode, hidden, length, first, ttft):
        self.reset(True)
        outputs, packets, events, deliveries = [first], [], [], []
        current = [self.epoch, length - 1, 2047, first]
        initial = list(current)
        record = dict(request_id=question['dataset'] + '-' + question['question_id'] + '-' + mode,
                      dataset=question['dataset'], question_id=question['question_id'], backend=mode,
                      prompt_tokens=length, token_ids=outputs, TTFT_seconds=ttft,
                      TTFT_scope='measured common same-request live prefill',
                      initial_state=initial, packet_batches=packets, events=events, deliveries=deliveries,
                      status='RUNNING', max_new_tokens=2048, temperature=0)
        self.requests.append(record)
        adapter = self.adapters[mode]
        captured = {}
        started = time.monotonic_ns()
        raw_result = None
        try:
            if first not in self.eos:
                self.seed_history(hidden, length)
                self.state_upload(current)

                def capture(frame, cycles):
                    nonlocal current
                    raw = adapter.capture(frame, cycles)
                    group = [{'control': raw['control'][0][i], 'emitted': raw['emitted'][0][i]}
                             for i in range(cycles)]
                    row = dict(frame=frame, cycles=cycles, initial_state=list(current), raw=raw, packets=group)
                    packets.append(row)
                    try:
                        checked = check_packets(current, group, minimum_two=7)
                    except BaseException:
                        self.save()
                        raise
                    current = checked['final_state']
                    captured['frame'] = frame
                    return checked['outputs']

                def deliver(ids):
                    outputs.extend(ids)
                    deliveries.append(dict(ids=list(ids), request_ns=time.monotonic_ns() - started))
                    adapter.consume(captured.pop('frame'))

                def discard(ids):
                    deliveries.append(dict(discarded_after_EOS=list(ids)))
                    adapter.consume(captured.pop('frame'))

                raw_result = run_request(token_budget=2047, eos_token_ids=self.eos,
                                         submit=adapter.submit, capture_and_validate=capture,
                                         release_frame=adapter.release, deliver=deliver,
                                         discard_after_eos=discard, events=events)
            ended = time.monotonic_ns()
            count = len(outputs) - 1
            record.update(status='COMPLETE_SAVED_BEFORE_NUMERICAL_GATE',
                          actual_generated_tokens=len(outputs), actual_decode_tokens=count,
                          decode_wall_seconds=(ended - started) / 1e9,
                          TPOT_ms=(ended - started) / 1e6 / count if count else None,
                          finish_reason='stop' if outputs[-1] in self.eos else 'length',
                          initial_seed_and_state_upload_in_decode_clock=first not in self.eos,
                          raw_controller=raw_result, numerical_qualification=False)
            if not 1 <= len(outputs) <= 2048 or (record['finish_reason'] == 'length' and len(outputs) != 2048):
                raise RuntimeError('Request did not reach its actual EOS or full generation budget')
            if raw_result:
                record['final_state_all8'] = self.final_state(current)
                self.epoch = current[0]
            self.save()
            return record
        except BaseException as error:
            record.update(status='FAILED_PRESERVED', error=repr(error))
            self.save()
            raise

    def k0_request(self, question, length, first, ttft):
        self.reset(True)
        outputs = [first]
        record = dict(request_id=question['dataset'] + '-' + question['question_id'] + '-K0',
                      dataset=question['dataset'], question_id=question['question_id'], backend='K0',
                      prompt_tokens=length, token_ids=outputs, TTFT_seconds=ttft,
                      TTFT_scope='measured common same-request live prefill', status='RUNNING')
        self.requests.append(record)
        started = time.monotonic_ns()
        while len(outputs) < 2048 and outputs[-1] not in self.eos:
            token, _ = self.single_step(length + len(outputs) - 1, outputs[-1], features=False)
            outputs.append(token)
        ended = time.monotonic_ns()
        count = len(outputs) - 1
        record.update(status='COMPLETE_SAVED_BEFORE_NUMERICAL_GATE', actual_generated_tokens=len(outputs),
                      actual_decode_tokens=count, decode_wall_seconds=(ended - started) / 1e9,
                      TPOT_ms=(ended - started) / 1e6 / count if count else None,
                      finish_reason='stop' if outputs[-1] in self.eos else 'length', numerical_qualification=False)
        self.save()
        return record

    def run(self):
        durations = []
        for index, question in enumerate(self.records):
            elapsed = time.monotonic() - self.device_started
            if elapsed >= 960 or 1200 - elapsed < max([85, *durations]) + 60:
                self.save()
                raise TimeoutError('No new complete request triplet fits the remaining device window')
            began = time.monotonic()
            self.reset(False)
            hidden = {}
            prompt = question['prompt_ids']
            prefix_start = time.monotonic_ns()
            for position, token in enumerate(prompt):
                first, hidden[position] = self.single_step(position, token, features=True)
            ttft = (time.monotonic_ns() - prefix_start) / 1e9
            self.prefix_snapshot = {name: [read_shard(self.worker, self.common['s_' + name], rank)
                                          for rank in range(8)] for name in CACHES}
            if index == 0:
                for i, (mode, cycles, frame) in enumerate([
                    ('oldStreaming', 2, 0), ('newStreaming', 1, 0), ('newStreaming', 2, 1),
                    ('oldStreaming', 2, 1), ('newStreaming', 2, 0)]):
                    self.fixed_run(hidden, len(prompt), first, mode, cycles, frame, f'warm{i}', False)
                for i, mode in enumerate(['oldStreaming', 'newStreaming', 'oldStreaming', 'newStreaming']):
                    self.fixed_run(hidden, len(prompt), first, mode, 2, i // 2, f'functional{i}', True)
            triplet = [self.k0_request(question, len(prompt), first, ttft)]
            order = ['oldStreaming', 'newStreaming'] if index % 2 == 0 else ['newStreaming', 'oldStreaming']
            for mode in order:
                triplet.append(self.native_request(question, mode, hidden, len(prompt), first, ttft))
            equality = {r['backend']: r['token_ids'] == triplet[0]['token_ids'] for r in triplet[1:]}
            self.cases.append(dict(dataset=question['dataset'], question_id=question['question_id'],
                                   output_ID_equality=equality, same_native_IDs=triplet[1]['token_ids'] == triplet[2]['token_ids']))
            self.save()
            durations.append(time.monotonic() - began)
        self.plan.bundle.recheck()
        return self.export()

    def export(self):
        summary = {}
        for mode in ['K0', 'oldStreaming', 'newStreaming']:
            selected = [r for r in self.requests if r['backend'] == mode and r['status'].startswith('COMPLETE')]
            positive = [r for r in selected if r['actual_decode_tokens']]
            summary[mode] = dict(requests=len(selected),
                                 median_TPOT_ms=statistics.median(r['TPOT_ms'] for r in positive) if positive else None,
                                 token_weighted_TPOT_ms=1000 * sum(r['decode_wall_seconds'] for r in positive) /
                                 sum(r['actual_decode_tokens'] for r in positive) if positive else None)
        atomic_json(self.output / 'per-request.json', self.requests)
        fields = ['request_id', 'dataset', 'question_id', 'backend', 'prompt_tokens',
                  'actual_generated_tokens', 'actual_decode_tokens', 'TTFT_seconds',
                  'decode_wall_seconds', 'TPOT_ms', 'finish_reason', 'status']
        with (self.output / 'per-request.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self.requests)
        result = dict(summary=summary, complete_requests=len(self.requests),
                      request_ID_gates_passed=all(all(c['output_ID_equality'].values()) for c in self.cases),
                      full_Torch_numerical_qualification=False, all_samples_retained=True)
        atomic_json(self.output / 'summary.json', result)
        return result
