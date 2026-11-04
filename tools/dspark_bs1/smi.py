# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import re

def parse_smi(raw):
    devices={};idle_boards=set();processes=[];pending=None;proc_section=False
    for line in raw.splitlines():
        if 'Process id' in line:
            proc_section=True;continue
        if not line.startswith('|'):continue
        clean=re.search(r'No running processes found in NPU\s+(\d+)',line)
        if clean:
            idle_boards.add(int(clean.group(1)));continue
        cols=[x.strip() for x in line.split('|')]
        if proc_section:
            if not any(cols):continue
            m=re.fullmatch(r'(\d+)\s+(\d+)',cols[1]) if len(cols)>3 else None
            if m and cols[2].isdigit():
                processes.append({'npu_id':int(m.group(1)),'chip_id':int(m.group(2)),'pid':int(cols[2]),'raw':line});continue
            raise ValueError('Unrecognized process-table row: '+line)
        if len(cols)==5:
            m=re.fullmatch(r'(\d+)\s+(Ascend\S+)',cols[1])
            if m:
                pending={'npu_id':int(m.group(1)),'name':m.group(2),'health':cols[2]};continue
            ids=re.fullmatch(r'(\d+)\s+(\d+)',cols[1]);mem=re.fullmatch(r'(\d+)\s+(\d+)\s*/\s*(\d+)\s+(\d+)\s*/\s*(\d+)',cols[3])
            if ids and mem and re.fullmatch(r'[0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9]',cols[2]):
                if pending is None:raise ValueError('Chip row has no health header')
                chip,phy=map(int,ids.groups());used,total=int(mem.group(4)),int(mem.group(5))
                if phy in devices or not 0<=used<=total or total<=0:raise ValueError('Invalid or duplicate physical memory row')
                devices[phy]=dict(pending,chip_id=chip,smi_physical_id=phy,bus_id=cols[2].upper(),aicore_percent=int(mem.group(1)),hbm_used_bytes=used*2**20,hbm_total_bytes=total*2**20,free_bytes=(total-used)*2**20)
                pending=None
    if len(devices)!=16 or set(devices)!=set(range(16)):raise ValueError('Expected16 actual physical die IDs')
    if len({(x['npu_id'],x['chip_id']) for x in devices.values()})!=16:raise ValueError('Ambiguous board/chip coordinates')
    for p in processes:
        if p['npu_id'] in idle_boards:raise ValueError('Contradictory idle/process record')
        if (p['npu_id'],p['chip_id']) not in {(x['npu_id'],x['chip_id']) for x in devices.values()}:raise ValueError('Unknown process device')
    seen_boards=idle_boards|{p['npu_id'] for p in processes}
    for row in devices.values():
        row['processes']=[p for p in processes if p['npu_id']==row['npu_id'] and p['chip_id']==row['chip_id']]
        row['process_visibility_complete']=row['npu_id'] in seen_boards
        row['idle']=row['process_visibility_complete'] and not row['processes'] and row['aicore_percent']==0
    return devices


def bind_logical(devices,mapping):
    result={}
    for x in mapping['mapping']:
        if x['rc']!=0 or x['reverse_rc']!=0 or x['roundtrip_logical']!=x['logical_device_id']:raise ValueError('Bad actual HAL roundtrip')
        logical=x['logical_device_id'];phy=x['smi_physical_id']
        if logical in result or phy not in devices:raise ValueError('Bad logical mapping')
        result[logical]=dict(devices[phy],logical_device_id=logical)
    if len(result)!=16:raise ValueError('All16 logical mappings required')
    return result

