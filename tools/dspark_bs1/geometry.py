# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Absolute-anchor D metadata and dynamically sized target compressed history.

Only integer address/layout construction. Model inputs are still produced by
the actual request; no token/hidden/golden values are constructed here.
"""
import math

BANK_FIELDS = (
    'context_group_position_ids', 'context_group_slot_mapping',
    'query_group_position_ids', 'query_group_slot_mapping',
    'context_group_freqs_cos', 'context_group_freqs_sin',
    'query_freqs_cos', 'query_freqs_sin',
    'query_group_freqs_cos', 'query_group_freqs_sin',
)


def build_d_banks(torch, anchors, block_tables, rope_cos, rope_sin, *, shared=True):
    """Return the original10 bank fields, now indexed by absolute anchor.

For qualification anchors may be a selected list. Production passes range(N),
and its bank count N must cover every actually submitted parent/cycle anchor.
"""
    if not anchors or any(type(p)is not int or p<0 or p+5>=len(rope_cos) for p in anchors):
        raise ValueError('Real nonnegative anchors inside the actual RoPE table required')
    if tuple(block_tables.shape)!=(4,3,1,32768):raise ValueError('Original TP4/CP4 private ring tables required')
    n=len(anchors);a=torch.tensor(anchors,dtype=torch.int64)
    count=(a+1).clamp(max=128);first=a+1-count
    local=torch.full((n,4,128),-1,dtype=torch.int32)
    columns=torch.arange(128,dtype=torch.int64)
    local[:,0,:]=torch.where(columns[None,:]<count[:,None],first[:,None]+columns[None,:],-1).to(torch.int32)
    local[:,1:,0]=0  # Original independent computational filler owners.
    context_slots=torch.full((n,3,4,128),-1,dtype=torch.int64)
    for owner in range(4):
        pos=local[:,owner,:].to(torch.int64);safe=pos.clamp(min=0)
        for layer in range(3):
            page=block_tables[owner,layer,0].to(torch.int64)[safe//32]
            context_slots[:,layer,owner,:]=torch.where(pos>=0,page*32+safe%32,-1)
    query=torch.zeros((n,4,20),dtype=torch.int32)
    for owner in range(4):
        anchor=a if owner==0 else torch.zeros_like(a)
        query[:,owner,:5]=(anchor[:,None]+torch.arange(1,6,dtype=torch.int64)[None,:]).to(torch.int32)
    query_slots=torch.full((n,3,4,20),-1,dtype=torch.int64)
    for owner in range(4):
        pos=query[:,owner,:5].to(torch.int64)
        for layer in range(3):
            page=block_tables[owner,layer,0].to(torch.int64)[pos//32]
            query_slots[:,layer,owner,:5]=page*32+pos%32

    def replicated(x):return x.unsqueeze(0).repeat(4,*([1]*x.ndim)).contiguous()
    group_pos=local.reshape(n,512);group_query=query.reshape(n,80)
    result={
        'context_group_position_ids_bank':replicated(group_pos),
        'context_group_slot_mapping_bank':replicated(context_slots.reshape(n,3,512)),
        'query_group_position_ids_bank':replicated(group_query),
        'query_group_slot_mapping_bank':replicated(query_slots.reshape(n,3,80)),
    }
    def rope(table,pos):return table.index_select(0,pos.clamp(min=0).to(torch.int64).reshape(-1)).reshape(*pos.shape,64)
    for suffix,table in [('cos',rope_cos),('sin',rope_sin)]:
        result['context_group_freqs_'+suffix+'_bank']=replicated(rope(table,group_pos))
        result['query_group_freqs_'+suffix+'_bank']=replicated(rope(table,group_query))
        result['query_freqs_'+suffix+'_bank']=rope(table,query).permute(1,0,2,3).contiguous()
    if shared:
        for tensor in result.values():tensor.share_memory_()
    return result


def target_layout(max_start):
    """Capacity through every S6 position start..start+5, no compressed overwrite."""
    if type(max_start)is not int or not 0<=max_start<16379:
        raise ValueError('Existing actual16K table/RoPE ABI; speculative tail must fit')
    source_rows=max_start+6
    hca_rows=source_rows//128;csa_rows=source_rows//4
    # Keep at least the original2 pages: unchanged original short fixtures.
    counts=[2,max(2,hca_rows),max(2,math.ceil(csa_rows/32)),max(2,math.ceil(csa_rows/32)),17,4,4]
    table_specs=[('block_table',128,0,True),('hca_cmp_block_table',128,1,False),
                 ('csa_cmp_block_table',128,2,False),('idx_block_table',128,3,False),
                 ('hca_compress_state_block_table',2048,4,True),
                 ('csa_compress_state_block_table',4096,5,True),
                 ('csa_inner_compress_state_block_table',4096,6,True)]
    tables={name:[i%counts[index] if rolling else i if i<counts[index] else -1 for i in range(width)]
            for name,width,index,rolling in table_specs}
    pools={
        'kv_cache':([43*counts[0],128,1,512],'BF16'),
        'hca_cmp_kv':([20*counts[1],1,1,512],'BF16'),
        'csa_cmp_kv':([21*counts[2],32,1,512],'BF16'),
        'idx_kv_cache':([21*counts[3],32,1,128],'I8'),
        'idx_kv_scale':([21*counts[3],32,1,1],'F32'),
        'hca_compress_state':([20*counts[4],8,1024],'F32'),
        'csa_compress_state':([21*counts[5],4,2048],'F32'),
        'csa_inner_compress_state':([21*counts[6],4,512],'F32'),
    }
    sizes={'BF16':2,'I8':1,'F32':4}
    return dict(max_start=max_start,last_speculative_position=max_start+5,
                block_counts=counts,tables=tables,
                cache_pools={k:dict(shape=s,dtype=d,bytes=math.prod(s)*sizes[d]) for k,(s,d) in pools.items()},
                bytes_per_rank=sum(math.prod(s)*sizes[d] for s,d in pools.values()),
                compressed_history_rows={'hca':hca_rows,'csa':csa_rows,'indexer':csa_rows})
