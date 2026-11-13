"""Strata GPU stream partitions and separate host ledgers; never subtract clocks."""
import collections
import json
from pathlib import Path
import statistics


def window(w):
    s=w['stamps']
    if len(s)!=48*33+4:raise ValueError('unexpected stamp geometry')
    segments=collections.Counter()
    boundaries=[(0,17,'dense_attention_router'),(17,18,'shared_quant'),(18,19,'plan_wait_copy'),
                (19,20,'resident_experts'),(20,21,'staging_wait_fetch'),(21,22,'staged_experts'),
                (22,23,'cpu_wait'),(23,24,'result_copy_combine')]
    for layer in range(48):
        for a,b,name in boundaries:
            start,end=s[layer*33+a],s[layer*33+b]
            if not start or end<start:raise ValueError(f'missing/nonmonotonic GPU stamp {layer}:{a}/{b}')
            segments[name]+=(end-start)/1e6
        if layer<47:
            gap=s[(layer+1)*33]-s[layer*33+24]
            if gap<0:raise ValueError('overlapping layers')
            segments['inter_layer_gap']+=gap/1e6
    head_start,head_end=s[48*33:48*33+2]
    if not head_start or head_end<head_start or head_start<s[47*33+24]:raise ValueError('invalid head stamps')
    segments['last_layer_to_head']=(head_start-s[47*33+24])/1e6
    segments['head']=(head_end-head_start)/1e6
    span=(head_end-s[0])/1e6
    if abs(sum(segments.values())-span)>1e-6:raise ValueError('partition fails closure')
    return dict(pos=w['pos'],gpu_stream_span_ms=span,segments_ms=dict(segments),tail_host_sync_ms=w['tail_sync_ms'])


def strata(path):
    groups=[];host=[];gpu=[]
    for line in Path(path).read_text().splitlines():
        e=json.loads(line)
        if e.get('kind')=='gpu_window':
            if e['rows']==1:gpu.append(window(e))
        elif e.get('request_boundary'):
            h=[x for x in host if x['rows']==1]
            if len(gpu)!=128 or len(h)!=128*48 or {x['pos'] for x in h}!={x['pos'] for x in gpu}:
                raise ValueError('request step count/identity mismatch')
            if len({(x['pos'],x['layer']) for x in h})!=128*48:raise ValueError('duplicate host layer')
            groups.append(dict(steps=128,gpu_mean_ms={k:statistics.mean(x['segments_ms'][k] for x in gpu) for k in gpu[0]['segments_ms']},
                               gpu_span_mean_ms=statistics.mean(x['gpu_stream_span_ms'] for x in gpu),
                               host_reach_wait_mean_ms=sum(x['reach_wait_ms'] for x in h)/128,
                               host_pool_callback_mean_ms=sum(x['pool_ms'] for x in h)/128,
                               host_tail_sync_mean_ms=statistics.mean(x['tail_host_sync_ms'] for x in gpu),
                               per_step=gpu))
            host=[];gpu=[]
        else:host.append(e)
    if host or gpu or len(groups)!=3:raise ValueError('incomplete request dump')
    return groups


def naiwa(path):
    from analyze import schedule
    result=schedule(path)
    groups=collections.defaultdict(dict)
    for line in Path(path).read_text().splitlines():
        e=json.loads(line)
        if 'phase' in e and e.get('tick',0)>=3 and e['phase'] in ('route_ready','dma_enqueued','cpu_submit_return','cpu_worker','cpu_compute_return','result_publish_begin','result_publish_return'):
            groups[e['layer'],e['tick']][e['phase']]=e['ns']
    def average(a,b):
        vals=[(v[b]-v[a])/1e6 for v in groups.values() if a in v and b in v]
        if not vals or min(vals)<0:raise ValueError('missing/invalid host interval')
        return dict(samples=len(vals),mean_ms_per_observed_interval=statistics.mean(vals),
                    summed_ms_per_48_layer_windows=sum(vals)/len(groups)*48)
    return dict(layer_windows=len(groups),route_to_dma=average('route_ready','dma_enqueued'),
                dma_to_cpu_submit_return=average('dma_enqueued','cpu_submit_return'),
                worker_to_return=average('cpu_worker','cpu_compute_return'),
                result_publication=average('result_publish_begin','result_publish_return'),
                warning='Host-clock means; conditional CPU-job mean x48 is NOT all-layer total. Route-ready excludes GPU dense/router and top-k read; worker interval includes activation acquisition. No request alignment; no exposed GPU wait inferred.')


if __name__=='__main__':
    import sys
    root=Path(sys.argv[1])
    print(json.dumps(dict(rounds=[dict(round=r,strata=strata(root/f'r{r}-strata/strata-ledger.jsonl'),
                                       naiwa=naiwa(root/f'r{r}-cpu_first/schedule.jsonl')) for r in (1,2,3)],
                          warning='Strata GPU segments form a disjoint compute-stream partition including waits; not device activity unions or pure kernel durations. Staging-wait-fetch combines flag wait, fetch and rebase. Host work overlaps GPU and MUST NOT be added. Profiler changes shared expert stream behavior. Naiwa data host-only; GPU capture from other runs NOT directly subtractable. No clean TPOT attribution.'),indent=2))
