#!/usr/bin/env python3
"""Analyze A/B manifests, explicit GPU windows, or legacy routing traces."""
import argparse
import ast
import collections
import csv
import json
from pathlib import Path
import re
import statistics
from common import digest


def union(intervals):
    out=[]
    for a,b in sorted(intervals):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1]=(out[-1][0], max(b,out[-1][1]))
        else:
            out.append((a,b))
    return out


def length(intervals):
    return sum(b-a for a,b in union(intervals))


def gpu(path, start, end, steps, gap_min):
    if end <= start or steps <= 0:
        raise ValueError('explicit positive window and actual decode step count required')
    events=[]
    with open(path, newline='') as f:
        rows=csv.reader(f)
        for header in rows:
            if 'Start (ns)' in header and 'Duration (ns)' in header and 'Name' in header:
                break
        else:
            raise ValueError('nsys cuda_gpu_trace CSV header not found')
        for row in rows:
            d=dict(zip(header,row))
            try:
                a=int(d['Start (ns)']); b=a+int(d['Duration (ns)'])
            except (KeyError,ValueError):
                continue
            if a < end and b > start:
                name=d['Name']
                kind='copy' if 'CUDA memcpy' in name else ('spin' if 'doorbell_wait' in name else 'kernel')
                events.append((max(a,start),min(b,end),kind,name))
    all_iv=union((a,b) for a,b,_,_ in events)
    busy=length(all_iv)
    groups={k:[(a,b) for a,b,c,_ in events if c==k] for k in ('copy','spin','kernel')}
    # Sum durations and union are explicitly different quantities.
    scale=1e6*steps
    result={'window_ns':[start,end], 'actual_decode_steps':steps,
            'span_ms_per_step':(end-start)/scale,'busy_union_ms_per_step':busy/scale,
            'idle_ms_per_step':(end-start-busy)/scale, 'classes':{}}
    for k,iv in groups.items():
        without=[(a,b) for a,b,c,_ in events if c!=k]
        result['classes'][k]={'duration_sum_ms_per_step':sum(b-a for a,b in iv)/scale,
                             'union_ms_per_step':length(iv)/scale,
                             'exclusive_ms_per_step':(busy-length(without))/scale}
    counts=collections.Counter(); totals=collections.Counter()
    # Boundary names come from the event actually extending each merged component's end.
    components=[]
    for a,b,c,n in sorted(events):
        if components and a <= components[-1][1]:
            if b > components[-1][1]:
                components[-1][1]=b; components[-1][3]=n
        else:
            components.append([a,b,n,n])
    for prev,nxt in zip(components,components[1:]):
        gap=nxt[0]-prev[1]
        if gap >= gap_min:
            key=(prev[3],nxt[2]);counts[key]+=1;totals[key]+=gap
    result['gap_classes']=[{'before':k[0],'after':k[1], 'count':counts[k],
                            'ms_per_step':v/scale} for k,v in totals.most_common()]
    result['warning']='Explicit window must be decode-only on this capture clock. Adjacency is not causality. Exclusive occupancy is not a TPOT speedup bound. Copy engine busy does not imply compute stream busy.'
    return result


def routes(path):
    records=[]
    for line in Path(path).read_text().splitlines():
        if line.startswith('MOE_WORK '):
            records.append(json.loads(line[len('MOE_WORK '):]))
    keys=[(r['layer'],r['tick']) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate layer/tick: concatenate neither processes nor tick resets')
    return records


def compare_routes(a,b):
    left,right=routes(a),routes(b)
    result={'left_records':len(left),'right_records':len(right),'equal':left==right,
            'left_sha256':digest(left),'right_sha256':digest(right),
            'scope':'ordered records in entire supplied logs, including prefill if logged; activation identity NOT measured'}
    for i in range(max(len(left),len(right))):
        l=left[i] if i<len(left) else None; r=right[i] if i<len(right) else None
        if l!=r:
            result['first_difference']={'index':i,'left':l,'right':r}
            break
    if not left or not right:
        result['equal']=False;result['error']='missing MOE_WORK records; equality unproven'
    return result


def cpu(path):
    tasks=[]; windows=[]; previous=0
    for line in Path(path).read_text().splitlines():
        if line.startswith('CPU_TASK '):
            fields=dict(re.findall(r'(\w+)=([^ ]+)',line))
            required=('layer','tick','rows','submit_us','worker_us','compute_return_us')
            if not all(k in fields for k in required):
                raise ValueError('incomplete CPU_TASK record')
            vals={k:int(fields[k]) for k in required}
            vals['submit_to_worker_us']=vals['worker_us']-vals['submit_us']
            vals['worker_to_return_us']=vals['compute_return_us']-vals['worker_us']
            if min(vals['submit_to_worker_us'],vals['worker_to_return_us']) < 0:
                raise ValueError('negative task interval')
            tasks.append(vals)
        elif line.startswith('ROUND_STATS '):
            fields=line.split()
            if len(fields)==7 and fields[1].isdigit():
                cumulative=int(fields[1]); delta=cumulative-previous
                if delta<=0:
                    raise ValueError('ROUND_STATS cumulative pass count reset/duplicate')
                windows.append({'cumulative_passes':cumulative,'window_passes':delta,
                                'cpu_job_wall_ms':float(fields[6]),
                                'cpu_job_wall_ms_per_pass':float(fields[6])/delta,
                                'boundary_window':previous==0})
                previous=cumulative
    def distribution(key):
        values=sorted(t[key] for t in tasks)
        if not values:return None
        return {'count':len(values),'mean':statistics.mean(values),'p50':statistics.median(values),
                'p95':values[max(0,__import__('math').ceil(.95*len(values))-1)],'max':max(values)}
    return {'tasks':tasks,'submit_to_worker_us':distribution('submit_to_worker_us'),
            'worker_to_return_us':distribution('worker_to_return_us'),'round_windows':windows,
            'warning':'CPU_TASK timing is diagnostic. Worker->return includes activation acquisition and fused compute, not pure kernel compute. Task sums are not exposed GPU wait. First/boundary windows may include prefill; four-pass counters can straddle in-flight jobs. RET_DBG window means are not task percentiles.'}


def schedule(path):
    groups=collections.defaultdict(dict)
    dropped=0; flushes=0
    for line in Path(path).read_text().splitlines():
        e=json.loads(line)
        if e.get('flush'):
            flushes+=1; dropped+=e['dropped']; continue
        if e['phase'] in ('fill_publish','promotion_publish','resident_use'):
            continue
        if e['phase'] in ('staged_h2d','cache_fill_h2d'):
            if type(e.get('bytes')) is not int or e['bytes']<=0:
                raise ValueError('invalid transfer bytes')
            continue
        key=(e['layer'],e['tick'])
        if e['phase'] in groups[key]:
            raise ValueError(f'duplicate schedule phase {key}')
        groups[key][e['phase']]=e['ns']
    if not groups or not flushes or dropped:
        raise ValueError('missing/unflushed/dropped schedule events')
    for key,g in groups.items():
        if 'route_ready' not in g or 'dma_enqueued' not in g or 'cpu_submit_return' not in g:
            raise ValueError(f'incomplete layer {key}')
        if 'cpu_submit' in g and not all(k in g for k in ('cpu_worker','cpu_compute_return')):
            raise ValueError(f'incomplete CPU job {key} (generic path is not instrumented)')
    result={}
    for name,a,b in [('route_to_submit','route_ready','cpu_submit'),
                     ('route_to_worker','route_ready','cpu_worker'),
                     ('route_to_return','route_ready','cpu_compute_return'),
                     ('worker_to_return','cpu_worker','cpu_compute_return'),
                     ('route_to_dma','route_ready','dma_enqueued'),
                     ('result_publish','result_publish_begin','result_publish_return')]:
        values=[(g[b]-g[a])/1000 for (layer,tick),g in groups.items() if tick>=3 and a in g and b in g]
        if values:
            if min(values)<0:raise ValueError('negative schedule interval')
            result[name]={'count':len(values),'mean_us':statistics.mean(values),'p50_us':statistics.median(values)}
    return {'intervals':result,'layer_records':len(groups),'dropped':dropped,
            'warning':'Host-clock buffered diagnostics only, tick>=3 filter is layer-local not request/decode identity. DMA enqueue is not DMA completion; compute return is not publication. Not GPU-exposed waiting.'}


def decompose_count_cost(n0, total0, n1, total1, denominator):
    if min(n0,n1,denominator)<=0:
        raise ValueError('positive counts/denominator required')
    c0,c1=total0/n0,total1/n1
    return {'count_component':(n1-n0)*(c0+c1)/2/denominator,
            'cost_component':(c1-c0)*(n0+n1)/2/denominator,
            'total_delta':(total1-total0)/denominator}


def report(path):
    m=json.loads(Path(path).read_text())
    samples=m['samples'];names=list(dict.fromkeys(s['arm'] for s in samples))
    if len(names)==3:
        rounds=[]
        for rnd in sorted({s['round'] for s in samples}):
            row={s['arm']:s['median_tpot_ms'] for s in samples if s['round']==rnd}
            if set(row)!=set(names):
                raise ValueError('incomplete three-arm round')
            rounds.append({'round':rnd,'medians':row})
        return {'status':m['status'],'mode':m['mode'],'rounds':rounds,
                'median_of_process_medians':{n:statistics.median(r['medians'][n] for r in rounds) for n in names},
                'paired_deltas':{f'{b} minus {a}':statistics.median(r['medians'][b]-r['medians'][a] for r in rounds)
                                 for i,a in enumerate(names) for b in names[i+1:]},
                'same_work_proven':False,
                'warning':'Three-arm rotated-order policy comparison; inspect drafting logs, text and configuration differences. No cross-session subtraction.'}
    if len(names)!=2:
        raise ValueError('two or three arms required')
    pairs=[]
    for rnd in sorted({s['round'] for s in samples}):
        row={s['arm']:s for s in samples if s['round']==rnd}
        if set(row)!=set(names):
            continue
        a,b=(row[n]['median_tpot_ms'] for n in names)
        pairs.append({'round':rnd,names[0]:a,names[1]:b,'delta_ms':b-a})
    return {'status':m['status'],'mode':m['mode'],'reference':names[0],'candidate':names[1],
            'pairs':pairs,'median_paired_delta_ms':statistics.median(p['delta_ms'] for p in pairs) if pairs else None,
            'performance_eligible':m['status']=='complete' and m['mode']=='clean' and len(pairs)>=3,
            'all_text_equal':m.get('all_text_equal'),
            'text_equal_by_request_position':m.get('text_equal_by_request_position'),
            'same_work_proven':False,
            'warning':'Performance eligibility is not same-work causality; inspect SHA256 and matched diagnostic traces. Never pool diagnostic TPOT with clean TPOT.'}


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('gpu');g.add_argument('csv');g.add_argument('--start-ns',type=int,required=True);g.add_argument('--end-ns',type=int,required=True);g.add_argument('--steps',type=int,required=True);g.add_argument('--gap-min-ns',type=int,default=10000)
    r=sub.add_parser('routes');r.add_argument('left');r.add_argument('right')
    s=sub.add_parser('schedule');s.add_argument('jsonl')
    c=sub.add_parser('cpu');c.add_argument('log')
    l=sub.add_parser('ledger');l.add_argument('directory')
    t=sub.add_parser('mtp');t.add_argument('directory')
    m=sub.add_parser('report');m.add_argument('manifest')
    a=p.parse_args()
    if a.command=='gpu':out=gpu(a.csv,a.start_ns,a.end_ns,a.steps,a.gap_min_ns)
    elif a.command=='routes':out=compare_routes(a.left,a.right)
    elif a.command=='cpu':out=cpu(a.log)
    elif a.command=='schedule':out=schedule(a.jsonl)
    elif a.command=='ledger':
        from ledger import compare
        out=compare(a.directory)
    elif a.command=='mtp':
        from mtp import compare
        out=compare(a.directory)
    else:out=report(a.manifest)
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
