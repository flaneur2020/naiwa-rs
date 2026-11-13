"""Boundary-explicit diagnostic comparison; no clean-gap causal allocation."""
import json
import statistics
from pathlib import Path
from step_costs import strata

PHASES=['route','plan_submit','preload_ready','resident_end','staging_ready','staged_end','promotion_end','cpu_ready','combine_end','next_route']

def naiwa(path):
    records=[json.loads(l) for l in Path(path).read_text().splitlines()]
    if any(x.get('flush') and x.get('dropped') for x in records):raise ValueError('dropped GPU rows')
    rows=[x for x in records if not x.get('flush')]
    groups=[];seq=[]
    for i,x in enumerate(rows):
        ep=x['endpoints_ms']
        if [a for a,b in ep]!=PHASES or any(b[1]<a[1] for a,b in zip(ep,ep[1:])):
            raise ValueError('invalid GPU event partition')
        if x['layer']==0:seq=[]
        seq.append(x)
        if (x['layer']==47 and len(seq)==48 and
            all(y['rows']==1 and y['layer']==j for j,y in enumerate(seq)) and
            i+1<len(rows) and rows[i+1]['rows']==1 and rows[i+1]['layer']==0):
            groups.append(seq)
    if not groups:raise ValueError('no complete decode groups')
    steps=[]
    for seq in groups:
        totals={p:0. for p in PHASES[1:]}
        for x in seq:
            for (a,t),(b,u) in zip(x['endpoints_ms'],x['endpoints_ms'][1:]):totals[b]+=u-t
        steps.append(totals)
    return dict(complete_groups=len(groups),segments_ms={p:statistics.mean(s[p] for s in steps) for p in PHASES[1:]},
                span_ms=statistics.mean(sum(s.values()) for s in steps),
                warning='Route-to-next-route rolling partition. next_route includes dense/shared/router/head/host/event collection and file I/O, not isolated dense compute. Structurally excludes prefill/cross-request groups; no actual request IDs. Last incomplete groups censored, not force-flushed.')


def compare(root):
    root=Path(root);rounds=[]
    for r in (1,2,3):
        n=naiwa(root/f'r{r}-cpu_first/gpu-partition.jsonl')
        s=strata(root/f'r{r}-strata/strata-ledger.jsonl')
        rounds.append(dict(round=r,naiwa=n,strata_gpu_ms={k:statistics.mean(q['gpu_mean_ms'][k] for q in s) for k in s[0]['gpu_mean_ms']}))
    return dict(rounds=rounds,warning='Diagnostic policies/work differ. Event allocation, elapsed queries, JSON and per-step writes perturb naiwa; Strata profiling changes stream concurrency. Diagnostic gap is not clean gap. No causal per-component clean-gap subtraction.')

if __name__=='__main__':
    import sys
    print(json.dumps(compare(sys.argv[1]),indent=2))
