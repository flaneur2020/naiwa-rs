"""Tail bracket analysis; exclude prompt/cross-request transitions by next model row shape."""
import json
import math
import statistics
from pathlib import Path

def analyze(path):
    records=[json.loads(l) for l in Path(path).read_text().splitlines()]
    samples=[r for r in records if r.get('kind')=='sampler_host']
    tails=[r for r in records if r.get('kind')!='sampler_host']
    rows=[r for r in tails if r['rows']==1 and r['next_rows']==1]
    if len(rows)<370:raise ValueError('missing head samples')
    for r in rows:
        e=r['endpoints_ms']
        if len(e)!=4 or any(not math.isfinite(v) for v in e) or any(b<a for a,b in zip(e,e[1:])):raise ValueError('invalid head endpoints')
    return dict(samples=len(rows),excluded=len(tails)-len(rows),
                sampler_calls=sum(r['calls'] for r in samples),
                sampler_device_greedy=sum(r['device_greedy'] for r in samples),
                sampler_mean_ms_per_call=(sum(r['total_ms'] for r in samples)/sum(r['calls'] for r in samples)) if sum(r['calls'] for r in samples) else None,
                mean_ms={name:statistics.mean(r['endpoints_ms'][i+1]-r['endpoints_ms'][i] for r in rows)
                         for i,name in enumerate(['final_hc','lm_head','head_to_next_model'])},
                warning='Row-shape excludes ordinary prompt transitions, not proven request IDs. Warmup may remain. Last2tails censored. GPU timestamps span host idle, not kernel-only. Tail event does not mark engine sampling boundary.')

if __name__=='__main__':
    import sys
    root=Path(sys.argv[1]);print(json.dumps([dict(round=r,**analyze(root/f'r{r}-head/head-partition.jsonl')) for r in (1,2,3)],indent=2))
