"""Model-level stream brackets, not pure arithmetic or causal gap allocation."""
import json
import statistics
from pathlib import Path
PHASES=['layer_start','attn_norm_end','attn_down','attn_act','attn_up','attn_mix_end','mixer_start','mixer_end','mlp_norm_end','mlp_down','mlp_act','mlp_up','mlp_mix_end','moe_start','routed_end','shared_end','moe_end','layer_end','next_layer']

def sparse(path):
    records=[json.loads(l) for l in Path(path).read_text().splitlines()]
    if not any(x.get('flush') for x in records) or any(x.get('dropped',0) for x in records):raise ValueError('unflushed/dropped model events')
    rows=[x for x in records if 'endpoints_ms' in x and x['rows']==1]
    out={}
    for layer in sorted({x['layer'] for x in rows}):
        selected=[x for x in rows if x['layer']==layer]
        sums={k:[] for k in PHASES[1:-1]}
        for x in selected:
            ep=x['endpoints_ms']
            if [p for p,t in ep]!=PHASES or any(b[1]<a[1] for a,b in zip(ep,ep[1:])):raise ValueError('invalid endpoints')
            for (a,t),(b,u) in zip(ep[:-1],ep[1:-1]):sums[b].append(u-t)
        out[layer]={'samples':len(selected),'mean_ms_per_layer':{k:statistics.mean(v) for k,v in sums.items()}}
    if not out:raise ValueError('no sparse samples')
    return out

def naiwa(path):
    records=[json.loads(l) for l in Path(path).read_text().splitlines()]
    if not any(x.get('flush') for x in records) or any(x.get('dropped',0) for x in records):raise ValueError('unflushed/dropped model events')
    rows=[x for x in records if not x.get('flush')];groups=[];seq=[]
    for i,x in enumerate(rows):
        ep=x['endpoints_ms']
        if [p for p,t in ep]!=PHASES or any(b[1]<a[1] for a,b in zip(ep,ep[1:])):raise ValueError('invalid endpoints')
        if x['layer']==0:seq=[]
        seq.append(x)
        if (len(seq)==48 and all(y['layer']==j and y['rows']==1 for j,y in enumerate(seq)) and
            i+1<len(rows) and rows[i+1]['layer']==0 and rows[i+1]['rows']==1):groups.append(seq)
    if not groups:raise ValueError('no complete decode groups')
    sums={p:0. for p in PHASES[1:]};kinds={k:[] for k in ['qsa','gdn']}
    for g in groups:
        for x in g:
            for (a,t),(b,u) in zip(x['endpoints_ms'],x['endpoints_ms'][1:]):sums[b]+=u-t
            ep=dict(x['endpoints_ms']);kinds['qsa' if (x['layer']+1)%4==0 else 'gdn'].append(ep['mixer_end']-ep['mixer_start'])
    return dict(groups=len(groups),ms_per_group={k:v/len(groups) for k,v in sums.items()},
                mixer_ms_per_layer={k:statistics.mean(v) for k,v in kinds.items()})

def strata(path):
    windows=[]
    for line in Path(path).read_text().splitlines():
        w=json.loads(line)
        if w.get('kind')!='gpu_window' or w['rows']!=1:continue
        s=w['stamps'];totals=dict(hc0=0.,mixer=0.,hc1_router=0.,shared=0.,head=0.)
        for l in range(48):
            for name,a,b in [('hc0',0,1),('mixer',1,16),('hc1_router',16,17),('shared',17,18)]:
                a,b=s[l*33+a],s[l*33+b]
                if not a or b<a:raise ValueError('invalid Strata boundary')
                totals[name]+=(b-a)/1e6
        totals['head']=(s[48*33+1]-s[48*33])/1e6
        windows.append(totals)
    if len(windows)!=384:raise ValueError('expected384decode windows')
    return {k:statistics.mean(w[k] for w in windows) for k in windows[0]}

if __name__=='__main__':
    import sys
    root=Path(sys.argv[1]);other=Path(sys.argv[2])
    print(json.dumps(dict(rounds=[dict(round=r,naiwa=naiwa(root/f'r{r}-model/model-partition.jsonl'),
                                      strata=strata(other/f'r{r}-strata/strata-ledger.jsonl')) for r in (1,2,3)],
                          warning='Separate diagnostic runs; no cross-session causal subtraction. Naiwa post-MoE shared vs Strata pre-expert shared changes overlap. hc1/router brackets differ (naiwa router in routed_end). next_layer includes head/sample/host gap. QSA assumed every fourth layer; validate model geometry before reuse.'),indent=2))
