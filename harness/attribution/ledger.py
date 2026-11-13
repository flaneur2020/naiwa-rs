"""Diagnostic byte ledger: different engines have deliberately distinct windows."""
import collections
import json
from pathlib import Path


def strata(path):
    requests=[];rows=[]
    for line in Path(path).read_text().splitlines():
        r=json.loads(line)
        if r.get('kind')=='gpu_window':continue
        if not r.get('request_boundary'):
            rows.append(r);continue
        # Current single-sequence no-drafting config: prefill rows>1, decode rows1.
        decode=[x for x in rows if x['rows']==1]
        positions=collections.defaultdict(list)
        for x in decode:positions[x['pos']].append(x)
        if len(positions)!=128 or sorted(positions)!=list(range(min(positions),min(positions)+128)):
            raise ValueError('expected128 consecutive decode positions')
        if any(sorted(x['layer'] for x in v)!=list(range(48)) for v in positions.values()):
            raise ValueError('missing/duplicate layers')
        requests.append(dict(windows=len(positions),staged_MB_per_window=sum(x['staged_bytes'] for x in decode)/128/1e6,
                             staged_groups_per_layer=sum(x['staged_groups'] for x in decode)/len(decode),
                             cpu_entries_per_layer=sum(10-x['gpu_entries'] for x in decode)/len(decode),
                             pool_callback_ms_per_window=sum(x['pool_ms'] for x in decode)/128))
        rows=[]
    if rows or len(requests)!=3:raise ValueError('unflushed/incomplete request ledger')
    return requests


def naiwa(path):
    from analyze import schedule
    schedule(path)  # reject drops/incomplete CPU timing
    records=[json.loads(line) for line in Path(path).read_text().splitlines()]
    rows=[x for x in records if x.get('tick',0)>=3]
    layers=sum(x.get('phase')=='route_ready' for x in rows)
    if not layers:raise ValueError('no layer windows')
    totals=collections.Counter()
    for x in rows:
        if x.get('phase') in ('staged_h2d','cache_fill_h2d'):totals[x['phase']]+=x['bytes']
    return dict(layer_windows=layers,MB_per_48_layer_equivalent={k:v/layers*48/1e6 for k,v in totals.items()},
                warning='tick>=3 is local layer filter, NOT request/decode ID. Cache fills may execute asynchronously across boundary. Does not account promotion D2D or fallback-owned uploads.')


def compare(root):
    root=Path(root)
    return dict(rounds=[dict(round=r,strata=strata(root/f'r{r}-strata/strata-ledger.jsonl'),
                            naiwa=naiwa(root/f'r{r}-cpu_first/schedule.jsonl')) for r in (1,2,3)],
                warning='Diagnostic only. Strata staged bytes are selected full-blob transfer payload (kernel PCIe loads, not memcpy). Naiwa is queued projection-byte payload. No equality of request windows/work, no bandwidth-to-TPOT causal subtraction. Single GPU/no drafting only. Strata all-resident/batch/pipeline paths not instrumented.')
