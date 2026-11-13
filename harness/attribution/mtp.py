"""MTP terminal-outcome accounting; sequential single-client runs only."""
import collections
import json
from pathlib import Path


def requests(directory, expected):
    directory=Path(directory)
    records=[json.loads(line) for line in (directory/'mtp.jsonl').read_text().splitlines()]
    if len(records)!=expected or len({r['seq_id'] for r in records})!=expected:
        raise ValueError('MTP request count or sequence IDs invalid')
    output=[]
    for i,record in enumerate(records,1):
        if not record.get('terminal') or not record.get('steps'):
            raise ValueError('nonterminal/empty MTP request')
        steps=record['steps']
        if any(len(s)!=3 or any(type(v) is not int or v<0 for v in s) or s[1]>s[0] for s in steps):
            raise ValueError('invalid proposed/accepted/keep_len')
        if any(b[2]<=a[2] for a,b in zip(steps,steps[1:])):
            raise ValueError('non-increasing keep_len; alignment unproven')
        req=json.loads((directory/f'request-{i}.json').read_text())
        tokens=req['usage']['completion_tokens']
        if tokens<=1:raise ValueError('invalid completion_tokens')
        proposed=sum(s[0] for s in steps);accepted=sum(s[1] for s in steps)
        if not proposed:raise ValueError('no proposals')
        duration=req['total_ms']-req['ttft_ms']
        output.append(dict(request=i,seq_id=record['seq_id'],steps=len(steps),proposed=proposed,
                           accepted=accepted,acceptance=accepted/proposed,
                           accepted_histogram=dict(collections.Counter(s[1] for s in steps)),
                           proposed_histogram=dict(collections.Counter(s[0] for s in steps)),
                           step_sha256=__import__('hashlib').sha256(json.dumps(steps,separators=(',',':')).encode()).hexdigest(),
                           completion_tokens=tokens,decode_intervals=tokens-1,
                           client_decode_ms=duration,amortized_ms_per_outcome=duration/len(steps),
                           tpot_ms=req['tpot_ms'],sha256=req['sha256']))
    return output


def compare(root):
    from analyze import decompose_count_cost
    root=Path(root)
    config=json.loads((root/'config.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['status']!='complete' or config['mode']!='diagnostic':
        raise ValueError('requires complete diagnostic run')
    arms=[a['name'] for a in config['arms']]
    pairs=[]
    for rnd in range(1,config['rounds']+1):
        left=requests(root/f'r{rnd}-{arms[0]}',config['requests'])
        right=requests(root/f'r{rnd}-{arms[1]}',config['requests'])
        for a,b in zip(left,right):
            if a['completion_tokens']!=b['completion_tokens']:
                raise ValueError('different output lengths; fixed-denominator decomposition invalid')
            components=decompose_count_cost(a['steps'],a['client_decode_ms'],b['steps'],b['client_decode_ms'],a['decode_intervals'])
            pairs.append(dict(round=rnd,request=a['request'],reference=a,candidate=b,
                              text_equal=a['sha256']==b['sha256'],steps_equal=a['step_sha256']==b['step_sha256'],
                              decomposition=components))
    return dict(schema=1,reference=arms[0],candidate=arms[1],pairs=pairs,
                warning='Sequential request order maps terminal seq records to requests; no concurrent clients/cancellations. Counts cover completed verification outcomes only. Cost includes drafting/bootstrap/transport, not pure verify kernel. Exact accounting is not causal same-work attribution. Diagnostic timings are not clean benchmark evidence.')
