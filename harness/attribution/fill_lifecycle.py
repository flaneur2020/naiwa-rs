"""Publication-to-resident-use diagnostic, not GPU consumption or request attribution."""
import collections
import json
from pathlib import Path
from analyze import schedule


def analyze(path):
    schedule(path)
    events=[json.loads(l) for l in Path(path).read_text().splitlines()]
    active={};queued=collections.defaultdict(list);done=[];unmatched=0
    for e in events:
        phase=e.get('phase')
        if phase not in ('cache_fill_h2d','fill_publish','promotion_publish','resident_use'):continue
        if e.get('slot') is None or e.get('expert') is None:raise ValueError('missing identity')
        key=(e['layer'],e['slot']);identity=(*key,e['expert'])
        if phase=='cache_fill_h2d':queued[identity].append(e);continue
        if phase=='resident_use':
            a=active.get(key)
            if a and a['expert']==e['expert']:
                a['uses']+=1
                if a['first_use_us'] is None:a['first_use_us']=(e['ns']-a['ns'])/1000
            continue
        old=active.pop(key,None)
        if old:old['end']='superseded_by_publication';done.append(old)
        if phase=='fill_publish':
            q=queued[identity]
            if not q:unmatched+=1;continue
            src=q.pop(0)
            active[key]=dict(expert=e['expert'],ns=e['ns'],tick=e['tick'],bytes=src['bytes'],uses=0,first_use_us=None)
        # Promotion invalidates tracked H2D slot generation; not itself an H2D fill.
    for a in active.values():a['end']='right_censored';done.append(a)
    selected=[a for a in done if a['tick']>=3]
    groups={}
    for end in ('superseded_by_publication','right_censored'):
        for used in (False,True):
            rows=[a for a in selected if a['end']==end and bool(a['uses'])==used]
            groups[f'{end}_{"used" if used else "unused"}']=dict(count=len(rows),bytes=sum(a['bytes'] for a in rows))
    total=sum(a['bytes'] for a in selected)
    return dict(fills=len(selected),bytes=total,groups=groups,
                unused_superseded_byte_fraction=groups['superseded_by_publication_unused']['bytes']/total if total else None,
                unpaired_publications=unmatched,unpublished_queues=sum(map(len,queued.values())),
                warning='No request/GPU-use IDs. tick>=3 local filter only. Resident-use means classified host hit, not GPU consumption. Superseded means next successful fill/promotion publication on slot, NOT exact eviction time. Borrow/unpublish/error paths not tracked; tail unused is censored, not wasted. Diagnostic lifecycle capture may alter policy. No TPOT savings inferred.')


if __name__=='__main__':
    import sys
    print(json.dumps(analyze(sys.argv[1]),indent=2))
