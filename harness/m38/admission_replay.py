#!/usr/bin/env python3
"""One-decision replay on observed staged sets, NOT a full cache simulation.
Select using past/current demand; score on next same-layer three-row call.
Does not change staging, model evictions, or estimate wall-time savings.
"""
import argparse
import ast
import collections
import json
import re


def analyze(path):
    layers = collections.defaultdict(list)
    for line in open(path):
        m = re.search(r'ROUTE_IDS layer=(\d+) rows=(\d+) ids=(\[.*\]) resident=(\[.*\])', line)
        if m:
            layers[int(m[1])].append(dict(rows=int(m[2]), demand=collections.Counter(
                e for row in ast.literal_eval(m[3]) for e in row), resident=set(ast.literal_eval(m[4])), chosen=[]))
        m = re.search(r'ADMISSION layer=(\d+) rows=(\d+) ranked=(\[.*\]) chosen=(\[.*\])', line)
        if m and layers[int(m[1])]:
            layers[int(m[1])][-1]['chosen'] = ast.literal_eval(m[4])
    out = collections.Counter()
    for seq in layers.values():
        counts = collections.Counter()
        for i, cur in enumerate(seq[:-1]):
            counts.update(cur['demand'])
            nxt = seq[i + 1]
            if cur['rows'] != 3 or nxt['rows'] != 3:
                continue
            candidates = [e for e in cur['chosen'] if counts[e] >= 4 and e not in cur['resident']]
            if not candidates:
                continue
            added = nxt['resident'] - cur['resident']
            # Strict filter: infer one likely staged promotion; other cache activity can still confound it.
            if len(added) != 1 or not added.issubset(candidates):
                out['ambiguous_or_no_admission'] += 1
                continue
            actual = next(iter(added))
            selected = min(candidates, key=lambda e: (-counts[e], e))
            out['comparisons'] += 1
            out['different_choice'] += selected != actual
            out['observed_next_pairs'] += nxt['demand'][actual]
            out['frequency_next_pairs'] += nxt['demand'][selected]
            out['observed_next_hits'] += nxt['demand'][actual] > 0
            out['frequency_next_hits'] += nxt['demand'][selected] > 0
            # Charging lost resident demand shows why raw admitted reuse is not net value.
            victims = cur['resident'] - nxt['resident']
            out['victim_next_pairs'] += sum(nxt['demand'][e] for e in victims)
    return dict(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('logs', nargs='+')
    args = p.parse_args()
    for path in args.logs:
        print(json.dumps({'path': path, **analyze(path)}, ensure_ascii=False))
