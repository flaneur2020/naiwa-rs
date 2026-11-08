#!/usr/bin/env python3
"""Launch-count KPI for M38: launches and inter-kernel gaps per layer call, from an nsys capture.

A layer call is the segment between two `doorbell_wait_copy_f32_kernel` instances (exactly one per
layer call in the decode path), so the slicing needs no timestamps of its own.

  harness/m38/gap_kpi.py <capture.nsys-rep>
"""
import csv, statistics as st, subprocess, sys, os

rep = sys.argv[1]
csvp = '/tmp/.gapkpi_trace.csv'
stem = rep.rsplit('.', 1)[0]
if not os.path.exists(stem + '.sqlite'):
    subprocess.run(['/usr/local/cuda/bin/nsys', 'export', '--type', 'sqlite', '--force-overwrite=true',
                    '-o', stem, rep], capture_output=True)
out = subprocess.run(['/usr/local/cuda/bin/nsys', 'stats', '--report', 'cuda_gpu_trace', '--format', 'csv', rep],
                     capture_output=True, text=True).stdout
with open(csvp, 'w') as f:
    f.write(out)

rows = []
with open(csvp, newline='') as f:
    for rec in csv.reader(f):
        if len(rec) < 21 or not rec[0].strip().isdigit():
            continue
        rows.append((int(rec[0]), int(rec[1]), rec[20]))
rows.sort()
db = [i for i, r in enumerate(rows) if 'doorbell_wait_copy' in r[2]]
if len(db) < 3:
    sys.exit("no doorbell_wait_copy instances: not a decode capture?")

per_layer = []
for a, b in zip(db, db[1:]):
    seg = rows[a:b]
    if len(seg) < 5:
        continue
    gaps = [(seg[i + 1][0] - (seg[i][0] + seg[i][1])) / 1e3 for i in range(len(seg) - 1)]
    per_layer.append((len(seg), sum(g for g in gaps if g > 0)))

n = [p[0] for p in per_layer]
g = [p[1] for p in per_layer]
print(f"  layer calls {len(per_layer)}")
print(f"  launches / layer   median {st.median(n):.0f}   mean {sum(n) / len(n):.1f}")
print(f"  gap / layer        median {st.median(g):.0f} us   mean {sum(g) / len(g):.0f} us")
print(f"  gap share of a 1.6 ms layer: {100 * st.median(g) / 1600:.0f}%")
