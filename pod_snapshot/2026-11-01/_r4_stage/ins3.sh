#!/bin/bash
python3 - <<'PY' > /workspace/_r4_stage/ins3.out 2>&1

import sys
F = '/workspace/fasterwam_a3_bridge/bridge/_next/run_matrix.py'
lines = open(F, newline='').read().split('\n')
marks = ['safety_layer', 'estop_steps', 'binfloor_steps', '"duration_s"',
         'post_w = pre.copy', 'env.step(post)', 'upper = name.upper',
         'safety_st = safety_wrap', 'default="C0,C1,C2"']
hits = set()
for i, l in enumerate(lines):
    for m in marks:
        if m in l:
            for k in range(max(0, i - 3), min(len(lines), i + 6)):
                hits.add(k)
for i in sorted(hits):
    print(i + 1, repr(lines[i]))
print('--- total lines', len(lines))

PY
echo END >> /workspace/_r4_stage/ins3.out