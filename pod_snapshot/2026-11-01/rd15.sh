cd /workspace
python3 - <<'PY'
import collections
lines = [l.split() for l in open('/workspace/a3_train_runs/a3_lora_r5/windows.log') if not l.startswith('#') and l.strip()]
vals = [int(p[1]) for p in lines]
print('logged draws', len(vals), 'distinct windows', len(set(vals)))
print('min idx', min(vals), 'max idx', max(vals))
seen = set(vals)
n = 39481
print('coverage of the 39481 windows:', round(100.0*len(seen)/n, 1), 'pct')
print('never drawn in r5:', n - len(seen))
hist = collections.Counter()
for i in range(n):
    pass
print('resume sample_offset was 18000; indices below that belong to r1-r4 draws')
print('r5 draws below 18000:', sum(1 for v in vals if v < 18000))
print('distinct below 18000:', len({v for v in vals if v < 18000}))
PY
echo '=== r5 loss trace ==='
python3 - <<'PY'
import re
rows = []
for l in open('/workspace/a3_train_runs/a3_lora_r5/train.log'):
    m = re.search(r'step=(d+)/8000 loss=([0-9.]+) loss_action=([0-9.]+) loss_video=([0-9.]+) lr=([0-9.e-]+)', l)
    if m:
        rows.append([int(m.group(1))] + [float(x) for x in m.groups()[1:]])
print('logged points', len(rows), 'first', rows[0], 'last', rows[-1])
import statistics
for k, name in ((5, 'first quintile'), (6, 'mid'), (7, 'last quintile')):
    pass
n = len(rows)
for i in range(5):
    seg = rows[i*n//5:(i+1)*n//5]
    print('seg', i, 'steps', seg[0][0], '-', seg[-1][0], 'mean loss %.4f' % statistics.mean(r[1] for r in seg),
          'mean action %.5f' % statistics.mean(r[2] for r in seg), 'mean video %.4f' % statistics.mean(r[3] for r in seg),
          'lr %.3e' % seg[-1][4])
print('lr min', min(r[4] for r in rows), 'lr max', max(r[4] for r in rows))
print('best action loss', min(r[2] for r in rows), 'at step', min(rows, key=lambda r: r[2])[0])
PY
ls /workspace/a3_train_runs/a3_lora_r5/checkpoints/light/
