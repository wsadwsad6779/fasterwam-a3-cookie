cd /workspace
python3 - <<'PY'
import collections, re, statistics
rows = []
pat = re.compile('step=([0-9]+)/8000 loss=([0-9.]+) loss_action=([0-9.]+) loss_video=([0-9.]+) lr=([0-9.e-]+)')
for l in open('/workspace/a3_train_runs/a3_lora_r5/train.log'):
    m = pat.search(l)
    if m:
        rows.append([int(m.group(1))] + [float(x) for x in m.groups()[1:]])
print('logged points', len(rows), 'first', rows[0], 'last', rows[-1])
n = len(rows)
for i in range(5):
    seg = rows[i*n//5:(i+1)*n//5]
    print('seg', i, 'steps', seg[0][0], '-', seg[-1][0], 'mean loss %.4f' % statistics.mean(r[1] for r in seg),
          'mean action %.5f' % statistics.mean(r[2] for r in seg), 'mean video %.4f' % statistics.mean(r[3] for r in seg),
          'lr_end %.3e' % seg[-1][4])
print('lr min %.3e max %.3e' % (min(r[4] for r in rows), max(r[4] for r in rows)))
b = min(rows, key=lambda r: r[2])
print('best action loss %.5f at step %d' % (b[2], b[0]))
lines = [l.split() for l in open('/workspace/a3_train_runs/a3_lora_r5/windows.log') if not l.startswith('#') and l.strip()]
vals = [int(p[1]) for p in lines]
tot = 39481
print('r5 logged draws', len(vals), 'distinct', len(set(vals)), 'coverage %.1f pct' % (100.0*len(set(vals))/tot))
print('indices below the resume offset 18000:', sum(1 for v in vals if v < 18000), 'distinct', len({v for v in vals if v < 18000}))
rep = collections.Counter(v >= 18000 for v in vals)
print('above/below split', dict(rep))
PY
ls /workspace/a3_train_runs/a3_lora_r5/checkpoints/light/
tail -c 900 /workspace/gate.out
