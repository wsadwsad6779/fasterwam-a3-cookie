'''Task 1 reporting: pull the LR schedule, loss distribution and window coverage out of
the r5 run so docs_task1 sections 4 and 5 can be filled with measured numbers.

Read-only. It parses /workspace/a3_train_runs/a3_lora_r5/train.log (the lines the trainer
already wrote) and windows.log (one 'pass_in_epoch window_index' record per sampled
window), and compares coverage against the r4 run it continued from.
'''
import argparse
import re
from collections import Counter
from pathlib import Path

LINE = re.compile(r'step=(\d+)/(\d+) loss=([\d.eE+-]+) loss_action=([\d.eE+-]+) '
                  r'loss_video=([\d.eE+-]+) lr=([\d.eE+-]+)')


def parse_train(log):
    rows = []
    for m in LINE.finditer(Path(log).read_text(encoding='utf-8', errors='replace')):
        rows.append({'step': int(m.group(1)), 'loss': float(m.group(3)),
                     'action': float(m.group(4)), 'video': float(m.group(5)),
                     'lr': float(m.group(6))})
    return rows


def q(vals, k):
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(int(len(s) * k), len(s) - 1)], 5)


def coverage(winlog):
    idx = []
    for line in Path(winlog).read_text(encoding='utf-8', errors='replace').splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            idx.append(int(parts[1]))
    c = Counter(idx)
    return {'records': len(idx), 'distinct': len(c), 'max_index': max(idx) if idx else -1,
            'repeats_min': min(c.values()) if c else 0, 'repeats_max': max(c.values()) if c else 0,
            'never_seen_above': sum(1 for v in c.values() if v == 1)}


def main():
    ap = argparse.ArgumentParser(description='r5 training report')
    ap.add_argument('--run', default='/workspace/a3_train_runs/a3_lora_r5')
    ap.add_argument('--compare', default='/workspace/a3_train_runs/a3_lora_r4')
    args = ap.parse_args()
    run = Path(args.run)
    rows = parse_train(str(run / 'train.log'))
    print('logged_points', len(rows))
    if rows:
        print('first', rows[0])
        print('last', rows[-1])
        n = len(rows)
        for i in range(5):
            seg = rows[i * n // 5:(i + 1) * n // 5]
            print('quintile%d steps %d-%d mean_loss=%.5f mean_action=%.5f mean_video=%.5f' % (
                i + 1, seg[0]['step'], seg[-1]['step'],
                sum(s['loss'] for s in seg) / len(seg),
                sum(s['action'] for s in seg) / len(seg),
                sum(s['video'] for s in seg) / len(seg)))
        best = min(rows, key=lambda r: r['action'])
        print('min_action', best)
        print('lr_unique_endpoints', sorted({round(r['lr'], 12) for r in rows})[-3:])
    print('coverage_r5', coverage(str(run / 'windows.log')))
    cmp_path = Path(args.compare)
    if (cmp_path / 'windows.log').exists():
        print('coverage_r4', coverage(str(cmp_path / 'windows.log')))
    if (cmp_path / 'train.log').exists():
        r4 = parse_train(str(cmp_path / 'train.log'))
        if r4:
            print('r4_last', r4[-1])


main()
