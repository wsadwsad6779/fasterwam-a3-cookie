"""Post-12k degradation analysis across milestone eval summaries.
Reads /workspace/fwv1/eval_fingertip_v1/step_*/SUMMARY_step_*.json and reports
changes in exact_fill, in-target count, grasps, safety stops and gripper behavior."""
import glob
import json
import os
import time

ROOT = '/workspace/fwv1/eval_fingertip_v1'
OUT = '/workspace/fwv1/p_items/DEGRADATION_REPORT.json'
MD = '/workspace/fwv1/p_items/DEGRADATION_REPORT.md'
MILESTONES = [3000, 6000, 9000, 12000, 15000, 18000, 21000]


def load(step):
    p = os.path.join(ROOT, 'step_%06d' % step, 'SUMMARY_step_%d.json' % step)
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    rs = d.get('results', [])
    return {'step': step, 'exact_fill_rate': d.get('exact_fill_rate'),
            'exact_fill_count': d.get('exact_fill_count'),
            'avg_in_target': d.get('avg_in_target'),
            'safety_stop_count': d.get('safety_stop_count'),
            'grasps_mean': (sum((r.get('grasps') or 0) for r in rs) / max(1, len(rs))),
            'seeds': d.get('seeds'),
            'summary': p}


def main():
    rows = [x for x in (load(s) for s in MILESTONES if load(s)) if x]
    out = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'milestones': rows, 'warnings': [], 'verdict': 'insufficient_milestones'}
    if len(rows) >= 2:
        base = rows[0]
        late = [r for r in rows if r['step'] >= 12000]
        warns = []
        for r in rows[1:]:
            drop = (base['exact_fill_rate'] or 0) - (r['exact_fill_rate'] or 0)
            if r['step'] >= 12000 and drop > 0.2 * max(1e-9, base['exact_fill_rate'] or 0):
                warns.append('exact_fill dropped >20%% vs %dk at %dk (%.3f -> %.3f)' % (
                    base['step'] // 1000, r['step'] // 1000,
                    base['exact_fill_rate'] or 0, r['exact_fill_rate'] or 0))
            if (r['avg_in_target'] or 0) < (base['avg_in_target'] or 0) - 1.0 and r['step'] >= 12000:
                warns.append('avg_in_target dropped >1 block at %dk' % (r['step'] // 1000))
        out['warnings'] = warns
        out['verdict'] = 'degradation_flagged' if warns else ('pass' if len(rows) >= 7 else 'partial')
        out['compare_12k_onward'] = [r for r in rows if r['step'] >= 9000]
    json.dump(out, open(OUT, 'w'), indent=1)
    lines = ['# DEGRADATION_REPORT', '']
    for r in rows:
        lines.append('- %dk: exact_fill %s/%s (%.3f), avg_in_target %.2f, grasps %.2f, safety %s' % (
            r['step'] // 1000, r['exact_fill_count'], len(r['seeds'] or []),
            r['exact_fill_rate'] or 0, r['avg_in_target'] or 0, r['grasps_mean'], r['safety_stop_count']))
    if out['warnings']:
        lines += ['', '## Warnings'] + ['- ' + w for w in out['warnings']]
    lines += ['', 'Verdict: %s' % out['verdict']]
    open(MD, 'w').write('\n'.join(lines))
    print(json.dumps({'rows': len(rows), 'warnings': out['warnings'], 'verdict': out['verdict']}))


if __name__ == '__main__':
    main()