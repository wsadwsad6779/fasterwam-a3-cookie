"""S5 final independent bare-model test + force ablation on the selected best checkpoint."""
import glob
import json
import os
import subprocess
import sys
import time

P = '/workspace/fwv1/p_items'
ROOT = '/workspace/fwv1/eval_fingertip_v1'
RUN = os.environ.get('T7_RUN_DIR', '/workspace/fwv1/train_runs/t7_21k_v3run')
EVAL = '/workspace/_t7_eval.py'
OUT_SEL = P + '/BEST_CHECKPOINT_SELECTION.json'
OUT_FORCE = P + '/FORCE_CONTRIBUTION.json'
OUT_MD = P + '/FINAL_TEST_REPORT.md'
OUT_MAN = P + '/VIDEO_MANIFEST_final.json'
SEEDS = ','.join(str(7900 + i) for i in range(10))
MILESTONES = [3000, 6000, 9000, 12000, 15000, 18000, 21000]


def load_summary(step):
    p = os.path.join(ROOT, 'step_%06d' % step, 'SUMMARY_step_%d.json' % step)
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def select_best():
    rows = []
    for s in MILESTONES:
        d = load_summary(s)
        if d is None:
            continue
        rows.append({'step': s, 'exact_fill_rate': d.get('exact_fill_rate'),
                     'exact_fill_count': d.get('exact_fill_count'),
                     'avg_in_target': d.get('avg_in_target'),
                     'safety_stop_count': d.get('safety_stop_count'),
                     'summary': os.path.join(ROOT, 'step_%06d' % s, 'SUMMARY_step_%d.json' % s)})
    if not rows:
        raise SystemExit('no milestone summaries yet')
    rows.sort(key=lambda r: (-(r['exact_fill_rate'] or 0), -(r['avg_in_target'] or 0), r['step']))
    sel = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'rule': 'max exact_fill_rate, then avg_in_target, then earlier step; 21k is not assumed best',
           'candidates': rows, 'selected': rows[0]}
    json.dump(sel, open(OUT_SEL, 'w'), indent=1)
    return rows[0]


def run_eval(step, force_zero, subdir):
    ckpt = os.path.join(RUN, 'step_%06d.pt' % step)
    env = dict(os.environ)
    env.update({'T7_CKPT': ckpt, 'T7_STEP': 'final_%d' % step,
                'T7_OUT_SUBDIR': subdir, 'T7_SEEDS': SEEDS,
                'T7_MAX_STEPS': '600', 'T7_DENOISE': '5',
                'T7_FORCE_ZERO': '1' if force_zero else '0'})
    p = subprocess.run([sys.executable, EVAL], env=env, capture_output=True, text=True)
    return p.returncode, p.stdout[-2000:] + p.stderr[-2000:]


def main():
    sel = select_best()
    step = sel['step']
    rc1, log1 = run_eval(step, False, 'final_step_%d' % step)
    rc2, log2 = run_eval(step, True, 'final_step_%d_forcezero' % step)
    s1p = os.path.join(ROOT, 'final_step_%d' % step, 'SUMMARY_step_final_%d.json' % step)
    s2p = os.path.join(ROOT, 'final_step_%d_forcezero' % step, 'SUMMARY_step_final_%d.json' % step)
    s1 = json.load(open(s1p)) if os.path.exists(s1p) else {}
    s2 = json.load(open(s2p)) if os.path.exists(s2p) else {}
    compare = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'checkpoint_step': step, 'independent_seeds': [int(x) for x in SEEDS.split(',')],
               'f2_with_force': {'exact_fill_rate': s1.get('exact_fill_rate'),
                                 'exact_fill_count': s1.get('exact_fill_count'),
                                 'avg_in_target': s1.get('avg_in_target'),
                                 'safety_stop_count': s1.get('safety_stop_count')},
               'f2_force_zero': {'exact_fill_rate': s2.get('exact_fill_rate'),
                                 'exact_fill_count': s2.get('exact_fill_count'),
                                 'avg_in_target': s2.get('avg_in_target'),
                                 'safety_stop_count': s2.get('safety_stop_count')},
               'force_gain_exact_fill': (s1.get('exact_fill_rate') or 0) - (s2.get('exact_fill_rate') or 0),
               'force_gain_in_target': (s1.get('avg_in_target') or 0) - (s2.get('avg_in_target') or 0),
               'rc_with_force': rc1, 'rc_force_zero': rc2}
    json.dump(compare, open(OUT_FORCE, 'w'), indent=1)
    vids = []
    for r in s1.get('results', []):
        if 'videos' in r:
            vids.append({'seed': r['episode'], 'front': r['videos']['front'],
                         'left_wrist': r['videos']['left_wrist'], 'frames': r['steps'],
                         'exact_fill': r['exact_fill'], 'in_target': r.get('final_in_target')})
    json.dump({'worker': 'task7', 'checkpoint_step': step, 'seeds': compare['independent_seeds'],
               'videos': vids, 'root': os.path.join(ROOT, 'final_step_%d' % step)}, open(OUT_MAN, 'w'), indent=1)
    md = ['# FINAL_TEST_REPORT (independent bare model)', '',
          '- selection: %s' % sel['rule'],
          '- selected checkpoint: step %d (exact_fill %s/%d, avg_in_target %s)' % (
              step, sel['exact_fill_count'], len(compare['independent_seeds']), sel['avg_in_target']),
          '- independent seeds (never used before): %s' % compare['independent_seeds'],
          '',
          '## F2 with force', '- exact_fill %s/%d, avg_in_target %s, safety stops %s' % (
              s1.get('exact_fill_count'), len(compare['independent_seeds']), s1.get('avg_in_target'),
              s1.get('safety_stop_count')),
          '## F2 force-zero ablation', '- exact_fill %s/%d, avg_in_target %s, safety stops %s' % (
              s2.get('exact_fill_count'), len(compare['independent_seeds']), s2.get('avg_in_target'),
              s2.get('safety_stop_count')),
          '- force gain: exact_fill %+.3f, avg_in_target %+.2f' % (
              compare['force_gain_exact_fill'], compare['force_gain_in_target']),
          '', '- videos: %s' % OUT_MAN]
    open(OUT_MD, 'w').write('\n'.join(md))
    print(json.dumps({'selected': step, 'force_gain': compare['force_gain_exact_fill']}))


if __name__ == '__main__':
    main()