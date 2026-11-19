import json
from pathlib import Path

OUT = Path('/workspace/eval_restore_20261117/reconstructed_normalizer/c_class_eval_v3_step8000')
a = json.loads((OUT / 'A_summary.json').read_text(encoding='utf-8'))
c = json.loads((OUT / 'C_summary.json').read_text(encoding='utf-8'))
MARKERS = {k: v for k, v in c.items() if k not in ('arm', 'facts', 'load', 'episodes')}

PAIR_KEYS = ('cookies_final', 'cookies_max', 'source_final', 'exact_fill_any',
             'force_envelope', 'jam_steps', 'eef_path_mm', 'success_frames',
             'grasps_confirmed', 'gripper_close_events', 'grasp_success_rate')

cmp = {'per_seed': []}
for ea in a['episodes']:
    seed = ea['record']['seed']
    ec = next(x for x in c['episodes'] if x['record']['seed'] == seed)
    cmp['per_seed'].append({
        'seed': seed,
        'A': {k: ea.get(k) for k in PAIR_KEYS},
        'C': {k: ec.get(k) for k in PAIR_KEYS},
        'delta_cookies': ec['cookies_final'] - ea['cookies_final'],
        'delta_force_absmax': ec['force_envelope']['absmax'] - ea['force_envelope']['absmax'],
    })
cmp['C_beats_A_on_any_seed'] = any(p['delta_cookies'] > 0 for p in cmp['per_seed'])
cmp['C_exact_fill_any_seed'] = any(e['exact_fill_any'] for e in c['episodes'])
cmp['C_success_any_seed'] = any(e['record']['success'] for e in c['episodes'])


def brief(e):
    return {'seed': e['record']['seed'], 'success': e['record']['success'],
            'cookies_final': e['cookies_final'], 'cookies_max': e['cookies_max'],
            'source_final': e['source_final'], 'exact_fill_any': e['exact_fill_any'],
            'force_absmax': e['force_envelope']['absmax'], 'jam_steps': e['jam_steps'],
            'safety_abort': e['record']['safety_abort'],
            'grasps_confirmed': e.get('grasps_confirmed'),
            'gripper_close_events': e.get('gripper_close_events'),
            'grasp_success_rate': e.get('grasp_success_rate'),
            'video': e.get('video')}


summary = {**MARKERS, 'complete': True, 'seeds': [29, 30], 'resumed': True,
           'A': [brief(e) for e in a['episodes']],
           'C': [brief(e) for e in c['episodes']],
           'paired': cmp}
(OUT / 'paired_A_vs_C.json').write_text(json.dumps({**MARKERS, **cmp}, indent=2, default=str))
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, default=str))
print('REBUILD_OK')
for p in cmp['per_seed']:
    print('seed', p['seed'], 'A grasps', p['A']['grasps_confirmed'], 'C grasps',
          p['C']['grasps_confirmed'], 'C close_events', p['C']['gripper_close_events'],
          'C force_absmax', round(p['C']['force_envelope']['absmax'], 2),
          'C video', (p['C'].get('video') or {}).get('path'))
