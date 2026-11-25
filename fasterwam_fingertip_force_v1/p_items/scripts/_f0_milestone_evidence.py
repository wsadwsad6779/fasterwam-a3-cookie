import hashlib
import json
import os
import time

P = '/workspace/fwv1/p_items'
EV = '/workspace/fwv1/eval_fingertip_v1'
STEP = os.environ['F0_STEP']
SUB = os.environ['F0_SUBDIR']
LABEL = int(STEP) // 1000
TS = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


summ = EV + '/' + SUB + '/SUMMARY_step_%s.json' % STEP
zp = EV + '/' + SUB + '/videos_f0_step_%s.zip' % STEP
man = EV + '/' + SUB + '/VIDEO_MANIFEST_f0_step_%s.json' % STEP
d = json.load(open(summ))
rs = d.get('results', [])
n = len(rs)
fm = [((r.get('force_stats') or {}).get('force_max') or 0) for r in rs]
out = {'ts': TS, 'artifact': 'F0_EVAL_%dK_DONE' % LABEL, 'eval_dir': EV + '/' + SUB,
       'summary': {'path': summ, 'sha256': sha(summ), 'seeds': n,
                   'avg_in_target': d.get('avg_in_target'),
                   'exact_fill_rate': d.get('exact_fill_rate'),
                   'grasps_mean': sum((r.get('grasps') or 0) for r in rs) / n if n else None,
                   'open_cmd_mean': (d.get('action_stats_mean') or {}).get('open_cmd_frac'),
                   'close_cmd_frac_mean': (d.get('action_stats_mean') or {}).get('grip_close_frac'),
                   'force_max_mean': sum(fm) / n if n else None},
       'checkpoint': d.get('checkpoint'), 'checkpoint_sha256': d.get('checkpoint_sha256'),
       'videos': {'count': len((json.load(open(man)).get('videos', [])) if os.path.exists(man) else []),
                  'zip': zp,
                  'zip_bytes': os.path.getsize(zp) if os.path.exists(zp) else None,
                  'zip_sha256': sha(zp) if os.path.exists(zp) else None},
       'frozen_env': {
           'eval_script_sha256': sha('/workspace/_t7_eval.py'),
           'sim_code_sha256': sha('/workspace/fwv1/sim_repo/src/a3_dual_arm_sim/tasks/cookie_transfer.py'),
           'sim_config_sha256': sha('/workspace/fwv1/sim_repo/configs/cookie_batch.yaml'),
           'scene_manifest_sha256': sha('/workspace/fwv1/p_items/EVAL_SCENE_MANIFEST_V3_1.json'),
           'freeze_sha256': json.load(open('/workspace/fwv1/p_items/EVAL_SCENE_MANIFEST_V3_1.json')).get('freeze_sha256')},
       'scene_all_match': (d.get('scene_manifest') or {}).get('all_match'),
       'note': 'F0 %s-step official milestone evidence; zip 10 composites' % STEP}
json.dump(out, open(P + '/F0_EVAL_%dK_DONE.json' % LABEL, 'w'), indent=1)
try:
    st = json.load(open(P + '/POD_JOB_STATE_F0.json'))
    st['last_milestone_eval'] = {'step': int(STEP), 'sub': SUB, 'summary_sha256': out['summary']['sha256'],
                                 'videos': out['videos']['count'], 'zip_sha256': out['videos']['zip_sha256'],
                                 'ts': out['ts']}
    json.dump(st, open(P + '/POD_JOB_STATE_F0.json', 'w'), indent=1)
except Exception:
    pass
ms = []
if os.path.exists(P + '/F0_MILESTONES.json'):
    try:
        ms = json.load(open(P + '/F0_MILESTONES.json'))
    except Exception:
        ms = []
ms.append({'step': int(STEP), 'sub': SUB, 'evidence': 'p_items/F0_EVAL_%dK_DONE.json' % LABEL,
           'summary_sha256': out['summary']['sha256'], 'zip_sha256': out['videos']['zip_sha256'],
           'videos': out['videos']['count'], 'ts': out['ts']})
json.dump(ms, open(P + '/F0_MILESTONES.json', 'w'), indent=1)
print('WROTE F0_EVAL_%dK_DONE.json videos=%s zip=%s' % (
    LABEL, out['videos']['count'], out['videos']['zip_bytes']))