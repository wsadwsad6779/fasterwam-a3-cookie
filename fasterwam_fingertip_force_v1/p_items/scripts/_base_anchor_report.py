import hashlib
import json
import os
import time

P = '/workspace/fwv1/p_items'
EV = '/workspace/fwv1/eval_fingertip_v1'
TS = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def means(d):
    rs = (d or {}).get('results', [])
    if not rs:
        return None
    n = len(rs)
    return {'n': n,
            'mean_in_target': sum((r.get('final_in_target') or 0) for r in rs) / n,
            'exact_fill_rate': sum(1 for r in rs if r.get('exact_fill')) / n,
            'mean_grasps': sum((r.get('grasps') or 0) for r in rs) / n,
            'mean_carry': sum(((r.get('force_stats') or {}).get('carry_force_frac_gt_0p5') or 0) for r in rs) / n,
            'open_cmd_mean': sum(((r.get('action_stats') or {}).get('open_cmd_frac') or 0) for r in rs) / n}


base = load(EV + '/base_v31_thinpads/SUMMARY_step_base.json')
f0 = load(EV + '/cmp_f0_3k/SUMMARY_step_3000.json')
f2 = load(EV + '/cmp_f2_3k/SUMMARY_step_3000.json')
mb, m0, m2 = means(base), means(f0), means(f2)
out = {'ts': TS, 'artifact': 'BASE_ANCHOR_V31',
       'protocol': 'official base step_021700, no trainable checkpoint, V3.1 thin pad, seeds 7100-7109',
       'base': {'checkpoint_sha256': '01be4efca565d1890fd8a7dc1507b01c00c8cc63c16ae252287492ba9ac7550d',
                'summary': EV + '/base_v31_thinpads/SUMMARY_step_base.json',
                'summary_checkpoint_sha256': (base or {}).get('checkpoint_sha256'),
                'evidence_note': ('base run came from the cmp_f2_3k eval whose F2 checkpoint path was missing; '
                                  'the summary checkpoint_sha256 is null, so build_model used the official base'),
                'means': mb,
                'videos': len((load(EV + '/base_v31_thinpads/VIDEO_MANIFEST_step_base.json') or {}).get('videos', []))},
       'f0_3k': {'summary': EV + '/cmp_f0_3k/SUMMARY_step_3000.json', 'means': m0},
       'f2_3k': {'summary': EV + '/cmp_f2_3k/SUMMARY_step_3000.json', 'means': m2},
       'frozen_env': {
           'eval_script_sha256': sha('/workspace/_t7_eval.py'),
           'sim_code_sha256': sha('/workspace/fwv1/sim_repo/src/a3_dual_arm_sim/tasks/cookie_transfer.py'),
           'sim_config_sha256': sha('/workspace/fwv1/sim_repo/configs/cookie_batch.yaml'),
           'scene_manifest_sha256': sha('/workspace/fwv1/p_items/EVAL_SCENE_MANIFEST_V3_1.json'),
           'freeze_sha256': (load('/workspace/fwv1/p_items/EVAL_SCENE_MANIFEST_V3_1.json') or {}).get('freeze_sha256')},
       'reading': 'anchor for whether the 3k F0/F2 numbers are above or below the untrained official base on the same 10 scenes'}
json.dump(out, open(P + '/BASE_ANCHOR_V31.json', 'w'), indent=1)
lines = ['# Base anchor V3.1 (official step_021700, seeds 7100-7109)', '',
         '| model | in_target | exact | grasps | carry>0.5 | open_cmd |', '|---|---|---|---|---|---|',
         '| base | %s | %s | %s | %s | %s |' % tuple(
             None if mb is None else round(mb[k], 4) for k in
             ('mean_in_target', 'exact_fill_rate', 'mean_grasps', 'mean_carry', 'open_cmd_mean')),
         '| F0 3k | %s | %s | %s | %s | %s |' % tuple(
             None if m0 is None else round(m0[k], 4) for k in
             ('mean_in_target', 'exact_fill_rate', 'mean_grasps', 'mean_carry', 'open_cmd_mean')),
         '| F2 3k | %s | %s | %s | %s | %s |' % tuple(
             None if m2 is None else round(m2[k], 4) for k in
             ('mean_in_target', 'exact_fill_rate', 'mean_grasps', 'mean_carry', 'open_cmd_mean'))]
open(P + '/BASE_ANCHOR_V31.md', 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('WROTE', P + '/BASE_ANCHOR_V31.json', 'base_means', mb)