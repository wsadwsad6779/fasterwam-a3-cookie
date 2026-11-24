"""G-B regression: verify the frozen eval protocol and action decoding.

Checks (frozen protocol):
  * label->env scaling: translation x2 (0.05/0.025), rotation x5 (0.5/0.10)
  * gripper decode: g = 2*opening - 1, clipped to [-1,1]
  * replan = 10, action horizon = 32
  * exact_fill: info.cookies_in_target == 10 and cookies_in_source == 70, held 20 steps
  * no T7_REPLAN environment override in the file
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/fwv1/repo/stage_40k/fw/src')

OUT = '/workspace/fwv1/p_items/G_B_EVAL_REGRESSION.json'


def main():
    import t7_eval as E
    checks = []

    def ck(name, ok, detail):
        checks.append({'name': name, 'ok': bool(ok), 'detail': detail})

    ck('trans_scale', E.TRANS_SCALE == 0.025, E.TRANS_SCALE)
    ck('rot_scale', E.ROT_SCALE == 0.10, E.ROT_SCALE)
    ck('label_trans_scale', E.LABEL_TRANS_SCALE == 0.05, E.LABEL_TRANS_SCALE)
    ck('label_rot_scale', E.LABEL_ROT_SCALE == 0.5, E.LABEL_ROT_SCALE)
    ck('trans_factor', E.LABEL_TRANS_SCALE / E.TRANS_SCALE == 2.0,
       E.LABEL_TRANS_SCALE / E.TRANS_SCALE)
    ck('rot_factor', E.LABEL_ROT_SCALE / E.ROT_SCALE == 5.0,
       E.LABEL_ROT_SCALE / E.ROT_SCALE)
    ck('replan', E.REPLAN == 10, E.REPLAN)
    ck('horizon', E.ACTION_HORIZON == 32, E.ACTION_HORIZON)

    a = np.array([0.1, -0.2, 0.05, 0.1, 0.0, -0.2, 0.75], np.float64)
    trans = np.clip(a[:3] * (E.LABEL_TRANS_SCALE / E.TRANS_SCALE), -1, 1)
    rot = np.clip(a[3:6] * (E.LABEL_ROT_SCALE / E.ROT_SCALE), -1, 1)
    grip = float(np.clip(2.0 * a[6] - 1.0, -1.0, 1.0))
    ck('trans_values', np.allclose(trans, [0.2, -0.4, 0.1]), trans.tolist())
    ck('rot_values', np.allclose(rot, [0.5, 0.0, -1.0]), rot.tolist())
    ck('grip_value', abs(grip - 0.5) < 1e-9, grip)
    grips = {0.0: -1.0, 0.28: -0.44, 0.49: -0.02, 1.0: 1.0}
    ck('grip_decode_table',
       all(abs(float(np.clip(2.0 * k - 1.0, -1, 1)) - v) < 1e-9 for k, v in grips.items()),
       {str(k): float(np.clip(2.0 * k - 1.0, -1, 1)) for k in grips})

    src = open(E.__file__, 'r', encoding='utf-8').read()
    ck('exact_fill_condition', "info.get('cookies_in_target') == 10 and src == 70" in src,
       'cookies_in_target==10 and cookies_in_source==70')
    ck('exact_fill_hold', 'if hold >= 20:' in src, 'hold>=20')
    ck('replan_loop', 'for k in range(min(ACTION_HORIZON, REPLAN)):' in src, 'min(horizon, replan)')
    ck('no_replan_env', 'T7_REPLAN' not in src, 'no env override')
    ck('grip_decode_src', '2.0 * a[6] - 1.0' in src, 'g=2a-1')
    ck('safety_guard_present', 'safety_reason' in src, 'records safety_reason')

    h = hashlib.sha256(open(E.__file__, 'rb').read()).hexdigest()
    out = {'worker': 'supervisor', 'checks': checks, 't7_eval': E.__file__,
           't7_eval_sha256': h,
           'verdict': 'pass' if all(c['ok'] for c in checks) else 'fail'}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(json.dumps({'verdict': out['verdict'], 'failed': [c for c in checks if not c['ok']],
                      't7_eval_sha256': h}, indent=1))
    if out['verdict'] != 'pass':
        raise SystemExit(1)


if __name__ == '__main__':
    main()