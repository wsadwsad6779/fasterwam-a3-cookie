import glob
import json
import os
import subprocess
import time

P = '/workspace/fwv1/p_items'
EV = '/workspace/fwv1/eval_fingertip_v1'
TS = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def eval_procs():
    out = subprocess.run(['pgrep', '-af', '_t7_eval'], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if l.strip()]


def env_of(pid):
    try:
        raw = open('/proc/%d/environ' % pid, 'rb').read().split(b'\0')
        d = {}
        for kv in raw:
            if b'=' in kv:
                k, v = kv.split(b'=', 1)
                d[k.decode(errors='replace')] = v.decode(errors='replace')
        return d
    except Exception:
        return {}


ps = eval_procs()
state = {'ts': TS, 'stage': 'idle', 'pid': None, 'status': 'idle',
         'log_path': '/workspace/fwv1/logs/f0_eval.log',
         'expected_artifacts': ['cmp_f0_3k', 'cmp_f2_3k',
                                'cmp_f0_3k/videos_f0_step_3000.zip',
                                'cmp_f2_3k/videos_f2_step_3000.zip'],
         'next': ''}
if ps:
    pid = int(ps[0].split()[0])
    e = env_of(pid)
    sub = e.get('T7_OUT_SUBDIR', '?')
    step = e.get('T7_STEP', '?')
    root = os.path.join(EV, sub)
    seeds = sorted(glob.glob(root + '/seed_*')) if os.path.isdir(root) else []
    try:
        st = subprocess.run(['ps', '-o', 'lstart=', '-p', str(pid)],
                            capture_output=True, text=True).stdout.strip()
        start_epoch = time.mktime(time.strptime(st))
    except Exception:
        start_epoch = 0.0
    done = 0
    for sd in seeds:
        ep = os.path.join(sd, 'events.json')
        if os.path.exists(ep) and os.path.getmtime(ep) >= start_epoch:
            done += 1
    state.update({'stage': sub, 'step': step, 'pid': pid, 'status': 'running',
                  'seeds_done': done, 'seed_dirs_total': len(seeds),
                  'expected_artifacts': [sub], 'checkpoint': e.get('T7_CKPT', ''),
                  'next': 'remaining seeds, then next eval phase'})
else:
    f0 = os.path.exists(EV + '/cmp_f0_3k/SUMMARY_step_3000.json')
    f2 = os.path.exists(EV + '/cmp_f2_3k/SUMMARY_step_3000.json')
    if f2:
        state.update({'stage': '3k_evals_done', 'status': 'done', 'seeds_done': 10,
                      'next': 'cmp 3k + zip + FROZEN_ENV_3K_CHECK'})
    elif f0:
        state.update({'stage': 'f0_3k_done_waiting_f2', 'status': 'between', 'seeds_done': 10,
                      'next': 'F2 3k paired eval'})
    else:
        state.update({'stage': 'between_evals', 'status': 'idle', 'next': 'F0 3k eval start'})
tmp = P + '/POD_JOB_STATE_EVAL_3K.json.tmp'
json.dump(state, open(tmp, 'w'), indent=1)
os.replace(tmp, P + '/POD_JOB_STATE_EVAL_3K.json')
print(json.dumps(state))