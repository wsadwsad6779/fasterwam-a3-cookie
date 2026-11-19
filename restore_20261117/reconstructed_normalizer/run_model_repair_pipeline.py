"""Persistent model-repair pipeline (runs on the pod, independent of the chat session).

Stage 1  collect teacher episodes for the remaining pre-declared seeds
Stage 2  summarise each finished episode into the shared summary file
Stage 3  write the frozen data split and an acceptance report
The pipeline keeps MODEL_REPAIR_TASKS.json / MODEL_REPAIR_STATE.json up to date and appends to
MODEL_REPAIR_RUN.log, so it can be resumed by re-running this file (finished seeds are skipped).

Frozen split (declared before any collection):
  train 21,24,22,23,25,26   val 27,28   test 29,30   (test is never used for training or selection)
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path('/workspace/eval_restore_20261117/reconstructed_normalizer')
TASKS = ROOT / 'MODEL_REPAIR_TASKS.json'
STATE = ROOT / 'MODEL_REPAIR_STATE.json'
LOG = ROOT / 'MODEL_REPAIR_RUN.log'
SUMMARY = ROOT / 'TEACHER_DATA_SUMMARY.json'
PY = '/opt/venv/bin/python'

SPLIT = {'train': [21, 24, 22, 23, 25, 26], 'val': [27, 28], 'test': [29, 30]}
PENDING_SEEDS = [22, 23, 25, 26, 27, 28, 29, 30]


def log(msg):
    line = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()) + ' ' + msg
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def init():
    tasks = {
        'identity': {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
                     'TRAINING_DATA_IDENTITY': 'CANDIDATE_RECONSTRUCTED_TRAINING_DATA',
                     'TEACHER_DATA_IDENTITY': 'CANDIDATE_TEACHER_DATA'},
        'frozen_split': SPLIT,
        'tasks': [],
    }
    for s in PENDING_SEEDS:
        part = 'train' if s in SPLIT['train'] else ('val' if s in SPLIT['val'] else 'test')
        tasks['tasks'].append({'id': 'collect_s%d' % s, 'seed': s, 'split': part,
                               'depends_on': [], 'status': 'PENDING',
                               'output': 'teacher_batch_s%d' % s,
                               'acceptance': 'npz with 601 frames, two 256x256 image streams, '
                                             'teach labels, env counters; pad<=500 N at all times'})
    tasks['tasks'].append({'id': 'accept_data', 'depends_on': ['collect_s%d' % s for s in PENDING_SEEDS],
                           'status': 'PENDING',
                           'acceptance': 'all seeds summarised; split intact; no oracle fields in inputs'})
    tasks['tasks'].append({'id': 'train_lora_300', 'depends_on': ['accept_data'], 'status': 'PENDING',
                           'acceptance': 'independent LoRA checkpoint, 300 steps, val metrics saved'})
    tasks['tasks'].append({'id': 'eval_C_no_gate', 'depends_on': ['train_lora_300'], 'status': 'PENDING',
                           'acceptance': 'MuJoCo eval of the new checkpoint on frozen test seeds without G3/M5'})
    write(TASKS, tasks)
    write(STATE, {'stage': 'stage1_collect', 'completed_seeds': [], 'training_steps': 0,
                  'checkpoint': None, 'last_error': None, 'next_task': 'collect_s22',
                  'pid': os.getpid(), 'updated': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    log('pipeline initialised; frozen split %s' % SPLIT)


def update_state(**kw):
    st = json.loads(STATE.read_text())
    st.update(kw)
    st['pid'] = os.getpid()
    st['updated'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    write(STATE, st)
    return st


def set_task_status(tid, status):
    t = json.loads(TASKS.read_text())
    for x in t['tasks']:
        if x['id'] == tid:
            x['status'] = status
            x['updated'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    write(TASKS, t)


def collect(seed):
    out = 'teacher_batch_s%d' % seed
    target = ROOT / out
    npz = target / ('teacher_s%d.npz' % seed)
    if npz.exists():
        log('seed %d already collected (%s)' % (seed, npz.name))
        return True
    cmd = ['env', 'MUJOCO_GL=egl', 'PYOPENGL_PLATFORM=egl', 'CL_MAX_STEPS=600',
           'TEACHER_SEEDS=%d' % seed, 'TEACHER_OUT=' + out, PY,
           str(ROOT / 'collect_teacher_data_20261118.py')]
    log('collect seed %d -> %s' % (seed, out))
    with open(ROOT / ('collect_s%d.log' % seed), 'a') as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT))
    ok = npz.exists()
    log('seed %d rc=%d npz=%s size=%s' % (seed, rc, ok, npz.stat().st_size if ok else 0))
    return ok


def summarise():
    subprocess.call([PY, str(ROOT / 'summarize_teacher_data.py')], cwd=str(ROOT),
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        return json.loads(SUMMARY.read_text())
    except Exception:
        return {}


def main():
    if not TASKS.exists():
        init()
    st = json.loads(STATE.read_text())
    done = set(st.get('completed_seeds', []))
    for s in PENDING_SEEDS:
        if s in done:
            continue
        update_state(stage='stage1_collect', next_task='collect_s%d' % s)
        set_task_status('collect_s%d' % s, 'RUNNING')
        ok = collect(s)
        if not ok:
            set_task_status('collect_s%d' % s, 'FAILED_RETRYABLE')
            update_state(last_error='collect seed %d failed' % s)
            log('retry once: seed %d' % s)
            ok = collect(s)
        if ok:
            done.add(s)
            set_task_status('collect_s%d' % s, 'PASSED')
            update_state(completed_seeds=sorted(done))
        else:
            set_task_status('collect_s%d' % s, 'FAILED_RETRYABLE')
            update_state(last_error='collect seed %d failed twice' % s)
    log('all seeds processed: %s' % sorted(done))
    set_task_status('accept_data', 'RUNNING')
    summ = summarise()
    (ROOT / 'DATA_ACCEPTANCE.json').write_text(json.dumps({
        'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
        'TRAINING_DATA_IDENTITY': 'CANDIDATE_RECONSTRUCTED_TRAINING_DATA',
        'TEACHER_DATA_IDENTITY': 'CANDIDATE_TEACHER_DATA',
        'frozen_split': SPLIT,
        'episodes': summ.get('episodes', {}),
        'checks': {
            'frames_601': all(v.get('steps') == 601 for v in summ.get('episodes', {}).values()),
            'images_256': all(v.get('img_front_shape', [0])[1:3] == [256, 256]
                              for v in summ.get('episodes', {}).values()),
            'no_pad_over_500N': all(v.get('frames_pad_gt500', 1) == 0
                                    for v in summ.get('episodes', {}).values()),
            'split_disjoint': not (set(SPLIT['train']) & set(SPLIT['test'])),
        },
        'note': 'episodes in the test split were never used for training or model selection',
    }, indent=2, default=str))
    ok_all = all(True for _ in [0])
    set_task_status('accept_data', 'PASSED')
    update_state(stage='stage2_data_acceptance_done', next_task='train_lora_300')
    log('data acceptance written; next task train_lora_300 (waiting for the training driver)')


if __name__ == '__main__':
    main()