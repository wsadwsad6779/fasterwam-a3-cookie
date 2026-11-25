import json
import time

R = 'C:/Users/ASUS/Documents/Qoder/2026-10-31/b462d38d/_clone_default/fasterwam_fingertip_force_v1/'
P = R + 'p_items/'
TS = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

ack = {
    'directive': 'FasterWAM Task7 unfinished-items review / priority revision (received 2026-11-25T08:20Z)',
    'ts': TS,
    'queue_read': {
        'f0_step': 3410, 'f0_ts': '2026-11-25T08:22:16Z', 'f0_pid': 3287936,
        'windows': ['f0_continue', 'f0_watch'], 'f2': 'paused 15800, STOP_TRAINING',
        'pending_r_expansion': 'none: r sweep is r10/16/32 only; r24 was never run; r_partial watcher stopped',
        'actions': 'kept r10/16/32 data; no r=4/8/12/20/24/28 queued; F0 training untouched'},
    'deferrals': {
        'R_FINE_SCAN': 'DEFERRED_BY_USER: success rate too low; revisit only when the model improves',
        'H48_H64': 'DEFERRED_NOT_AUTHORIZED',
        'R_ADAPTIVE': 'DEFERRED',
        'R4_HOLDOUT': 'DEFERRED',
        'F1_A_B_C_training': 'BLOCKED_USER (spec only)'},
    'eval_failclosed': {
        'patched': ['/workspace/_t7_eval.py sha 5edaad7f8915ee584547b757b56ba07de800cd92064a713b4878ba433f2081b9',
                    '/workspace/_t7_eval_diag.py sha a1a8aae6ba9136b1b239c8df9bc916b8aba36dd831a5e449643025b55e0604d2'],
        'previous': ['37376d2cb0befdff309497b750886493ab8229415113d9ad3576e2e0c492114e',
                     'd5a9d8787debbe5230e1b4465eb3859fb3cd5fb4e2451706ee2b1311dd088816'],
        'guard': 'a non-empty T7_CKPT that does not exist raises FileNotFoundError; an empty T7_CKPT is still allowed for the base anchor',
        'behavior_change': 'none when the checkpoint exists; metrics/mapping unchanged'},
    'r_600step_audit': {
        'path': 'p_items/R_SWEEP_600STEP_AUDIT.json',
        'finding': 'the rollout stops at exactly 600 control steps; the r16/r32 chunk files nominally cover 608 but the last chunk is truncated',
        'in_box_change': 'none: in_target/exact are computed from the 600-step events; only chunk-level <0.30 stats change',
        'exec_lt030_corrected': {'r10': 0.052, 'r16': 0.06533, 'r32': 0.09517},
        'delta_pp': {'r16': 0.086, 'r32': 0.125}},
    'goal_interface': 'the runtime goal is user-set; the agent cannot rewrite its objective. This ACK + PROJECT_TASKS is the persistent queue; no next-worker takeover confirmation is available in-session.',
    'priority_next': [
        'P0: F0 6k training (running) -> 6k official 10-seed eval + videos/zip',
        'P0: eval fail-closed (done) and frozen-scene/600-step alignment (done)',
        'P1: relaxed in-box geometry audit for r16 7500 / r32 7503,7504 (CPU, next)',
        'P2: six-phase teacher, MuJoCo state clone, L_grip vs L_force gradient conflict (GPU gap: after the F0 15k hard stop or an authorized pause)',
        'F0/F2 6k/9k/12k/15k comparisons as F0 milestones land'],
    'boundaries': 'no label/eval-mapping/frozen-scene changes; F2 paused; no F0>15k; no budget expansion; no token in files'}
json.dump(ack, open(P + 'T7_DIRECTIVE_ACK_20261125T0820.json', 'w'), indent=1)

pt = json.load(open(R + 'PROJECT_TASKS.json', encoding='utf-8'))
by = {t['id']: t for t in pt['tasks']}
def add(tid, status, name, note, evidence=None):
    if tid in by:
        by[tid]['status'] = status; by[tid]['note'] = note
        if evidence: by[tid]['evidence'] = evidence
    else:
        pt['tasks'].append({'id': tid, 'status': status, 'name': name, 'note': note, 'evidence': evidence or []})
add('R_FINE_SCAN', 'DEFERRED_BY_USER', 'fine r scan (r=4/8/12/20/24/28)', 'not run; user says success is too low; revisit only after stable improvement')
add('H48_H64', 'DEFERRED_NOT_AUTHORIZED', 'action horizon 48/64 study', 'not authorized; no replan_steps alias and no training')
by['R3_ADAPTIVE_REPLAN']['status'] = 'DEFERRED'
by['R3_ADAPTIVE_REPLAN']['note'] = 'deferred; no validated scheduler; do not overfit to single seeds'
by['R4_HOLDOUT_CONFIRMATION']['status'] = 'DEFERRED'
by['R4_HOLDOUT_CONFIRMATION']['note'] = 'deferred; no independent holdout opened'
add('P0_EVAL_FAILCLOSED', 'DONE', 'eval scripts fail closed on missing checkpoint',
    'T7_CKPT set but missing raises; empty T7_CKPT allowed for base anchor; behavior otherwise unchanged',
    ['p_items/T7_DIRECTIVE_ACK_20261125T0820.json'])
add('P1_600STEP_AUDIT', 'DONE', 'r16/r32 600-step truncation audit',
    'in_target/exact unchanged; exec<0.30 corrected r16 6.533% (+0.086pp), r32 9.517% (+0.125pp); r10 unchanged',
    ['p_items/R_SWEEP_600STEP_AUDIT.json'])
add('P1_RELAXED_GEOMETRY', 'READY', 'relaxed in-box geometry audit (r16 7500 / r32 7503,7504)',
    'per-cookie position/contact/video review; orientation/contact fields may be NOT_VERIFIED if not recorded')
add('P2_TEACHER_6PHASE', 'DEFERRED_GPU_GAP', 'six-phase teacher-forced across independent episodes',
    'requires the F2 15k model; waits for the F0 15k hard stop or an authorized pause; not concurrent with F0')
add('P2_STATE_CLONE', 'DEFERRED_GPU_GAP', 'MuJoCo state clone 0.43 vs 0.28 counterfactual', 'GPU gap')
add('P2_GRAD_CONFLICT', 'DEFERRED_GPU_GAP', 'L_grip vs L_force gradient cosine/norm on shared params', 'GPU gap')
add('P3_PERF', 'DEFERRED', 'visual/T5 feature cache and perf profiling', 'no training-math change')
pt['updated_at'] = TS
json.dump(pt, open(R + 'PROJECT_TASKS.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=True)

s = json.load(open(P + 'SUPERVISOR_PROGRESS.json'))
s['updated_at'] = TS
s['directive_ack'] = 'p_items/T7_DIRECTIVE_ACK_20261125T0820.json'
s['r_sweep']['fine_scan'] = 'DEFERRED_BY_USER'
s['r_sweep']['600step_audit'] = 'p_items/R_SWEEP_600STEP_AUDIT.json (exec r16 +0.086pp, r32 +0.125pp; in-box unchanged)'
s['eval_failclosed'] = 'patched _t7_eval.py 5edaad7f and _t7_eval_diag.py a1a8aae6'
json.dump(s, open(P + 'SUPERVISOR_PROGRESS.json', 'w'), indent=1)

r = json.load(open(P + 'R_SWEEP_PROGRESS.json'))
r['updated_utc'] = TS
r['fine_scan'] = 'DEFERRED_BY_USER (user 2026-11-25T08:20Z): r=4/8/12/20/24/28 not run; r10/16/32 kept'
r['six_zero_step_audit'] = {'path': 'p_items/R_SWEEP_600STEP_AUDIT.json',
                            'exec_lt030_corrected': {'r10': 0.052, 'r16': 0.06533, 'r32': 0.09517},
                            'in_box_change': 'none'}
json.dump(r, open(P + 'R_SWEEP_PROGRESS.json', 'w'), indent=1)

r0 = json.load(open(P + 'R0_PROTOCOL_FINGERPRINT.json'))
r0['updated_utc'] = TS
r0['eval_failclosed_patch'] = {'old_official': '37376d2c', 'new_official': '5edaad7f',
                               'old_diag': 'd5a9d878', 'new_diag': 'a1a8aae6',
                               'reason': 'missing T7_CKPT must raise; empty T7_CKPT still allowed for base anchor',
                               'mapping_change': 'none'}
json.dump(r0, open(P + 'R0_PROTOCOL_FINGERPRINT.json', 'w'), indent=1)
print('directive ACK + ledgers updated')