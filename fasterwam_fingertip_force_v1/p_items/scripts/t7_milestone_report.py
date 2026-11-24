"""Milestone report/gate for the approved V3 run.

At 3000 steps it enforces the G-D contract (close<=0.95, open>=0.05, reopen>=1,
carry force frac>0.5N >=0.95) plus task success (exact_fill>=1), runs a
root-cause scan (scene / labels / sampler / eval mapping / checkpoint / metrics),
builds a downloadable video manifest (per-file URLs + one zip), and touches
/workspace/fwv1/STOP_TRAINING on failure. Exit codes: 0 pass, 2 gate fail, 1
fundamental error/missing data.
"""
import glob
import hashlib
import json
import math
import os
import sys
import time
import zipfile
from urllib.parse import quote

STEP = int(os.environ.get('T7_STEP', '3000'))
LABEL = os.environ.get('T7_STEP_LABEL', str(STEP))
RUN = os.environ.get('T7_RUN_DIR', '/workspace/fwv1/train_runs/t7_21k_v3run')
ROOT = os.environ.get('T7_EVAL_ROOT', '/workspace/fwv1/eval_fingertip_v1')
SUBDIR = os.environ.get('T7_EVAL_SUBDIR', 'step_%06d' % STEP)
EVAL_DIR = os.path.join(ROOT, SUBDIR)
SUM = os.path.join(EVAL_DIR, 'SUMMARY_step_%s.json' % LABEL)
VM = os.path.join(EVAL_DIR, 'VIDEO_MANIFEST_step_%s.json' % LABEL)
POD = os.environ.get('T7_POD_URL', 'https://jupyter.anruicloud.com/instances/rgapi-one-click-24316-57b6b870')
DRY = os.environ.get('T7_REPORT_DRY', '0') == '1'
PREFLIGHT = os.environ.get('T7_REPORT_PREFLIGHT', '0') == '1'
PRE_OUT = '/workspace/fwv1/p_items/PREFLIGHT_3K.json'
OUTDIR = '/tmp' if DRY else '/workspace/fwv1/p_items'
TAG = ('DRY_' if DRY else '') + ('G_D_3K' if STEP == 3000 else 'REPORT_step_%d' % STEP)
OUT = os.path.join(OUTDIR, TAG + '.json')
MD = os.path.join(OUTDIR, TAG + '.md')
VOUT = os.path.join(OUTDIR, ('DRY_' if DRY else '') + 'VIDEO_DOWNLOAD_step_%d.json' % STEP)
STOP = '/workspace/fwv1/STOP_TRAINING'
BASE_SHA = '01be4efca565d1890fd8a7dc1507b01c00c8cc63c16ae252287492ba9ac7550d'
LABELS = '/workspace/fwv1/p_items/A3_845_7D_LABELS_V3.npz'
MANIFEST = '/workspace/fwv1/p_items/EVAL_SCENE_MANIFEST_V3.json'


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def url_of(rel):
    return POD + '/api/contents/' + quote(rel, safe='/') + '?download=1'


def preflight():
    """Static fail-closed checks against the real pod files (no 3k artifacts needed)."""
    ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    fund = []

    def fadd(name, ok, detail):
        fund.append({'name': name, 'ok': bool(ok), 'detail': detail})

    labels_sha = sha256(LABELS) if os.path.exists(LABELS) else None
    fadd('labels_v3_sha', labels_sha == 'c4090dd2cb994a8b3524b4c2614b869035acb89dccb7c8eed0e9dbe63e4affff', labels_sha)
    files = {}
    for p in ('/workspace/t7_data.py', '/workspace/t7_train.py', '/workspace/_t7_eval.py',
              '/workspace/t7_milestone_report.py'):
        files[p] = {'sha256': sha256(p) if os.path.exists(p) else None,
                    'bytes': os.path.getsize(p) if os.path.exists(p) else None}
    data_src = open('/workspace/t7_data.py', encoding='utf-8').read() if os.path.exists('/workspace/t7_data.py') else ''
    fadd('sampler_v3_no_pm1', ('g >= 0.8' in data_src) and ('g <= -0.8' not in data_src)
         and ('release_rise' in data_src) and ('T7_FORCE_DROPOUT' in data_src), '/workspace/t7_data.py')
    train_src = open('/workspace/t7_train.py', encoding='utf-8').read() if os.path.exists('/workspace/t7_train.py') else ''
    fadd('force_head_train', ('force_head' in train_src) and ('loss_force' in train_src)
         and ("bb['grip_raw']" in train_src), '/workspace/t7_train.py')
    ev_src = open('/workspace/_t7_eval.py', encoding='utf-8').read() if os.path.exists('/workspace/_t7_eval.py') else ''
    fadd('eval_mapping_frozen', all(x in ev_src for x in [
        '2.0 * a[6] - 1.0', 'REPLAN = 10', 'LABEL_ROT_SCALE = 0.5', 'ROT_SCALE = 0.10',
        "info.get('cookies_in_target') == 10 and src == 70"]), '/workspace/_t7_eval.py')
    try:
        m = json.load(open(MANIFEST))
        canonical = hashlib.sha256(json.dumps(
            {'base_options': m['base_options'],
             'scenes': [{'seed': x['seed'], 'hash': x['hash']} for x in m['scenes']]},
            sort_keys=True).encode()).hexdigest()
        fadd('scene_manifest_freeze', canonical == m.get('freeze_sha256')
             and len(m['scenes']) == 80 and len({x['hash'] for x in m['scenes']}) == 80, canonical)
    except Exception as exc:
        fadd('scene_manifest_freeze', False, str(exc))
    metrics = []
    mp = os.path.join(RUN, 'metrics.jsonl')
    if os.path.exists(mp):
        try:
            for line in open(mp, encoding='utf-8').read().strip().split('\n')[-100:]:
                if line.strip():
                    metrics.append(json.loads(line))
        except Exception:
            metrics = []
    finite = bool(metrics) and all(
        math.isfinite(float(r.get('loss', 0.0))) and math.isfinite(float(r.get('loss_force') or 0.0))
        for r in metrics)
    mean_train_open = (sum(r.get('open_frac', 0) for r in metrics) / len(metrics)) if metrics else None
    mean_train_close = (sum(r.get('close_frac', 0) for r in metrics) / len(metrics)) if metrics else None
    diff_frac = (sum(1 for r in metrics if abs(r.get('close_frac', 0) - r.get('open_frac', 0)) > 0.05)
                 / len(metrics)) if metrics else None
    fadd('train_metrics_finite', finite, {'records': len(metrics),
                                          'last_step': metrics[-1].get('step') if metrics else None})
    fadd('train_sampler_balanced', (mean_train_open is not None and mean_train_open >= 0.03
                                    and diff_frac is not None and diff_frac >= 0.5),
         {'mean_open': mean_train_open, 'mean_close': mean_train_close, 'close_neq_open_frac': diff_frac})
    out = {'worker': 'task7', 'ts': ts, 'kind': '3k_preflight', 'run_dir': RUN,
           'metrics_path': mp, 'checks': fund, 'files': files,
           'all_ok': all(x['ok'] for x in fund)}
    json.dump(out, open(PRE_OUT, 'w'), indent=1)
    print(json.dumps(out, indent=1))
    return 0 if out['all_ok'] else 1


def main():
    if PREFLIGHT:
        return preflight()
    ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    checks = []
    per_seed = []
    videos = []
    zip_info = None
    if not os.path.exists(SUM):
        verdict = {'worker': 'task7', 'ts': ts, 'step': STEP, 'verdict': 'failed',
                   'reason': 'summary_missing', 'summary': SUM}
        json.dump(verdict, open(OUT, 'w'), indent=1)
        open(MD, 'w').write('# step %d report\n\nsummary missing: %s\n' % (STEP, SUM))
        if not DRY:
            open(STOP, 'w').write('step %d report missing %s\n' % (STEP, ts))
        print(json.dumps({'verdict': 'failed', 'reason': 'summary_missing'}))
        return 2

    s = json.load(open(SUM))
    results = s.get('results', []) or []
    close = (s.get('action_stats_mean') or {}).get('grip_close_frac')
    openf = (s.get('action_stats_mean') or {}).get('open_cmd_frac')
    reopen_vals = [((r.get('action_stats') or {}).get('reopen_events')) for r in results
                   if r.get('action_stats')]
    reopen_vals = [v for v in reopen_vals if v is not None]
    carry = (s.get('force_stats_mean') or {}).get('carry_force_frac_gt_0p5')
    exact = s.get('exact_fill_count')
    scene = s.get('scene_manifest') or {}
    protocol = s.get('protocol') or {}

    def add(name, ok, detail):
        checks.append({'name': name, 'ok': bool(ok), 'detail': detail})

    gd = {}
    if STEP == 3000:
        gd = {
            'close_le_0.95': {'value': close, 'pass': close is not None and close <= 0.95},
            'open_ge_0.05': {'value': openf, 'pass': openf is not None and openf >= 0.05},
            'reopen_ge_1': {'value': {'min': min(reopen_vals) if reopen_vals else None,
                                      'mean': (sum(reopen_vals) / len(reopen_vals)) if reopen_vals else None},
                            'pass': bool(reopen_vals) and min(reopen_vals) >= 1},
            'carry_force_ge_0.95': {'value': carry, 'pass': carry is not None and carry >= 0.95},
            'exact_fill_ge_1': {'value': exact, 'pass': (exact or 0) >= 1},
        }

    expected_seeds = [7100 + i for i in range(10)] if STEP == 3000 else None
    seeds = [r.get('episode') for r in results]
    errs = [r for r in results if r.get('error')]
    add('ten_results', len(results) == 10, len(results))
    if expected_seeds:
        add('expected_seeds', seeds == expected_seeds, seeds)
    add('no_eval_errors', not errs, [r.get('episode') for r in errs])
    add('scene_all_match', bool(scene.get('all_match')), scene.get('checks'))
    add('protocol_frozen', (protocol.get('max_steps') == 600 and protocol.get('denoise_steps') == 5
                            and protocol.get('replan') == 10 and protocol.get('action_horizon') == 32
                            and s.get('force_zero') is False), protocol)

    for r in results:
        a = r.get('action_stats') or {}
        f = r.get('force_stats') or {}
        per_seed.append({'seed': r.get('episode'), 'exact_fill': r.get('exact_fill'),
                         'in_target': r.get('final_in_target'), 'grasps': r.get('grasps'),
                         'close_frac': a.get('grip_close_frac'), 'open_frac': a.get('open_cmd_frac'),
                         'reopen_events': a.get('reopen_events'),
                         'carry_force_gt_0p5': f.get('carry_force_frac_gt_0p5'),
                         'dp_absmean': a.get('dp_absmean'),
                         'scene_match': (r.get('scene') or {}).get('match'),
                         'safety': r.get('safety_reason'), 'error': bool(r.get('error'))})

    # ---- root-cause scan -------------------------------------------------
    fund = []
    def fadd(name, ok, detail):
        fund.append({'name': name, 'ok': bool(ok), 'detail': detail})

    labels_sha = sha256(LABELS) if os.path.exists(LABELS) else None
    fadd('labels_v3_sha', labels_sha == 'c4090dd2cb994a8b3524b4c2614b869035acb89dccb7c8eed0e9dbe63e4affff', labels_sha)
    data_src = open('/workspace/t7_data.py', encoding='utf-8').read() if os.path.exists('/workspace/t7_data.py') else ''
    fadd('sampler_v3_no_pm1', ('g >= 0.8' in data_src) and ('g <= -0.8' not in data_src)
         and ('release_rise' in data_src) and ('T7_FORCE_DROPOUT' in data_src), 't7_data.py')
    train_src = open('/workspace/t7_train.py', encoding='utf-8').read() if os.path.exists('/workspace/t7_train.py') else ''
    fadd('force_head_train', ('force_head' in train_src) and ('loss_force' in train_src)
         and ("bb['grip_raw']" in train_src), 't7_train.py')
    ev_src = open('/workspace/_t7_eval.py', encoding='utf-8').read() if os.path.exists('/workspace/_t7_eval.py') else ''
    fadd('eval_mapping_frozen', all(x in ev_src for x in [
        '2.0 * a[6] - 1.0', 'REPLAN = 10', 'LABEL_ROT_SCALE = 0.5', 'ROT_SCALE = 0.10',
        "info.get('cookies_in_target') == 10 and src == 70"]), 't7_eval.py')
    try:
        m = json.load(open(MANIFEST))
        canonical = hashlib.sha256(json.dumps(
            {'base_options': m['base_options'],
             'scenes': [{'seed': x['seed'], 'hash': x['hash']} for x in m['scenes']]},
            sort_keys=True).encode()).hexdigest()
        fadd('scene_manifest_freeze', canonical == m.get('freeze_sha256')
             and len(m['scenes']) == 80 and len({x['hash'] for x in m['scenes']}) == 80, canonical)
    except Exception as exc:
        fadd('scene_manifest_freeze', False, str(exc))

    metrics = []
    mp = os.path.join(RUN, 'metrics.jsonl')
    if os.path.exists(mp):
        try:
            for line in open(mp, encoding='utf-8').read().strip().split('\n')[-100:]:
                if line.strip():
                    metrics.append(json.loads(line))
        except Exception:
            metrics = []
    finite = bool(metrics) and all(
        math.isfinite(float(r.get('loss', 0.0))) and math.isfinite(float(r.get('loss_force') or 0.0))
        for r in metrics)
    mean_train_close = (sum(r.get('close_frac', 0) for r in metrics) / len(metrics)) if metrics else None
    mean_train_open = (sum(r.get('open_frac', 0) for r in metrics) / len(metrics)) if metrics else None
    diff_frac = (sum(1 for r in metrics if abs(r.get('close_frac', 0) - r.get('open_frac', 0)) > 0.05)
                 / len(metrics)) if metrics else None
    fadd('train_metrics_finite', finite, {'records': len(metrics),
                                          'last_step': metrics[-1].get('step') if metrics else None})
    fadd('train_sampler_balanced', (mean_train_open is not None and mean_train_open >= 0.03
                                    and diff_frac is not None and diff_frac >= 0.5),
         {'mean_open': mean_train_open, 'mean_close': mean_train_close, 'close_neq_open_frac': diff_frac})

    ckpt = os.environ.get('T7_CKPT') or os.path.join(RUN, 'step_%06d.pt' % STEP)
    ck_info = {'path': ckpt, 'exists': os.path.exists(ckpt)}
    if os.path.exists(ckpt):
        try:
            import torch
            ck = torch.load(ckpt, map_location='cpu', weights_only=False)
            ck_info.update({'keys': sorted(ck.keys()), 'step': ck.get('step'),
                            'force_head': 'force_head' in ck,
                            'base_sha_ok': ck.get('base_sha256') == BASE_SHA,
                            'sha256': sha256(ckpt), 'run_dir': RUN})
            del ck
        except Exception as exc:
            ck_info['error'] = str(exc)[:300]
    fadd('checkpoint_ok', ck_info.get('exists') and ck_info.get('step') == STEP
         and ck_info.get('force_head') and ck_info.get('base_sha_ok'), ck_info)

    # ---- videos + zip ----------------------------------------------------
    vm = {'videos': []}
    if os.path.exists(VM):
        try:
            vm = json.load(open(VM))
        except Exception:
            vm = {'videos': []}
    for e in vm.get('videos', []):
        try:
            sd = os.path.join(EVAL_DIR, 'seed_%05d' % int(e.get('seed')))
        except Exception:
            continue
        if not (e.get('composite') and os.path.exists(e.get('composite'))):
            cand = os.path.join(sd, 'composite.mp4')
            if os.path.exists(cand):
                e['composite'] = cand
        if not (e.get('composite') and os.path.exists(e.get('composite'))):
            legacy = e.get('front') if (e.get('front') and os.path.exists(e.get('front'))) else os.path.join(sd, 'front.mp4')
            if os.path.exists(legacy):
                e['composite'] = legacy
                e['legacy_front_only'] = True
    if not any(e.get('composite') and os.path.exists(e.get('composite')) for e in vm.get('videos', [])):
        for d in sorted(glob.glob(os.path.join(EVAL_DIR, 'seed_*'))):
            try:
                seed = int(os.path.basename(d).split('_')[-1])
            except Exception:
                continue
            c = os.path.join(d, 'composite.mp4')
            f = os.path.join(d, 'front.mp4')
            p = c if os.path.exists(c) else (f if os.path.exists(f) else None)
            if p:
                vm['videos'].append({'seed': seed, 'composite': p,
                                     'frames': None, 'exact_fill': None, 'in_target': None})
    zip_path = (os.path.join('/tmp', 'videos_step_%d.zip' % STEP) if DRY
                else os.path.join(EVAL_DIR, 'videos_step_%d.zip' % STEP))
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_STORED) as z:
            for e in vm.get('videos', []):
                seed = e.get('seed')
                p = e.get('composite')
                if p and os.path.exists(p):
                    rel = os.path.relpath(p, '/workspace')
                    arc = 'seed_%05d_composite.mp4' % int(seed)
                    z.write(p, arcname=arc)
                    videos.append({'seed': seed, 'camera': 'composite', 'path': p, 'rel': rel,
                                   'url': url_of(rel), 'bytes': os.path.getsize(p),
                                   'sha256': sha256(p), 'frames': e.get('frames'),
                                   'exact_fill': e.get('exact_fill'), 'in_target': e.get('in_target'),
                                   'legacy_front_only': bool(e.get('legacy_front_only'))})
        zip_info = {'path': zip_path, 'rel': os.path.relpath(zip_path, '/workspace'),
                    'url': url_of(os.path.relpath(zip_path, '/workspace')),
                    'bytes': os.path.getsize(zip_path), 'sha256': sha256(zip_path)}
    except Exception as exc:
        zip_info = {'error': str(exc)[:300]}
    expected_n = len({r.get('episode') for r in results}) or 10
    fadd('videos_present', len(videos) >= expected_n and len({v['seed'] for v in videos}) >= expected_n,
         {'videos': len(videos), 'seeds': len({v['seed'] for v in videos}), 'expected': expected_n,
          'kind': 'composite (left_wrist|front|force ch12/ch13)'})

    fundamental_ok = all(x['ok'] for x in fund)
    if STEP == 3000:
        passed = fundamental_ok and all(v['pass'] for v in gd.values())
    else:
        passed = fundamental_ok
    verdict = 'pass' if passed else ('failed_gate' if STEP == 3000 else 'failed_fundamental')

    if not passed and not DRY:
        open(STOP, 'w').write('step %d %s %s\n' % (STEP, verdict, ts))

    hints = []
    if STEP == 3000:
        if not gd['close_le_0.95']['pass'] or not gd['open_ge_0.05']['pass'] or not gd['reopen_ge_1']['pass']:
            hints.append('gripper-only-close signature: check V3 label pipeline and eval decode')
        if not gd['carry_force_ge_0.95']['pass']:
            hints.append('carry contact force missing: grasp not established or force path ineffective')
        if not gd['exact_fill_ge_1']['pass']:
            hints.append('no exact_fill at 3k: may be early training; see behavior metrics')
    if not (scene.get('all_match')):
        hints.append('eval scene mismatch/default reset: eval wiring issue')
    if metrics and mean_train_open is not None and mean_train_open < 0.03:
        hints.append('sampler balance lost: check t7_data.py phase_open/release_rise')
    if ck_info.get('exists') and not ck_info.get('force_head'):
        hints.append('checkpoint missing force_head: training code drift')
    out = {'worker': 'task7', 'ts': ts, 'step': STEP, 'eval_dir': EVAL_DIR,
           'summary': SUM, 'verdict': verdict, 'g_d': gd, 'checks': checks,
           'fundamental': fund, 'per_seed': per_seed,
           'train_metrics': {'mean_open': mean_train_open, 'mean_close': mean_train_close,
                             'close_neq_open_frac': diff_frac, 'records': len(metrics)},
           'checkpoint': ck_info, 'video_manifest': VOUT, 'videos': len(videos), 'zip': zip_info,
           'stop_flag': STOP if (not passed and not DRY) else None, 'hints': hints}
    json.dump(out, open(OUT, 'w'), indent=1)
    json.dump({'step': STEP, 'eval_dir': EVAL_DIR, 'pod_base': POD,
               'zip': zip_info, 'videos': videos}, open(VOUT, 'w'), indent=1)

    lines = ['# step %d milestone report' % STEP, '',
             '- verdict: **%s**' % verdict,
             '- eval dir: `%s`' % EVAL_DIR,
             '- zip: %s' % ((zip_info or {}).get('url') or 'n/a'), '']
    if STEP == 3000:
        lines += ['## G-D', '', '| criterion | value | pass |', '|---|---|---|']
        for k, v in gd.items():
            lines.append('| %s | %s | %s |' % (k, json.dumps(v['value']), v['pass']))
        lines.append('')
    lines += ['## fundamental checks', '', '| check | ok | detail |', '|---|---|---|']
    for x in fund:
        lines.append('| %s | %s | %s |' % (x['name'], x['ok'], json.dumps(x['detail'])[:160]))
    lines += ['', '## per seed', '', '| seed | exact | in_target | close | open | reopen | carry_force>0.5 | scene |', '|---|---|---|---|---|---|---|---|']
    for r in per_seed:
        lines.append('| %s | %s | %s | %s | %s | %s | %s | %s |' % (
            r['seed'], r['exact_fill'], r['in_target'], r['close_frac'], r['open_frac'],
            r['reopen_events'], r['carry_force_gt_0p5'], r['scene_match']))
    lines += ['', '## videos', '', '- zip: %s' % ((zip_info or {}).get('url') or 'n/a')]
    for v in videos:
        lines.append('- %s %s: %s' % (v['seed'], v['camera'], v['url']))
    if hints:
        lines += ['', '## hints', ''] + ['- ' + h for h in hints]
    open(MD, 'w').write('\n'.join(lines) + '\n')

    print(json.dumps({'step': STEP, 'verdict': verdict, 'g_d': gd,
                      'fundamental_ok': fundamental_ok, 'videos': len(videos),
                      'zip': (zip_info or {}).get('url'), 'hints': hints}, default=str))
    if DRY:
        return 0
    return 0 if passed else (2 if STEP == 3000 else 1)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        import traceback
        traceback.print_exc()
        try:
            open(OUT + '.error', 'w').write(traceback.format_exc()[:3000])
        except Exception:
            pass
        sys.exit(1)