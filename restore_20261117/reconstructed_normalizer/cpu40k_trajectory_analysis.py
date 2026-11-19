"""CPU-only, evidence-scoped trajectory analysis; no model/simulator imports.
Run: D:/APP/Python313/python.exe <absolute path to this file>
"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'forensics_download'
IDENTITY = 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL'
EPS = 1e-4


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def clean(x):
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [clean(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def stats(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return dict(n=len(a), min=float(a.min()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                mean=float(a.mean()) if len(a) else None,
                p95=float(np.percentile(a, 95)) if len(a) else None,
                max=float(a.max()) if len(a) else None)


def first(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else None


def movement(delta):
    return 'opening' if delta > EPS else 'closing' if delta < -EPS else 'stationary_eps1e-4'


def raw_class(a, b):
    if abs(a) <= .1 and abs(b) <= .1:
        return 'small'
    if abs(a) >= .3 and abs(b) >= .3:
        return 'large'
    return 'mixed'


def phase(frame, contact):
    if contact is None:
        return 'no_contact_observed'
    return 'pre' if frame < contact - 10 else 'post' if frame > contact + 10 else 'near_pm10'


def angle(q):
    norm = np.linalg.norm(q, axis=1)
    if np.any(norm < 1e-12):
        raise ValueError('zero quaternion')
    q = q / norm[:, None]
    return np.degrees(2 * np.arccos(np.clip(np.abs(q @ q[0]), 0, 1)))


def rates(rows):
    result = {}
    # Denominators are eligible adjacent executed-command pairs, not frames or flips.
    for split in [(), ('location',), ('phase',), ('raw_class',),
                  ('location', 'phase'), ('location', 'raw_class'),
                  ('phase', 'raw_class'), ('location', 'phase', 'raw_class')]:
        buckets = {}
        for r in rows:
            key = '|'.join(f'{k}={r[k]}' for k in split) or 'all'
            buckets.setdefault(key, []).append(r)
        for key, vals in buckets.items():
            flips = [r for r in vals if r['flip']]
            result[key] = dict(opportunities=len(vals), flips=len(flips),
                               rate=len(flips) / len(vals),
                               raw_absolute_jump_all=stats([r['raw_abs_jump'] for r in vals]),
                               raw_absolute_jump_flips=stats([r['raw_abs_jump'] for r in flips]))
    # Explicit zero-opportunity groups make absent categories auditable.
    for k in ('within', 'boundary'):
        result.setdefault('location=' + k, dict(opportunities=0, flips=0, rate=None))
    for k in ('small', 'large', 'mixed'):
        result.setdefault('raw_class=' + k, dict(opportunities=0, flips=0, rate=None))
    for k in ('pre', 'near_pm10', 'post', 'no_contact_observed'):
        result.setdefault('phase=' + k, dict(opportunities=0, flips=0, rate=None))
    return result


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(clean(v), ensure_ascii=False) if isinstance(v, (dict, list, tuple, np.ndarray)) else clean(v) for k, v in row.items()})


def analyze(seed, folder):
    obs_path = folder / f'cpu_seed{seed}_state_force.npz'
    with np.load(obs_path, allow_pickle=False) as z:
        obs = {k: z[k].copy() for k in z.files}
    assert str(obs['NORMALIZER_IDENTITY']) == IDENTITY
    state, force, pose, velocity, time = [obs[k] for k in
        ('observation.state', 'observation.force', 'observation.eef_pose', 'observation.velocity', 'time')]
    n = len(time) - 1
    assert state.shape == velocity.shape == (n + 1, 16)
    assert force.shape == (n + 1, 18) and pose.shape == (n + 1, 14)
    assert all(np.isfinite(x).all() for x in (state, force, pose, velocity, time))
    assert np.all(np.diff(time) > 0)
    missing, checks = [], {}
    opening = state[:, 7].astype(float)
    if seed in (11, 17):
        with np.load(folder / f'seed{seed}_steps.npz', allow_pickle=False) as z:
            steps = {k: z[k].copy() for k in z.files}
        with np.load(folder / f'seed{seed}_original_chunks.npz', allow_pickle=False) as z:
            chunks = {k: z[k].copy() for k in z.files}
        cid, row = steps['executed_chunk'].astype(int), steps['executed_chunk_row'].astype(int)
        raw7 = np.stack([chunks[f'chunk_{c:03d}_model_out'][r] for c, r in zip(cid, row)])
        denorm = np.stack([chunks[f'chunk_{c:03d}_denorm'].reshape(-1, 7)[r] for c, r in zip(cid, row)])
        pm1 = np.stack([chunks[f'chunk_{c:03d}_pm1'][r] for c, r in zip(cid, row)])
        adapter = np.stack([chunks[f'chunk_{c:03d}_adapter_out'][r] for c, r in zip(cid, row)])
        handed, requested = steps['handed_14d'], steps['requested_14d']
        target = steps['applied_joint_16d'][:, 7]
        checks.update(frame_mapping=bool(np.array_equal(steps['frame'], np.arange(1, n + 1))),
                      actual_opening_matches_recorded=bool(np.allclose(opening[1:], steps['opening_after'], atol=1e-6, rtol=0)),
                      actual_pads_match_recorded=bool(np.allclose(force[:, 12:16], steps['pad_forcen'], atol=1e-5, rtol=0)),
                      adapter_equals_handed=bool(np.array_equal(adapter, handed)),
                      handed_equals_requested=bool(np.array_equal(handed, requested)))
        raw = raw7[:, 6]
        checks.update(raw_denorm_gripper_exact=bool(np.array_equal(raw, denorm[:, 6])),
                      pm1_threshold_matches_raw=bool(np.array_equal(pm1[:, 6], np.where(raw >= 0, 1, -1))),
                      adapter_gripper_negative_pm1=bool(np.array_equal(adapter[:, 6], -pm1[:, 6])))
        source = 'steps.npz executed_chunk/row -> original_chunks.npz; observation from cpu_state_force'
    else:
        attribution = load_json(folder / 'seed12_gripper_attribution.json')['rows']
        plan = load_json(folder / 'seed12_plan32_comparison.json')['seed12_prediction_plan']
        assert n == 10 and len(attribution) == 10
        cid = np.array([r['chunk'] for r in attribution], dtype=int)
        row = np.array([r['row'] for r in attribution], dtype=int)
        raw = np.array([r['raw'] for r in attribution], dtype=float)
        target = np.array([r['applied_opening_target'] for r in attribution])
        handed = requested = adapter = raw7 = denorm = pm1 = None
        checks.update(plan_first10_equals_attributed_raw=bool(np.array_equal(raw, plan['raw32'][:10])),
                      actual_opening_matches_attribution=bool(np.allclose(opening[1:], [r['opening_after'] for r in attribution], atol=1e-6, rtol=0)),
                      actual_opening_before_matches_attribution=bool(np.allclose(opening[:-1], [r['opening_before'] for r in attribution], atol=1e-6, rtol=0)))
        missing.append('seed12 full executed XYZ/first6 command arrays unavailable; rail cannot be reconstructed from gripper JSON')
        source = 'seed12_gripper_attribution.json + plan32.raw32[:10]; rows10..31 excluded from execution'
    assert len(raw) == len(cid) == len(row) == n
    pads = np.max(np.abs(force[:, 12:16]), axis=1)
    wrist = np.linalg.norm(force[:, :3], axis=1)
    left_angle, right_angle = angle(pose[:, 3:7]), angle(pose[:, 10:14])
    contact = first(pads > 1)

    def snapshot(f):
        return dict(frame=int(f), time_s=float(time[f]), opening=float(opening[f]),
                    wrist_norm_cols0_3_N=float(wrist[f]), wrist_vector_cols0_3_N=force[f, :3],
                    pads_cols12_16_N=force[f, 12:16], pads_maxabs_N=float(pads[f]),
                    left_angle_reset_deg=float(left_angle[f]), right_angle_reset_deg=float(right_angle[f]),
                    left_xyz=pose[f, :3], right_xyz=pose[f, 7:10],
                    left_xyz_delta_reset=pose[f, :3]-pose[0, :3],
                    right_xyz_delta_reset=pose[f, 7:10]-pose[0, 7:10],
                    velocity16=velocity[f])

    def window(lo, hi):
        idx = np.arange(max(0, lo), min(n, hi) + 1)
        return dict(frames=idx, opening=stats(opening[idx]), pads_maxabs_N=stats(pads[idx]),
                    wrist_norm_N=stats(wrist[idx]), left_angle_reset_deg=stats(left_angle[idx]),
                    left_xyz_mean=np.mean(pose[idx, :3], axis=0))

    opportunities, flips = [], []
    for i in range(1, n):
        f = i + 1
        delta = float(opening[f] - opening[f-1])
        prev_delta = float(opening[f-1] - opening[f-2])
        cmd_before = float(handed[i-1, 6]) if handed is not None else float(attribution[i-1]['handed'])
        cmd_after = float(handed[i, 6]) if handed is not None else float(attribution[i]['handed'])
        r = dict(seed=seed, step_zero_based=i, command_frame=f, observation_before_frame=f-1,
                 previous_chunk=int(cid[i-1]), chunk=int(cid[i]), chunk_row=int(row[i]),
                 location='boundary' if cid[i] != cid[i-1] else 'within',
                 phase=phase(f, contact), first_contact_frame=contact,
                 raw_class=raw_class(raw[i-1], raw[i]), raw_before=float(raw[i-1]), raw_after=float(raw[i]),
                 raw_signed_jump=float(raw[i]-raw[i-1]), raw_abs_jump=float(abs(raw[i]-raw[i-1])),
                 flip=bool((raw[i-1] >= 0) != (raw[i] >= 0)),
                 executed_command_flip=bool(cmd_before != cmd_after), command_before=cmd_before, command_after=cmd_after,
                 opening_before=float(opening[f-1]), opening_after=float(opening[f]),
                 opening_delta=delta, measured_motion=movement(delta), previous_opening_delta=prev_delta,
                 previous_measured_motion=movement(prev_delta), applied_opening_target=float(target[i]),
                 target_is_not_measurement=True)
        opportunities.append(r)
        if r['flip']:
            r = dict(r)
            for label, frame in [('before', f-1), ('after', f)]:
                for k, v in snapshot(frame).items():
                    r[label + '_' + k] = v
            r['pre_window_inclusive'] = window(f-10, f-1)
            r['post_window_inclusive'] = window(f, f+10)
            r['offset_snapshots'] = {str(offset): snapshot(f+offset) if 0 <= f+offset <= n else None for offset in (-10, -1, 0, 1, 10)}
            flips.append(r)
    checks['all_raw_flips_match_executed_gripper_flips'] = all(r['flip'] == r['executed_command_flip'] for r in opportunities)

    def threshold_events(values):
        result = {}
        for t in (1, 10, 50, 100, 300):
            mask = values > t
            starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
            ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
            result[str(t)] = dict(first_frame=first(mask), frames_above=int(mask.sum()),
                                  first_time_s=float(time[first(mask)]) if mask.any() else None,
                                  episodes=[dict(start_frame=int(a), end_frame=int(b), start_time_s=float(time[a]), end_time_s=float(time[b])) for a, b in zip(starts, ends)])
        return result

    rail = {}
    for stage, commands in [('handed', handed), ('requested', requested)]:
        if commands is None:
            rail[stage] = None
            continue
        rail[stage] = {}
        for name, cols in [('XYZ', 3), ('first6', 6)]:
            mask = np.any(np.abs(commands[:, :cols]) >= 1 - 1e-6, axis=1)
            rail[stage][name] = dict(first_command_frame=int(np.flatnonzero(mask)[0] + 1) if mask.any() else None,
                                    count=int(mask.sum()), command_frames=(np.flatnonzero(mask) + 1).tolist(),
                                    per_dimension_count=np.sum(np.abs(commands[:, :cols]) >= 1-1e-6, axis=0),
                                    absmax_per_dimension=np.max(np.abs(commands[:, :cols]), axis=0))
    chunk_rows = []
    for c in sorted(set(cid.tolist())):
        indices = np.flatnonzero(cid == c)
        lo, hi = int(indices[0]), int(indices[-1])
        eligible = [r for r in opportunities if r['chunk'] == c]
        rr = dict(seed=seed, chunk=c, first_command_frame=lo+1, last_command_frame=hi+1,
                  executed_rows=len(indices), predicted_rows=32, excluded_prediction_rows=32-len(indices),
                  raw_min=float(raw[indices].min()), raw_max=float(raw[indices].max()),
                  raw_abs_stats=stats(np.abs(raw[indices])),
                  raw_flip_count=sum(r['flip'] for r in eligible), rates=rates(eligible),
                  opening_start=float(opening[lo]), opening_end=float(opening[hi+1]),
                  measured_opening_net_delta=float(opening[hi+1]-opening[lo]),
                  measured_motion_counts=dict(Counter(movement(v) for v in np.diff(opening[lo:hi+2]))),
                  pads_maxabs_N=float(pads[lo:hi+2].max()), wrist_norm_max_N=float(wrist[lo:hi+2].max()),
                  left_angle_reset_start_deg=float(left_angle[lo]), left_angle_reset_end_deg=float(left_angle[hi+1]),
                  left_xyz_start=pose[lo, :3], left_xyz_end=pose[hi+1, :3],
                  incoming_boundary_assigned_to_this_chunk=True)
        if handed is not None:
            rr['XYZ_rail_commands'] = int(np.any(np.abs(handed[indices, :3]) >= 1-1e-6, axis=1).sum())
            rr['first6_rail_commands'] = int(np.any(np.abs(handed[indices, :6]) >= 1-1e-6, axis=1).sum())
        chunk_rows.append(rr)
    measured_reversal = np.flatnonzero(np.diff(opening)[:-1] * np.diff(opening)[1:] < -EPS**2) + 2
    summary = dict(seed=seed, identity=str(obs['NORMALIZER_IDENTITY']), original_stats_sha256=str(obs['original_stats_sha256']),
                   observation_frames=n+1, executed_commands=n, source=source, checks=checks,
                   alignment_status='PASS' if all(checks.values()) else 'MISMATCH_DO_NOT_MERGE_WITHOUT_REVIEW',
                   first_contact_frame=contact, rates=rates(opportunities),
                   flip_count=len(flips), command_motion_counts=dict(Counter(movement(v) for v in np.diff(opening))),
                   measured_reversal_frames_eps1e4=measured_reversal,
                   timeline=dict(pads_maxabs_N=threshold_events(pads), wrist_norm_cols0_3_N=threshold_events(wrist),
                                 left_orientation={str(t): first(left_angle > t) for t in (5, 15, 30, 60, 90)},
                                 command_rail=rail, safety_stop_frames=np.flatnonzero(obs['safety_stop'])),
                   observation_summary=dict(opening=stats(opening), wrist_norm_N=stats(wrist), pads_maxabs_N=stats(pads),
                                            left_angle_reset_deg=stats(left_angle), right_angle_reset_deg=stats(right_angle),
                                            xyz_left_min=pose[:, :3].min(axis=0), xyz_left_max=pose[:, :3].max(axis=0),
                                            xyz_right_min=pose[:, 7:10].min(axis=0), xyz_right_max=pose[:, 7:10].max(axis=0),
                                            velocity_absmax_per_col=np.max(np.abs(velocity), axis=0)),
                   reset=snapshot(0), end=snapshot(n), missing=missing,
                   video_estimated_events=[], video_estimates_status='No structured video-estimated events supplied; not inferred or mixed with observation timeline')
    return summary, flips, chunk_rows, opportunities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT)
    args = parser.parse_args()
    folder, out = args.input_dir.resolve(strict=True), args.output_dir.resolve(strict=True)
    results, flips, chunks, opportunities = [], [], [], []
    for seed in (11, 12, 17):
        result, ff, cc, oo = analyze(seed, folder)
        results.append(result); flips.extend(ff); chunks.extend(cc); opportunities.extend(oo)
    identity = load_json(folder / 'forensics_identity.json')
    inputs = [folder / f'cpu_seed{s}_state_force.npz' for s in (11, 12, 17)]
    inputs += [folder / f'seed{s}_{kind}.npz' for s in (11, 17) for kind in ('steps', 'original_chunks')]
    inputs += [folder / name for name in ('seed12_gripper_attribution.json', 'seed12_plan32_comparison.json', 'forensics_identity.json')]
    payload = dict(schema_version=1, execution_device='CPU; NumPy only; CUDA_VISIBLE_DEVICES empty',
                   NORMALIZER_IDENTITY=IDENTITY, original_stats_sha256='UNKNOWN',
                   report_scope='在可追溯重建 normalizer 条件下的 40k 诊断结果，不代表原始 normalizer 身份',
                   identity_evidence=identity,
                   provenance=[dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs],
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   definitions=dict(frame='frame0=reset; command index i produces observation frame i+1; flip assigned to newer command frame',
                                    flip='adjacent actually executed raw gripper threshold classes differ; nonnegative -> +1; negative -> -1; actual handed flips cross-checked',
                                    opportunities='N-1 adjacent executed-command pairs; exclude reset-to-first-command; boundary if chunk differs',
                                    phase='first maxabs(observation.force[:,12:16])>1 N is c; pre f<c-10, near c-10<=f<=c+10, post f>c+10; absent c separate',
                                    raw_class='small: both abs<=0.1; large: both abs>=0.3; otherwise mixed; classified for all eligible pairs',
                                    opening='observation.state[:,7] measured normalized opening; delta>1e-4 opening, <-1e-4 closing, otherwise stationary',
                                    force='wrist norm is ONLY observation.force[:,0:3]; NOT all six model force channels; pads=12:16; no full18 norm across mixed force/torque channels',
                                    posture='both EEF xyz and normalized quaternion geodesic from reset, absolute quaternion dot; no Euler-angle inference',
                                    rail='descriptive normalized command rail abs>=1-1e-6 in handed/requested XYZ or first6 ONLY; excludes gripper dims6,13; not physical workspace/joint limit',
                                    windows='for newer flip command frame f, before=f-1, after=f; pre window f-10..f-1, post f..f+10, truncated to observed bounds',
                                    chunk='incoming boundary belongs to destination chunk; each opportunity counted once'),
                   pooled_rates=rates(opportunities), results=results,
                   missing=[dict(seed=r['seed'], fields=r['missing']) for r in results],
                   video_estimated_events=[], limitations=['descriptive associations, not causal attribution',
                       'CPU observation replays are separate artifacts: numerical opening/pad matches do not establish complete recorder no-side-effect equivalence or weight identity',
                       'no original stats checksum; do not upgrade reconstructed identity',
                       'video estimates not provided as structured events; no visual estimates invented'],
                   flips=flips, chunks=chunks)
    paths = {name: out / ('cpu40k_trajectory_' + suffix) for name, suffix in
             [('json', 'analysis.json'), ('flips', 'flips.csv'), ('chunks', 'chunks.csv'), ('md', 'report_zh.md')]}
    paths['json'].write_text(json.dumps(clean(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    write_csv(paths['flips'], flips); write_csv(paths['chunks'], chunks)
    lines = ['# CPU 40k 执行轨迹取证分析', '', '## 身份与证据边界',
             f'- NORMALIZER_IDENTITY：`{IDENTITY}`；original_stats_sha256：`UNKNOWN`。',
             '- 仅描述可追溯重建 normalizer 条件下的 40k，不冒充原始 normalizer 或原始评估结果。NumPy 本地 CPU 分析，无 GPU、无重新推理。',
             '- 新 NPZ 是真实 observation 的独立记录；与旧记录逐项对齐校验，不以局部一致推断记录器无副作用或完整实验身份等价。',
             '- 原始预测、实际执行命令、IK/限速 opening 目标、实测 opening 分离。开/闭仅由实测 opening 差定义，命令翻转不等于实际开合反转。',
             '- wrist 指 force 的 0:3 三列范数，并非模型全部六个力通道；pads 只用 12:16。不能把包含力矩的 force18 直接整体求牛顿范数。', '',
             '## 统计口径',
             '- frame0 为 reset；命令 i（0-based）完成后为 frame i+1。翻转记在较新命令的完成帧，初始命令没有前一条执行命令，不计机会。',
             '- 块内/边界率=翻转数/对应相邻执行命令对数，不用翻转构成比代替机会率；边界计入新 chunk。',
             '- 首次 maxabs(pads)>1N 为 c；互斥 pre：f<c-10、附近：c-10≤f≤c+10、post：f>c+10；未接触单列。按完成帧分层，不表示命令执行前已接触。',
             '- raw 小：两端 abs≤0.1；大：两端 abs≥0.3；其他混合。所有机会均分类，JSON 包含交叉分层及幅度 min/median/mean/p95/max。',
             '- seed12 仅执行前10行；32行计划的后22行不计机会、flip 或轨迹。',
             '- rail 是 handed/requested 归一化命令 abs≥1−1e−6 的描述，分别统计 XYZ 和前6维；不把 gripper 的 ±1 当异常，也不称实测 XYZ 越界。', '',
             '## 执行翻转结果', '', '|seed|执行命令|flip|块内 flip/机会（率）|边界 flip/机会（率）|首次接触帧|对齐|', '|---|---:|---:|---|---|---:|---|']
    for r in results:
        def fmt(key):
            a = r['rates'][key]
            return f"{a['flips']}/{a['opportunities']} ({a['rate']:.2%})" if a['rate'] is not None else '0/0（不可估）'
        lines.append(f"|{r['seed']}|{r['executed_commands']}|{r['flip_count']}|{fmt('location=within')}|{fmt('location=boundary')}|{r['first_contact_frame']}|{r['alignment_status']}|")
    for r in results:
        lines += ['', f"### seed {r['seed']}", '- 实测开合命令步计数：' + json.dumps(r['command_motion_counts'], ensure_ascii=False),
                  '- 分层机会率：', '', '|分层|flip|机会|率|', '|---|---:|---:|---:|']
        for key in ['phase=pre', 'phase=near_pm10', 'phase=post', 'phase=no_contact_observed', 'raw_class=small', 'raw_class=large', 'raw_class=mixed']:
            a = r['rates'][key]
            rate = f"{a['rate']:.2%}" if a['rate'] is not None else '不可估'
            lines.append(f"|{key}|{a['flips']}|{a['opportunities']}|{rate}|")
        lines += ['', '- 力阈值首次超出帧（严格 >）：', '', '|N|pads maxabs 首帧|wrist 0:3 norm 首帧|', '|---|---:|---:|']
        for t in ('1', '10', '50', '100', '300'):
            lines.append(f"|{t}|{r['timeline']['pads_maxabs_N'][t]['first_frame']}|{r['timeline']['wrist_norm_cols0_3_N'][t]['first_frame']}|")
        rail = r['timeline']['command_rail']['handed']
        lines += ['', '- handed rail：' + (json.dumps({k: dict(first_command_frame=v['first_command_frame'], count=v['count']) for k, v in rail.items()}, ensure_ascii=False) if rail else '缺失完整 XYZ/前6维执行命令，不估算。'),
                  '- 对齐检查：' + json.dumps(r['checks'], ensure_ascii=False),
                  '- flip 前后实测 opening/运动、wrist、pads、左右 EEF 相对 reset 姿态角与 XYZ、±10帧快照/窗口在逐 flip CSV 和 JSON 中；时间线全部超阈值区间、rail 全帧列表在 JSON。',
                  '- 每 chunk 原始幅度、机会率、开合净差、姿态和 XYZ、rail 数量在逐 chunk CSV。']
    lines += ['', '## 缺失与视频估计（不混入实测时间线）',
              '- seed12 缺少完整 XYZ/前6维执行链数组，不能据 gripper JSON 恢复 rail；其余真实 observation 按11帧使用。',
              '- 无结构化视频估计事件输入，本次未重新做视觉判读；视频事件列表为空不代表视频里没有事件。抓取、掉落、碰撞等视觉估计不得由力阈值替代或伪造。',
              '- None 表示观察范围内未超阈值或字段不可用，具体以 missing 和对应字段定义区分；短 seed12 的未观察到不代表长期不发生。', '',
              '## 解释限制',
              '- 翻转、接触、姿态漂移、rail 的时序共现只构成相关描述，不证明任何一项导致失败；阈值不是任务失败标准。',
              '- 独立 CPU observation 与已有 executed 链匹配，只对已校验字段有效；不外推模型六通道力、未执行预测或记录器无副作用。', '',
              '## 复核与产物',
              f'- 脚本：`{Path(__file__).resolve()}`',
              '- 输入绝对路径及 SHA256、身份原文、计算定义和全量结果均在 JSON；无随机采样。']
    lines += [f'- {k}：`{p}`' for k, p in paths.items()]
    paths['md'].write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(dict(outputs={k: str(p) for k, p in paths.items()},
                          seeds=[dict(seed=r['seed'], commands=r['executed_commands'], flips=r['flip_count'],
                                      contact=r['first_contact_frame'], alignment=r['alignment_status']) for r in results],
                          flip_rows=len(flips), chunk_rows=len(chunks)), ensure_ascii=True, indent=2))
    if any(r['alignment_status'] != 'PASS' for r in results):
        raise SystemExit('Outputs saved; alignment mismatch requires review')


if __name__ == '__main__':
    main()
