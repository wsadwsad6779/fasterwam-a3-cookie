import json, time
import numpy as np
import pyarrow.parquet as pq

NPZ = '/workspace/fwv1/p_items/A3_845_7D_LABELS.npz'
NPZ_V2 = '/workspace/fwv1/p_items/A3_845_7D_LABELS_V2.npz'
PARQ = '/workspace/fwv1/p_items/hf_multi/data/chunk-000/file-000.parquet'
PARQ2 = '/workspace/fwv1/p_items/hf_multi/data/chunk-000/file-001.parquet'
OUT = '/workspace/fwv1/p_items/JOINT_TO_7D_VERIFICATION.json'


def rotvec(q0, q1):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    cw, cx, cy, cz = w0, -x0, -y0, -z0
    w = w1 * cw - x1 * cx - y1 * cy - z1 * cz
    x = w1 * cx + x1 * cw + y1 * cz - z1 * cy
    y = w1 * cy - x1 * cz + y1 * cw + z1 * cx
    z = w1 * cz + x1 * cy - y1 * cx + z1 * cw
    if w < 0:
        w, x, y, z = -w, -x, -y, -z
    ang = 2 * np.arccos(np.clip(w, -1, 1))
    s = np.sqrt(max(1e-12, 1 - w * w))
    if ang < 1e-8:
        return np.zeros(3)
    return np.array([x, y, z]) / s * ang


def pm1(g):
    n = len(g)
    pm = np.zeros(n)
    for i in range(n):
        if g[i] <= 0.6:
            pm[i] = 1
        elif g[i] >= 0.8:
            pm[i] = -1
        else:
            pm[i] = pm[i - 1] if i else 1
    for _ in range(3):
        i = 0
        while i < n:
            j = i
            while j < n and pm[j] == pm[i]:
                j += 1
            if j - i < 5:
                pm[i:j] = pm[i - 1] if i > 0 else (pm[j] if j < n else pm[i])
            i = j
    return pm


def episode_label(pos, q, g):
    n = len(g)
    dp = np.clip(np.diff(pos, axis=0) / 0.05, -1, 1)
    rv = np.stack([rotvec(q[i], q[i + 1]) / 0.5 for i in range(n - 1)]) if n > 1 else np.zeros((0, 3))
    if len(rv):
        rv = np.clip(rv, -1, 1)
    p = pm1(g)
    lab = np.zeros((n - 1, 7))
    lab[:, :3] = dp
    lab[:, 3:6] = rv
    lab[:, 6] = p[:n - 1]
    return lab


z = np.load(NPZ)
L = z['label'].astype(np.float64)
es = z['ep_start'].astype(np.int64)
el = z['ep_len'].astype(np.int64)
n_ep = len(es)
lab_start = es - np.arange(n_ep)
out = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
       'file': NPZ, 'shape': list(L.shape), 'episodes': int(n_ep),
       'frames_sum': int(el.sum()), 'checks': {}}
c = out['checks']
c['shape_rows_match_ep_len'] = bool(int(el.sum()) == int(L.shape[0]))
c['no_nan_inf'] = bool(np.isfinite(L).all())
c['ep_start_contiguous'] = bool(np.array_equal(es[1:], es[:-1] + el[:-1] + 1))
c['frame_offsets_start_zero'] = bool(es[0] == 0)
out['defect_found'] = {
    'field': 'ep_start',
    'issue': 'ep_start stores cumulative FRAME offsets, but each episode contributes len-1 label rows; using ep_start directly as a label row offset misaligns every episode k>0 by k rows and overruns the array at the tail (ep844: 458114+1062 > 458332)',
    'fix': 'label row offset = ep_start - episode_index (saved as ep_start_label in V2 npz)',
}
c['label_offsets_contiguous'] = bool(lab_start[0] == 0 and
                                    np.array_equal(lab_start[1:], (lab_start + el)[:-1]) and
                                    int(lab_start[-1] + el[-1]) == int(L.shape[0]))
dp_absmax = float(np.abs(L[:, :3]).max()) if len(L) else 0.0
rv_absmax = float(np.abs(L[:, 3:6]).max()) if len(L) else 0.0
out['dp_absmax_label'] = dp_absmax
out['dp_max_mm_per_step'] = round(dp_absmax * 0.05 * 1000, 3)
out['rv_absmax_label'] = rv_absmax
out['rv_absmax_rad_per_step'] = round(rv_absmax * 0.5, 5)
c['dp_step_under_50mm'] = bool(dp_absmax * 0.05 <= 0.05)
c['rv_step_under_pi_rad'] = bool(rv_absmax * 0.5 <= np.pi)
uniq = np.unique(np.round(L[:, 6], 6))
c['gripper_pm1_only'] = bool(np.all(np.isin(uniq, [-1.0, 1.0])))
out['gripper_unique'] = [float(x) for x in uniq]
out['gripper_close_frac'] = round(float((L[:, 6] > 0).mean()), 5) if len(L) else None
ep_dp = []
for k, (s, l) in enumerate(zip(lab_start, el)):
    if int(l) > 0:
        seg = L[int(s):int(s) + int(l), :3]
        ep_dp.append((float(np.abs(seg).max()), int(k), int(l)))
ep_dp.sort(reverse=True)
out['top5_episodes_by_dp'] = ep_dp[:5]

t = pq.read_table([PARQ, PARQ2], columns=['episode_index', 'observation.eef_pose', 'action'])
E = np.asarray(t['episode_index'])
X = np.asarray(t['observation.eef_pose'].to_pylist(), np.float64)
A = np.asarray(t['action'].to_pylist(), np.float64)
order = np.argsort(E, kind='stable')
E = E[order]
X = X[order]
A = A[order]
b = np.flatnonzero(np.diff(E)) + 1
st_ = np.r_[0, b]
en_ = np.r_[b, len(E)]
out['parquet_episodes'] = int(len(st_))
c['parquet_episode_count_845'] = bool(len(st_) == n_ep)
c['frame_boundaries_match_parquet'] = bool(len(st_) == n_ep and np.array_equal(st_, es)
                                           and np.array_equal(en_ - st_ - 1, el))
dp_max_all = 0.0
grip_mis_all = 0
for k in range(n_ep):
    s0, e0 = int(st_[k]), int(en_[k])
    ls = int(lab_start[k])
    ref_dp = L[ls:ls + int(el[k]), :3]
    dp = np.clip(np.diff(X[s0:e0, :3], axis=0) / 0.05, -1, 1)
    if dp.size and dp.shape == ref_dp.shape:
        dp_max_all = max(dp_max_all, float(np.abs(dp - ref_dp).max()))
    pm = pm1(A[s0:e0, 7])[:e0 - s0 - 1]
    grip_mis_all += int((pm != L[ls:ls + int(el[k]), 6]).sum())
out['global_dp_max_abs_diff'] = dp_max_all
out['global_gripper_mismatch'] = grip_mis_all
c['global_dp_matches_lt_1e6'] = bool(dp_max_all < 1e-6)
c['global_gripper_matches'] = bool(grip_mis_all == 0)
picks = sorted(set([0, n_ep // 2, n_ep - 1]))
recs = []
max_diff = 0.0
for k in picks:
    s, e = int(st_[k]), int(en_[k])
    pos = X[s:e, :3]
    q = X[s:e, 3:7]
    g = A[s:e, 7]
    lab = episode_label(pos, q, g)
    ref = L[int(lab_start[k]):int(lab_start[k]) + int(el[k])]
    d_ = float(np.abs(lab - ref).max()) if lab.size == ref.size else 999.0
    max_diff = max(max_diff, d_)
    recs.append({'episode': int(k), 'frames': int(e - s), 'labels': int(len(lab)),
                 'ref_rows': int(len(ref)), 'max_abs_diff': d_,
                 'gripper_mismatch': int((lab[:, 6] != ref[:, 6]).sum()) if lab.size == ref.size else -1})
out['recompute_picks'] = recs
out['recompute_max_abs_diff'] = max_diff
c['recompute_matches_lt_1e5'] = bool(max_diff < 1e-5)
c['episode_count_845'] = bool(n_ep == 845)
c['frames_458332'] = bool(int(el.sum()) == 458332)
out['v2_npz'] = NPZ_V2
out['v2_fields'] = ['label', 'ep_start_label', 'ep_start_frame', 'ep_len']
np.savez_compressed(NPZ_V2, label=z['label'], ep_start_label=lab_start.astype(np.int64),
                    ep_start_frame=es, ep_len=el)
out['verdict'] = 'pass' if all(c.values()) else 'fail'
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps({'verdict': out['verdict'], 'checks': c, 'recompute_max_abs_diff': max_diff}))