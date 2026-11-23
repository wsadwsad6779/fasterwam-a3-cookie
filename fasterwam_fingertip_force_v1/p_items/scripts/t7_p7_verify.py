import json, time
import numpy as np
import pyarrow.parquet as pq

NPZ = '/workspace/fwv1/p_items/A3_845_7D_LABELS.npz'
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
out = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
       'file': NPZ, 'shape': list(L.shape), 'episodes': int(len(es)),
       'frames_sum': int(el.sum()), 'checks': {}}
c = out['checks']
c['shape_rows_match_ep_len'] = bool(int(el.sum()) == int(L.shape[0]))
c['no_nan_inf'] = bool(np.isfinite(L).all())
c['ep_start_contiguous'] = bool(len(es) > 0 and es[0] == 0 and
                                np.array_equal(es[1:], es[:-1] + el[:-1]))
dp_absmax = float(np.abs(L[:, :3]).max()) if len(L) else 0.0
rv_absmax = float(np.abs(L[:, 3:6]).max()) if len(L) else 0.0
out['dp_absmax_label'] = dp_absmax
out['dp_absmax_mps'] = dp_absmax
out['dp_max_mm_per_step'] = round(dp_absmax * 0.05 * 1000, 3)
out['rv_absmax_label'] = rv_absmax
out['rv_absmax_rad_per_step'] = round(rv_absmax * 0.5, 5)
c['dp_velocity_within_cap_1'] = bool(dp_absmax <= 1.0)
c['dp_step_under_50mm'] = bool(dp_absmax * 0.05 <= 0.05)
c['rv_step_under_pi_rad'] = bool(rv_absmax * 0.5 <= np.pi)
uniq = np.unique(np.round(L[:, 6], 6))
c['gripper_pm1_only'] = bool(np.all(np.isin(uniq, [-1.0, 1.0])))
out['gripper_unique'] = [float(x) for x in uniq]
close_frac = float((L[:, 6] > 0).mean()) if len(L) else None
out['gripper_close_frac'] = round(close_frac, 5) if close_frac is not None else None
first_dp = [float(np.abs(L[int(s):int(s) + 1, :3]).max()) for s in es if int(s) < len(L)]
out['ep_first_dp_max'] = max(first_dp) if first_dp else None
c['ep_first_dp_zero'] = bool(max(first_dp) < 1e-6) if first_dp else False
ep_dp = []
for k, (s, l) in enumerate(zip(es, el)):
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
c['parquet_episode_count_845'] = bool(len(st_) == len(es))
c['ep_boundaries_match_parquet'] = bool(len(st_) == len(es) and np.array_equal(st_, es)
                                        and np.array_equal(en_ - st_ - 1, el))
picks = sorted(set([0, len(es) // 2, len(es) - 1]))
recs = []
max_diff = 0.0
for k in picks:
    s, e = int(st_[k]), int(en_[k])
    pos = X[s:e, :3]
    q = X[s:e, 3:7]
    g = A[s:e, 7]
    lab = episode_label(pos, q, g)
    ref = L[int(es[k]):int(es[k]) + int(el[k])]
    d_ = float(np.abs(lab - ref).max()) if lab.size else 0.0
    max_diff = max(max_diff, d_)
    recs.append({'episode': int(k), 'frames': int(e - s), 'labels': int(len(lab)),
                 'max_abs_diff': d_,
                 'gripper_mismatch': int((lab[:, 6] != ref[:, 6]).sum()) if lab.size else 0})
out['recompute_picks'] = recs
out['recompute_max_abs_diff'] = max_diff
c['recompute_matches_lt_1e5'] = bool(max_diff < 1e-5)
c['episode_count_845'] = bool(len(es) == 845)
c['frames_458332'] = bool(int(el.sum()) == 458332)
out['verdict'] = 'pass' if all(c.values()) else 'fail'
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps({'verdict': out['verdict'], 'checks': c, 'recompute_max_abs_diff': max_diff}))