import glob, json, os
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from io import BytesIO
from scipy.spatial.transform import Rotation
from concurrent.futures import ThreadPoolExecutor

ROOT = "/workspace/a3_data/a3-front-close-left-100"
OUT = "/workspace/a3_data/a3_fw"
IMG = 224
POS_SCALE, ROT_SCALE = 0.05, 0.5
GRIP_EPS = 1e-4
os.makedirs(OUT, exist_ok=True)

files = sorted(glob.glob(ROOT + "/data/chunk-000/*.parquet"))
print("[load] parquet files:", len(files), flush=True)

rec = []
for i, f in enumerate(files):
    d = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state",
                                  "observation.eef_pose", "action"]).to_pydict()
    for j in range(len(d["episode_index"])):
        rec.append((int(d["episode_index"][j]), int(d["frame_index"][j]),
                    np.asarray(d["observation.state"][j], dtype=np.float64),
                    np.asarray(d["observation.eef_pose"][j], dtype=np.float64),
                    np.asarray(d["action"][j], dtype=np.float64)))
    if (i + 1) % 13 == 0:
        print(f"[load] {i+1}/{len(files)} rows={len(rec)}", flush=True)

rec.sort(key=lambda r: (r[0], r[1]))
N = len(rec)
print("[load] total frames:", N, flush=True)

gidx = {(r[0], r[1]): k for k, r in enumerate(rec)}
frames = np.memmap(os.path.join(OUT, "frames.dat"), dtype=np.uint8, mode="w+",
                   shape=(N, IMG, IMG * 2, 3))

def decode(b):
    return np.asarray(Image.open(BytesIO(b)).convert("RGB").resize((IMG, IMG), Image.BILINEAR), dtype=np.uint8)

for i, f in enumerate(files):
    d = pq.read_table(f, columns=["episode_index", "frame_index",
                                  "observation.images.front",
                                  "observation.images.left_wrist"]).to_pydict()
    n = len(d["episode_index"])
    idx = [gidx[(int(d["episode_index"][j]), int(d["frame_index"][j]))] for j in range(n)]
    fb = [d["observation.images.front"][j]["bytes"] for j in range(n)]
    lb = [d["observation.images.left_wrist"][j]["bytes"] for j in range(n)]
    def work(j):
        frames[idx[j], :, :IMG] = decode(fb[j])
        frames[idx[j], :, IMG:] = decode(lb[j])
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(work, range(n)))
    frames.flush()
    print(f"[img] {i+1}/{len(files)}", flush=True)

ep_arr = np.array([r[0] for r in rec], dtype=np.int64)
bounds, start = [], 0
for k in range(1, N + 1):
    if k == N or ep_arr[k] != ep_arr[start]:
        bounds.append((int(ep_arr[start]), start, k))
        start = k
lens = np.array([b - a for _, a, b in bounds])
print("[ep] count:", len(bounds), "len min/mean/max:", lens.min(), lens.mean(), int(lens.max()), flush=True)

def quat_wxyz_to_aa(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 1], q[..., 2], q[..., 3], q[..., 0]
    w = np.clip(w, -1.0, 1.0)
    den = np.sqrt(np.maximum(0.0, 1.0 - w * w))
    ang = 2.0 * np.arccos(w)
    out = np.zeros(q.shape[:-1] + (3,))
    ok = den > 1e-8
    out[ok] = np.stack([x, y, z], -1)[ok] * (ang[ok] / den[ok])[:, None]
    return out

act = np.zeros((N, 7), dtype=np.float32)
prop = np.zeros((N, 8), dtype=np.float32)
opening = np.zeros(N, dtype=np.float32)
joint_cmd = np.zeros(N, dtype=np.float32)
trans_hist = []

for ep, a, b in bounds:
    L = b - a
    eef = np.stack([r[3] for r in rec[a:b]])
    st = np.stack([r[2] for r in rec[a:b]])
    ac = np.stack([r[4] for r in rec[a:b]])
    p, quat = eef[:, :3], eef[:, 3:7]
    R = Rotation.from_quat(np.stack([quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0]], -1)).as_matrix()
    dR = R[1:] @ np.transpose(R[:-1], (0, 2, 1))
    pos = np.diff(p, axis=0) / POS_SCALE
    rot = Rotation.from_matrix(dR).as_rotvec() / ROT_SCALE
    op = st[:, 7]
    d = np.diff(op)
    g = np.full(L, -1.0, dtype=np.float32)
    for t in range(L - 1):
        if d[t] < -GRIP_EPS: g[t] = 1.0
        elif d[t] > GRIP_EPS: g[t] = -1.0
        else: g[t] = g[t - 1] if t > 0 else -1.0
    g[L - 1] = g[L - 2] if L > 1 else -1.0
    act[a:a + L - 1, :3] = pos
    act[a:a + L - 1, 3:6] = rot
    act[a:a + L - 1, 6] = g[:L - 1]
    act[a + L - 1] = act[a + L - 2]
    prop[a:b, :3] = p
    prop[a:b, 3:6] = quat_wxyz_to_aa(quat)
    prop[a:b, 6] = 0.04 * op
    prop[a:b, 7] = -0.04 * op
    opening[a:b] = op
    joint_cmd[a:b] = ac[:, 7]
    trans_hist.append(int(np.sum(np.diff(g) != 0)) + 1)

sat = np.mean(np.abs(act[:, :6]) >= 0.999)
act[:, :6] = np.clip(act[:, :6], -1.0, 1.0)
print("[act] mean|a|:", np.round(np.abs(act).mean(0), 4).tolist(), flush=True)
print("[act] sat_fraction(before clip):", round(float(sat), 5), flush=True)
print("[act] gripper +1 frac:", round(float((act[:, 6] > 0).mean()), 3), "segments per ep:", np.bincount(trans_hist).tolist(), flush=True)
print("[raw] opening range:", round(float(opening.min()), 4), round(float(opening.max()), 4),
      "| stored action[7] range:", round(float(joint_cmd.min()), 4), round(float(joint_cmd.max()), 4), flush=True)
print("[raw] per-frame |dp| m: mean %.4f p99 %.4f" % (np.abs(np.diff(np.concatenate([[0], prop[:, 0]]))).mean(),
      np.percentile(np.abs(np.diff(prop[:, 2])), 99)), flush=True)

np.savez_compressed(os.path.join(OUT, "arrays.npz"),
                    action=act, proprio=prop, opening=opening, joint_cmd=joint_cmd,
                    episode_index=ep_arr, ep_start=np.array([a for _, a, _ in bounds], dtype=np.int64),
                    ep_len=lens.astype(np.int64), ep_ids=np.array([e for e, _, _ in bounds], dtype=np.int64))

amin = [-1.0] * 7
amax = [1.0] * 7
smin, smax = prop.min(0), prop.max(0)
smean, sstd = prop.mean(0), prop.std(0)
stats = {"action": {"default": {"global_min": amin, "global_max": amax,
                                "global_mean": np.zeros(7).tolist(), "global_std": np.ones(7).tolist(),
                                "global_q01": [-1.0] * 7, "global_q99": [1.0] * 7}},
         "state": {"default": {"global_min": smin.tolist(), "global_max": smax.tolist(),
                               "global_mean": smean.tolist(), "global_std": sstd.tolist(),
                               "global_q01": np.percentile(prop, 1, axis=0).tolist(),
                               "global_q99": np.percentile(prop, 99, axis=0).tolist()}},
         "num_episodes": len(bounds), "num_transition": int(N)}
json.dump(stats, open(os.path.join(OUT, "dataset_stats.json"), "w"), indent=2)
json.dump({"empirical_action_min": act.min(0).tolist(), "empirical_action_max": act.max(0).tolist(),
           "task": "transfer 10 cookies into target box", "fps": 20, "num_frames": int(N),
           "num_episodes": len(bounds)}, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print("[done]", OUT, "frames:", frames.shape, "bytes:", 12 * IMG * IMG * 3 * N // 10**9, "GB", flush=True)
