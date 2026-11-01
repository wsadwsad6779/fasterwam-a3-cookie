import glob, json, os
import numpy as np
import pyarrow.parquet as pq
CAND = ["/workspace/a3_data/a3-front-close-left-100", "/workspace/data/a3_front_close_left_100"]
ROOT = next((c for c in CAND if os.path.isdir(c)), None)
print("ROOT", ROOT)
files = sorted(glob.glob(ROOT + "/data/chunk-000/*.parquet"))
print("files", len(files))
ep, fr, ts, fo, st = [], [], [], [], []
for f in files:
    d = pq.read_table(f, columns=["episode_index", "frame_index", "timestamp", "observation.force", "observation.state"]).to_pydict()
    ep += [int(v) for v in d["episode_index"]]
    fr += [int(v) for v in d["frame_index"]]
    ts += [float(v) for v in d["timestamp"]]
    fo += [np.asarray(v, dtype=np.float32) for v in d["observation.force"]]
    st += [np.asarray(v, dtype=np.float32) for v in d["observation.state"]]
o = np.lexsort((np.array(fr), np.array(ep)))
ep = np.array(ep)[o]; fr = np.array(fr)[o]; ts = np.array(ts, dtype=np.float32)[o]
FO = np.stack(fo)[o]; ST = np.stack(st)[o]
print("rows", ep.shape, "eps", ep.min(), ep.max(), "force", FO.shape)
np.savez("/workspace/_r4_stage/expert_force.npz", episode_index=ep, frame_index=fr, timestamp=ts, force=FO, state=ST)
z = np.load("/workspace/a3_data/a3_fw/arrays.npz", mmap_mode="r")
print("arrays rows", z["action"].shape, "ep_start", z["ep_start"][:5], "ep_len", z["ep_len"][:5])
a_ep = np.asarray(z["episode_index"]); a_open = np.asarray(z["opening"])
print("ep identical:", bool(np.array_equal(a_ep, ep)), "len identical:", a_ep.shape == ep.shape)
sd = float(np.abs(a_open - ST[:, 7]).max())
print("max|opening - state7| =", sd)
print("ts monotone within ep0:", bool(np.all(np.diff(ts[ep == ep.min()]) > 0)), "dt median", float(np.median(np.diff(ts[:400]))))
