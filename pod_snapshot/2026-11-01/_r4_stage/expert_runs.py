import numpy as np, csv
E = np.load("/workspace/_r4_stage/expert_force.npz")
A = np.load("/workspace/a3_data/a3_fw/arrays.npz", mmap_mode="r")
FO = E["force"].astype(np.float64)
EP = E["episode_index"]
op = np.asarray(A["opening"], dtype=np.float64)
pr = np.asarray(A["proprio"], dtype=np.float64)
pad = np.maximum(FO[:, 12], FO[:, 13])
act = FO[:, 16]
wf = np.linalg.norm(FO[:, 0:3], axis=1)
THR = 0.05
GAP = 8
runs = []
for e in range(100):
    w = np.flatnonzero(EP == e)
    p = pad[w]
    a = act[w]
    o = op[w]
    z = pr[w, 2]
    xy = pr[w, 0:2]
    wl = wf[w]
    hit = p > THR
    d = np.flatnonzero(np.diff(np.r_[0, hit.astype(int), 0].astype(int)) != 0)
    seg = [(int(d[i]), int(d[i + 1] - 1)) for i in range(0, len(d) - 1, 2)]
    merged = []
    for s0, s1 in seg:
        if merged and s0 - merged[-1][1] - 1 <= GAP:
            merged[-1] = (merged[-1][0], s1)
        else:
            merged.append((s0, s1))
    for s0, s1 in merged:
        if s1 - s0 + 1 < 3:
            continue
        b0 = max(0, s0 - 20)
        b1 = min(len(w) - 1, s1 + 20)
        oo = o[b0:b1 + 1]
        rows_slice = slice(s0, s1 + 1)
        rec = {}
        rec["ep"] = e
        rec["start"] = s0
        rec["len"] = s1 - s0 + 1
        rec["pad_onset"] = float(p[s0])
        rec["pad_peak"] = float(p[rows_slice].max())
        rec["pad_mean"] = float(p[rows_slice].mean())
        rec["pad_last"] = float(p[s1])
        rec["pad_after5"] = float(p[min(s1 + 5, len(w) - 1)])
        rec["pad_after10"] = float(p[min(s1 + 10, len(w) - 1)])
        rec["act_peak"] = float(a[rows_slice].max())
        rec["act_min"] = float(a[rows_slice].min())
        rec["open_min"] = float(o[rows_slice].min())
        rec["open_before"] = float(o[max(0, s0 - 15)])
        rec["open_after"] = float(o[min(len(w) - 1, s1 + 15)])
        rec["z_at_start"] = float(z[s0])
        rec["z_at_end"] = float(z[s1])
        rec["z_max_after"] = float(z[s1:min(s1 + 60, len(w))].max()) if s1 + 2 < len(w) else float(z[s1])
        rec["lift_after"] = float(rec["z_max_after"] - z[rows_slice].min())
        rec["xy_after"] = float(np.linalg.norm(xy[min(s1 + 60, len(w) - 1)] - xy[s0]))
        rec["wf_mean"] = float(wl[rows_slice].mean())
        rec["wf_peak"] = float(wl[rows_slice].max())
        runs.append(rec)
print("contact runs", len(runs), "per-episode mean %.2f" % (len(runs) / 100.0))
cnt = np.bincount([r["ep"] for r in runs], minlength=100)
print("runs/episode p10/p50/p90", np.percentile(cnt, [10, 50, 90]), "min/max", cnt.min(), cnt.max())
cols = list(runs[0].keys())
fh = open("/workspace/_r4_stage/expert_runs.csv", "w", newline="")
wr = csv.DictWriter(fh, fieldnames=cols)
wr.writeheader()
wr.writerows(runs)
fh.close()
def q(name, key, scale=1.0):
    v = np.array([r[key] for r in runs], dtype=np.float64) * scale
    a = np.percentile(v, [5, 25, 50, 75, 95, 99])
    print("%-16s p5=%8.3f p25=%8.3f p50=%8.3f p75=%8.3f p95=%8.3f p99=%8.3f max=%8.3f" % (name, a[0], a[1], a[2], a[3], a[4], a[5], v.max()))
q("pad_onset", "pad_onset")
q("pad_peak", "pad_peak")
q("pad_mean", "pad_mean")
q("pad_last", "pad_last")
q("pad_after5", "pad_after5")
q("pad_after10", "pad_after10")
q("act_peak", "act_peak")
q("len_frames", "len")
q("open_min", "open_min")
q("open_before", "open_before")
q("open_after", "open_after")
q("lift_after_mm", "lift_after", 1000.0)
q("xy_after_mm", "xy_after", 1000.0)
q("wf_mean", "wf_mean")
q("wf_peak", "wf_peak")
om = np.array([r["open_min"] for r in runs])
print("open_min hist 0..1 by 0.05:", np.histogram(om, bins=np.arange(0, 1.05, 0.05))[0].tolist())
pk = np.array([r["pad_peak"] for r in runs])
print("pad_peak bands: >1N %d >2N %d >3N %d >5N %d" % ((pk > 1).sum(), (pk > 2).sum(), (pk > 3).sum(), (pk > 5).sum()))
print("global act range p1/p99/p999 %.3f %.3f %.3f" % tuple(np.percentile(act, [1, 99, 99.9])))
print("pad global p99/p999/max %.4f %.4f %.4f" % (np.percentile(pad, 99), np.percentile(pad, 99.9), pad.max()))
