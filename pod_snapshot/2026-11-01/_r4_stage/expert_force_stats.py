import numpy as np, csv, math
E = np.load("/workspace/_r4_stage/expert_force.npz")
A = np.load("/workspace/a3_data/a3_fw/arrays.npz", mmap_mode="r")
FO = E["force"].astype(np.float64)
EP = E["episode_index"]
TS = E["timestamp"].astype(np.float64)
ac = np.asarray(A["action"], dtype=np.float64)
pr = np.asarray(A["proprio"], dtype=np.float64)
op = np.asarray(A["opening"], dtype=np.float64)
pad = np.maximum(FO[:, 12], FO[:, 13])
padR = np.maximum(FO[:, 14], FO[:, 15])
wf = np.linalg.norm(FO[:, 0:3], axis=1)
wt = np.linalg.norm(FO[:, 3:6], axis=1)
wfr = np.linalg.norm(FO[:, 6:9], axis=1)
af = FO[:, 16]
free = op > 0.5
thr = max(float(np.percentile(pad[free], 99)) + 1e-6, 0.05)
print("frames", pad.size, "free_frames", int(free.sum()))
p50 = float(np.percentile(pad[free], 50))
p90 = float(np.percentile(pad[free], 90))
p99 = float(np.percentile(pad[free], 99))
print("freespace pad p50/p90/p99 %.4f %.4f %.4f thr %.4f" % (p50, p90, p99, thr))
print("freespace p99: Lwrist %.3f N  Rwrist %.3f N  Rpad %.3f N  Ltors %.3f Nm" % (
    np.percentile(wf[free], 99), np.percentile(wfr[free], 99), np.percentile(padR[free], 99), np.percentile(wt[free], 99)))
rows = []
nan = float("nan")
for e in range(100):
    w = np.flatnonzero(EP == e)
    L = w.size
    if L < 20:
        continue
    g = ac[w, 6]
    o = op[w]
    z = pr[w, 2]
    pd_ = pad[w]
    cl = np.flatnonzero((g[1:] > 0) & (g[:-1] <= 0)) + 1
    if cl.size == 0:
        continue
    for s0 in cl:
        nxt = np.flatnonzero((g[s0 + 1:] < 0) & (g[s0:-1] >= 0))
        e1 = int(s0 + 1 + nxt[0]) if nxt.size else L - 1
        c = slice(int(s0), e1 + 1)
        oc = o[c]; zc = z[c]; pc = pd_[c]
        lc = e1 - int(s0) + 1
        hit = np.flatnonzero(pc > thr)
        on = int(hit[0]) if hit.size else -1
        amin = int(np.argmin(oc))
        d = {}
        d["ep"] = e
        d["t0"] = round(float(TS[w[int(s0)]]), 3)
        d["dur"] = lc
        d["contact"] = 1 if on >= 0 else 0
        d["onset_rel"] = on
        d["pad_onset"] = float(pc[on]) if on >= 0 else nan
        d["pad_peak"] = float(pc.max())
        d["pad_ramp5"] = float(pc[on:on + 5].max()) if on >= 0 else nan
        d["pad_hold"] = float(np.median(pc[amin + 4:lc - 6])) if (lc - amin - 10) >= 6 else nan
        d["pad_pre_open"] = float(pc[-1])
        d["pad_after_open8"] = float(pd_[min(e1 + 8, L - 1)])
        d["wf_peak"] = float(wf[w[c]].max())
        d["wt_peak"] = float(wt[w[c]].max())
        d["act_peak"] = float(af[w[c]].max())
        d["open_start"] = float(oc[0])
        d["open_settle"] = float(oc[-6:].mean())
        d["z_min"] = float(zc.min())
        d["z_max"] = float(zc.max())
        rows.append(d)
print("cycles", len(rows))
cols = list(rows[0].keys())
fh = open("/workspace/_r4_stage/expert_cycles.csv", "w", newline="")
wr = csv.DictWriter(fh, fieldnames=cols)
wr.writeheader()
wr.writerows(rows)
fh.close()
def q(name, key, scale=1.0):
    v = np.array([r[key] for r in rows], dtype=np.float64) * scale
    v = v[~np.isnan(v)]
    a = np.percentile(v, [10, 50, 90, 99])
    print("%-20s n=%5d p10=%8.3f p50=%8.3f p90=%8.3f p99=%8.3f max=%9.3f" % (name, v.size, a[0], a[1], a[2], a[3], v.max()))
q("pad_onset_N", "pad_onset")
q("pad_ramp5_N", "pad_ramp5")
q("pad_peak_N", "pad_peak")
q("pad_hold_N", "pad_hold")
q("pad_pre_open_N", "pad_pre_open")
q("pad_after_open_N", "pad_after_open8")
q("wrist_peak_N", "wf_peak")
q("wrist_tor_peak_Nm", "wt_peak")
q("finger_act_peak_N", "act_peak")
q("open_settle_m", "open_settle")
q("z_lift_mm", None) if False else None
zl = np.array([(r["z_max"] - r["z_min"]) * 1000 for r in rows])
a = np.percentile(zl, [10, 50, 90, 99])
print("%-20s n=%5d p10=%8.3f p50=%8.3f p90=%8.3f p99=%8.3f max=%9.3f" % ("z_lift_mm", zl.size, a[0], a[1], a[2], a[3], zl.max()))
hv = np.array([r["pad_peak"] for r in rows])
print("pad_peak bands: >5N %d >10N %d >20N %d >25N %d >40N %d total %d" % (
    (hv > 5).sum(), (hv > 10).sum(), (hv > 20).sum(), (hv > 25).sum(), (hv > 40).sum(), hv.size))
os_ = np.array([r["open_settle"] for r in rows])
bins = np.arange(0.0, 0.0625, 0.005)
print("open_settle hist", np.histogram(os_, bins=bins)[0].tolist())
print("contact-less cycles", int(sum(1 for r in rows if r["contact"] == 0)))
