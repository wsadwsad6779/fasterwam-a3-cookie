import glob, numpy as np
E = np.load("/workspace/_r4_stage/expert_force.npz")
A = np.load("/workspace/a3_data/a3_fw/arrays.npz", mmap_mode="r")
FO = E["force"].astype(np.float64); EP = E["episode_index"]
op = np.asarray(A["opening"], float); pad = np.maximum(FO[:, 12], FO[:, 13])
THR = 0.05; GAP = 8
rec = []
for e in range(100):
    w = np.flatnonzero(EP == e)
    p = pad[w]; o = op[w]
    hit = p > THR
    d = np.flatnonzero(np.diff(np.r_[0, hit.astype(int), 0].astype(int)) != 0)
    seg = [(int(d[i]), int(d[i + 1] - 1)) for i in range(0, len(d) - 1, 2)]
    mg = []
    for s0, s1 in seg:
        if mg and s0 - mg[-1][1] - 1 <= GAP:
            mg[-1] = (mg[-1][0], s1)
        else:
            mg.append((s0, s1))
    for s0, s1 in mg:
        if s1 - s0 + 1 < 3:
            continue
        pre = p[max(0, s0 - 20):s0]
        hold = p[s0 + 8:s1 - 4]
        idx = np.flatnonzero(p[s0:] >= 1.0)
        t10 = int(idx[0]) if idx.size else -1
        tail = p[s1 + 1:]
        und = np.flatnonzero(tail <= THR)
        decay = int(und[0]) + 1 if und.size else -1
        rec.append(dict(peak=float(p[s0:s1 + 1].max()), onset=float(p[s0]),
                        press=float(pre.max()) if pre.size else 0.0,
                        hold=float(np.median(hold)) if hold.size else float("nan"),
                        t_to_1N=t10, decay=decay, dur=s1 - s0 + 1,
                        open_min=float(o[s0:s1 + 1].min()), open_max=float(o[max(0, s0 - 25):s0].max()) if s0 > 0 else float(o[0])))
def P(name, v, unit=""):
    v = np.array([x for x in v if x is not None], dtype=np.float64)
    v = v[~np.isnan(v)]
    a = np.percentile(v, [5, 50, 95])
    print("%-22s n=%4d  p5=%7.3f  p50=%7.3f  p95=%7.3f  min=%7.3f  max=%7.3f %s" % (name, v.size, a[0], a[1], a[2], v.min(), v.max(), unit))
print("=== EXPERT (100 eps, %d grasp runs, batch_size=5 x2) ===" % len(rec))
P("press_before_contact", [r["press"] for r in rec], "N")
P("pad_at_contact_onset", [r["onset"] for r in rec], "N")
P("pad_hold_median", [r["hold"] for r in rec], "N")
P("pad_peak", [r["peak"] for r in rec], "N")
P("frames_to_reach_1N", [r["t_to_1N"] for r in rec if r["t_to_1N"] >= 0], "frames@20Hz")
P("contact_dur", [r["dur"] for r in rec], "frames")
P("frames_to_zero_after", [r["decay"] for r in rec if r["decay"] >= 0], "frames")
P("gripper_open_while_clamped", [r["open_min"] for r in rec])
P("gripper_open_before_close", [r["open_max"] for r in rec])
print("runs with decay==-1 (never released inside ep):", sum(1 for r in rec if r["decay"] < 0))
print("runs reaching >=1.0N:", sum(1 for r in rec if r["t_to_1N"] >= 0), " >=1.3N target:", sum(1 for r in rec if r["peak"] >= 1.3))
print()
print("=== POLICY r4 rollouts (same threshold 0.05 N) ===")
OUT = "/workspace/a3_eval/r4_b0_c0_paired"
for ctl in ("B0", "C0"):
    for sd in (11, 21, 22):
        c = [f for f in glob.glob(OUT + "/**/*%d*.npz" % sd, recursive=True) if ctl in f]
        if not c:
            print(ctl, sd, "NOFILE")
            continue
        z = np.load(c[0], mmap_mode="r")
        pd_ = np.maximum(np.asarray(z["pad_force_inner"], float), np.asarray(z["pad_force_outer"], float))
        wr = np.asarray(z["wrist_force"], float)
        wn = np.linalg.norm(wr, axis=1)
        h = pd_ > THR
        print("%s/%s pad p50=%6.2f p95=%7.2f p99=%7.2f max=%8.2f N | contact frames %3d/600 | >20N %3d | >40N %3d | wrist p50=%6.2f max=%7.2f N" % (
            ctl, sd, np.percentile(pd_, 50), np.percentile(pd_, 95), np.percentile(pd_, 99), pd_.max(),
            int(h.sum()), int((pd_ > 20).sum()), int((pd_ > 40).sum()), np.median(wn), wn.max()))
