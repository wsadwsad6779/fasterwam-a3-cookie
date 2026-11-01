import sys, glob, json, os
import numpy as np

PHASES = ("open", "preclose", "descend", "close", "hold", "release", "done", "raw")
CH = ["Lx", "Ly", "Lz", "Lrx", "Lry", "Lrz", "Lgrip",
      "Rx", "Ry", "Rz", "Rrx", "Rry", "Rrz", "Rgrip"]


def prog(rp, seed):
    epf = "/workspace/expert_seed%d.npz" % seed
    if not os.path.exists(epf):
        return None, None
    ep = np.load(epf, allow_pickle=True)["eef"][:, :3].astype(float)
    a = ep[:-1]
    ab = ep[1:] - a
    L2 = (ab * ab).sum(1)
    P = rp[:, None, :]
    num = ((P - a[None]) * ab[None]).sum(-1)
    t = np.clip(num / np.where(L2 == 0, 1.0, L2)[None], 0.0, 1.0)
    q = a[None] + t[..., None] * ab[None]
    dd = np.linalg.norm(P - q, axis=-1)
    j = dd.argmin(1)
    cnt = j + t[np.arange(len(rp)), j]
    return (round(float(cnt[-1]) / (len(ep) - 1) * 100, 1),
            round(float(np.median(dd.min(1))) * 1000, 1))


def breakdown(R):
    pre = np.asarray(R["pre_14d"], float)
    po = np.asarray(R["post_14d"], float)
    pw = np.asarray(R["post_wrap_14d"], float) if "post_wrap_14d" in R else pre.copy()
    ph = np.asarray(R["phase"], dtype=int) if "phase" in R else np.full(len(pre), -1)
    out = {}
    for pval in sorted(set(ph.tolist())):
        m = ph == pval
        name = PHASES[pval] if 0 <= pval < len(PHASES) else ("unwrapped" if pval == -1 else str(pval))
        dw = np.abs(pw[m] - pre[m])
        ds = np.abs(po[m] - pw[m])
        row = {}
        for c in range(pre.shape[1]):
            ew = int((dw[:, c] > 1e-9).sum())
            es = int((ds[:, c] > 1e-9).sum())
            if ew or es:
                row[CH[c]] = [ew, es, round(float(dw[:, c].mean()), 6), round(float(ds[:, c].mean()), 6)]
        out[name] = {"n": int(m.sum()), "ch": row}
    return out


def cookie_metrics(R, rp, op):
    o = {}
    if "cookie_xyz" not in R:
        return o
    c = np.asarray(R["cookie_xyz"], float)
    c0 = np.asarray(R["cookie_xyz0"], float) if "cookie_xyz0" in R else c[0]
    T, n, _ = c.shape
    disp = np.linalg.norm(c - c0[None], axis=-1)
    lift = c[:, :, 2] - c0[None, :, 2]
    o["n_cookies"] = int(n)
    o["cookie_max_disp_mm"] = round(float(disp.max()) * 1000, 1)
    o["cookie_max_lift_mm"] = round(float(lift.max()) * 1000, 1)
    o["cookies_moved_gt20mm"] = int((disp.max(0) > 0.02).sum())
    d = np.linalg.norm(c - rp[:, None, :], axis=-1)
    near = (d.min(1) < 0.04) & (op < 0.9)
    o["closed_steps"] = int((op < 0.9).sum())
    o["closed_near_steps"] = int(near.sum())
    if near.sum() == 0:
        o["real_grasp"] = False
        return o
    cand = np.unique(d[near].argmin(1)); held = int(cand[np.argmax(disp[near][:, cand].max(0))])
    o["held_cookie"] = held
    o["held_disp_mm"] = round(float(disp[near, held].max()) * 1000, 1)
    o["held_end_disp_mm"] = round(float(disp[-1, held]) * 1000, 1)
    o["held_lift_mm"] = round(float(lift[near, held].max()) * 1000, 1)
    ch = c[:, held, :]
    ok = near[1:] & near[:-1]
    dc = np.diff(ch, axis=0)
    de = np.diff(rp, axis=0)
    fe = np.linalg.norm(dc - de, axis=1)
    rel = (ch[1:] - rp[1:]) - (ch[:-1] - rp[:-1])
    if ok.any():
        o["held_follow_err_mm"] = round(float(np.median(fe[ok])) * 1000, 2)
        o["held_slip_mm"] = round(float(np.median(np.linalg.norm(rel, axis=1)[ok])) * 1000, 2)
    rigid = ok & (fe < 0.003) & (np.diff(ch[:, 2]) > 0.0005)
    best = run = 0
    for v in rigid:
        run = run + 1 if v else 0
        best = max(best, run)
    o["follow_rise_run"] = int(best)
    o["real_grasp"] = bool(best >= 10 and o["held_lift_mm"] >= 10.0)
    return o



# Static calibration-time geometry (configs/cookie_batch.yaml + config.py
# defaults): target-bin centre, inner half extents after wall thickness, and
# the interior z band (floor top -> rim). Fixed geometry, not simulator state.
TARGET_BOX_CENTER = (0.095, 0.100, 0.753)
TARGET_BOX_INNER_HALF = (0.057, 0.029)
TARGET_BOX_Z = (0.759, 0.779)
TARGET_REGION_Z = (0.748, 0.818)
TARGET_TOL = 0.004


def box_metrics(R, held=None):
    o = {}
    if "cookie_xyz" not in R:
        o["box_dist"] = "N/A"
        return o
    c = np.asarray(R["cookie_xyz"], float)
    c0 = np.asarray(R["cookie_xyz0"], float) if "cookie_xyz0" in R else c[0]
    lift = c[:, :, 2] - c0[None, :, 2]
    op = np.asarray(R["opening"], float)
    cx, cy, _ = TARGET_BOX_CENTER
    hx, hy = TARGET_BOX_INNER_HALF
    z0, z1 = TARGET_BOX_Z
    dx = np.maximum(np.abs(c[..., 0] - cx) - hx, 0.0)
    dy = np.maximum(np.abs(c[..., 1] - cy) - hy, 0.0)
    dz = np.maximum(np.maximum(z0 - c[..., 2], c[..., 2] - z1), 0.0)
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    dxy = np.sqrt(dx * dx + dy * dy)
    k = np.unravel_index(int(np.argmin(dist)), dist.shape)
    o["box_min_dist_mm"] = round(float(dist[k]) * 1000, 1)
    o["box_min_xy_dist_mm"] = round(float(dxy.min()) * 1000, 1)
    o["box_min_dist_step"] = int(k[0])
    o["box_min_dist_cookie"] = int(k[1])
    rz0, rz1 = TARGET_REGION_Z
    inside = ((np.abs(c[..., 0] - cx) <= hx + TARGET_TOL)
              & (np.abs(c[..., 1] - cy) <= hy + TARGET_TOL)
              & (c[..., 2] >= rz0) & (c[..., 2] <= rz1))
    o["ever_in_target_region"] = int(inside.any(0).sum())
    if held is not None and 0 <= int(held) < c.shape[1]:
        h = int(held)
        o["held_box_min_dist_mm"] = round(float(dist[:, h].min()) * 1000, 1)
        o["held_box_min_xy_dist_mm"] = round(float(dxy[:, h].min()) * 1000, 1)
        o["held_ever_in_target_region"] = bool(inside[:, h].any())
        run = 0
        for v in inside[::-1, h]:
            if v:
                run += 1
            else:
                break
        o["held_target_region_tail_steps"] = int(run)
        lk = lift[:, h]
        clo = op < 0.9
        o["held_final_lift_mm"] = round(float(lk[-1]) * 1000, 1)
        if clo.any():
            peak_step = int(np.argmax(np.where(clo, lk, -np.inf)))
            peak = float(lk[peak_step])
            after = lk[peak_step:][clo[peak_step:]]
            drop = max(0.0, peak - float(after.min())) if after.size else 0.0
            o["held_peak_lift_mm"] = round(peak * 1000, 1)
            o["held_drop_mm"] = round(drop * 1000, 1)
            o["dropped_after_lift"] = bool(peak >= 0.010 and drop >= 0.008)
        else:
            o["held_peak_lift_mm"] = 0.0
            o["held_drop_mm"] = 0.0
            o["dropped_after_lift"] = False
    return o


def one(f):
    R = np.load(f, allow_pickle=True)
    seed = int(f[:-4].split("_")[-1])
    rp = np.asarray(R["eef_xyz"], float)
    op = np.asarray(R["opening"], float)
    pad = np.maximum(np.abs(R["pad_force_inner"]), np.abs(R["pad_force_outer"]))
    o = {"steps": int(len(rp))}
    o["prog"], o["offpath"] = prog(rp, seed)
    tail = rp[-200:]
    o["tail_span_mm"] = round(float(np.linalg.norm(tail.max(0) - tail.min(0))) * 1000, 1)
    o["pad_max"] = round(float(pad.max()), 1)
    o["pad_gt10_pct"] = round(100 * float((pad > 10).mean()), 1)
    hit = np.where((pad > 5.0) & (op < 0.9))[0]
    if hit.size:
        k = int(hit[0])
        o["contact_step"] = k
        o["tcp_lift_mm"] = round(float(rp[k:, 2].max() - rp[k, 2]) * 1000, 1)
    else:
        o["contact_step"] = None
        o["tcp_lift_mm"] = None
    pre = np.asarray(R["pre_14d"], float)
    po = np.asarray(R["post_14d"], float)
    pw = np.asarray(R["post_wrap_14d"], float) if "post_wrap_14d" in R else pre.copy()
    ch = np.any(np.abs(po - pre) > 1e-9, axis=1)
    wr = np.any(np.abs(pw - pre) > 1e-9, axis=1)
    sf = np.any(np.abs(po - pw) > 1e-9, axis=1)
    o["rewrite_pct"] = round(100 * float(ch.mean()), 1)
    o["wrapper_pct"] = round(100 * float(wr.mean()), 1)
    o["safety_pct"] = round(100 * float(sf.mean()), 1)
    o["both_pct"] = round(100 * float((wr & sf).mean()), 1)
    o["has_split"] = bool("post_wrap_14d" in R)
    o.update(cookie_metrics(R, rp, op))
    o.update(box_metrics(R, o.get("held_cookie")))
    sm = os.path.join(os.path.dirname(f), "summary_lora.json")
    if os.path.exists(sm):
        sj = json.load(open(sm))
        row = [x for x in sj if x.get("seed") == seed]
        if row:
            o["cookies"] = row[0].get("cookies_in_target")
            o["verbatim"] = row[0].get("verbatim_pct")
    return seed, o, R


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print("==", d)
        mf = os.path.join(d, "manifest_lora.json")
        if os.path.exists(mf):
            print("  ckpt", json.dumps(json.load(open(mf)).get("checkpoint")))
        for f in sorted(glob.glob(os.path.join(d, "lora_*_*.npz"))):
            seed, o, R = one(f)
            print("  seed", seed, json.dumps(o, ensure_ascii=False))
            print("    split", json.dumps(breakdown(R), ensure_ascii=False))
            keys = ("real_grasp", "held_cookie", "held_peak_lift_mm", "held_final_lift_mm",
                    "held_drop_mm", "dropped_after_lift", "held_slip_mm", "follow_rise_run",
                    "box_min_xy_dist_mm", "held_box_min_xy_dist_mm",
                    "held_ever_in_target_region", "held_target_region_tail_steps",
                    "cookies", "wrapper_pct", "safety_pct", "both_pct")
            print("    answer", json.dumps({k: o.get(k, "N/A") for k in keys}, ensure_ascii=False))
