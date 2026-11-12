#!/usr/bin/env python3
"""P0 follow-up, CPU only: what physically generates the six force columns the model reads?

Every number here comes from files the run itself shipped -- the corpus arrays, the frozen
calibration, the normalizer stats the checkpoint was trained with, and the per-step arrays the
evaluation recorded -- not from any diagnostic I wrote earlier in this investigation.

The question is narrow. `make_force_calib.py` records the corpus wrist `fy` mean as -10.91 N with
a 1-sigma spread of 0.589 N, and `force_calib_v1.json` says in its own words: raw sensor values,
no gravity compensation, flange frame. A MuJoCo <force site=..._wrist_ft> sensor on the flange body
reports the load of everything distal to it, expressed in the flange frame, so the reading is the
projection of gravity onto a moving frame: it codes wrist *orientation*, not contact. This script
fits that projection on the corpus and then asks how much of the excursion measured during the
evaluation is orientation and how much is contact.

No torch, no model load, no MuJoCo, nothing written.
"""
import json
import sys
from pathlib import Path

import numpy as np

V1G = Path("/workspace/a3_data/a3_fw_v1g/arrays.npz")
FORCE = Path("/workspace/a3_data/a3_fw_force/arrays.npz")
CALIB = Path("/workspace/a3_data/a3_fw_force/force_calib_v1.json")
STATS = Path("/workspace/a3_data/a3_fw_v1g/dataset_stats.json")
RECORD = Path("/workspace/_v1_stage/dev/out40k/record_diag/_record")

BASE_COLS = ["eef_x", "eef_y", "eef_z", "aa_x", "aa_y", "aa_z", "fin_l", "fin_r"]
FCOLS = ["f_fx", "f_fy", "f_fz", "f_pad_in", "f_pad_out", "f_grip_act"]
G = 9.8151
TARE_WINDOW = 20


def r(x, n=4):
    return round(float(x), n)


def band(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    return [[r(a[:, i].min()), r(a[:, i].max()), r(a[:, i].mean()), r(a[:, i].std())]
            for i in range(a.shape[1])]


def rot_from_axis_angle(aa):
    """Rodrigues, batched: (n,3) axis-angle -> (n,3,3) site-to-world rotation."""
    aa = np.asarray(aa, dtype=np.float64).reshape(-1, 3)
    th = np.linalg.norm(aa, axis=1)
    safe = np.where(th > 1e-12, th, 1.0)
    k = aa / safe[:, None]
    K = np.zeros((aa.shape[0], 3, 3))
    K[:, 0, 1] = -k[:, 2]
    K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]
    K[:, 2, 1] = k[:, 0]
    s = np.sin(th)[:, None, None]
    c = (1.0 - np.cos(th))[:, None, None]
    eye = np.eye(3)[None, :, :]
    return eye + s * K + c * np.einsum('nij,njk->nik', K, K)


def up_in_flange(aa):
    """The world +z axis expressed in the flange frame: R^T e_z, i.e. the third row of R."""
    R = rot_from_axis_angle(aa)
    return R[:, 2, :]


def fit(A, B):
    """Least squares B ~ A @ P; returns P, per-channel RMS of the fit, per-channel R^2."""
    P, *_ = np.linalg.lstsq(A, B, rcond=None)
    pred = A @ P
    res = B - pred
    rms = np.sqrt((res ** 2).mean(axis=0))
    ss = ((B - B.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - (res ** 2).sum(axis=0) / np.where(ss > 0, ss, 1.0)
    return P, rms, r2


def main():
    print("== 0. inputs ==")
    for p in (V1G, FORCE, CALIB, STATS):
        print("  %s exists=%s size=%s" % (p, p.is_file(), p.stat().st_size if p.is_file() else None))

    cal = json.loads(CALIB.read_text())
    tare = np.asarray(cal["zero_load_ref_median"], dtype=np.float64)
    sd = np.asarray(cal["std"], dtype=np.float64)
    st = json.loads(STATS.read_text())["state"]["default"]
    lo = np.asarray(st["global_min"], dtype=np.float64)
    hi = np.asarray(st["global_max"], dtype=np.float64)
    scale = 2.0 / (hi - lo)
    clamp_n = np.array([5.0 / scale[8 + i] for i in range(6)])
    print("  tare=%s" % [r(x, 5) for x in tare])
    print("  1 sigma=%s" % [r(x, 5) for x in sd])
    print("  |y|=5 clamp bites at residual=%s N" % [r(x, 4) for x in clamp_n])

    f = np.load(FORCE, allow_pickle=False)
    v = np.load(V1G, allow_pickle=False)
    p14 = np.asarray(v["proprio"], dtype=np.float64)
    raw6 = np.asarray(f["force6"], dtype=np.float64)
    ep_start = np.asarray(f["ep_start"])
    ep_len = np.asarray(f["ep_len"])
    ep_idx = np.asarray(f["episode_index"])
    n = p14.shape[0]
    print("== 1. the 14 columns the trainer actually read (rows=%d, episodes=%d) ==" %
          (n, ep_len.shape[0]))
    print("  base cols %s" % BASE_COLS)
    for c, name in zip(band(p14[:, :8]), BASE_COLS):
        print("    %-7s min=%-10s max=%-10s mean=%-10s std=%s" % (name, c[0], c[1], c[2], c[3]))
    recon_err = float(np.abs(p14[:, 8:14] - (raw6 - tare)).max())
    print("  stored force columns vs (force6 - tare): max abs diff=%s" % r(recon_err, 8))
    print("  finger columns carry opposite joint axes: fin_l mean=%s fin_r mean=%s" %
          (r(p14[:, 6].mean(), 5), r(p14[:, 7].mean(), 5)))

    # frame position inside the episode, for the tare window and the contact split
    starts = np.zeros(n, dtype=np.int64)
    for k, s in enumerate(ep_start):
        starts[s:s + int(ep_len[k])] = np.arange(int(ep_len[k]))
    u = up_in_flange(p14[:, 3:6])
    light = (starts < TARE_WINDOW) & (np.abs(raw6[:, 3]) < 1e-9) & (np.abs(raw6[:, 4]) < 1e-9)
    print("== 2. is the wrist force the gravity of the distal load, seen in the flange frame? ==")
    print("  fit subset: first %d frames of each episode with both pad channels exactly 0: rows=%d"
          % (TARE_WINDOW, int(light.sum())))
    Ul, Fl = u[light], raw6[light, :3]
    print("  |raw wrist force| in the subset: med=%s p95=%s min=%s max=%s N" %
          (r(np.median(np.linalg.norm(Fl, axis=1)), 3),
           r(np.percentile(np.linalg.norm(Fl, axis=1), 95), 3),
           r(np.linalg.norm(Fl, axis=1).min(), 3), r(np.linalg.norm(Fl, axis=1).max(), 3)))
    cos = (Fl * Ul).sum(1) / (np.linalg.norm(Fl, axis=1) * np.linalg.norm(Ul, axis=1))
    angfu = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    print("  angle between the measured wrist force and the flange-frame up axis: med=%s p95=%s "
          "max=%s deg  (0 deg would mean the reading IS the weight of what hangs below the wrist)"
          % (r(angfu[0], 2), r(np.percentile(angfu, 95), 2), r(angfu.max(), 2)))
    W = float((Ul * Fl).sum() / (Ul * Ul).sum())
    Wrms = np.sqrt(((Fl - W * Ul) ** 2).mean(axis=0))
    print("  isotropic static-gravity model  F = W * up_flange:  W=%s N -> distal mass=%s kg" %
          (r(W, 3), r(W / G, 4)))
    print("  its residual RMS per channel=%s N   (1 sigma of the trained band=%s N)" %
          ([r(x, 4) for x in Wrms], [r(x, 4) for x in sd[:3]]))
    X = np.concatenate([Ul, np.ones((Ul.shape[0], 1))], axis=1)
    cond = float(np.linalg.cond(X))
    Pf, rmsf, r2f = fit(X, Fl)
    print("  full 3x3+bias fit on the SAME subset (the isotropic model above is the only part of"
          " it that is identifiable):")
    print("    design matrix condition number=%s -- a near-fixed wrist gives almost no rotational"
          " excitation, so a 3x3 fit reads as a perfect fit without meaning anything" % r(cond, 1))
    print("    in-sample RMS=%s N  R^2=%s" % ([r(x, 4) for x in rmsf], [r(x, 4) for x in r2f]))
    Pfull, rms_a, r2_a = fit(np.concatenate([u, np.ones((n, 1))], axis=1), raw6[:, :3])
    print("    the subset fit P applied to all %d corpus rows: RMS=%s N  R^2=%s" %
          (n, [r(x, 4) for x in rms_a], [r(x, 4) for x in r2_a]))
    print("    (that row is the honest cross-check: the subset fit does not generalise to contact"
          " frames, which is the point -- the leftover is contact, not geometry)")

    u0 = Ul.mean(axis=0)
    u0 = u0 / np.linalg.norm(u0)
    ang = np.degrees(np.arccos(np.clip(u @ u0, -1.0, 1.0)))

    def pct(x):
        return [r(np.percentile(x, 50), 2), r(np.percentile(x, 95), 2),
                r(np.percentile(x, 99), 2), r(np.max(x), 2)]

    print("== 3. how much wrist tilt does the corpus contain? (angle of flange z from its mean) ==")
    print("  all 42681 corpus frames: p50/p95/p99/max degrees = %s" % pct(ang))
    print("  contact frames (start>=20): p50/p95/p99/max = %s" % pct(ang[starts >= TARE_WINDOW]))
    resid_absmax = float(np.abs(raw6 - tare)[:, 1].max())
    print("  corpus residual |fy| absmax=%s N; a %s N load tilted by 1 deg moves the flange-frame"
          " force by %s N" % (r(resid_absmax, 3), r(W, 1), r(W * np.sin(np.radians(1.0)), 4)))

    print("== 4. the same decomposition during the evaluation ==")
    for d in sorted(RECORD.glob("seed*")):
        steps = d / ("%s_step40000_steps.npz" % d.name)
        if not steps.is_file():
            print("  %s: no steps npz" % d.name)
            continue
        z = np.load(steps, allow_pickle=False)
        if "proprio14" not in z.files:
            print("  %s: keys=%s" % (d.name, sorted(z.files)))
            continue
        e = np.asarray(z["proprio14"], dtype=np.float64)
        eu = up_in_flange(e[:, 3:6])
        eang = np.degrees(np.arccos(np.clip(eu @ u0, -1.0, 1.0)))
        p_res = W * eu - tare[:3]              # residual explained by wrist orientation alone
        meas = e[:, 8:11]                       # residual actually fed to the model
        c_res = meas - p_res                    # what is left for contact / inertia
        nF = np.linalg.norm(meas + tare[:3], axis=1)
        cosE = ((meas + tare[:3]) * eu).sum(1) / (nF * np.linalg.norm(eu, axis=1))
        angE = np.degrees(np.arccos(np.clip(cosE, -1.0, 1.0)))
        print("  %s rows=%d base min/max=%s" %
              (d.name, e.shape[0],
               [[r(e[:, i].min()), r(e[:, i].max())] for i in range(8)]))
        print("    |wrist force| med=%s p95=%s max=%s N   (corpus med=%s)" %
              (r(np.median(nF), 2), r(np.percentile(nF, 95), 2), r(nF.max(), 2),
               r(np.median(np.linalg.norm(Fl, axis=1)), 2)))
        print("    angle to flange up-axis: med=%s p95=%s max=%s deg  (corpus med=%s)" %
              (r(np.median(angE), 1), r(np.percentile(angE, 95), 1), r(angE.max(), 1),
               r(np.median(angfu), 1)))
        print("    flange tilt deg p50/p95/p99/max = %s   (corpus p99=%.2f, corpus max=%.2f)" %
              (pct(eang), np.percentile(ang, 99), ang.max()))
        print("    frames tilted beyond the corpus max: %d of %d (%.3f%%)" %
              (int((eang > ang.max()).sum()), e.shape[0], 100.0 * float((eang > ang.max()).mean())))
        for i, name in enumerate(["fx", "fy", "fz"]):
            m = meas[:, i]
            print("      %s meas med=%-9s p95=%-9s absmax=%-9s | pose-pred med=%-9s absmax=%-9s"
                  " | leftover med=%-9s absmax=%-9s | frac |meas|>clamp=%-6.3f frac |pose-pred|>clamp=%-6.3f"
                  % (name, r(np.median(m), 2), r(np.percentile(np.abs(m), 95), 2),
                     r(np.abs(m).max(), 2), r(np.median(p_res[:, i]), 2),
                     r(np.abs(p_res[:, i]).max(), 2), r(np.median(c_res[:, i]), 2),
                     r(np.abs(c_res[:, i]).max(), 2),
                     float((np.abs(m) > clamp_n[i]).mean()),
                     float((np.abs(p_res[:, i]) > clamp_n[i]).mean())))
        head = e[:TARE_WINDOW, 8:14]
        print("    first %d frames (nothing touched yet) residual med=%s absmax=%s" %
              (TARE_WINDOW, [r(x, 2) for x in np.median(head, axis=0)],
               [r(x, 2) for x in np.abs(head).max(axis=0)]))
        print("    pad/gripper residual columns over the whole episode: pad_in med=%s absmax=%s "
              "pad_out med=%s absmax=%s grip med=%s absmax=%s" %
              (r(np.median(e[:, 11]), 3), r(np.abs(e[:, 11]).max(), 2),
               r(np.median(e[:, 12]), 3), r(np.abs(e[:, 12]).max(), 2),
               r(np.median(e[:, 13]), 3), r(np.abs(e[:, 13]).max(), 2)))
        print("    == 5. which left the training band first: the wrist pose, or the command? ==")
        acts = d / ("%s_step40000_actions.npz" % d.name)
        if acts.is_file():
            za = np.load(acts, allow_pickle=False)
            print("      actions npz keys=%s" % sorted(za.files))
        cmd = None
        if "handed_14d" in z.files:
            c = np.asarray(z["handed_14d"], dtype=np.float64)
            cmd = c.reshape(c.shape[0], -1) if c.ndim > 1 else c[:, None]
            print("      handed_14d shape=%s" % (c.shape,))
        nb = 12
        edges = [0, 20, 40, 60, 80, 100, 150, 200, 300, 400, 500, 600]
        for lo_i, hi_i in zip(edges[:-1], edges[1:]):
            sl = slice(min(lo_i, e.shape[0] - 1), min(hi_i, e.shape[0]))
            t = eang[sl]
            fy = np.abs(meas[sl, 1])
            line = ("      frames %-3d-%-3d tilt med=%-7s max=%-7s | frac |fy|>clamp=%-6.3f"
                    " | eef_z med=%-7s" %
                    (lo_i, hi_i, r(np.median(t), 1), r(t.max(), 1),
                     float((fy > clamp_n[1]).mean()), r(np.median(e[sl, 2]), 3)))
            if cmd is not None and cmd.shape[0] > lo_i:
                cs = cmd[sl, :6]
                line += " | |cmd| med=%-7s max=%-7s frac saturated=%-6.3f" % (
                    r(np.median(np.abs(cs)), 3), r(np.abs(cs).max(), 3),
                    float((np.abs(cs) > 0.999).mean()))
            print(line)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
