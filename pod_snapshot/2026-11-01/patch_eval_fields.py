p = "/workspace/r2_metrics.py"
s = open(p).read()

geom = '''
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
'''

assert s.count("def one(f):\n") == 1
s = s.replace("def one(f):\n", geom, 1)

anchor = "    o.update(cookie_metrics(R, rp, op))\n"
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + '    o.update(box_metrics(R, o.get("held_cookie")))\n')

anchor2 = '            print("    split", json.dumps(breakdown(R), ensure_ascii=False))\n'
answer = '''            keys = ("real_grasp", "held_cookie", "held_peak_lift_mm", "held_final_lift_mm",
                    "held_drop_mm", "dropped_after_lift", "held_slip_mm", "follow_rise_run",
                    "box_min_xy_dist_mm", "held_box_min_xy_dist_mm",
                    "held_ever_in_target_region", "held_target_region_tail_steps",
                    "cookies", "wrapper_pct", "safety_pct", "both_pct")
            print("    answer", json.dumps({k: o.get(k, "N/A") for k in keys}, ensure_ascii=False))
'''
assert s.count(anchor2) == 1
s = s.replace(anchor2, anchor2 + answer)

open(p, "w").write(s)
print("METRICS_PATCH_OK")
