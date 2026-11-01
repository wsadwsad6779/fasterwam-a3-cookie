cd /workspace/_r5_stage
cat > _clamp_method.txt <<'CEOF'
def clamp_stats(a, b, phase, pad, xyz, eef, z0, opening, carried_rest_z):
    idx = range(a, b)
    win = lambda names: [i for i in idx if phase[i] in names]
    ap, cl, cy = win(APPROACH_PHASES), win(CLOSE_PHASES), win(CARRY_PHASES)
    rel = win(("OPEN", "VERIFY_RELEASE", "RETRACT"))
    rise = xyz[:, :, 2] - z0[:, 2]
    peak_rise = rise[a:b].max(axis=0)
    travel = np.linalg.norm(xyz[a:b, :, :2] - z0[:, :2], axis=2).max(axis=0)
    top = int(np.argmax(peak_rise))
    stats = {"steps": b - a, "phases": sorted({phase[i] for i in idx})}
    touched = [i for i in idx if pad[i] > CONTACT_N]
    stats["first_contact_pad_n"] = round(float(pad[touched[0]]), 3) if touched else None
    stats["closure_peak_pad_n"] = round(float(max((pad[i] for i in cl), default=0.0)), 3)
    stats["hold_mean_pad_n"] = round(float(np.mean([pad[i] for i in cy])), 3) if cy else None
    stats["carry_peak_pad_n"] = round(float(max((pad[i] for i in cy), default=0.0)), 3)
    if rel:
        stats["pre_release_pad_n"] = round(float(max(pad[max(rel[0] - 10, 0):rel[0] + 1])), 3)
        stats["post_release_pad_n"] = round(float(np.mean(pad[rel[0]:min(rel[-1] + 11, b)])), 3)
    n_at_grip = [int(np.sum((np.linalg.norm(xyz[i, :, :2] - eef[i, :2], axis=1) < HOLD_XY_M)
                            & (rise[i] > LIFT_Z_M))) for i in cy]
    stats["max_cookies_at_grip"] = int(max(n_at_grip, default=0))
    stats["cookies_raised_20mm"] = int((peak_rise > 0.020).sum())
    stats["min_opening"] = round(float(min(opening[a:b])), 4)
    stats["max_opening"] = round(float(max(opening[a:b])), 4)
    stats["carried_index"] = top
    stats["carried_lift_mm"] = round(float(peak_rise[top]) * 1000, 1)
    stats["carried_distance_mm"] = round(float(travel[top]) * 1000, 1)
    end = xyz[b - 1, top]
    stats["placed_in_bin_xy"] = bool(abs(end[0] - BIN_XY[0]) < BIN_INNER_HALF[0]
                                     and abs(end[1] - BIN_XY[1]) < BIN_INNER_HALF[1])
    stats["placed_upright"] = bool(abs(end[2] - z0[top, 2]) < 0.004)
    stats["accept"] = bool(stats["max_cookies_at_grip"] == 1
                           and stats["carried_lift_mm"] >= 60.0
                           and (stats["carry_peak_pad_n"] or 0.0) <= 2.0
                           and stats["placed_in_bin_xy"]
                           and stats["placed_upright"])
    return stats

CEOF
python3 - <<'PYEOF'
from pathlib import Path
src = Path('replay_forces.py').read_text(encoding='utf-8')
new = Path('_clamp_method.txt').read_text(encoding='utf-8')
start = src.index('def clamp_stats')
stop = src.index('def run_episode')
src = src[:start] + new.rstrip() + chr(10) + chr(10) + chr(10) + src[stop:]
Path('replay_forces.py').write_text(src, encoding='utf-8')
print('spliced')
PYEOF
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
python3 -m py_compile replay_forces.py && echo COMPILE_OK
sha256sum replay_forces.py
tail -c 700 /workspace/gate.out
