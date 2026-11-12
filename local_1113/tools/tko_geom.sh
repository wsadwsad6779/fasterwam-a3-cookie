#!/usr/bin/env bash
set -u
S=/workspace/_r4_stage
{
  date -u +%FT%TZ
  echo "== where the metric is defined"
  grep -rn "cookies_in_target" /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/sim/ | head -20
  echo "== function bodies mentioning in_target"
  grep -rn -A6 "def .*in_target\|def .*target.*mask" /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/sim/env.py | head -60
  echo "== final cookie positions dump (seed22 video run)"
  python3 -u - <<'PY'
import numpy as np
m = np.load('/workspace/a3_eval/r4_c0_seed22_video/lora_C0_22.npz')
xyz = m['cookie_xyz'].astype(float)
x0 = m['cookie_xyz0'].astype(float)
BX, BY, HX, HY = 0.095, 0.100, 0.063, 0.035
last = xyz[-1]
dx = np.abs(last[:, 0] - BX)
dy = np.abs(last[:, 1] - BY)
o = np.argsort(dx + dy)[:12]
print("idx  final_xyz            |dx|    |dy|   moved_xy_mm  dz_mm   start_xy")
for i in o:
    print(int(i), np.round(last[i], 4).tolist(), round(float(dx[i]), 4), round(float(dy[i]), 4),
          round(1000 * float(np.linalg.norm(last[i, :2] - x0[i, :2])), 1),
          round(1000 * float(last[i, 2] - x0[i, 2]), 1), np.round(x0[i, :2], 4).tolist())
print("strict_footprint_count", int(((dx < HX) & (dy < HY)).sum()),
      "swapped_count", int(((dx < HY) & (dy < HX)).sum()))
print("cookies whose z differs from start by >2mm", int((np.abs(last[:, 2] - x0[:, 2]) > 0.002).sum()))
print("unique final z (rounded mm)", sorted(set(np.round(last[:, 2] * 1000, 1).tolist()))[:12])
print("max xy displacement per group",
      round(1000 * float(np.abs(xyz[:, :, :2] - x0[None, :, :2]).max(2).max(0).max()), 1))
moved = np.abs(xyz[:, :, :2] - x0[None, :, :2]).max(2).max(0)
print("cookies moved >20mm", int((moved > 0.02).sum()), "indices", np.where(moved > 0.02)[0].tolist())
print("cookies moved >5mm", int((moved > 0.005).sum()))
PY
  date -u +%FT%TZ
} > $S/tko_geom.log 2>&1
echo __GEOM_DONE__ >> $S/tko_geom.log
