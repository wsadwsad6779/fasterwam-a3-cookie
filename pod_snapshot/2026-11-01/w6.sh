cat > /workspace/_r5_stage/pilot_sweep.py <<'PY'
"""Sweep the two knobs that decide whether a single-cookie clamp survives transport.

The five-cookie experts never cap the lift speed on this path and hold the bundle
by wedge force, so a 6 mm cookie pinched at 0.45 N slips out before the place
phase and the 16 mm transport check calls it a failure. The 2x10 expert already
carries the settings for a one-cookie stack; this sweep finds the smallest change
that lets the single-cookie run reach DONE in the 2x5 training scene.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/_r5_stage")
import json

import single_lowforce_probe as P

BASE = P.SingleLowForce


def variant(name, lift, target, stable):
    def reset(self, context=None):
        BASE.reset(self, context)
        if lift is not None:
            self._max_lift_speed = lift

    def _select_batch(self):
        BASE._select_batch(self)
        self.target_grip_force = target
        self.stable_grip_force = stable

    return type(name, (BASE,), {"reset": reset, "_select_batch": _select_batch})


GRID = [
    ("A_lift0012_f045", 0.0012, 0.45, 0.35),
    ("B_lift0022_f045", 0.0022, 0.45, 0.35),
    ("C_lift0012_f070", 0.0012, 0.70, 0.55),
    ("D_nocap_f070", None, 0.70, 0.55),
]

rows = []
for name, lift, target, stable in GRID:
    cls = variant(name, lift, target, stable)
    P.SingleLowForce = cls
    out = Path("/workspace/a3_eval/pilot_sweep_" + name)
    if out.exists() and any(out.glob("*")):
        print("skip, non-empty:", out, flush=True)
        continue
    out.mkdir(parents=True, exist_ok=True)
    row = P.run_seed(6001, 3200, 0.0, out)
    row["variant"] = name
    row["lift"] = lift
    rows.append(row)
    print(json.dumps(row, default=float), flush=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

print(json.dumps([[r["variant"], r["steps"], r["stop"], r["in_target_final"],
                  round(r["pad_peak_this_run"], 3), len(r["batch_reports"])] for r in rows],
                 indent=1), flush=True)
PY
python3 -m py_compile /workspace/_r5_stage/pilot_sweep.py && echo COMPILE_OK
cd /workspace && export PYTHONPATH=/workspace/a3_dual_arm_sim/src
nohup timeout 1800 python3 /workspace/_r5_stage/pilot_sweep.py > /workspace/sweep.out 2>&1 &
sleep 5; tail -3 /workspace/sweep.out
echo DONE