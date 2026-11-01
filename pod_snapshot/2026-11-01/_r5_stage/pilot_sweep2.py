"""Round two: the transport check was tripped by the place target, not the pinch.

All four force/lift variants released the first cookie and then failed the second
clamp at the same place. The base expert aims every release at the mean row of a
target column, so a single-cookie run drops cookie 2 onto cookie 1 and the 16 mm
tool-frame offset check reads that contact as a slip. Give each clamp its own row.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/_r5_stage")
import json

import numpy as np

import single_lowforce_probe as P
from a3_dual_arm_sim.controllers.same_column_batch_expert import A3SameColumnBatchExpert

BASE = P.SingleLowForce


def row_aware_place(lift=None, target=0.45, stable=0.35):
    def reset(self, context=None):
        BASE.reset(self, context)
        if lift is not None:
            self._max_lift_speed = lift

    def _select_batch(self):
        BASE._select_batch(self)
        self.target_grip_force = target
        self.stable_grip_force = stable

    def _place_pose(self, clearance=0.0, column_index=None):
        if column_index is None:
            order = getattr(self.env, "demonstration_column_order", (0, 1))
            column_index = order[0] if self.batch_index < 5 else order[1]
        position, rotation = A3SameColumnBatchExpert._place_pose(
            self, clearance, column_index
        )
        slots = np.asarray(self.env.TARGET_SLOTS_LOCAL)
        column_x = np.unique(slots[:, 0])[column_index]
        rows = np.sort(slots[np.isclose(slots[:, 0], column_x), 1])
        offset = rows[self.batch_index % len(rows)] - rows.mean()
        box_r = self.data.xmat[self._tb_id].reshape(3, 3)
        return position + box_r @ np.array([0.0, offset, 0.0]), rotation

    return type("RowAware", (BASE,), {"reset": reset, "_select_batch": _select_batch,
                                     "_place_pose": _place_pose})


GRID = [
    ("E_rows_lift0012_f045", 0.0012, 0.45, 0.35),
    ("F_rows_nocap_f045", None, 0.45, 0.35),
    ("G_rows_lift0022_f070", 0.0022, 0.70, 0.55),
]

rows = []
for name, lift, target, stable in GRID:
    P.SingleLowForce = row_aware_place(lift, target, stable)
    out = Path("/workspace/a3_eval/pilot_single_" + name)
    if out.exists() and any(out.glob("*")):
        print("skip, non-empty:", out, flush=True)
        continue
    out.mkdir(parents=True, exist_ok=True)
    row = P.run_seed(6001, 3200, 0.0, out)
    row["variant"] = name
    rows.append(row)
    print(json.dumps({k: row[k] for k in ("variant", "steps", "stop", "in_target_final",
                                          "pad_peak_this_run", "success")}, default=float),
          flush=True)
    print("   reports:", [(b.get("cookies"), b.get("lifted"), b.get("released"))
                          for b in row["batch_reports"]], flush=True)
    (out / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
