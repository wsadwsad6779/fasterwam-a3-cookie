"""Resume the five-Cookie batch expert from a mid-episode grasp state.

The parent expert always starts at SELECT_COOKIE.  A policy takeover can leave
the pads already clamped around a row of Cookies, where re-approaching the
source column only jams the pads against them.  This subclass rebuilds the
grasp bookkeeping from the live simulator state and hands control to the
parent's CLOSE/LIFT -> MOVE_TO_SLOT -> OPEN -> VERIFY_RELEASE machinery.

Held Cookies and the resume phase come from A3_RESUME_HELD / A3_RESUME_PHASE
so that the unchanged dagger_recover entry point can be reused.
"""
from __future__ import annotations

import os

import numpy as np

from a3_dual_arm_sim.controllers.expert import CookiePhase
from a3_dual_arm_sim.controllers.same_column_batch_expert import (
    A3SameColumnBatchExpert,
)


class A3ResumeHeldExpert(A3SameColumnBatchExpert):
    def reset(self, context=None):
        super().reset(context)
        spec = os.environ.get("A3_RESUME_HELD", "").strip()
        if not spec:
            return
        held = [int(v) for v in spec.replace(",", " ").split()]
        phase = os.environ.get("A3_RESUME_PHASE", "close").strip() or "close"
        info = self.begin_resume(held, phase)
        print(f"[resume] {info}", flush=True)

    def begin_resume(self, held_indices, start_phase="close"):
        held = [int(i) for i in held_indices]
        if not held:
            raise RuntimeError("resume needs at least one held Cookie")
        if start_phase not in ("close", "lift"):
            raise ValueError(f"unknown resume phase {start_phase!r}")
        in_target = [
            i
            for i in range(self.env.task_config.cookie_count)
            if self.env.privileged_cookie_in_target(i)
        ]
        self.completed_cookie_indices = list(in_target)
        self.batch_index = 1 if len(in_target) >= self.batch_size else 0
        self.batch_indices = held
        self.current_cookie_index = held[0]
        self.target_slot_index = self.batch_index
        self._grasp_rotation = self.data.site_xmat[self._l_site].reshape(3, 3).copy()
        self._opening = float(np.clip(self.env.current_joint_action[7], 0.0, 1.0))
        self._pick_eef = self.data.site_xpos[self._l_site].copy()
        self._high_eef = self._pick_eef + np.array([0.0, 0.0, 0.082])
        self._grip_reference = self._positions().copy()
        self._lift_start_eef = self.data.site_xpos[self._l_site].copy()
        self._held_offsets = None
        self._aligned_gap_center = None
        self._aligned_grasp = False
        self._rim_contact_stop = False
        self._stable = 0
        self._jam_count = 0
        self.failure_phase = None
        self.failure_reason = None
        self._advance(CookiePhase.CLOSE if start_phase == "close" else CookiePhase.LIFT)
        return {
            "held": held,
            "start_phase": start_phase,
            "opening": round(self._opening, 4),
            "already_in_target": list(in_target),
            "batch_index": int(self.batch_index),
        }
