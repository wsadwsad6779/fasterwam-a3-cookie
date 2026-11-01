"""dagger_recover with the resume-held expert swapped in.

Held Cookie indices come from A3_RESUME_HELD (space or comma separated) and the
resume phase from A3_RESUME_PHASE.  Everything else is dagger_recover.
"""
from __future__ import annotations

from a3_dual_arm_sim.controllers import same_column_batch_expert as _scbe

import dagger_recover
from resume_held_expert import A3ResumeHeldExpert

_scbe.A3SameColumnBatchExpert = A3ResumeHeldExpert

if __name__ == "__main__":
    dagger_recover.main()
