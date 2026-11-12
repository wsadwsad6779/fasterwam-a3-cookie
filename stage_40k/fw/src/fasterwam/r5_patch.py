"""r5: extend the r4 LoRA run by 4,000 optimiser steps (cumulative 16,000).

`scripts/train_light.py` and `r4_patch` already restore adapter, optimizer,
scheduler, RNG, `global_step` and sampler position from a light-state
directory, so the resume itself is not new here. r5 changes exactly one thing,
and it has to be explicit because it is a deliberate departure from a verbatim
restore:

r4's saved scheduler is fully annealed -- `CosineAnnealingLR(T_max=3800)` with
`last_epoch=3800` and `eta_min=5e-7`, i.e. the run ended at its floor. Stepping
that restored object for another 4,000 iterations does not anneal further; the
cosine is periodic, so the learning rate would instead *climb back* to the
5e-5 peak by step 8,000 and finish training at maximum lr.

`FW_LR_RESTART=1` therefore rebuilds the schedule for the remaining steps with
the shape r4 itself used: linear warmup over 5% of the segment, then cosine from
`cfg.learning_rate` down to 1% of it. The first restarted step is 2.5e-7, which
is continuous with the 5e-7 the checkpoint was left at.

Order of operations, kept auditable:

1. `light_state.load_light_state` restores adapter, optimizer moments, scheduler
   state, RNG streams and the sampler counters unchanged, and the trainer picks
   up `global_step` / `epoch` / `batch_in_epoch`.
2. The restored scheduler state is snapshotted.
3. A new schedule is built for `max_steps - global_step` steps and installed.
4. Both the snapshot and the new schedule are written to `lr_restart.json`.

Nothing else moves: the same 100 demonstrations and 39,481 windows, batch size
6, the same action representation and input structure, no wrapper and no safety
rule, light-state checkpoints every 500 steps, and r4's own directory is only
ever read.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def install():
    """Wrap the r4-patched resume hook with the LR restart. Opt-in by env."""
    from fasterwam import light_state
    from fasterwam.trainer import Wan22Trainer

    if getattr(Wan22Trainer, "_r5_patched", False):
        return False
    if os.environ.get("FW_LR_RESTART", "") != "1":
        logger.info("r5 LR restart not requested (FW_LR_RESTART unset)")
        return False

    orig_resume = Wan22Trainer._resume_or_load_checkpoint

    def _resume_or_load_checkpoint(self):
        orig_resume(self)
        step = int(self.global_step)
        segment = int(self.max_steps) - step
        if segment <= 0:
            logger.warning("FW_LR_RESTART set but max_steps=%d <= resumed step=%d",
                           self.max_steps, step)
            return

        peak = float(self.learning_rate)
        groups = self.optimizer.param_groups
        for group in groups:
            # A scheduler takes its base from `initial_lr`; the restart is keyed
            # to cfg.learning_rate, so pin the two together before building.
            group["initial_lr"] = peak
        restored_lr = float(groups[0]["lr"])
        restored_state = getattr(self.scheduler, "state_dict", lambda: {})()

        warmup = max(int(segment * 0.05), 1)
        new_scheduler = self._build_scheduler(
            str(self.cfg.lr_scheduler_type), segment, warmup
        )
        if getattr(self.scheduler, "scheduler", None) is None:
            self.scheduler = new_scheduler
        else:
            # `accelerator.prepare` wrapped it; keep that object and swap inside.
            self.scheduler.scheduler = new_scheduler
        lr_after = float(groups[0]["lr"])

        record = {
            "resume_dir": str(self.resume),
            "resumed_global_step": step,
            "max_steps": int(self.max_steps),
            "segment_steps": segment,
            "epoch_at_resume": int(self.epoch),
            "batch_in_epoch_at_resume": int(self.batch_in_epoch),
            "dataset_len": (len(self.train_dataset)
                            if hasattr(self.train_dataset, "__len__") else None),
            "sampler": light_state._sampler_state(self.train_sampler),
            "lr_restored_from_checkpoint": restored_lr,
            "lr_first_step_after_restart": lr_after,
            "restart_peak_lr": peak,
            "restart_warmup_steps": warmup,
            "restart_cosine_t_max": max(segment - warmup, 1),
            "restart_eta_min": peak * 0.01,
            "scheduler_type": str(self.cfg.lr_scheduler_type),
            "restored_scheduler_state": _plain(restored_state),
        }
        out = Path(self.output_dir) / "lr_restart.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2))
        logger.info(
            "LR restart: step=%d->%d warmup=%d peak=%.3e eta_min=%.3e "
            "(restored lr=%.3e, first restarted lr=%.3e) -> %s",
            step, self.max_steps, warmup, peak, peak * 0.01, restored_lr,
            lr_after, out,
        )

    Wan22Trainer._resume_or_load_checkpoint = _resume_or_load_checkpoint
    Wan22Trainer._r5_patched = True
    logger.info("r5 patch installed (FW_LR_RESTART=1)")
    return True


def _plain(obj):
    """JSON-safe copy of a scheduler state dict (it carries nested dicts/lists)."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)
