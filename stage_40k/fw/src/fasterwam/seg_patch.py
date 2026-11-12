"""Segmented training: stop every N steps without breaking the single cosine.

The 40k run is delivered in eight 5k segments so each boundary can be uploaded
and evaluated. Two things must survive the segmentation:

1. The LR schedule is ONE cosine over 40,000 steps. ``Wan22Trainer.__init__``
   builds it from ``cfg.max_steps`` before ``train()`` is ever called, so every
   segment is launched with ``max_steps=40000`` and only the *loop bound* is
   lowered to the segment end. Lowering ``cfg.max_steps`` instead would rebuild a
   short cosine that anneals to its floor eight times -- the r5 defect.

2. The saved state must contain every tensor the optimizer stepped. The v1 light
   state selected on the name pattern ``lora_`` and dropped ``proprio_encoder``;
   v3 selects by optimizer membership. This module swaps the v3 pair into
   ``fasterwam.light_state`` so the r4 runtime patch calls it unchanged.

Nothing here edits an upstream file. ``FW_LR_RESTART`` is refused outright: it
belongs to the periodic-restart experiment and would climb the LR back up.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def install_v3_light_state():
    from fasterwam import light_state
    from fasterwam import light_state_v3

    if getattr(light_state, "_v3_installed", False):
        return False
    light_state.save_light_state = light_state_v3.save_light_state
    light_state.load_light_state = light_state_v3.load_light_state
    light_state._v3_installed = True
    logger.info("light_state v3 installed (optimizer-membership selection)")
    return True


def _ledger_path():
    return os.environ.get("FW_SEG_LEDGER", "")


def _ledger_append(payload):
    path = _ledger_path()
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _previous_segment_end():
    """Last `segment_end` record, or None.

    A resumed segment must start at exactly the LR its predecessor ended at; that
    is the only ledger-internal way to see a rebuilt schedule. Comparing the
    segment's own start and end LR cannot do it -- during the 2000-step warmup the
    LR legitimately climbs within every segment.
    """
    path = _ledger_path()
    if not path:
        return None
    last = None
    try:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("event") == "segment_end":
                    last = event
    except OSError:
        return None
    return last


def install_stop_step():
    """Cap the training loop at FW_STOP_STEP, leaving the scheduler alone."""
    raw = os.environ.get("FW_STOP_STEP", "")
    if not raw:
        return False
    stop = int(raw)
    if stop <= 0:
        raise RuntimeError("FW_STOP_STEP must be a positive integer, got %r" % raw)
    if os.environ.get("FW_LR_RESTART", ""):
        raise RuntimeError("FW_LR_RESTART is set; it belongs to the periodic-restart "
                           "experiment and would break the single 40k cosine.")

    from fasterwam.trainer import Wan22Trainer

    if getattr(Wan22Trainer.train, "_seg_wrapped", False):
        return False
    orig_train = Wan22Trainer.train

    def train(self):
        total = int(self.max_steps)
        cfg_total = self.cfg.get("max_steps")
        if cfg_total is None or int(cfg_total) != total:
            raise RuntimeError(
                "segmented training requires an explicit cfg.max_steps equal to the "
                "scheduler horizon; got cfg.max_steps=%r trainer.max_steps=%d"
                % (cfg_total, total))
        if stop > total:
            raise RuntimeError("FW_STOP_STEP=%d exceeds the scheduler horizon %d" % (stop, total))
        start = int(self.global_step)
        if start >= stop:
            raise RuntimeError(
                "resumed at global_step=%d which is already at/past FW_STOP_STEP=%d; "
                "refusing to write a duplicate boundary checkpoint." % (start, stop))
        lr_start = float(self.optimizer.param_groups[0]["lr"])
        prev = _previous_segment_end()
        if prev is not None and prev.get("global_step") == start:
            prev_lr = float(prev["lr"])
            if lr_start < prev_lr * 0.999:
                logger.error("LR RESET at resume: segment %d ended at %.6e, this one "
                             "starts at %.6e -- the scheduler was rebuilt.",
                             start, prev_lr, lr_start)
        logger.info("segment: steps %d -> %d of a single %d-step cosine (lr=%.6e)",
                    start, stop, total, lr_start)
        _ledger_append({"event": "segment_start", "ts": time.time(), "global_step": start,
                        "stop_step": stop, "scheduler_total": total, "lr": lr_start,
                        "epoch": int(self.epoch), "batch_in_epoch": int(self.batch_in_epoch),
                        "prev_segment_end_step": None if prev is None else prev.get("global_step"),
                        "prev_segment_end_lr": None if prev is None else float(prev["lr"])})
        self.max_steps = stop
        try:
            return orig_train(self)
        finally:
            self.max_steps = total
            end = int(self.global_step)
            lr_end = float(self.optimizer.param_groups[0]["lr"])
            if end != stop:
                logger.error("segment ended at global_step=%d, expected %d", end, stop)
            _ledger_append({"event": "segment_end", "ts": time.time(), "global_step": end,
                            "stop_step": stop, "scheduler_total": total, "lr": lr_end,
                            "epoch": int(self.epoch), "batch_in_epoch": int(self.batch_in_epoch),
                            "hit_stop": end == stop, "lr_moved_up": lr_end > lr_start})
            logger.info("segment done: global_step=%d lr=%.6e", end, lr_end)

    train._seg_wrapped = True
    Wan22Trainer.train = train
    logger.info("FW_STOP_STEP=%d installed", stop)
    return True


def install():
    if os.environ.get("FW_LR_RESTART", ""):
        raise RuntimeError("FW_LR_RESTART must not be set for the segmented 40k run.")
    install_v3_light_state()
    install_stop_step()
    return True