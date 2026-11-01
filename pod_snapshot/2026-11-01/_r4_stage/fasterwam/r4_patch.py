"""Runtime patches for r4: unvisited-window sampling + light-state checkpoints.

Installed by `scripts/train_light.py` *before* the trainer is constructed, so no
existing FasterWAM source file is rewritten. Two behaviours are changed:

1. `ResumableEpochSampler` gains `sample_offset`: on epoch 0 it skips that many
   windows in addition to any resume offset. r4 uses `FW_SAMPLE_OFFSET=18000`
   with the same generator seed as r2/r3, so its first batch is window 18,000 of
   the identical permutation instead of replaying 0..17,999 for a third time.
   A resumed epoch-0 run (same-process crash) uses the larger of the two skips.

2. `Wan22Trainer.save_checkpoint` writes adapter + optimizer + scheduler + RNG +
   counters (see `light_state.py`) instead of an 11 GB merged model, and a
   second call at the same global step is a no-op (the stock loop saves twice
   when `save_every` divides `max_steps`). `resume` accepts a light-state
   directory; the frozen base is reloaded from `FW_BASE_CKPT` first, then the
   adapter on top.

Full training-state resume (`accelerator.save_state`) is never written by this
patch, so nothing here can masquerade as a bit-exact continuation of r3: r3 has
no optimizer state, and r4 is a warm restart by construction.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _skip_for_epoch(sampler):
    if sampler.epoch != 0:
        return 0
    resume = sampler.resume_batch_offset * sampler.batch_size * sampler.num_processes
    return max(int(resume), int(getattr(sampler, "sample_offset", 0) or 0))


def install(sample_offset=None):
    """Patch sampler + trainer in place. Safe to call once per process."""
    from fasterwam import light_state

    from fasterwam.trainer import Wan22Trainer
    from fasterwam.utils.samplers import ResumableEpochSampler

    if getattr(ResumableEpochSampler, "_r4_patched", False):
        return

    def __iter__(self):
        import torch

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch + self.epoch_offset)
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        return iter(indices[_skip_for_epoch(self):])

    def set_sample_offset(self, count):
        self.sample_offset = int(count)

    ResumableEpochSampler.__iter__ = __iter__
    ResumableEpochSampler.set_sample_offset = set_sample_offset
    ResumableEpochSampler.sample_offset = 0
    ResumableEpochSampler._r4_patched = True

    orig_build_loader = Wan22Trainer._build_loader

    def _build_loader(self, dataset, worker_init_fn=None):
        loader = orig_build_loader(self, dataset, worker_init_fn=worker_init_fn)
        # A DataLoader iterator draws its worker base_seed from the *global* RNG
        # when no generator is set. That one extra draw would make a resumed run
        # take a different diffusion-noise stream than an uninterrupted one, so
        # pin it to its own generator and leave the global stream to the model.
        import torch

        loader.generator = torch.Generator(device="cpu").manual_seed(int(self.seed))
        offset = sample_offset if sample_offset is not None else os.environ.get("FW_SAMPLE_OFFSET")
        if offset:
            self.train_sampler.set_sample_offset(int(offset))
        return loader

    Wan22Trainer._build_loader = _build_loader

    def save_checkpoint(self):
        step = int(self.global_step)
        if getattr(self, "_light_last_step", None) == step:
            return self._light_last_info
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return {"weights_path": None, "state_path": None}

        base = os.environ.get("FW_BASE_CKPT", "")
        model = self.accelerator.unwrap_model(self.model)
        out_dir = Path(self.output_dir) / "checkpoints" / "light" / f"step_{step:06d}"
        manifest = light_state.save_light_state(
            out_dir,
            model=model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            sampler=self.train_sampler,
            global_step=step,
            epoch=self.epoch,
            batch_in_epoch=self.batch_in_epoch,
            extra={
                "base_checkpoint": base,
                "instruction": os.environ.get("FASTERWAM_A3_INSTRUCTION", ""),
                "lr": float(self.optimizer.param_groups[0].get("lr", 0.0)),
            },
            adapter_meta={"base_checkpoint": base},
        )
        logger.info("[light] step=%d adapter=%s rng/counters=%s",
                    step, out_dir / "adapter.pt", out_dir / "trainer_state.json")
        self._light_last_step = step
        self._light_last_info = {"weights_path": str(out_dir / "adapter.pt"),
                                 "state_path": str(out_dir)}
        self.accelerator.wait_for_everyone()
        return self._light_last_info

    Wan22Trainer.save_checkpoint = save_checkpoint

    orig_resume = Wan22Trainer._resume_or_load_checkpoint

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        path = Path(str(resume))
        if path.is_dir() and (path / "trainer_state.json").exists():
            base = os.environ.get("FW_BASE_CKPT")
            if not base:
                raise RuntimeError("resuming a light-state directory requires FW_BASE_CKPT")
            model = self.accelerator.unwrap_model(self.model)
            logger.info("Loading frozen base first: %s", base)
            model.load_checkpoint(str(base), optimizer=None)
            state = light_state.load_light_state(
                path, model=model, optimizer=self.optimizer,
                scheduler=self.scheduler, sampler=self.train_sampler)
            self.global_step = int(state["global_step"])
            self.epoch = int(state["epoch"])
            self.batch_in_epoch = int(state["batch_in_epoch"])
            logger.info("Resumed light state %s at step=%d epoch=%d batch_in_epoch=%d",
                        path, self.global_step, self.epoch, self.batch_in_epoch)
            self.accelerator.wait_for_everyone()
            return
        return orig_resume(self)

    Wan22Trainer._resume_or_load_checkpoint = _resume_or_load_checkpoint
    logger.info("r4 patch installed (sample_offset=%s)", sample_offset)
    return True
