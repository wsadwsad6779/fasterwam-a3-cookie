"""Runtime patches for r4: unvisited-window sampling + light-state checkpoints.

Installed by `scripts/train_light.py` *before* the trainer is constructed, so no
existing FasterWAM source file is rewritten. Two behaviours are changed:

1. `ResumableEpochSampler` gains `sample_offset`: on the first pass it skips
   that many windows in addition to any resume offset. r4 uses
   `FW_SAMPLE_OFFSET=18000` with the same generator seed as r2/r3, so its first
   batch is window 18,000 of the identical permutation instead of replaying
   0..17,999 for a third time. Passes are keyed on the trainer's epoch counter
   (not the sampler's stale `epoch`), so once the 21,481 remaining windows are
   consumed a run continues on a *new* permutation rather than rebuilding the
   same one; a resume mid-pass skips only the batches already consumed.

2. `Wan22Trainer.save_checkpoint` writes adapter + optimizer + scheduler + RNG +
   counters (see `light_state.py`) instead of an 11 GB merged model, and a
   second call at the same global step is a no-op (the stock loop saves twice
   when `save_every` divides `max_steps`). `resume` accepts a light-state
   directory; the frozen base is reloaded from `FW_BASE_CKPT` first, then the
   adapter on top.

Full training-state resume (`accelerator.save_state`) is never written by this
patch, so nothing here can masquerade as a bit-exact continuation of r3: r3 has
no optimizer state, and r4 is a warm restart by construction.

3. LoRA's random `lora_A` draw is made reproducible. Check 3 found two fresh
   processes with `seed=42` producing different adapters, because
   `apply_lora` initialises `lora_A` from the *global* torch RNG whose position
   at that moment is not pinned; the resumed run restored the saved adapter
   exactly, so the defect is in the starting draw, not the resume. With
   `FW_LORA_INIT_SEED` set, each wrapped layer re-initialises from its own
   generator keyed by a fixed sequence number.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def _passes_done(sampler):
    """Completed passes over the dataset, as counted by the trainer.

    The trainer's `epoch` increments exactly when its own iterator raises
    StopIteration, which is immune to DataLoader prefetch (prefetch can drain
    the sampler while the trainer is still mid-pass). Without a trainer -- the
    standalone sampler test -- the sampler's own `epoch` field is the only
    source.
    """
    trainer = getattr(sampler, "_trainer", None)
    return int(getattr(trainer, "epoch", sampler.epoch))


def _resume_skip(sampler):
    return sampler.resume_batch_offset * sampler.batch_size * sampler.num_processes


def _skip_for_pass(sampler, passes_done):
    """Samples to skip at the head of the pass `passes_done` builds.

    The stock sampler only honours `resume_batch_offset` on epoch 0 and the
    trainer never propagates its own epoch counter into the sampler, so a run
    that exhausts its index list rebuilds the identical permutation (and, after
    this patch, re-applies `sample_offset`, which can empty it). The pass
    counter replaces that: pass 0 starts at `sample_offset`, later passes use a
    new seed and only skip the batches already consumed before a resume.
    """
    skip = int(_resume_skip(sampler))
    if passes_done == 0:
        skip = max(skip, int(getattr(sampler, "sample_offset", 0) or 0))
    return skip


def lora_init_seed():
    """The recorded `lora_A` seed, or None when the patch is not active."""
    from fasterwam import lora as lora_module

    return getattr(lora_module.LoRALinear, "_r4_seed", None)


def install_seeded_lora_init():
    """Give every LoRA layer a draw that does not depend on the global RNG.

    `runtime.run_training` wraps the model with `apply_lora` before constructing
    the trainer, and the trainer is what calls `set_global_seed(cfg.seed)`, so up
    to r4 the LoRA start point was never covered by the run seed. Opt-in through
    `FW_LORA_INIT_SEED`.
    """
    import math

    import torch
    import torch.nn as nn

    from fasterwam import lora as lora_module

    raw = os.environ.get("FW_LORA_INIT_SEED", "")
    if not raw:
        return False
    seed = int(raw)
    base_cls = lora_module.LoRALinear
    if getattr(base_cls, "_r4_seed", None) is not None:
        return False

    class SeededLoRALinear(base_cls):
        _seq = 0

        def __init__(self, base, r, alpha, dropout=0.0):
            super().__init__(base, r, alpha, dropout)
            idx = SeededLoRALinear._seq
            SeededLoRALinear._seq += 1
            gen = torch.Generator(device=self.lora_A.device).manual_seed(seed + idx)
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5), generator=gen)

    SeededLoRALinear._r4_seed = seed
    lora_module.LoRALinear = SeededLoRALinear
    logger.info("LoRA init pinned: FW_LORA_INIT_SEED=%d", seed)
    return True


def install(sample_offset=None):
    """Patch sampler + trainer in place. Safe to call once per process."""
    from fasterwam import light_state

    from fasterwam.trainer import Wan22Trainer
    from fasterwam.utils.samplers import ResumableEpochSampler

    if getattr(ResumableEpochSampler, "_r4_patched", False):
        return

    install_seeded_lora_init()

    def __iter__(self):
        import torch

        passes_done = _passes_done(self)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch_offset + passes_done)
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()
        return iter(indices[_skip_for_pass(self, passes_done):])

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
        self.train_sampler._trainer = self
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
        # The sampler only learns its own position from `resume_batch_offset`,
        # which the stock trainer sets while *resuming*, never while saving, so
        # the field is 0 at save time and a light state would restart the pass
        # from its head. Publish the trainer's consumed-batch counter for the
        # duration of the capture and put the sampler back afterwards.
        sampler = self.train_sampler
        prev_offset = int(getattr(sampler, "resume_batch_offset", 0))
        sampler.set_resume_batch_offset(int(self.batch_in_epoch))
        try:
            manifest = light_state.save_light_state(
                out_dir,
                model=model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                sampler=sampler,
                global_step=step,
                epoch=self.epoch,
                batch_in_epoch=self.batch_in_epoch,
                extra={
                    "base_checkpoint": base,
                    "instruction": os.environ.get("FASTERWAM_A3_INSTRUCTION", ""),
                    "lr": float(self.optimizer.param_groups[0].get("lr", 0.0)),
                    "lora_init_seed": lora_init_seed(),
                },
                adapter_meta={"base_checkpoint": base},
            )
        finally:
            if prev_offset:
                sampler.set_resume_batch_offset(prev_offset)
            else:
                sampler.clear_resume_batch_offset()
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
