"""Tier-2 admission tests: real corpus, real model geometry, real resume path.

Run on the instance (it reads /workspace/a3_data/...). A GPU is not required:
every check is about wiring, units and state, and each one runs at the *frozen*
dimensions (action expert hidden 1024 / 30 layers / horizon 32 / hist_len 16 /
6 force channels), not at a shrunk proxy. Shrunk proxies are Tier-1's job
(smoke_force_v1.py).

  R1  corpus + geometry .... the 14-dim V1 corpus and the 18-column force table
      describe the same frames; the dataset gathers exactly the channels the
      model config names, and windows obey the padding convention without ever
      borrowing force across an episode boundary (§六).
  R2  leak gate ............. `FastWAM.assert_future_frames_are_actually_future`
      over real windows with the dataset's independent retriever, plus a
      negative control (+1 shift must be rejected).
  R2b proprio units ......... `proprio[:, 8:]` must be raw-minus-GLOBAL-zero. If
      it came from a per-episode median of the first 20 frames it would not
      exist at online step 0 (§四).
  R3  zero-init equivalence . bitwise, at real dims, through the real pre_dit,
      the real 30-block forward and the real action head -- plus the proof that
      the path is live rather than dead.
  R4  gradient ladder ....... under the real joint loss step 0 moves only the
      two zero-init output layers and every other force tensor is exactly zero;
      three steps later none is frozen. Force loss alone is the negative
      control -- it cannot open the injection gate.
  R5  joint flow target ..... the force target is the rectified-flow velocity,
      invertible back to the clean window, not the clean value itself; at init
      the prediction is exactly zero.
  R6  calibration ........... the real calib JSON loads only with version +
      matching channels, lands in the buffers, and a mismatch is fatal.
  R7  train/infer units ..... `_prepare_force_inputs` (training) and
      `_build_force_infer_state` (roll-out) produce the SAME normalized history
      bitwise from the same newtons (§四.6); normalize is invertible.
  R8  light state v3 ........ save -> wipe -> restore returns every force tensor
      and its Adam moments bitwise, and a calibration change at resume is
      refused (buffers are not in the adapter).
  R9  sampler continuity .... 5k segments vs one uninterrupted run produce the
      identical window-index sequence, mid-pass and at an exact pass boundary,
      using the r4 patched sampler and the real state capture/restore; losing
      the resume offset is shown to repeat windows (negative control).
  R10 LR continuity ......... seven resumes through the real v3 light state keep
      ONE 40k cosine built by the trainer's own `_build_scheduler`; rebuilding a
      short cosine per segment (the r5 defect) is detected.

Usage:
    python smoke_force_v1_real.py
    python smoke_force_v1_real.py --only R2,R9
    python smoke_force_v1_real.py --report out.json

Exit code 0 means every requested check passed. Failures raise with the number
that broke; nothing is caught and reported as a pass.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

RESULTS: dict[str, str] = {}
NOTES: dict[str, dict] = {}

FORCE_DIR = Path("/workspace/a3_data/a3_fw_force")
HIST_LEN = 16
HORIZON = 32
FORCE_DIM = 6
CHANNELS = [0, 1, 2, 12, 13, 16]
TOTAL_STEPS = 40000
SEGMENT = 5000


def ok(name, detail="", notes=None):
    RESULTS[name] = detail or "pass"
    if notes:
        NOTES[name] = notes
    print("PASS %s %s" % (name, detail), flush=True)


# --------------------------------------------------------------------- scaffolding
def to_py(node):
    from omegaconf import OmegaConf

    return OmegaConf.to_container(node, resolve=True)


def load_cfg():
    from hydra import compose, initialize_config_dir

    from fasterwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    import fasterwam

    configs = Path(fasterwam.__file__).resolve().parent.parent.parent / "configs"
    if not (configs / "train.yaml").is_file():
        raise FileNotFoundError("configs/train.yaml not found next to the package: %s" % configs)
    with initialize_config_dir(config_dir=str(configs), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["task=a3_force_40k"])
    return cfg


def force_spec_from_cfg(cfg):
    fc = cfg.model.force.conditioner
    return {
        "force_dim": int(fc.history_encoder.force_dim),
        "hist_len": int(fc.history_encoder.hist_len),
        "horizon": int(fc.future_encoder.horizon),
        "hidden_dim": int(fc.history_encoder.hidden_dim),
        "inject_num_heads": int(fc.inject.num_heads),
        "action_hidden_dim": int(fc.inject.action_hidden_dim),
        "channels": tuple(int(c) for c in fc.channels),
    }


def load_dataset(cfg):
    from hydra.utils import instantiate

    return instantiate(cfg.data.train)


def batch_windows(ds, idxs):
    """Stack `ds.force_windows(i)` the way the collate layer would."""
    parts = [ds.force_windows(int(i)) for i in idxs]
    return {
        "force_history": torch.stack([p["force_history"] for p in parts]),
        "force_history_mask": torch.stack([p["force_history_mask"] for p in parts]),
        "force_future": torch.stack([p["force_future"] for p in parts]),
        "force_future_mask": torch.stack([p["force_future_mask"] for p in parts]),
        "window_offset": [int(p["window_offset"]) for p in parts],
    }


def build_experts(cfg, seed=0):
    """The frozen action expert, with and without the force conditioner.

    Both are built from the same seed so every non-force tensor is identical
    without any copying: the forced expert's new modules are the only addition,
    which is exactly the claim R3 is testing.
    """
    from fasterwam.models.wan22.sparse_action_dit import SparseActionDiT

    dit_cfg = to_py(cfg.model.action_dit_config)
    force_cfg = to_py(cfg.model.force.conditioner)
    torch.manual_seed(seed)
    plain = SparseActionDiT(**dit_cfg, force_config=None)
    torch.manual_seed(seed)
    forced = SparseActionDiT(**dit_cfg, force_config=force_cfg)

    src, dst = plain.state_dict(), forced.state_dict()
    absent = [k for k in src if k not in dst]
    if absent:
        raise AssertionError("forced expert lost plain keys: %s" % absent[:5])
    differ = [k for k in src if not torch.equal(src[k], dst[k])]
    if differ:
        raise AssertionError("seeded build did not match on %d non-force tensors: %s"
                             % (len(differ), differ[:5]))
    plain.eval()
    forced.eval()
    return plain, forced


def set_real_calibration(expert, calib):
    enc = expert.force_conditioner["history_encoder"]
    enc.set_norm_stats(calib["mean"], calib["std"], scale=float(calib.get("scale", 1.0)))
    return enc


def get_calib(cfg, ctx):
    """The calibration the run will use, loaded through the real production reader.

    R6 checks this file in detail; every other check only needs it to exist, and
    `--only` must be able to run them in isolation.
    """
    if "calib" not in ctx:
        from fasterwam.runtime import _load_force_norm_stats

        ctx["calib"] = _load_force_norm_stats(to_py(cfg.model.force.norm_stats),
                                              to_py(cfg.model.force.conditioner))
    return ctx["calib"]


def _experts(cfg, ctx):
    if "experts" not in ctx:
        ctx["experts"] = build_experts(cfg)
    return ctx["experts"]


class _Stub:
    """Hosts the real FastWAM force methods without instantiating the 5B model.

    Only the attributes those methods actually read are provided, so a method
    that starts depending on something else fails loudly here instead of
    silently running against a different object.
    """

    def __init__(self, expert, cfg, device="cpu"):
        from fasterwam.models.wan22.fastwam import FastWAM
        from fasterwam.models.wan22.schedulers.scheduler_continuous import (
            WanContinuousFlowMatchScheduler,
        )

        self.force_conditioner = expert.force_conditioner
        self.force_spec = dict(expert.force_conditioner.spec)
        self.device = torch.device(device)
        self.torch_dtype = torch.float32
        self.loss_lambda_force = float(cfg.model.force.lambda_force)
        nts = int(cfg.model.action_scheduler.num_train_timesteps)
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=nts, shift=float(cfg.model.action_scheduler.train_shift)
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=nts, shift=float(cfg.model.action_scheduler.infer_shift)
        )
        for name in (
            "_require_force_training_inputs",
            "_prepare_force_inputs",
            "assert_future_frames_are_actually_future",
            "_require_force_inference_inputs",
            "_build_force_infer_state",
            "_decode_force_state",
            "_advance_force_state",
        ):
            setattr(self, name, getattr(FastWAM, name).__get__(self, _Stub))


def forward_with_force(expert, action_tokens, timestep, context, context_mask, force):
    """Run the expert's REAL forward loop with force fed through its real pre_dit.

    `forward()` calls `self.pre_dit(...)` without force kwargs (the video/MoT
    path passes them explicitly), so the instance attribute is shadowed for the
    duration of the call. Everything measured afterwards is upstream code.
    """
    orig = expert.pre_dit

    def patched(*a, **kw):
        kw.update(force)
        return orig(*a, **kw)

    expert.pre_dit = patched
    try:
        with torch.no_grad():
            return expert.forward(action_tokens, timestep, context, context_mask)
    finally:
        expert.pre_dit = orig


def joint_loss(expert, action_tokens, timestep, context, context_mask, force,
               action_target, force_target, action_weight=1.0):
    """One differentiable graph shaped like `FastWAM.training_loss`.

    `training_loss` reads the force head off the post-block tokens the action
    head already consumed (fastwam.py:828-831) and sums
    `lambda_action * loss_action + lambda_force * loss_force` (862-881). The
    block loop below is `SparseActionDiT.forward` verbatim; it is written out
    because `forward()` neither takes the force kwargs nor returns the state the
    force head needs.

    The shape matters for what R4 proves: the injection gate and the force
    head's final layer are *both* zero-init, so `loss_force` alone cannot put
    gradient on the gate -- only `loss_action`, back-propagating through the
    injection into the trunk, opens it. Setting `action_weight = 0` is the
    negative control for that claim, and is the failure mode a run with a
    detached action loss would hit: the history encoder would stay frozen for
    all 40,000 steps.
    """
    import torch.nn.functional as F

    if expert.use_gradient_checkpointing:
        raise AssertionError("this helper mirrors the non-checkpointed forward; "
                             "gradient checkpointing must stay off for the ladder test")
    pre = expert.pre_dit(action_tokens, timestep, context, context_mask, **force)
    x = pre["tokens"]
    for block in expert.blocks:
        x = block(x, pre["context"], pre["t_mod"], pre["freqs"],
                  context_mask=pre["context_mask"])
    vel = expert.post_dit(x, pre)
    force_tokens = expert.apply_force_injection_post_dit(x, pre)
    pred = expert.predict_force_velocity(force_tokens, pre)
    return action_weight * F.mse_loss(vel, action_target) + F.mse_loss(pred, force_target)


# ----------------------------------------------------------------------------- R1
def r1_corpus(cfg, ctx):
    ds = ctx.setdefault("ds", load_dataset(cfg))
    spec = force_spec_from_cfg(cfg)
    if tuple(ds._force_channels) != spec["channels"]:
        raise AssertionError("dataset gathers %s, model config names %s"
                             % (list(ds._force_channels), list(spec["channels"])))
    if ds._force.shape[1] != 18:
        raise AssertionError("force table is %d wide, expected the untouched 18-column table"
                             % ds._force.shape[1])
    if ds._force_hist_len != spec["hist_len"] or ds._force_horizon != spec["horizon"]:
        raise AssertionError("dataset window geometry %s/%s != model %s/%s"
                             % (ds._force_hist_len, ds._force_horizon,
                                spec["hist_len"], spec["horizon"]))
    if ds._force_horizon != ds.num_frames - 1:
        raise AssertionError("force horizon must be one step per action token (num_frames-1)")
    if ds._proprio.shape[1] != 14:
        raise AssertionError("corpus proprio is %d wide, the frozen V1 input is 14"
                             % ds._proprio.shape[1])
    if int(cfg.model.proprio_dim) != 14:
        raise AssertionError("model proprio_dim is %s, expected 14" % cfg.model.proprio_dim)
    if ds._action.shape[1] != 7:
        raise AssertionError("action is %d wide, expected 7" % ds._action.shape[1])
    if ds._force.shape[0] != ds._proprio.shape[0] or ds._force.shape[0] != ds._action.shape[0]:
        raise AssertionError("force rows do not line up with the video corpus rows")

    starts = np.asarray(ds._window_ep_start)
    ends = np.asarray(ds._window_ep_end)
    cur = np.asarray(ds._windows)
    if (cur < starts).any() or (cur >= ends).any():
        raise AssertionError("a window's current frame lies outside its own episode")

    rng = np.random.default_rng(0)
    idxs = sorted(set(range(0, len(cur), max(1, len(cur) // 400))).union(
        set(rng.integers(0, len(cur), size=200).tolist())))
    n_left = n_right = 0
    for i in idxs:
        b = batch_windows(ds, [i])
        hm = b["force_history_mask"][0].numpy()
        fm = b["force_future_mask"][0].numpy()
        hist = b["force_history"][0].numpy()
        fut = b["force_future"][0].numpy()
        nl, nf = int(hm.sum()), int(fm.sum())
        if nl == 0 or nf == 0:
            raise AssertionError("window %d has an entirely masked force window" % i)
        if not hm[HIST_LEN - nl:].all() or hm[:HIST_LEN - nl].any():
            raise AssertionError("window %d history mask is not the trailing valid suffix" % i)
        if not fm[:nf].all() or fm[nf:].any():
            raise AssertionError("window %d future mask is not the leading valid prefix" % i)
        if np.any(hist[:HIST_LEN - nl] != 0.0):
            raise AssertionError("window %d left-padded history rows are not zero" % i)
        if np.any(fut[nf:] != 0.0):
            raise AssertionError("window %d right-padded future rows are not zero" % i)
        s, ep_start, ep_end = int(cur[i]), int(starts[i]), int(ends[i])
        # Inverting the dataset slice: nl rows covering lo..s inclusive means lo = s-nl+1.
        lo = s - nl + 1
        if lo < ep_start:
            raise AssertionError("window %d history reaches into the previous episode" % i)
        if s + nf > ep_end - 1:
            raise AssertionError("window %d future reaches into the next episode" % i)
        n_left += int(nl < HIST_LEN)
        n_right += int(nf < HORIZON)
        rows = ds.force_rows(np.arange(lo, s + 1))
        if not np.array_equal(rows, hist[HIST_LEN - nl:]):
            raise AssertionError("window %d history != independent force_rows() read" % i)
        rows = ds.force_rows(np.arange(s + 1, s + 1 + nf))
        if not np.array_equal(rows, fut[:nf]):
            raise AssertionError("window %d future != independent force_rows() read" % i)
    ok("R1_corpus_and_geometry",
       "%d windows / %d episodes, 18-col table, channels %s, proprio 14; %d of %d sampled "
       "windows left-padded, %d right-padded; no cross-episode borrow"
       % (len(cur), len(set(starts.tolist())), list(spec["channels"]), n_left, len(idxs), n_right),
       {"n_windows": int(len(cur)), "n_episodes": len(set(starts.tolist())),
        "n_rows": int(ds._force.shape[0]), "sampled": len(idxs)})


# ---------------------------------------------------------------------------- R2b
def r2b_proprio_units(cfg, ctx):
    ds = ctx.setdefault("ds", load_dataset(cfg))
    stats_path = FORCE_DIR / "force_stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError("missing %s: the proprio force columns cannot be attributed"
                                % stats_path)
    stats = json.loads(stats_path.read_text())
    if list(stats["channels"]) != CHANNELS:
        raise AssertionError("force_stats.json channels %s != %s" % (stats["channels"], CHANNELS))
    glob = np.asarray(stats["zero_load_ref_median"], dtype=np.float64)
    if glob.shape != (FORCE_DIM,):
        raise AssertionError("zero_load_ref_median shape %s" % (glob.shape,))

    corpus = Path(str(cfg.data.train.dataset_dirs[0]))
    arr = np.load(corpus / "arrays.npz", allow_pickle=True)
    starts = np.asarray(arr["ep_start"], dtype=np.int64)
    lens = np.asarray(arr["ep_len"], dtype=np.int64)

    raw6 = ds._force[:, CHANNELS].astype(np.float64)
    stored = ds._proprio[:, 8:].astype(np.float64)
    d_global = float(np.abs((raw6 - glob) - stored).max())

    ep_zero = np.zeros((len(starts), FORCE_DIM), dtype=np.float64)
    ep_of_row = np.zeros(raw6.shape[0], dtype=np.int64)
    for k, (st, ln) in enumerate(zip(starts, lens)):
        ep_zero[k] = np.median(raw6[st:st + min(20, int(ln))], axis=0)
        ep_of_row[st:st + int(ln)] = k
    d_episode = float(np.abs((raw6 - ep_zero[ep_of_row]) - stored).max())

    if d_global > 1e-5:
        raise AssertionError(
            "proprio force columns are not raw-minus-global-reference (max abs diff %.3e): the "
            "corpus was built with a zero the calibration does not record" % d_global)
    if d_episode <= d_global:
        raise AssertionError(
            "proprio[:, 8:] matches the per-episode first-20-frame median (diff %.3e) as well as "
            "or better than the global reference (%.3e). That zero point cannot be computed at "
            "online step 0 (§四)." % (d_episode, d_global))
    ok("R2b_proprio_force_units",
       "global-reference residual %.2e (accepted); per-episode-median residual %.2e (rejected, "
       "would not exist at step 0)" % (d_global, d_episode),
       {"d_global": d_global, "d_per_episode": d_episode,
        "global_zero_ref": [float(v) for v in glob]})


# ------------------------------------------------------------------------------ R2
def r2_leak_gate(cfg, ctx):
    ds = ctx.setdefault("ds", load_dataset(cfg))
    plain, forced = _experts(cfg, ctx)
    set_real_calibration(forced, get_calib(cfg, ctx))
    stub = _Stub(forced, cfg)

    n_win = len(ds._windows)
    firsts = sorted(set(int(s) for s in ds._window_ep_start))[:4]
    idxs = firsts + [0, n_win // 3, n_win // 2, n_win - 1]
    b = batch_windows(ds, idxs)
    res = stub.assert_future_frames_are_actually_future(
        b["force_future"], b["force_future_mask"], b["window_offset"], ds.force_rows)
    if res["compared_steps"] <= 0:
        raise AssertionError("leak gate compared nothing")
    if float(res["max_abs_diff"]) != 0.0:
        raise AssertionError("real force_future differs from reindexed truth by %.3e"
                             % res["max_abs_diff"])

    shifted = b["force_future"].clone()
    shifted[:, :-1] = shifted[:, 1:]
    try:
        stub.assert_future_frames_are_actually_future(
            shifted, b["force_future_mask"], b["window_offset"], ds.force_rows)
    except AssertionError:
        pass
    else:
        raise AssertionError("gate accepted a +1 forward label shift on real windows")

    # The gate also enforces that the masked tail is exactly zero, i.e. what the
    # model is fed contains no hidden future value. Windows are only built when the
    # whole 32-step future exists inside the episode (R1 counts 0 right-padded), so
    # the clipped case has to be synthesised by masking a real window's tail.
    clip = batch_windows(ds, [n_win - 1])
    keep = HORIZON - 4
    cmask = torch.zeros_like(clip["force_future_mask"])
    cmask[:, :keep] = True
    cfut = clip["force_future"] * cmask[:, :, None]
    stub.assert_future_frames_are_actually_future(
        cfut, cmask, clip["window_offset"], ds.force_rows)
    leaked = cfut.clone()
    leaked[:, keep] = 1.0
    try:
        stub.assert_future_frames_are_actually_future(
            leaked, cmask, clip["window_offset"], ds.force_rows)
    except AssertionError:
        pass
    else:
        raise AssertionError("gate accepted a nonzero value hidden behind the future mask")
    ok("R2_leak_gate_real",
       "windows %s: %d steps exact against force_rows(); +1 shift rejected"
       % ([idxs[0], "...", idxs[-1]], res["compared_steps"]), res)


# ------------------------------------------------------------------------------ R3
def r3_zero_init_equivalence(cfg, ctx):
    plain, forced = _experts(cfg, ctx)
    set_real_calibration(forced, get_calib(cfg, ctx))
    spec = forced.force_spec
    want = force_spec_from_cfg(cfg)
    for key in ("force_dim", "hist_len", "horizon", "action_hidden_dim", "channels"):
        if spec[key] != want[key]:
            raise AssertionError("built expert %s=%s != frozen config %s" % (key, spec[key], want[key]))

    B, T = 2, HORIZON
    g = torch.Generator().manual_seed(7)
    action_dim = int(cfg.model.action_dit_config.action_dim)
    ctx_len = int(cfg.data.train.context_len)
    text_dim = int(cfg.model.action_dit_config.text_dim)
    action_tokens = torch.randn(B, T, action_dim, generator=g)
    timestep = torch.rand(B, generator=g) * 999.0
    context = torch.randn(B, ctx_len, text_dim, generator=g)
    context_mask = torch.ones(B, ctx_len, dtype=torch.bool)
    force = {
        "force_history": torch.randn(B, HIST_LEN, FORCE_DIM, generator=g),
        "force_history_mask": torch.ones(B, HIST_LEN, dtype=torch.bool),
        "noisy_force_future": torch.randn(B, T, FORCE_DIM, generator=g),
        "force_future_mask": torch.ones(B, T, dtype=torch.bool),
    }

    with torch.no_grad():
        pre_plain = plain.pre_dit(action_tokens, timestep, context, context_mask)
        pre_forced = forced.pre_dit(action_tokens, timestep, context, context_mask, **force)
    for key in ("tokens", "t", "t_mod", "context"):
        if not torch.equal(pre_plain[key], pre_forced[key]):
            raise AssertionError("pre_dit '%s' differs at zero init (max %.3e)"
                                 % (key, (pre_plain[key] - pre_forced[key]).abs().max()))
    if pre_forced["force_kv"] is None or pre_forced["force_kv"].shape[1] != HIST_LEN + HORIZON:
        raise AssertionError("force kv block missing or wrong length: %s"
                             % (None if pre_forced["force_kv"] is None
                                else tuple(pre_forced["force_kv"].shape)))

    with torch.no_grad():
        out_plain = plain.forward(action_tokens, timestep, context, context_mask)
    out_forced = forward_with_force(forced, action_tokens, timestep, context, context_mask, force)
    if not torch.equal(out_plain, out_forced):
        raise AssertionError("the 30-block action output differs at zero init (max %.3e)"
                             % (out_plain - out_forced).abs().max())

    post = forced.apply_force_injection_post_dit(pre_forced["tokens"], pre_forced)
    if not torch.equal(post, pre_forced["tokens"]):
        raise AssertionError("post-DiT injection perturbed tokens at zero init")
    with torch.no_grad():
        pred = forced.predict_force_velocity(post, pre_forced)
    if tuple(pred.shape) != (B, T, FORCE_DIM):
        raise AssertionError("force head shape %s" % (tuple(pred.shape),))
    if float(pred.abs().max()) != 0.0:
        raise AssertionError("force head is not zero-initialised: max %.3e" % pred.abs().max())

    inject = forced.force_conditioner["inject"]
    with torch.no_grad():
        inject.attn.out_proj.weight.normal_(0, 0.02)
        moved = forced.apply_force_injection_post_dit(pre_forced["tokens"], pre_forced)
        if torch.equal(moved, pre_forced["tokens"]):
            raise AssertionError("injection is still a no-op with a non-zero out_proj -> dead module")
        after = forward_with_force(forced, action_tokens, timestep, context, context_mask, force)
        if torch.equal(after, out_plain):
            raise AssertionError("the pre_dit injection never reaches the action head -> dead module")
        inject.attn.out_proj.weight.zero_()
        inject.attn.out_proj.bias.zero_()
        restored = forward_with_force(forced, action_tokens, timestep, context, context_mask, force)
    if not torch.equal(restored, out_plain):
        raise AssertionError("zeroing out_proj again did not restore equivalence")
    ok("R3_zero_init_equivalence",
       "pre_dit, 30-block forward and post-DiT injection bitwise equal at init; both injection "
       "points live once out_proj is non-zero")


# ------------------------------------------------------------------------------ R4
def r4_gradient_ladder(cfg, ctx):
    # R4 takes real optimizer steps, so it must not use the cached pair: every
    # check ordered after it asserts properties of a model *at init*.
    plain, forced = build_experts(cfg)
    set_real_calibration(forced, get_calib(cfg, ctx))
    B, T = 2, HORIZON
    g = torch.Generator().manual_seed(11)
    action_dim = int(cfg.model.action_dit_config.action_dim)
    ctx_len = int(cfg.data.train.context_len)
    text_dim = int(cfg.model.action_dit_config.text_dim)
    action_tokens = torch.randn(B, T, action_dim, generator=g)
    timestep = torch.rand(B, generator=g) * 999.0
    context = torch.randn(B, ctx_len, text_dim, generator=g)
    context_mask = torch.ones(B, ctx_len, dtype=torch.bool)
    history = torch.randn(B, HIST_LEN, FORCE_DIM, generator=g)
    future = torch.randn(B, T, FORCE_DIM, generator=g)
    target = torch.randn(B, T, FORCE_DIM, generator=g)
    action_target = torch.randn(B, T, action_dim, generator=g)
    force = {
        "force_history": history,
        "force_history_mask": torch.ones(B, HIST_LEN, dtype=torch.bool),
        "noisy_force_future": future,
        "force_future_mask": torch.ones(B, T, dtype=torch.bool),
    }

    def backward_once(action_weight=1.0):
        forced.zero_grad(set_to_none=True)
        loss = joint_loss(forced, action_tokens, timestep, context, context_mask, force,
                          action_target, target, action_weight)
        loss.backward()
        return {n: p.grad for n, p in forced.named_parameters()}, loss

    named = dict(forced.named_parameters())
    force_names = sorted(n for n in named if n.startswith("force_conditioner."))
    if not force_names:
        raise AssertionError("the built expert has no force_conditioner parameters")

    grads, loss0 = backward_once()
    if not torch.isfinite(loss0):
        raise AssertionError("joint loss is not finite at init")

    stepped = sorted(n for n in force_names
                     if grads.get(n) is not None and float(grads[n].abs().max()) > 0.0)
    allowed0 = ("force_conditioner.inject.attn.out_proj", "force_conditioner.force_head.linear.2")
    if "force_conditioner.inject.attn.out_proj.weight" not in stepped:
        raise AssertionError("step 0 left the injection gate at zero gradient: %s" % stepped)
    if "force_conditioner.force_head.linear.2.weight" not in stepped:
        raise AssertionError("step 0 left the force head output layer at zero gradient: %s"
                             % stepped)
    # The exhaustive form of the claim, so a tensor the test forgot to name cannot
    # slip through: outside the two zero-init output layers the step-0 gradient
    # must be exactly zero, and every force tensor must be in the graph at all.
    # Written as a rule over `force_names` rather than a list because the real
    # `nn.MultiheadAttention` splits `in_proj_weight` into q/k/v as soon as
    # kdim != embed_dim, which is the case here (force hidden 256 vs action 1024).
    bad = []
    for n in force_names:
        gr = grads.get(n)
        if gr is None:
            bad.append((n, "not in the graph"))
            continue
        if not torch.isfinite(gr).all():
            bad.append((n, "non-finite gradient"))
            continue
        magnitude = float(gr.abs().max())
        if n.startswith(allowed0):
            if magnitude == 0.0:
                bad.append((n, "zero inside the gate"))
        elif magnitude != 0.0:
            bad.append((n, "%.3e outside the gate" % magnitude))
    if bad:
        raise AssertionError("step-0 gradient does not match the zero-init gate: %s; the "
                             "equivalence argument in the freeze document is wrong" % bad[:6])

    # Negative control: the gate opens because of the action term, not because of
    # anything the force loss does. Remove the action term and it must be frozen.
    grads_force_only, _ = backward_once(action_weight=0.0)
    if float(grads_force_only["force_conditioner.inject.attn.out_proj.weight"].abs().max()) \
            != 0.0:
        raise AssertionError("force-loss-only already moves the gate, so the joint loss is "
                            "not what opens the encoder path and the claim above is untested")

    opt = torch.optim.AdamW([named[n] for n in force_names], lr=1e-4, betas=(0.9, 0.95))
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        joint_loss(forced, action_tokens, timestep, context, context_mask, force,
                   action_target, target).backward()
        opt.step()
    grads1, loss1 = backward_once()
    if not torch.isfinite(loss1):
        raise AssertionError("joint loss is not finite after three steps")
    lit = sorted(n for n in force_names if grads1[n] is None or float(grads1[n].abs().max()) == 0.0)
    if lit:
        raise AssertionError("force tensors still frozen at zero gradient after the gate "
                             "opened: %s" % lit[:8])
    never = [n for n in force_names if grads[n] is None]
    if never:
        raise AssertionError("force tensors with no gradient at all (not in the graph): %s" % never[:8])
    ok("R4_gradient_ladder",
       "step 0 subset = %s; force-only freezes the gate (negative control); after 3 joint "
       "steps every one of %d force tensors (%d elements) has non-zero gradient"
       % (stepped, len(force_names), sum(int(named[n].numel()) for n in force_names)),
       {"force_tensors": len(force_names),
        "force_elements": sum(int(named[n].numel()) for n in force_names),
        "step0_stepped": stepped})



# ------------------------------------------------------------------------------ R5
def r5_joint_flow_target(cfg, ctx):
    plain, forced = _experts(cfg, ctx)
    ds = ctx.setdefault("ds", load_dataset(cfg))
    set_real_calibration(forced, get_calib(cfg, ctx))
    stub = _Stub(forced, cfg)
    nts = int(cfg.model.action_scheduler.num_train_timesteps)

    b = batch_windows(ds, [0, len(ds._windows) // 2])
    clean = stub._prepare_force_inputs(dict(b))["future"]
    if not torch.isfinite(clean).all():
        raise AssertionError("normalized force future is not finite")

    torch.manual_seed(3)
    for t_val in (50.0, 500.0, 999.0):
        noise = torch.randn_like(clean)
        timestep = torch.full((clean.shape[0],), float(t_val))
        noisy = stub.train_action_scheduler.add_noise(clean, noise, timestep)
        target = stub.train_action_scheduler.training_target(clean, noise, timestep)
        if not torch.equal(target, noise - clean):
            raise AssertionError("force target is not the rectified-flow velocity noise-x0 at t=%s"
                                 % t_val)
        if torch.allclose(target, clean, atol=1e-4):
            raise AssertionError("the force target IS the clean value: a bolted-on MSE, not flow "
                                 "matching (§三-C)")
        sigma = timestep.view(-1, 1, 1) / float(nts)
        recon = noisy - sigma * target
        d = float((recon - clean).abs().max())
        if d > 1e-5:
            raise AssertionError("x0 is not recoverable from (noisy, target) at t=%s (diff %.3e)"
                                 % (t_val, d))

    # Jointly with the action stream: one `t` for both, independent noise.
    action = torch.randn_like(clean)
    timestep = torch.rand(clean.shape[0]) * 999.0
    noisy_a = stub.train_action_scheduler.add_noise(action, torch.randn_like(action), timestep)
    noisy_f = stub.train_action_scheduler.add_noise(clean, torch.randn_like(clean), timestep)
    if noisy_a.shape != noisy_f.shape:
        raise AssertionError("action and force streams must be token-aligned")
    if float((noisy_a - noisy_f).abs().max()) == 0.0:
        raise AssertionError("the two streams share a noise realization; they would not be "
                             "separately predictable")

    pre = forced.pre_dit(torch.randn(2, HORIZON, int(cfg.model.action_dit_config.action_dim)),
                         torch.rand(2) * 999.0,
                         torch.randn(2, int(cfg.data.train.context_len),
                                     int(cfg.model.action_dit_config.text_dim)),
                         torch.ones(2, int(cfg.data.train.context_len), dtype=torch.bool),
                         force_history=torch.randn(2, HIST_LEN, FORCE_DIM),
                         force_history_mask=torch.ones(2, HIST_LEN, dtype=torch.bool),
                         noisy_force_future=clean[:2],
                         force_future_mask=torch.ones(2, HORIZON, dtype=torch.bool))
    with torch.no_grad():
        pred = forced.predict_force_velocity(
            forced.apply_force_injection_post_dit(pre["tokens"], pre), pre)
    if float(pred.abs().max()) != 0.0:
        raise AssertionError("force velocity prediction is not exactly zero at init")
    e_target = float((torch.randn_like(clean) - clean).pow(2).mean())
    ok("R5_joint_flow_target",
       "target == noise - x0 (invertible to 1e-5), never the clean value; pred == 0 at init so "
       "loss_force starts at E||target||^2 ~ %.3f" % e_target,
       {"E_target_sq_estimate": e_target})


# ------------------------------------------------------------------------------ R6
def r6_calibration(cfg, ctx):
    from fasterwam.runtime import _load_force_norm_stats

    fc = to_py(cfg.model.force.conditioner)
    calib_path = Path(str(cfg.model.force.norm_stats.file))
    if not calib_path.is_file():
        raise FileNotFoundError(
            "%s is missing: run make_force_calib.py before this test -- a force stream without "
            "measured units cannot be trained (§四)" % calib_path)
    calib = _load_force_norm_stats(to_py(cfg.model.force.norm_stats), fc)
    if list(calib["channels"]) != CHANNELS:
        raise AssertionError("calibration channels %s != frozen %s" % (calib["channels"], CHANNELS))
    if len(calib["mean"]) != FORCE_DIM or len(calib["std"]) != FORCE_DIM:
        raise AssertionError("calibration widths %d/%d != %d"
                             % (len(calib["mean"]), len(calib["std"]), FORCE_DIM))
    if min(calib["std"]) <= 0 or not all(math.isfinite(v) for v in calib["std"] + calib["mean"]):
        raise AssertionError("calibration mean/std not finite and positive: %s" % calib["std"])
    if not calib.get("version"):
        raise AssertionError("calibration has no version")
    ctx["calib"] = calib

    for label, mutate in (
        ("foreign channels", lambda d: d.__setitem__("channels", [0, 1, 2, 3, 4, 5])),
        ("no version", lambda d: d.__setitem__("version", None)),
        ("wrong width", lambda d: d.__setitem__("std", d["std"][:5])),
    ):
        broken = dict(calib)
        mutate(broken)
        try:
            _load_force_norm_stats(broken, fc)
        except ValueError:
            continue
        raise AssertionError("`_load_force_norm_stats` accepted %s" % label)

    plain, forced = _experts(cfg, ctx)
    enc = set_real_calibration(forced, calib)
    if not torch.equal(enc.force_mean, torch.tensor(calib["mean"], dtype=torch.float32)):
        raise AssertionError("force_mean buffer != calibration")
    if not torch.equal(enc.force_std, torch.tensor(calib["std"], dtype=torch.float32)):
        raise AssertionError("force_std buffer != calibration")
    if float(enc.stats_are_set) != 1.0:
        raise AssertionError("stats_are_set buffer not raised")
    if enc.scale != float(calib.get("scale", 1.0)):
        raise AssertionError("calibration scale not applied")
    ok("R6_calibration_provenance",
       "%s version=%s mean=%s std=%s; foreign/width-less/version-less variants refused"
       % (calib_path.name, calib["version"],
          [round(v, 3) for v in calib["mean"]], [round(v, 4) for v in calib["std"]]),
       {k: calib.get(k) for k in ("version", "units", "mean", "std", "channels", "scale",
                                  "corpus", "n_rows", "source_sha256")})


# ------------------------------------------------------------------------------ R7
def r7_units_consistency(cfg, ctx):
    plain, forced = _experts(cfg, ctx)
    ds = ctx.setdefault("ds", load_dataset(cfg))
    set_real_calibration(forced, get_calib(cfg, ctx))
    stub = _Stub(forced, cfg)

    idx = int(np.random.default_rng(0).integers(0, len(ds._windows)))
    sample = dict(ds.force_windows(idx))          # one unbatched sample, as the dataset yields it
    batched = batch_windows(ds, [idx])            # what the collate layer hands the model
    train = stub._prepare_force_inputs(batched)
    infer = stub._build_force_infer_state(
        batched["force_history"], batched["force_history_mask"],
        num_inference_steps=10, sigma_shift=None, seed=0, rand_device="cpu")

    if not torch.equal(train["history"], infer["history"]):
        raise AssertionError("training and roll-out normalize the history differently (max %.3e)"
                             % (train["history"] - infer["history"]).abs().max())
    if not torch.equal(train["history_mask"], infer["history_mask"]):
        raise AssertionError("history masks differ between paths")
    if infer["future_mask"].shape != (1, HORIZON) or not bool(infer["future_mask"].all()):
        raise AssertionError("a roll-out must treat every future step as a real target")

    enc = forced.force_conditioner["history_encoder"]
    back = enc.denormalize(train["future"])
    d = float((back[0] - sample["force_future"]).abs().max())
    if d > 1e-4:
        raise AssertionError("normalize is not invertible to 1e-4 newton (diff %.3e)" % d)
    d2 = float((enc.denormalize(infer["history"])[0] - sample["force_history"]).abs().max())
    if d2 > 1e-4:
        raise AssertionError("history denormalize differs from the raw window (%.3e)" % d2)

    # The two normalisations in the model must be two, not one applied twice.
    arr = np.load(Path(str(cfg.data.train.dataset_dirs[0])) / "arrays.npz", allow_pickle=True)
    s = int(sample["window_offset"])
    stored = np.asarray(arr["proprio"])[s][8:14].astype(float)
    raw = np.asarray(ds._force[s][CHANNELS], dtype=float)
    zscore = (raw - np.asarray(get_calib(cfg, ctx)["mean"])) / np.asarray(
        get_calib(cfg, ctx)["std"])
    if np.allclose(stored, zscore, atol=1e-4):
        raise AssertionError("the proprio force columns equal the force stream: the documented "
                             "min/max and z-score normalisations are not distinct")
    if np.allclose(stored, raw, atol=1e-4):
        raise AssertionError("proprio force columns are raw newtons: the processor would min/max "
                            "scale a physical unit as if it were a joint position")
    if float(train["future"].abs().max()) > 50.0:
        raise AssertionError("normalized force stream reaches %.1f sigma -- calibration is wrong "
                            "for these channels" % float(train["future"].abs().max()))
    ok("R7_train_infer_units",
       "history bitwise equal across the two paths; denormalize within %.1e N; the min/max "
       "proprio stream and the z-score force stream are distinct" % d,
       {"proprio_cols_at_s": [round(float(v), 4) for v in stored],
        "raw_newtons_at_s": [round(float(v), 3) for v in raw],
        "zscore_at_s": [round(float(v), 4) for v in zscore],
        "force_stream_absmax": float(train["future"].abs().max())})


# ------------------------------------------------------------------------------ R8
def r8_light_state_round_trip(cfg, ctx):
    from fasterwam import light_state as v1
    from fasterwam import light_state_v3 as v3

    plain, forced = _experts(cfg, ctx)
    set_real_calibration(forced, get_calib(cfg, ctx))
    named = dict(forced.named_parameters())
    force_names = sorted(n for n in named if n.startswith("force_conditioner."))
    other_names = ["action_encoder.weight", "head.weight"]

    # Mirror the real run's parameter policy. Under LoRA the base weights are not
    # trainable, so `requires_grad` is exactly the set the optimizer must own; a
    # bare SparseActionDiT has ~600 trainable tensors and the coverage assertion
    # below would measure the harness instead of the checkpoint.
    train_set = set(force_names) | set(other_names)
    for n, p in named.items():
        p.requires_grad_(n in train_set)

    torch.manual_seed(5)
    with torch.no_grad():
        for n in force_names + other_names:
            named[n].normal_()
    opt = torch.optim.AdamW([named[n] for n in force_names + other_names],
                            lr=1e-4, betas=(0.9, 0.95))
    with torch.no_grad():
        for n in force_names + other_names:
            named[n].grad = torch.randn_like(named[n])
    opt.step()

    saved = {n: named[n].detach().clone() for n in force_names}
    saved_moments = {n: opt.state[named[n]]["exp_avg"].clone() for n in force_names}

    class _Sampler:
        seed = 42
        epoch = 3
        epoch_offset = 0
        resume_batch_offset = 1420
        sample_offset = 0
        batch_size = 6
        num_processes = 1

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    with tempfile.TemporaryDirectory() as td:
        manifest = v3.save_light_state(
            td, model=forced, optimizer=opt, scheduler=sched, sampler=_Sampler(),
            global_step=5000, epoch=3, batch_in_epoch=1420,
            extra={"lr": float(cfg.learning_rate)}, adapter_meta={"segment": 1})
        fp = manifest.get("force_calibration")
        if not fp:
            raise AssertionError("light state written without a force calibration fingerprint")
        cov = manifest["coverage"]
        missing = [n for n in force_names if n not in cov["saved_nonlora_names"]]
        if missing:
            raise AssertionError("%d force tensors left out of the checkpoint: %s"
                                 % (len(missing), missing[:5]))
        if cov["requires_grad_but_not_optimised"]:
            raise AssertionError("%d trainable tensors are not optimised"
                                 % cov["requires_grad_but_not_optimised"])
        # Negative control: this counter is what would have caught r4's lost
        # proprio_encoder. Make one base tensor trainable without handing it to the
        # optimizer and the coverage report must see exactly that tensor.
        stray = next(n for n in named if n not in train_set)
        named[stray].requires_grad_(True)
        nc = v3.coverage(forced, opt)
        if nc["requires_grad_but_not_optimised"] != 1:
            raise AssertionError("coverage missed a trainable-but-unoptimised tensor (%s): it "
                                 "cannot detect the r4 weight-loss defect" % stray)
        if nc["requires_grad_but_not_optimised_elements"] != int(named[stray].numel()):
            raise AssertionError("coverage counted the stray tensor but not its elements")
        named[stray].requires_grad_(False)

        with torch.no_grad():
            for n in force_names:
                named[n].zero_()
                opt.state[named[n]]["exp_avg"].zero_()
        fresh = _Sampler()
        fresh.resume_batch_offset = 0
        state = v3.load_light_state(td, model=forced, optimizer=opt, scheduler=sched,
                                    sampler=fresh)
        bad = [n for n in force_names if not torch.equal(named[n], saved[n])]
        if bad:
            raise AssertionError("restored force tensors differ: %s" % bad[:5])
        bad_opt = [n for n in force_names
                   if not torch.equal(opt.state[named[n]]["exp_avg"], saved_moments[n])]
        if bad_opt:
            raise AssertionError("restored Adam moments differ: %s" % bad_opt[:5])
        if state["global_step"] != 5000 or state["epoch"] != 3:
            raise AssertionError("counters wrong: %s" % {k: state[k] for k in
                                                         ("global_step", "epoch", "batch_in_epoch")})
        if v1._sampler_state(fresh) != v1._sampler_state(_Sampler()):
            raise AssertionError("sampler state lost a field: %s vs %s"
                                 % (v1._sampler_state(fresh), v1._sampler_state(_Sampler())))

        # Buffers are not in the adapter, so a resume with different calibration
        # must be refused rather than quietly re-uniting the stream.
        enc = forced.force_conditioner["history_encoder"]
        keep = enc.force_std.detach().clone()
        try:
            with torch.no_grad():
                enc.force_std.mul_(1.5)
            v3.load_light_state(td, model=forced, optimizer=opt, scheduler=sched,
                                sampler=_Sampler())
        except RuntimeError as exc:
            if "calibration" not in str(exc).lower():
                raise AssertionError("refused for the wrong reason: %s" % exc)
        else:
            raise AssertionError("resume accepted a different force calibration")
        finally:
            with torch.no_grad():
                enc.force_std.copy_(keep)
    ok("R8_light_state_v3",
       "%d force tensors + Adam moments restored bitwise, fingerprint enforced, adapter %d B"
       % (len(force_names), manifest["adapter_bytes"]),
       {"adapter_bytes": manifest["adapter_bytes"],
        "projected_bytes": cov.get("saved_elements"),
        "fingerprint": fp})


# ------------------------------------------------------------------------------ R9
class _SimLoop:
    """The trainer's data-consumption bookkeeping, minus the model.

    Mirrors trainer.py:697-706: batches are drawn from a fresh iterator, and on
    StopIteration the trainer's own epoch advances, batch_in_epoch resets and the
    resume offset is cleared. `drop_last` is False upstream, so a pass ends with
    a short batch. `_passes_done` reads `epoch` off this object, exactly as the
    patched sampler reads it off the real trainer.
    """

    def __init__(self, sampler, batch_size):
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.epoch = 0
        self.batch_in_epoch = 0
        self._it = None
        self._ended = False
        sampler._trainer = self

    def _iter_pass(self):
        """Begin the pass the sampler's current counters describe.

        Built here rather than in `__init__` because the patched `__iter__` slices
        its permutation eagerly -- a resumed run must have its light-state counters
        in place before the first draw.
        """
        self._it = iter(self.sampler)
        self._ended = False

    def _roll(self):
        """Exactly trainer.py:701-706: the pass is over, so start the next one.

        `batch_in_epoch` is reset at the roll, not when the short tail is handed out,
        which is the only ordering under which `resume_batch_offset * batch_size`
        equals the number of samples the pass has really given up.
        """
        self.epoch += 1
        self.batch_in_epoch = 0
        self.sampler.clear_resume_batch_offset()
        self._iter_pass()

    def _next_batch(self):
        if self._it is None:
            self._iter_pass()
        elif self._ended:
            self._roll()
        buf = []
        while len(buf) < self.batch_size:
            try:
                buf.append(int(next(self._it)))
            except StopIteration:
                self._ended = True
                break
        return buf

    def steps(self, n_steps):
        seq = []
        drawn = 0
        while drawn < n_steps:
            batch = self._next_batch()
            if not batch:
                continue  # the trainer `continue`s on an exhausted pass without counting it
            seq.extend(batch)
            drawn += 1
            self.batch_in_epoch += 1
        return seq


def _sampler_for(n_windows, batch_size, seed=42):
    from fasterwam.utils.samplers import ResumableEpochSampler

    class _DS:
        def __len__(self):
            return n_windows

    return ResumableEpochSampler(_DS(), seed=seed, batch_size=batch_size, num_processes=1)


def r9_sampler_continuity(cfg, ctx):
    from fasterwam import light_state as v1
    from fasterwam import r4_patch
    from fasterwam.utils.samplers import ResumableEpochSampler

    r4_patch.install()
    if not getattr(ResumableEpochSampler, "_r4_patched", False):
        raise AssertionError("the r4 sampler patch is not installed")

    n_win = len(ctx.setdefault("ds", load_dataset(cfg))._windows)
    batch = int(cfg.batch_size)
    steps = 2 * SEGMENT
    half = steps // 2

    cont = _SimLoop(_sampler_for(n_win, batch), batch)
    continuous = cont.steps(steps)

    def snapshot(loop):
        """What r4's save_checkpoint publishes: the consumed-batch counter."""
        prev = int(getattr(loop.sampler, "resume_batch_offset", 0))
        loop.sampler.set_resume_batch_offset(int(loop.batch_in_epoch))
        try:
            return v1._sampler_state(loop.sampler), loop.epoch, loop.batch_in_epoch
        finally:
            if prev:
                loop.sampler.set_resume_batch_offset(prev)
            else:
                loop.sampler.clear_resume_batch_offset()

    cases = {}
    for label, first_steps in (("mid-pass", half), ("pass-end", None)):
        if first_steps is None:
            first_steps = math.ceil(n_win / batch)  # boundary lands exactly at a pass end
        seg = _SimLoop(_sampler_for(n_win, batch), batch)
        first = seg.steps(first_steps)
        state, ep, bie = snapshot(seg)
        resumed = _SimLoop(_sampler_for(n_win, batch), batch)
        v1._apply_sampler_state(resumed.sampler, state)
        resumed.epoch = ep
        resumed.batch_in_epoch = bie
        second = resumed.steps(steps - first_steps)
        segmented = first + second
        if segmented != continuous:
            n_bad = sum(1 for a, b in zip(segmented, continuous) if a != b)
            at = next(i for i, (a, b) in enumerate(zip(segmented, continuous)) if a != b)
            raise AssertionError(
                "%s boundary: segmented order diverges at sample %d (%d of %d differ); resume "
                "would repeat or skip windows" % (label, at, n_bad, len(continuous)))
        cases[label] = {"first_steps": int(first_steps), "epoch": ep, "batch_in_epoch": bie}

    # Negative control: without the published offset (pre-r4 behaviour) the same
    # window range is served twice.
    seg = _SimLoop(_sampler_for(n_win, batch), batch)
    first = seg.steps(half)
    state, ep, bie = snapshot(seg)
    lost = dict(state)
    lost["resume_batch_offset"] = 0
    bad = _SimLoop(_sampler_for(n_win, batch), batch)
    v1._apply_sampler_state(bad.sampler, lost)
    bad.epoch = ep
    tail = bad.steps(200)
    overlap = len(set(tail).intersection(set(first)))
    if overlap == 0:
        raise AssertionError("the negative control showed no repetition: the test is vacuous, "
                            "i.e. the resume offset is not what makes the sequence continuous")
    ok("R9_sampler_continuity",
       "%d windows x %d steps: mid-pass and pass-end boundaries both reproduce the uninterrupted "
       "order exactly; without the offset %d of %d windows repeat"
       % (n_win, steps, overlap, len(tail)),
       {"n_windows": n_win, "cases": cases, "control_overlap": overlap,
        "batches_per_pass": math.ceil(n_win / batch)})


# ----------------------------------------------------------------------------- R10
def _make_sched(opt, cfg, total_steps):
    from fasterwam.trainer import Wan22Trainer

    host = type("_SchedHost", (), {"optimizer": opt,
                                   "learning_rate": float(cfg.learning_rate)})()
    return Wan22Trainer._build_scheduler(host, str(cfg.lr_scheduler_type),
                                         int(total_steps),
                                         warmup_steps=int(round(total_steps * 0.05)))


def _tiny_model():
    m = nn.Module()
    m.a = nn.Linear(4, 4)
    return m


def r10_lr_continuity(cfg, ctx):
    """Seven resumes through the real v3 light state must keep ONE 40k cosine."""
    from fasterwam import light_state_v3 as v3

    lr_peak = float(cfg.learning_rate)
    total = TOTAL_STEPS
    if int(cfg.max_steps) != total:
        raise AssertionError("cfg.max_steps=%s is not the single %d-step horizon"
                             % (cfg.max_steps, total))
    if int(cfg.save_every) != SEGMENT:
        raise AssertionError("cfg.save_every=%s != %d" % (cfg.save_every, SEGMENT))
    if int(cfg.gradient_accumulation_steps) != 1:
        raise AssertionError(
            "gradient_accumulation_steps=%s: this test steps the scheduler once per optimizer "
            "step, which is only the training-loop semantics when accumulation is 1"
            % cfg.gradient_accumulation_steps)
    warm = int(round(total * 0.05))
    bounds = list(range(SEGMENT, total + 1, SEGMENT))

    def fresh_opt():
        m = _tiny_model()
        return m, torch.optim.AdamW(m.parameters(), lr=lr_peak, betas=(0.9, 0.95))

    class _Sampler:
        seed = 42
        epoch = 0
        epoch_offset = 0
        resume_batch_offset = 0
        sample_offset = 0
        batch_size = 6
        num_processes = 1

    model, opt = fresh_opt()
    sched = _make_sched(opt, cfg, total)
    continuous = []
    for _ in range(total):
        continuous.append(float(opt.param_groups[0]["lr"]))
        opt.step()
        sched.step()
    final_lr = float(opt.param_groups[0]["lr"])

    model2, opt2 = fresh_opt()
    sched2 = _make_sched(opt2, cfg, total)
    segmented = []
    step = 0
    with tempfile.TemporaryDirectory() as td:
        for seg_end in bounds:
            while step < seg_end:
                segmented.append(float(opt2.param_groups[0]["lr"]))
                opt2.step()
                sched2.step()
                step += 1
            if seg_end == bounds[-1]:
                break
            d = Path(td) / ("step_%06d" % seg_end)
            d.mkdir(parents=True)
            v3.save_light_state(d, model=model2, optimizer=opt2, scheduler=sched2,
                                sampler=_Sampler(), global_step=step, epoch=step // 100,
                                batch_in_epoch=step % 100)
            # Fresh process: rebuild from the config, then resume on top of it.
            model3, opt3 = fresh_opt()
            sched3 = _make_sched(opt3, cfg, total)
            state = v3.load_light_state(d, model=model3, optimizer=opt3, scheduler=sched3,
                                        sampler=_Sampler())
            if int(state["global_step"]) != seg_end:
                raise AssertionError("light state resumed at %s, expected %d"
                                     % (state["global_step"], seg_end))
            # The saved optimizer has already advanced past step `seg_end`, so its
            # group LR is the one the uninterrupted run uses for step `seg_end` --
            # comparing against `continuous[seg_end - 1]` would fail by one step.
            if abs(float(opt3.param_groups[0]["lr"]) - continuous[seg_end]) > 0.0:
                raise AssertionError("optimizer LR after resume is not the saved LR at step %d"
                                     % seg_end)
            model2, opt2, sched2 = model3, opt3, sched3

    if len(segmented) != len(continuous):
        raise AssertionError("segmented curve has %d points, continuous has %d"
                             % (len(segmented), len(continuous)))
    diffs = np.abs(np.asarray(segmented) - np.asarray(continuous))
    if float(diffs.max()) > 0.0:
        at = int(np.argmax(diffs))
        raise AssertionError("segmented LR diverges at step %d: %.9e vs %.9e -- the schedule "
                             "restarted instead of continuing" % (at + 1, segmented[at],
                                                                  continuous[at]))
    if not all(continuous[i] <= continuous[i - 1] + 1e-18 for i in range(warm + 1, total)):
        raise AssertionError("the post-warmup cosine is not monotonically decreasing")
    if continuous[0] >= lr_peak or continuous[warm - 1] > lr_peak + 1e-12:
        raise AssertionError("warmup does not climb to the peak LR: %r %r"
                             % (continuous[0], continuous[warm - 1]))
    if not math.isclose(final_lr, lr_peak * 0.01, rel_tol=1e-6):
        raise AssertionError("the 40k cosine ends at %.3e, expected eta_min %.3e"
                             % (final_lr, lr_peak * 0.01))

    # Negative control: the r5 defect -- a cosine rebuilt per segment.
    restart = []
    for _ in bounds:
        m, o = fresh_opt()
        s = _make_sched(o, cfg, SEGMENT)
        for _ in range(SEGMENT):
            restart.append(float(o.param_groups[0]["lr"]))
            o.step()
            s.step()
    worst = float(np.abs(np.asarray(restart[:total]) - np.asarray(continuous)).max())
    if worst < 1e-6:
        raise AssertionError("a per-segment cosine is indistinguishable from the 40k cosine: "
                             "the test cannot see the r5 defect")
    climbed = sum(1 for i in range(1, total) if restart[i] > restart[i - 1] + 1e-12)
    ok("R10_lr_continuity",
       "%d steps in %d segments bitwise equal to one continuous cosine (end %.3e = eta_min); "
       "a per-segment cosine would differ by up to %.3e and climb %d times"
       % (total, len(bounds), final_lr, worst, climbed),
       {"lr_at_5k": continuous[SEGMENT - 1], "lr_at_40k": final_lr,
        "warmup_steps": warm, "restart_curve_worst_diff": worst})


# ---------------------------------------------------------------------------- main
CHECKS = {key: fn for key, fn in [
    ("R1", r1_corpus),
    ("R2", r2_leak_gate),
    ("R2b", r2b_proprio_units),
    ("R3", r3_zero_init_equivalence),
    ("R4", r4_gradient_ladder),
    ("R5", r5_joint_flow_target),
    ("R6", r6_calibration),
    ("R7", r7_units_consistency),
    ("R8", r8_light_state_round_trip),
    ("R9", r9_sampler_continuity),
    ("R10", r10_lr_continuity),
]}
ORDER = ["R6", "R1", "R2b", "R2", "R3", "R4", "R5", "R7", "R8", "R9", "R10"]


def run_all(only=None, report=None):
    ctx = {}
    cfg = load_cfg()
    selected = ORDER if not only else [k for k in ORDER if k in set(only)]
    if not selected:
        raise SystemExit("no checks selected by %s; known names: %s" % (only, sorted(CHECKS)))
    failed = []
    for key in selected:
        try:
            CHECKS[key](cfg, ctx)
        except SystemExit:
            raise
        except Exception as exc:
            failed.append(key)
            print("FAIL %s: %s" % (key, exc), flush=True)
            traceback.print_exc()
    payload = {"passed": RESULTS, "failed": failed, "notes": NOTES}
    if report:
        out = Path(report)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(out) + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, out)
    if failed:
        raise SystemExit("Tier-2 checks FAILED: %s" % failed)
    print("Tier-2 complete: %d checks passed" % len(RESULTS), flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="comma-separated check names, e.g. R2,R2b,R9 (default: all)")
    ap.add_argument("--report", default=None, help="write the JSON result here")
    args = ap.parse_args()
    only = [t.strip() for t in args.only.split(",") if t.strip()] if args.only else None
    return run_all(only=only, report=args.report)


if __name__ == "__main__":
    sys.exit(main())
