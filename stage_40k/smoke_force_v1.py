"""Tier-1 admission tests for the V1 force conditioning -- small tensors, CPU ok.

Run on the instance:
    python smoke_force_v1.py            # tier 1 only (seconds)
    python smoke_force_v1.py --real     # also loads the real dataset + model (needs GPU)

These are the falsifiable version of the claims in docs_v1_force_freeze.md:

  T1  zero-init equivalence ........ the force path cannot change the action
      tokens on step 0, bitwise, in both injection points.
  T2  gradient ladder ............... under the joint loss (action + force) the
      first backward moves only `inject.out_proj` and the head's last layer, and
      everything behind them is exactly zero; three steps later every force
      tensor moves. Force loss alone is the negative control: the gate stays at
      zero gradient, because the head's output layer is zero-init as well.
  T3  mask polarity ................. `loss_force` averages exactly the observed
      steps, so a truncated window must change the value and a fully observed one
      must equal the plain MSE.
  T4  leak gate ..................... passes on a correct window, raises when the
      labels are shifted by one frame (negative control).
  T5  geometry ...................... mismatched force_dim / horizon / hidden_dim
      are rejected at build time, not in some forward later.
  T6  window math ................... dataset padding matches the encoder's
      mask convention (history = trailing valid suffix, future = leading prefix).

Exit code 0 means every check passed; a failure prints the check name and raises.
Nothing here is caught and swallowed.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
import torch.nn as nn

RESULTS: dict[str, str] = {}


def ok(name, detail=""):
    RESULTS[name] = detail or "pass"
    print("PASS %s %s" % (name, detail))


def _force_cfg(force_dim=6, hist_len=16, horizon=32, hidden=32, heads=4, a_hidden=64):
    return {
        "enabled": True,
        "channels": [0, 1, 2, 12, 13, 16][:force_dim],
        "history_encoder": {"force_dim": force_dim, "hidden_dim": hidden,
                            "hist_len": hist_len, "num_layers": 2, "num_heads": heads},
        "future_encoder": {"force_dim": force_dim, "hidden_dim": hidden, "horizon": horizon},
        "inject": {"action_hidden_dim": a_hidden, "force_hidden_dim": hidden, "num_heads": 4},
        "force_head": {"action_hidden_dim": a_hidden, "force_dim": force_dim,
                       "horizon": horizon, "hidden_dim": 64},
    }


def _tiny_action_dit(force_config=None, seed=0):
    from fasterwam.models.wan22.sparse_action_dit import SparseActionDiT

    torch.manual_seed(seed)
    return SparseActionDiT(
        action_dim=7, hidden_dim=64, ffn_dim=128, num_layers=2, num_heads=4,
        noncondition_num_heads=2, attn_head_dim=16, text_dim=32, freq_dim=8,
        eps=1.0e-6, condition_layers=[0], use_gradient_checkpointing=False,
        force_config=force_config,
    )


def _set_calibration(expert, force_dim=6, hist_len=16):
    enc = expert.force_conditioner["history_encoder"]
    mean = torch.tensor([0.5, -1.2, 3.0, 0.1, 0.2, 0.05][:force_dim])
    std = torch.tensor([1.7, 2.1, 3.3, 0.9, 1.1, 0.7][:force_dim])
    enc.set_norm_stats(mean, std, scale=1.0)
    return enc


# --------------------------------------------------------------------------- T1
def test_zero_init_equivalence():
    torch.manual_seed(1234)
    plain = _tiny_action_dit(None)
    forced = _tiny_action_dit(_force_cfg())
    missing, unexpected = [], None
    # Copy the plain expert into the forced one: every plain key must exist.
    src = plain.state_dict()
    dst = forced.state_dict()
    absent = [k for k in src if k not in dst]
    if absent:
        raise AssertionError("forced expert lost plain keys: %s" % absent[:5])
    forced.load_state_dict(src, strict=False)
    _set_calibration(forced)

    B, T = 2, 32
    noisy_action = torch.randn(B, T, 7)
    timestep = torch.rand(B) * 900.0
    context = torch.randn(B, 5, 32)
    context_mask = torch.ones(B, 5, dtype=torch.bool)
    force_history = torch.randn(B, 16, 6)
    force_history_mask = torch.ones(B, 16, dtype=torch.bool)
    noisy_force = torch.randn(B, T, 6)

    with torch.no_grad():
        pre_plain = plain.pre_dit(noisy_action, timestep, context, context_mask)
        pre_forced = forced.pre_dit(
            noisy_action, timestep, context, context_mask,
            force_history=force_history, force_history_mask=force_history_mask,
            noisy_force_future=noisy_force, force_future_mask=torch.ones(B, T, dtype=torch.bool),
        )
    if not torch.equal(pre_plain["tokens"], pre_forced["tokens"]):
        d = (pre_plain["tokens"] - pre_forced["tokens"]).abs().max().item()
        raise AssertionError("step-0 action tokens differ, max abs diff %r" % d)

    # Second injection point, after the blocks, must also be a no-op at init.
    tokens_out = torch.randn(B, T, 64)
    with torch.no_grad():
        again = forced.apply_force_injection_post_dit(tokens_out, pre_forced)
    if not torch.equal(again, tokens_out):
        raise AssertionError("post-DiT injection perturbed tokens at zero init")

    # ...and it must NOT stay a no-op once out_proj is non-zero, otherwise T1
    # would be proving that the module is dead rather than zero-initialised.
    with torch.no_grad():
        forced.force_conditioner["inject"].attn.out_proj.weight.normal_(0, 0.02)
        moved = forced.apply_force_injection_post_dit(tokens_out, pre_forced)
    if torch.equal(moved, tokens_out):
        raise AssertionError("injection still a no-op with non-zero out_proj: dead module")
    ok("T1_zero_init_equivalence", "both injection points bitwise equal at init, live after")


# --------------------------------------------------------------------------- T2
def _joint_losses(expert, action_tokens, timestep, context, context_mask, force,
                  action_target, force_target, action_weight=1.0):
    """One graph over the real forward path, shaped like `training_loss`.

    The loop is `SparseActionDiT.forward` plus the two head-only calls that
    `fastwam.training_loss` makes around it (fastwam.py:828), because the force
    head reads the same post-block tokens the action head consumed.

    `action_weight` exists for the negative control: the injection gate
    (`inject.attn.out_proj`) and the force head's final layer are *both*
    zero-init, so the force loss cannot put gradient on the gate -- only the
    action term, which backprops through the injection into the pretrained
    trunk, can. A run whose action loss is detached would therefore freeze the
    whole history encoder forever, and this helper is how that is caught.
    """
    import torch.nn.functional as F

    pre = expert.pre_dit(action_tokens, timestep, context, context_mask, **force)
    x = pre["tokens"]
    for block in expert.blocks:
        x = block(x, pre["context"], pre["t_mod"], pre["freqs"],
                  context_mask=pre["context_mask"])
    vel = expert.post_dit(x, pre)
    force_tokens = expert.apply_force_injection_post_dit(x, pre)
    pred = expert.predict_force_velocity(force_tokens, pre)
    loss_action = F.mse_loss(vel, action_target)
    loss_force = F.mse_loss(pred, force_target)
    return action_weight * loss_action + loss_force


def test_gradient_ladder():
    torch.manual_seed(7)
    expert = _tiny_action_dit(_force_cfg())
    _set_calibration(expert)
    B, T = 2, 32
    noisy_action = torch.randn(B, T, 7)
    timestep = torch.rand(B) * 900.0
    context = torch.randn(B, 5, 32)
    context_mask = torch.ones(B, 5, dtype=torch.bool)
    force = {
        "force_history": torch.randn(B, 16, 6),
        "force_history_mask": torch.ones(B, 16, dtype=torch.bool),
        "noisy_force_future": torch.randn(B, T, 6),
        "force_future_mask": torch.ones(B, T, dtype=torch.bool),
    }
    action_target = torch.randn(B, T, 7)
    force_target = torch.randn(B, T, 6)

    def backward_once(action_weight):
        expert.zero_grad(set_to_none=True)
        loss = _joint_losses(expert, noisy_action, timestep, context, context_mask, force,
                             action_target, force_target, action_weight)
        loss.backward()
        return {n: p.grad for n, p in expert.named_parameters()}, loss

    def absmax(name, grads):
        gr = grads.get(name)
        if gr is None:
            raise AssertionError("%s is not in the graph at all" % name)
        if not torch.isfinite(gr).all():
            raise AssertionError("%s gradient is not finite" % name)
        return float(gr.abs().max())

    grads, loss = backward_once(1.0)
    if not torch.isfinite(loss):
        raise AssertionError("joint loss is not finite at init")
    gate = absmax("force_conditioner.inject.attn.out_proj.weight", grads)
    head_last = absmax("force_conditioner.force_head.linear.2.weight", grads)
    if gate == 0.0:
        raise AssertionError("the injection gate has no gradient under the joint loss")
    if head_last == 0.0:
        raise AssertionError("the force head output layer has no gradient")

    # The exhaustive form of the claim, so it cannot be dodged by a tensor this
    # list forgets: outside the two zero-init output layers the gradient must be
    # *exactly* zero on step 0, and inside them it must be present everywhere.
    force_names = sorted(n for n in grads if n.startswith("force_conditioner."))
    allowed0 = ("force_conditioner.inject.attn.out_proj",
                "force_conditioner.force_head.linear.2")
    moved, bad_nonzero = [], []
    for name in force_names:
        got = absmax(name, grads)
        if name.startswith(allowed0):
            if got == 0.0:
                bad_nonzero.append((name, "zero"))
            else:
                moved.append(name)
        elif got != 0.0:
            bad_nonzero.append((name, "%.3e" % got))
    if bad_nonzero:
        raise AssertionError(
            "step-0 gradient does not match the zero-init gate: %s; the "
            "equivalence argument in the freeze document is wrong" % bad_nonzero[:6])

    # Negative control: with the action term removed, the gate itself is frozen,
    # which is what makes the joint loss a requirement and not a stylistic one.
    grads_force_only, _ = backward_once(0.0)
    if absmax("force_conditioner.inject.attn.out_proj.weight", grads_force_only) != 0.0:
        raise AssertionError("force-loss-only gave the gate gradient; the two-stage "
                             "zero-init argument above is not the reason it opens")

    opt = torch.optim.SGD((p for n, p in expert.named_parameters()
                           if n.startswith("force_conditioner.")), lr=1e-2)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        _joint_losses(expert, noisy_action, timestep, context, context_mask, force,
                      action_target, force_target).backward()
        opt.step()
    grads1, _ = backward_once(1.0)
    still_frozen = [n for n in force_names
                    if grads1.get(n) is None or float(grads1[n].abs().max()) == 0.0]
    if still_frozen:
        raise AssertionError("force tensors still at zero gradient after the gate opened: %s"
                             % still_frozen[:6])
    ok("T2_gradient_ladder",
       "step0: gate=%.2e head=%.2e, only %s move and the other %d force tensors are "
       "exactly zero; force-only freezes the gate (negative control); after 3 joint "
       "steps all %d force tensors have gradient"
       % (gate, head_last, sorted(moved), len(force_names) - len(moved), len(force_names)))



# --------------------------------------------------------------------------- T3
def test_mask_polarity():
    """Replicate training_loss's force reduction and pin its meaning."""
    B, T, F = 2, 32, 6
    pred = torch.randn(B, T, F)
    target = torch.randn(B, T, F)
    future_mask = torch.ones(B, T, dtype=torch.bool)
    future_mask[1, 20:] = False  # second row: only 20 observed steps

    valid = future_mask.to(pred.dtype)[:, :, None]
    steps = valid.sum(dim=(1, 2)).clamp(min=1.0) * pred.shape[-1]
    per_sample = ((pred - target) ** 2 * valid).sum(dim=(1, 2)) / steps

    full = ((pred[0] - target[0]) ** 2).mean()
    if not torch.allclose(per_sample[0], full, atol=1e-6):
        raise AssertionError("fully observed window is not the plain MSE")
    trunc = ((pred[1, :20] - target[1, :20]) ** 2).mean()
    if not torch.allclose(per_sample[1], trunc, atol=1e-6):
        raise AssertionError("masked tail still enters the mean (inverted mask)")
    ok("T3_mask_polarity", "True=observed; masked tail excluded from the mean")


# --------------------------------------------------------------------------- T4
def test_leak_gate():
    from fasterwam.models.wan22.fastwam import FastWAM

    class _Stub:
        """Just enough FastWAM to exercise the gate without a 5B model."""
        force_spec = {"horizon": 32}
        device = "cpu"
        assert_future_frames_are_actually_future = FastWAM.assert_future_frames_are_actually_future

    stub = _Stub()
    B, T, F = 2, 32, 6
    base = torch.arange(3 * T * F, dtype=torch.float32).reshape(3, T, F)
    flat = base.reshape(3 * T, F)
    offsets = [0, T]
    idx = torch.tensor(offsets)[:, None] + torch.arange(1, T + 1)[None, :]

    def retriever(rows):
        # Source read addressed by (episode, step), the way the real dataset
        # reindexes; `good` below is addressed by absolute row.
        r = torch.as_tensor(rows)
        return base[r // T, r % T].numpy()

    # Label j of a window whose current frame is row `offset` is the source row
    # `offset + j + 1`, i.e. strictly forward and never the current frame.
    good = flat[idx].numpy()
    mask = torch.ones(B, T, dtype=torch.bool)
    res = stub.assert_future_frames_are_actually_future(
        torch.from_numpy(good), mask, offsets, retriever)
    if res["compared_steps"] != B * T:
        raise AssertionError("gate compared %d steps, expected %d" % (res["compared_steps"], B * T))

    shifted = good.copy()
    shifted[:, :-1] = shifted[:, 1:]  # label at +j becomes +j+1: a real leak
    try:
        stub.assert_future_frames_are_actually_future(
            torch.from_numpy(shifted), mask, offsets, retriever)
    except AssertionError:
        pass
    else:
        raise AssertionError("leak gate accepted a one-frame-forward label shift")
    ok("T4_leak_gate", "passes aligned window, rejects +1 shift")


# --------------------------------------------------------------------------- T5
def test_geometry_guards():
    from fasterwam.models.wan22.force_condition import build_force_conditioner

    for label, mutate in [
        ("force_dim", lambda c: c["future_encoder"].__setitem__("force_dim", 5)),
        ("horizon", lambda c: c["force_head"].__setitem__("horizon", 16)),
        ("hidden_dim", lambda c: c["inject"].__setitem__("force_hidden_dim", 64)),
    ]:
        cfg = _force_cfg()
        mutate(cfg)
        try:
            build_force_conditioner(cfg)
        except ValueError:
            continue
        raise AssertionError("build accepted a mismatched %s" % label)

    # A horizon that does not equal the action token count must be refused, and
    # refused at the earliest call that can see it: `pre_dit` gathers the force
    # stream, so by the time `apply_force_injection_post_dit` would run the
    # misaligned labels are already in the graph.
    def call_pre_dit(expert, horizon):
        expert.pre_dit(
            torch.randn(B, 32, 7), torch.rand(B) * 900, torch.randn(B, 5, 32),
            torch.ones(B, 5, dtype=torch.bool),
            force_history=torch.randn(B, 8, 6),
            force_history_mask=torch.ones(B, 8, dtype=torch.bool),
            noisy_force_future=torch.randn(B, horizon, 6),
            force_future_mask=torch.ones(B, horizon, dtype=torch.bool),
        )

    B = 2
    short = _tiny_action_dit(_force_cfg(force_dim=6, hist_len=8, horizon=4))
    _set_calibration(short)
    try:
        call_pre_dit(short, 4)
    except ValueError as exc:
        if "must equal the action token count" not in str(exc):
            raise AssertionError("refused the short horizon for the wrong reason: %s" % exc)
    else:
        raise AssertionError("horizon shorter than the action token count was accepted")

    # ...and the guard is a guard, not a constant failure: the matching horizon
    # must pass through the same call.
    matched = _tiny_action_dit(_force_cfg(force_dim=6, hist_len=8, horizon=32))
    _set_calibration(matched)
    call_pre_dit(matched, 32)
    ok("T5_geometry_guards", "dim/horizon/hidden mismatches rejected at build; short "
                             "horizon rejected by pre_dit with the matching one accepted")


# --------------------------------------------------------------------------- T6
def test_window_padding():
    """The dataset's padding convention must match what the encoders assert."""
    hist_len, horizon, F = 16, 32, 6
    ep_len = 2000

    def windows(s):
        """Mirror of A3VideoDataset.force_windows (a3_video_dataset.py:246)."""
        lo = max(0, s - hist_len + 1)
        hist = np_full[lo:s + 1, :F]
        h = np.zeros((hist_len, F), dtype="float32")
        h[hist_len - hist.shape[0]:] = hist
        hmask = np.zeros(hist_len, dtype=bool)
        hmask[hist_len - hist.shape[0]:] = True
        hi = min(ep_len - 1, s + horizon)
        fut = np_full[s + 1:hi + 1, :F]
        f = np.zeros((horizon, F), dtype="float32")
        f[:fut.shape[0]] = fut
        fmask = np.zeros(horizon, dtype=bool)
        fmask[:fut.shape[0]] = True
        # The padded length is reported separately so the mask assertions below
        # cannot silently drift one frame away from the slicing.
        return h, hmask, f, fmask, hist.shape[0], fut.shape[0]

    global np_full
    np_full = np_arange((ep_len + 5) * 18).reshape(ep_len + 5, 18)

    expert = _tiny_action_dit(_force_cfg(hist_len=hist_len, horizon=horizon))
    _set_calibration(expert)
    enc_h = expert.force_conditioner["history_encoder"]
    enc_f = expert.force_conditioner["future_encoder"]

    for label, s in [("left-padded history", 3), ("both full", 100), ("right-padded future", ep_len - 5)]:
        h, hm, f, fm, n_hist, n_fut = windows(s)
        want_h = np.zeros(hist_len, dtype=bool)
        want_h[hist_len - n_hist:] = True
        want_f = np.zeros(horizon, dtype=bool)
        want_f[:n_fut] = True
        if not np.array_equal(hm, want_h):
            raise AssertionError("%s: history mask is not the trailing valid suffix" % label)
        if not np.array_equal(fm, want_f):
            raise AssertionError("%s: future mask is not the leading valid prefix" % label)
        if hm.sum() != n_hist or fm.sum() != n_fut:
            raise AssertionError("%s: mask valid count disagrees with the slice length" % label)
        # Padding must be exactly zero: the encoders treat a masked row as absent,
        # and a nonzero pad would leak the neighbouring frame's force into the mean.
        def pad_absmax(arr, mask):
            sel = arr[~mask]
            return 0.0 if sel.size == 0 else float(np.abs(sel).max())

        if pad_absmax(h, hm) != 0.0 or pad_absmax(f, fm) != 0.0:
            raise AssertionError("%s: padded rows are not zero-filled" % label)
        with torch.no_grad():
            enc_h(torch.from_numpy(h)[None].float(), torch.from_numpy(hm)[None])
            enc_f(torch.from_numpy(f)[None].float(), torch.from_numpy(fm)[None])

    # The encoder must reject the opposite convention rather than accept it quietly.
    h, hm, f, fm = windows(3)[:4]
    try:
        enc_h(torch.from_numpy(h)[None].float(), torch.from_numpy(~hm)[None])
    except ValueError:
        pass
    else:
        raise AssertionError("encoder accepted an inverted history mask")
    try:
        enc_f(torch.from_numpy(f)[None].float(), torch.from_numpy(~fm)[None])
    except ValueError:
        pass
    else:
        raise AssertionError("encoder accepted an inverted future mask")
    ok("T6_window_padding", "left-padded history / right-padded future agree with the "
                            "dataset slice at 3 window positions, inverted masks rejected")


def np_arange(n):
    import numpy
    return numpy.arange(n, dtype="float32").reshape(-1, 18)


def main():
    import numpy  # noqa: F401  (T6 needs it; keep the import local to the run)
    globals()["np"] = numpy
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="also run the dataset/model tier on GPU (smoke_force_v1_real)")
    args = ap.parse_args()

    test_zero_init_equivalence()
    test_gradient_ladder()
    test_mask_polarity()
    test_leak_gate()
    test_geometry_guards()
    test_window_padding()

    if args.real:
        from smoke_force_v1_real import run_all
        run_all()

    print(json.dumps({"passed": RESULTS}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
