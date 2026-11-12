"""Smoke one real merged node on the real A3 env, and watch what the diffusion seed becomes
inside a live `infer_action` call.

The CPU flow test can prove `noise_seed` is a good table and that `ForcePolicy` asks for one row
per replan; it cannot prove a loaded model *receives* that row. This script loads a merged
checkpoint the way `eval40k.py` loads it -- by importing that file's own loader, not by copying it
-- installs a recording proxy on `model.infer_action`, rolls the requested episodes closed-loop,
and asserts on what the proxy saw.

It writes one JSON evidence file and touches nothing else: no edit under `fw/`, no edit to a
checkpoint, no deletion. The loader it exercises is `eval40k.build_policy` itself, so what passes
here is the code the eight-node batch will run.

    cd /workspace/_v1_stage/dev/eval40k
    PYTHONDONTWRITEBYTECODE=1 python smoke40k.py --steps 40000 --seeds 11 12

Exit code 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval40k as E                                    # noqa: E402
from force_metrics import ForceErrorAccount            # noqa: E402
from force_stream import CHANNELS, HIST_LEN        # noqa: E402
from policy40k import noise_seed                       # noqa: E402
from rollout40k import run_episode                     # noqa: E402


def _as_np(x):
    """numpy of a tensor-or-array argument, wherever it lives.

    The recorder reads the model's own outputs, which are device tensors in the model dtype, so a
    plain `np.asarray` raises. No `torch` name is needed here: `detach().cpu().float()` is enough
    and keeps this file importable on a machine with no GPU in sight.
    """
    if x is None:
        return None
    if hasattr(x, "detach"):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float64)


def distinct_chunk_shapes(clog):
    """The shapes a chunk log contains, de-duplicated and sorted, as plain lists.

    The recorder stores every shape with `list(arr.shape)`, so aggregating with a set comprehension
    over those lists raises `TypeError: unhashable type: 'list'` -- after the episodes that were
    meant to be proved have already run. Tuple is the hashable form; converting back keeps the
    report indexable (`chunk_shapes[0][0]`) and JSON-clean.
    """
    return [list(s) for s in sorted({tuple(c["chunk_shape"]) for c in clog})]


def summarize_episode(rec, calls, clog, chunks):
    """Fold the two recording proxies' logs into the episode record, in place.

    Both halves of this function's contract are load-bearing: every key it reads is a key one of the
    spies (or `ForcePolicy.chunk`) really writes, and every key it writes is read by the checks that
    follow in `main`. Neither half is visible from the other file, and both have already cost a live
    smoke -- once a `TypeError` on an unhashable shape, once a `KeyError` on a gripper field that the
    recorder names differently -- each time *after* the GPU had run the episodes being proved, so the
    evidence was thrown away with the traceback. `flow_test.py` therefore drives this function with
    the real recorders over a stub model and policy.
    """
    pred = [c for c in chunks if c["pred_force"] is not None]
    rec["infer_action_calls"] = len(calls)
    rec["seeds_seen_by_model"] = [c["seed"] for c in calls]
    rec["distinct_seeds"] = len({c["seed"] for c in calls})
    rec["expected_seeds"] = [noise_seed(rec["seed"], i) for i in range(1, len(calls) + 1)]
    rec["meta_noise_seed"] = [c["meta"]["noise_seed"] for c in chunks]
    rec["chunk_log"] = clog
    rec["chunk_shapes"] = distinct_chunk_shapes(clog)
    rec["chunk_all_finite"] = all(c["all_finite"] for c in clog)
    rec["gripper_pm1_values"] = sorted({v for c in clog for v in c["gripper_unique"]})
    rec["gripper_is_pm1_binary"] = all(c["gripper_is_pm1_binary"] for c in clog)
    rec["force_shape_reported"] = None if not calls else list(calls[-1]["force_shape"])
    rec["action_shape_reported"] = None if not calls else list(calls[-1]["action_shape"])
    rec["model_action_shapes"] = sorted({str(list(c["action_shape"])) for c in calls})
    rec["action_requires_grad_any"] = any(c["action_requires_grad"] for c in calls)
    rec["proprio_min_all"] = min(c["proprio_min"] for c in calls)
    rec["proprio_max_all"] = max(c["proprio_max"] for c in calls)
    rec["proprio_base_min_all"] = min(c["proprio_base_min"] for c in calls)
    rec["proprio_base_max_all"] = max(c["proprio_base_max"] for c in calls)
    rec["proprio_force_min_all"] = min(c["proprio_force_min"] for c in calls)
    rec["proprio_force_max_all"] = max(c["proprio_force_max"] for c in calls)
    rec["proprio_widths"] = sorted({c["proprio_width"] for c in calls})
    rec["force_history_shapes"] = sorted({str(list(c["force_history_shape"])) for c in calls})
    rec["mask_true_sequence"] = [c["force_history_mask_true"] for c in calls]
    rec["num_inference_steps_seen"] = sorted({c["num_inference_steps"] for c in calls})
    rec["rand_device_seen"] = sorted({c["rand_device"] for c in calls})
    rec["action_absmax_max"] = max(float(c["meta"]["action_absmax"]) for c in chunks)
    rec["infer_s_mean"] = round(float(np.mean([c["meta"]["infer_s"] for c in chunks])), 3)
    rec["infer_s_max"] = round(float(np.max([c["meta"]["infer_s"] for c in chunks])), 3)
    rec["nonfinite_pred_force"] = int(sum(0 if np.all(np.isfinite(c["pred_force"])) else 1
                                          for c in pred))
    return rec


def install_recorder(model, log):
    """Replace `model.infer_action` with a wrapper that records and forwards, unchanged.

    `functools.wraps` is not decoration: `ForcePolicy._infer_kwargs` inspects this signature to
    decide whether to pass `num_video_frames`, so a naive `**kwargs` shim would silently drop an
    argument and change the rollout being measured.
    """
    original = model.infer_action
    params_before = list(inspect.signature(original).parameters)

    @functools.wraps(original)
    def spy(**kwargs):
        out = original(**kwargs)
        img = kwargs.get("input_image")
        prop = kwargs.get("proprio")
        fh = kwargs.get("force_history")
        fm = kwargs.get("force_history_mask")
        act = out.get("action")
        force = out.get("force_future_newton")
        pnp, fnp = _as_np(prop), _as_np(fh)
        log.append({
            "call": len(log) + 1,
            "seed": int(kwargs["seed"]),
            "rand_device": str(kwargs.get("rand_device")),
            "num_inference_steps": int(kwargs.get("num_inference_steps", -1)),
            "action_horizon": int(kwargs.get("action_horizon", -1)),
            "num_video_frames": int(kwargs.get("num_video_frames", -1)),
            "input_image_shape": None if img is None else tuple(img.shape),
            "action_shape": None if act is None else tuple(act.shape),
            "action_requires_grad": None if act is None else bool(act.requires_grad),
            "force_shape": None if force is None else tuple(force.shape),
            "force_feedback_applied": bool(out.get("force_feedback_applied", False)),
            # the three model inputs the acceptance list asks to be identical to the deployment's
            "proprio_shape": None if prop is None else tuple(prop.shape),
            "proprio_min": None if pnp is None else float(pnp.min()),
            "proprio_max": None if pnp is None else float(pnp.max()),
            # `min/max` mode maps the recorded band onto [-1, 1]; only the eight base columns are
            # expected inside it, because the force columns' recorded band is +/- residual std by
            # design and the corpus reaches ~4.9 there before the normaliser's +-5 clamp.
            "proprio_base_min": None if pnp is None else float(pnp.reshape(-1)[:E.PROPRIO_BASE_DIM].min()),
            "proprio_base_max": None if pnp is None else float(pnp.reshape(-1)[:E.PROPRIO_BASE_DIM].max()),
            "proprio_force_min": None if pnp is None else float(pnp.reshape(-1)[E.PROPRIO_BASE_DIM:].min()),
            "proprio_force_max": None if pnp is None else float(pnp.reshape(-1)[E.PROPRIO_BASE_DIM:].max()),
            "proprio_width": None if pnp is None else int(pnp.reshape(-1).shape[0]),
            "force_history_shape": None if fh is None else tuple(fh.shape),
            "force_history_mask_true": None if fm is None else int(np.asarray(fm).sum()),
        })
        return out

    model.infer_action = spy
    return original, params_before, list(inspect.signature(model.infer_action).parameters)


def install_chunk_recorder(policy, log):
    """Record the actions the policy hands the simulator, before the loop slices them.

    `ForcePolicy.chunk` is where denormalisation, the gripper rewrite and the adapter conversion
    all happen, and `run_episode` then keeps only `[:replan]` of the result. Recording the return
    value of `chunk` is therefore the only way to see the full generated chunk -- its real length,
    its action dimension and the gripper column as the deployment side would compute it -- without
    editing `policy40k.py`.
    """
    original = policy.chunk

    @functools.wraps(original)
    def spy(*a, **kw):
        actions, pred_force, meta = original(*a, **kw)
        arr = np.asarray([np.asarray(x, dtype=np.float64) for x in actions])
        log.append({"frame": int(meta["replan_from_frame"]), "noise_seed": int(meta["noise_seed"]),
                    "chunk_shape": list(arr.shape), "all_finite": bool(np.all(np.isfinite(arr))),
                    "gripper_unique": sorted(set(np.round(arr[:, -1], 6).tolist()))[-2:],
                    "gripper_is_pm1_binary": bool(np.all(np.isin(np.round(arr[:, -1], 6),
                                                                [-1.0, 1.0])))})
        return actions, pred_force, meta

    policy.chunk = spy
    return original


def _overlap(xa, xb):
    """How much of one flat draw another repeats.

    `common_prefix_len` is the number of leading elements that are equal: a long prefix means the
    two streams are reading the same Philox sequence, which is the thing item 5 asks about. The
    full-range count is reported next to it so a longer stream cannot hide a late divergence.
    """
    n = min(int(xa.numel()), int(xb.numel()))
    eq = (xa[:n] == xb[:n])
    diff = (~eq).nonzero().flatten()
    return {"compared_elements": n, "identical_elements": int(eq.sum()),
            "common_prefix_len": n if diff.numel() == 0 else int(diff[0]),
            "identical_fraction": round(int(eq.sum()) / n, 6)}


def three_stream_noise(model, seen, seed, force_horizon, force_dim):
    """Reproduce the initial latents of all three streams for one seed, on the live device.

    The expressions are copied from the frozen source -- `jointwam.py:183-196` for video and
    action, `fastwam.py:935-941` for force -- and the shapes come from the model and from what a
    real call actually passed, so this is a measurement of the same arithmetic, not instrumentation
    inside `fw/`. It answers one question: does one `seed` value give the three streams the same
    noise stream, i.e. is one stream's initial noise a prefix of another's.
    """
    import torch

    call = seen[-1]
    height, width = call["input_image_shape"][2], call["input_image_shape"][3]
    vae = model.vae
    latent_t = (call["num_video_frames"] - 1) // vae.temporal_downsample_factor + 1
    latent_h = height // vae.upsampling_factor
    latent_w = width // vae.upsampling_factor
    shapes = {
        "video": (1, vae.model.z_dim, latent_t, latent_h, latent_w),
        "action": (1, call["action_horizon"], int(model.action_expert.action_dim)),
        "force": (1, force_horizon, force_dim),
    }

    def draw(name, s):
        gen = torch.Generator(device=call["rand_device"]).manual_seed(int(s))
        return torch.randn(shapes[name], generator=gen, device=call["rand_device"],
                           dtype=torch.float32)

    out = {"device": call["rand_device"], "seed": int(seed), "shapes": {}}
    for name, shape in shapes.items():
        flat = draw(name, seed).flatten()
        out["shapes"][name] = list(shape)
        out.setdefault("flat_len", {})[name] = int(flat.numel())
        out.setdefault("first_6", {})[name] = [round(float(v), 6) for v in flat[:6]]
        out[name + "_sha256_64"] = hashlib.sha256(
            np.asarray(flat[:64].tolist(), dtype=np.float64).tobytes()).hexdigest()[:16]
    for a, b in (("video", "action"), ("video", "force"), ("action", "force")):
        out[f"{a}_vs_{b}"] = _overlap(draw(a, seed).flatten(), draw(b, seed).flatten())
    independent = {}
    stream_offset = {"video": 1, "action": 2, "force": 3}   # fixed, so two runs agree
    for name in shapes:                          # the post-40k proposal, measured the same way
        independent[name] = draw(name, noise_seed(seed, stream_offset[name])).flatten()
    out["independent_seed_variant"] = {
        "seeds": {n_: noise_seed(seed, stream_offset[n_]) for n_ in shapes},
        "note": "three distinct seeds, same shapes and the same frozen expressions: this is what "
                "an adapter that passes one seed per stream would produce, measured the same way",
        **{f"{a}_vs_{b}": _overlap(independent[a], independent[b])
           for a, b in (("video", "action"), ("video", "force"), ("action", "force"))},
    }
    return out


def preprocess_with_deployment_funcs(env_factory, W, policy, calib_facts, seed, args):
    """Compare, on one real observation, what the harness hands `infer_action` with what the
    deployment worker would hand it.

    `infer_a3_worker.main` wires a deployment as `instantiate(cfg.data.train.processor)` +
    `set_normalizer_from_stats` and then builds its arguments in `predict_action_chunk`. So the
    check is not "does our processor look plausible" but "does it produce the same numbers through
    the worker's own functions": image, proprio, action horizon, video-frame count, inference
    steps and prompt are each computed both ways and compared.

    Two things about the proprio are worth stating rather than assuming, because both were found
    by running this and neither is visible in the source of either side:

    * the normaliser's `min/max` mode maps onto `[-1, 1]`, not `[0, 1]`
      (`normalizer.py:94-95,115-120`: `output_min=-1`, `output_max=1`, then `clamp(-5, 5)`), so a
      value inside the recorded band comes out between -1 and 1 and a double application pushes it
      to roughly three times as far out -- which is what this comparison catches;
    * the six force columns are *not* inside a data range. `build_v1_dataset.py:144-145` records
      their band as `+/- residual_std` on purpose, so min/max mode computes
      `residual / residual_std` for them, and the corpus itself reaches about -1.9 to +4.9. Those
      columns are therefore compared against that analytic quantity, and only the eight base
      columns are required to sit inside `[-1, 1]`.

    The deployment side of the comparison is the worker's own `normalize_proprio` applied to the
    output of the shared `eval40k.build_proprio14` (worker's 8 columns + tared force columns), so
    what is being compared is the harness's policy path against the worker functions, not two
    copies of one builder.
    """
    import torch

    env = env_factory.make_env(position_noise_m=args.position_noise, yaw_noise_rad=args.yaw_noise,
                               render_cameras=True)
    try:
        obs, _info = env_factory.reset(env, seed, randomize_cookies=True,
                                       randomize_boxes=args.randomize_boxes,
                                       batch_offset_m=args.batch_offset)
        calib = calib_facts
        p14_deploy = E.build_proprio14(obs, W, calib)
        proprio = W.normalize_proprio(p14_deploy, policy.processor)
        base8 = np.asarray(W.build_proprio(obs), dtype=np.float64).reshape(-1)
        image = W.prepare_fasterwam_images_official(obs)[1]
        worker_image = torch.from_numpy(np.asarray(image)).to(
            device=policy.device, dtype=policy.model.torch_dtype)

        # the same observation as the *first* replan of a fresh episode would see it
        policy.stream.reset()
        policy.observe(obs, 0)
        history, mask = policy.stream.snapshot()
        mine = policy._infer_kwargs(obs, history, mask, noise_seed(seed, 1))
        cfg = policy.cfg
        worker_frames = (int(cfg.data.train.num_frames) - 1) \
            // int(cfg.data.train.action_video_freq_ratio) + 1
        worker_steps = (int(cfg.get("eval_num_inference_steps", 20))
                        if cfg.EVALUATION.get("num_inference_steps", None) is None
                        else int(cfg.EVALUATION.get("num_inference_steps")))
        pnp = _as_np(proprio).reshape(-1)
        mpnp = _as_np(mine["proprio"]).reshape(-1)
        std = np.asarray(calib["std"], dtype=np.float64)
        # The deployment normaliser is `x*scale + offset` then `clamp(-5, 5)`
        # (`normalizer.py:127-130`), and for the force columns scale is 1/std with offset 0 because
        # the stats band is +/- that same std. Measured on the corpus (2026-11-12): the analytic
        # value and the processor's output agree to 3.5e-07 -- float32 epsilon -- everywhere, and
        # 620 of 256,086 force elements (620 rows, 1.45 %) really do sit past the clamp, the most
        # extreme at -7.475. So the contract this harness asserts includes the clamp; dropping it
        # would be asserting a formula the code does not implement.
        analytic_force = np.clip(p14_deploy.astype(np.float64)[E.PROPRIO_BASE_DIM:] / std,
                                 -5.0, 5.0)
        force_clamped_here = bool(np.any(np.abs(p14_deploy.astype(np.float64)[E.PROPRIO_BASE_DIM:]
                                                / std) > 5.0))
        # The band comes from `check_stats_width`, which already refused to run if the file had no
        # state band at all. Reading it by index rather than by a guessed key name, and refusing a
        # file that describes more than one state field, keeps this from degrading into `[]`.
        bands = E.STATS_FACTS.get("bands") or {}
        if len(bands) != 1:
            raise SystemExit(
                "expected exactly one state band from %s, found %d (%s): the processor looks up "
                "one shape_meta state key, so a file describing several cannot be reported here "
                "without saying which one normalised the proprio"
                % (E.STATS_FACTS.get("source"), len(bands), sorted(bands)))
        band_key = sorted(bands)[0]
        lo = np.asarray(bands[band_key]["min"], dtype=np.float64)
        hi = np.asarray(bands[band_key]["max"], dtype=np.float64)
        return {"proprio_shape": list(np.asarray(proprio).shape),
                "proprio_finite": bool(np.all(np.isfinite(pnp))),
                "proprio_min": float(pnp.min()), "proprio_max": float(pnp.max()),
                "proprio_base_min": float(pnp[:E.PROPRIO_BASE_DIM].min()),
                "proprio_base_max": float(pnp[:E.PROPRIO_BASE_DIM].max()),
                "proprio_force_min": float(pnp[E.PROPRIO_BASE_DIM:].min()),
                "proprio_force_max": float(pnp[E.PROPRIO_BASE_DIM:].max()),
                "proprio_force_columns": pnp[E.PROPRIO_BASE_DIM:].tolist(),
                "proprio_force_equals_residual_over_std":
                    bool(np.allclose(pnp[E.PROPRIO_BASE_DIM:], analytic_force, atol=2e-4,
                                     rtol=0.0)),
                "proprio_force_analytic": analytic_force.tolist(),
                "proprio_force_at_the_clamp": force_clamped_here,
                "base8_from_worker_build_proprio": base8.tolist(),
                "base8_bit_identical_to_14_wide_builder":
                    bool(np.array_equal(base8.astype(np.float32),
                                        p14_deploy.astype(np.float32)[:E.PROPRIO_BASE_DIM])),
                "proprio14_raw": [float(x) for x in p14_deploy.tolist()],
                "image_shape": list(np.asarray(image).shape),
                "proprio_identical_to_deployment_call": bool(np.array_equal(mpnp, pnp)),
                "image_identical_to_deployment_call":
                    bool(np.array_equal(_as_np(mine["input_image"]), _as_np(worker_image))),
                "harness_action_horizon": int(mine["action_horizon"]),
                "deployment_action_horizon": int(cfg.data.train.num_frames) - 1,
                "harness_num_video_frames": int(mine.get("num_video_frames", -1)),
                "deployment_num_video_frames": int(worker_frames),
                "harness_num_inference_steps": int(mine["num_inference_steps"]),
                "deployment_num_inference_steps": int(worker_steps),
                "harness_prompt": str(mine["prompt"]),
                "deployment_default_prompt": str(W.DEFAULT_PROMPT.format(task=W.INSTRUCTION)),
                "prompt_uses_deployment_template":
                    bool(mine["prompt"] == W.DEFAULT_PROMPT.format(task=policy.instruction)),
                "force_history_shape": list(np.asarray(mine["force_history"]).shape),
                "force_history_mask_shape": list(np.asarray(mine["force_history_mask"]).shape),
                "force_history_mask_true": int(np.asarray(mine["force_history_mask"]).sum()),
                "force_channels": list(np.asarray(history).shape)[-1:],
                "stats_source": E.STATS_FACTS.get("source"),
                "stats_state_width": E.STATS_FACTS.get("widths"),
                "normaliser_output_band": [-1.0, 1.0],
                "stats_band_key": band_key,
                "recorded_base_column_band": [float(lo[:E.PROPRIO_BASE_DIM].min()),
                                              float(hi[:E.PROPRIO_BASE_DIM].max())],
                "recorded_force_column_band": [[float(x) for x in lo[E.PROPRIO_BASE_DIM:]],
                                               [float(x) for x in hi[E.PROPRIO_BASE_DIM:]]]}
    finally:
        if hasattr(env, "close"):
            env.close()


def cross_prompt(policy, W):
    """The exact string `policy40k.ForcePolicy._infer_kwargs` will hand `infer_action`.

    It is built from the worker's own `DEFAULT_PROMPT` and the policy's resolved instruction, so
    comparing it against the training prompt is a comparison of what the model is about to be fed,
    not of two copies of a template.
    """
    return W.DEFAULT_PROMPT.format(task=policy.instruction)


def corpus_proprio14_proof():
    """Prove the 14-wide assembly against the corpus, before a model or a GPU is touched.

    The claim being tested is arithmetic, not behaviour, and the corpus states it directly:
    `a3_fw_force/arrays.npz` stores the gathered channels as `force6` and `a3_fw_v1g/arrays.npz`
    stores the 14-wide `proprio` the training loader actually read. So `build_proprio14` is right
    for one observation only if, over every one of the 42,681 recorded rows,

      * `force18[:, [0, 1, 2, 12, 13, 16]]` is `force6` (the channel gather is the frozen one), and
        the frozen gather `force_stream.select_channels` applied row by row is that same gather,
      * `force6 - zero_load_ref_median`, cast to float32 once, is `proprio14[:, 8:14]` bit-for-bit
        (the state force columns are exactly the tare the builder applied),
      * `proprio14[:, 0:8]` is the 8-wide proprio that lives next to the force arrays, i.e. the
        same eight columns `infer_a3_worker.build_proprio` is written to produce.

    The second bullet is the one that had been aimed at the wrong array. `force6_zeroed`, the third
    array in the 18-channel table, is a separately-defined quantity: measured over the whole corpus
    it is *not* `force6 - zero_load_ref_median` (per-channel max |difference| ~0.06 N, and no row is
    untouched), so a proprio rebuilt from it would be off the sensor preload. It is reported here,
    with its deviation, and asserted against nothing -- the state columns are compared with the
    tare the calibration file actually names.

    That last comparison is against the stored 8-wide array rather than against the live worker,
    because the worker needs a simulator observation; the live-side equality is checked separately
    by `preprocess_with_deployment_funcs`. Nothing here loads a checkpoint.
    """
    import json

    from force_stream import select_channels

    calib = json.loads(Path(str(E.CALIB_PATH)).read_text())
    stats = json.loads(Path(str(E.V1G_STATS)).read_text())
    zero = np.asarray(calib["zero_load_ref_median"], dtype=np.float64)
    std = np.asarray(calib["std"], dtype=np.float64)
    v1g = np.load(str(E.V1G_ARRAYS))
    frc = np.load(str(E.FORCE_ARRAYS))
    p14 = v1g["proprio"]
    base8_stored = frc["proprio"]
    f18, f6, f6z = frc["force18"], frc["force6"], frc["force6_zeroed"]

    n_rows = int(p14.shape[0])
    probed_rows = np.unique(np.concatenate([
        np.arange(0, min(8, n_rows)),                    # the very first windows
        np.arange(max(0, n_rows - 8), n_rows),           # and the last ones
        np.linspace(0, n_rows - 1, 64).astype(np.int64),  # spread across the corpus
        np.random.default_rng(20261112).integers(0, n_rows, 512)]))  # deterministic sample
    gathered = np.asarray(f18[:, list(CHANNELS)])        # every row, in the frozen order
    probed = np.asarray([select_channels(f18[i])         # the real function, called as the live
                         for i in probed_rows])          # path calls it: one 18-vector at a time
    tared = (f6.astype(np.float64) - zero).astype(np.float32)   # exactly what build_proprio14 does
    state_force = p14[:, E.PROPRIO_BASE_DIM:]
    zf = np.abs(state_force.astype(np.float64) - f6z.astype(np.float64))
    force_diff = float(np.abs(state_force.astype(np.float64) - tared.astype(np.float64)).max())
    base_diff = float(np.abs(p14[:, :E.PROPRIO_BASE_DIM].astype(np.float64)
                             - base8_stored.astype(np.float64)).max())
    norm_force = state_force.astype(np.float64) / std
    return {
        "v1g_proprio_shape": list(p14.shape), "force_arrays_shape": list(f18.shape),
        "rows": n_rows,
        "gathered_channels_equal_force6": bool(np.array_equal(gathered, f6)),
        "gathered_channels_bit_equal_frac": float((gathered == f6).mean()),
        # `select_channels` flattens its input (`force_stream.py:76`, `.reshape(-1)`), so it is a
        # single-observation function -- feeding it the whole (N, 18) table returns six numbers from
        # row 0 and says nothing about the corpus. The row-wise calls below are what the live path
        # actually does; the vectorised gather is what covers all 42,681 rows.
        "select_channels_is_a_single_vector_function":
            bool(select_channels(f18[0]).shape == (len(CHANNELS),)
                 and select_channels(f18[:2]).shape == (len(CHANNELS),)),
        "select_channels_rowwise_equals_vectorised_gather":
            bool(np.array_equal(probed, gathered[probed_rows])),
        "rows_probed_rowwise": int(len(probed_rows)),
        "state_force_columns_equal_tared_channels": force_diff == 0.0,
        "state_force_columns_max_abs_diff": force_diff,
        "state_force_columns_bit_equal_frac": float((state_force == tared).mean()),
        "state_base_columns_equal_stored_8_wide_proprio": base_diff == 0.0,
        "state_base_columns_max_abs_diff": base_diff,
        "base8_width": int(base8_stored.shape[1]),
        # `force6_zeroed` is reported, never asserted: it is a different derived quantity in the
        # 18-channel table, not the tare the state columns use. Keeping the numbers here is what
        # stops the next reader from rebuilding the proprio out of it.
        "force6_zeroed_vs_tared_max_abs_diff": float(zf.max()),
        "force6_zeroed_vs_tared_per_col_max_abs_diff": [float(x) for x in zf.max(0)],
        "force6_zeroed_vs_tared_elements_equal": int((zf == 0.0).sum()),
        "force6_zeroed_vs_tared_of": int(zf.size),
        "force6_zeroed_vs_tared_rows_differing": int((zf > 0.0).any(axis=1).sum()),
        "force_column_range": [[float(x) for x in state_force.min(0)],
                               [float(x) for x in state_force.max(0)]],
        "normalised_force_column_range": [[float(x) for x in norm_force.min(0)],
                                          [float(x) for x in norm_force.max(0)]],
        "zero_load_ref_median": [float(x) for x in zero],
        "calib_version": calib["version"],
        # read straight off the stats file, so this holds before any processor exists
        "stats_state_width": len(stats["state"]["default"]["global_min"]),
        "stats_force_band_is_symmetric": all(
            abs(stats["state"]["default"]["global_min"][i]
                + stats["state"]["default"]["global_max"][i]) < 1e-9
            for i in range(E.PROPRIO_BASE_DIM, E.PROPRIO_DIM)),
        # the two files have to agree or the force columns are z-scored by a number that was never
        # computed for them: the stats band is +/- the calibration's own residual std.
        "force_halfwidth_minus_calib_std_max": max(
            abs(0.5 * (stats["state"]["default"]["global_max"][i]
                       - stats["state"]["default"]["global_min"][i]) - std[i - E.PROPRIO_BASE_DIM])
            for i in range(E.PROPRIO_BASE_DIM, E.PROPRIO_DIM)),
        "force_halfwidth_equals_calib_std": all(
            abs(0.5 * (stats["state"]["default"]["global_max"][i]
                       - stats["state"]["default"]["global_min"][i]) - std[i - E.PROPRIO_BASE_DIM])
            < 1e-6 for i in range(E.PROPRIO_BASE_DIM, E.PROPRIO_DIM)),
        "stats_source": str(E.V1G_STATS)}


def processor_facts(policy):
    """The processor's contract, read off the object the deployment functions actually use.

    Three different mistakes all look like a normaliser here: no stats attached at all, stats
    attached to a layout whose state/action keys are not the single merged key
    `normalize_proprio`/`denormalize_action` require, and a norm mode other than the one training
    used. `_is_train` is the record that the processor was built `.eval()`, the way the worker
    builds it.
    """
    pr = policy.processor
    state_meta, action_meta = pr.shape_meta["state"], pr.shape_meta["action"]
    return {"processor_class": type(pr).__name__,
            "proprio_output_dim": int(pr.proprio_output_dim),
            "action_output_dim": int(pr.action_output_dim),
            "num_obs_steps": int(pr.num_obs_steps),
            "norm_default_mode": str(pr.norm_default_mode),
            "normalizer_attached": pr._normalizer is not None,
            "state_keys": [m["key"] for m in state_meta],
            "action_keys": [m["key"] for m in action_meta],
            "state_raw_shape": [int(m["raw_shape"]) for m in state_meta],
            "action_raw_shape": [int(m["raw_shape"]) for m in action_meta],
            "is_train_flag": None if pr._is_train is None else bool(pr._is_train)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000, help="one node: this is a smoke")
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 12])
    ap.add_argument("--merged-dir", default=str(E.MERGED_DEFAULT))
    ap.add_argument("--merged", default=None, help="explicit artifact, otherwise the slot in "
                                                   "--merged-dir for this step")
    ap.add_argument("--config", default=str(E.DEFAULT_CFG))
    ap.add_argument("--calib", default=str(E.CALIB_PATH))
    ap.add_argument("--stats", default=str(E.V1G_STATS),
                    help="the stats `run40k/config.yaml` names as pretrained_norm_stats")
    ap.add_argument("--instruction", default=None,
                    help="default: the task string the training corpus was conditioned on, read "
                         "off the data by eval40k.resolve_prompt_facts. Anything else is refused.")
    ap.add_argument("--gripper-mode", default="pm1", choices=["pm1", "libero01"])
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--replan", type=int, default=10)
    ap.add_argument("--force-horizon", type=int, default=32)
    ap.add_argument("--out", default=str(HERE / "out" / "smoke"))
    ap.add_argument("--batch-offset", type=float, default=0.0)
    ap.add_argument("--position-noise", type=float, default=0.0)
    ap.add_argument("--yaw-noise", type=float, default=0.0)
    ap.add_argument("--randomize-boxes", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-noise-probe", action="store_true")
    args = ap.parse_args()

    step = int(args.steps)
    seeds = [int(s) for s in args.seeds]
    if len(set(seeds)) != len(seeds):
        raise SystemExit("repeated seed: the protocol is one episode per seed")
    out_dir = Path(args.out) / ("step_%06d" % step)
    out_dir.mkdir(parents=True, exist_ok=True)

    checks = []
    res = {"generated": time.strftime("%FT%TZ"), "step": step, "seeds": seeds,
           "max_steps": args.max_steps, "replan": args.replan, "checks": checks}

    st = E.node_state(step)
    if st is None:
        raise SystemExit("no light node for step %d" % step)
    if int(st["global_step"]) != step:
        raise SystemExit("node %d claims global_step=%s" % (step, st["global_step"]))
    calib_facts = E.resolve_calib(args.calib, st["force_calibration"])
    res["node"] = st
    res["calibration"] = calib_facts

    env_factory, W = E.build_env(args)      # anchors `fasterwam` on the way in, via E.use_stage_fw()
    import fasterwam
    import torch

    res["fw_import"] = dict(E.FW_IMPORT_FACTS)
    res["fw_import"]["after_build_env"] = str(fasterwam.__file__)
    fw_src = str(E.FW / "src")
    checks.append(("the anchor proves itself on the imported objects, not on `sys.path` order: "
                   "package __file__, runtime __file__, the factory's own source file all under "
                   f"{fw_src}, and the factory really takes `force`",
                   all(str(res["fw_import"].get(k, "")).startswith(fw_src + "/")
                       for k in ("package_file", "runtime_file", "create_fasterwam_file"))
                   and res["fw_import"]["create_fasterwam_accepts_force"] is True))
    checks.append(("the bridge did not re-point `fasterwam`: after build_env() the package is "
                   "still the force-aware copy these weights were trained with",
                   str(fasterwam.__file__).startswith(fw_src)))

    # The 14-wide proprio is the one input this harness has to build itself, so its arithmetic is
    # proved against the corpus on CPU before a checkpoint is opened or a GPU is touched.
    proof = corpus_proprio14_proof()
    res["proprio14_corpus_proof"] = proof
    checks.append(("the state layout the model was trained on is reproducible from the recorded "
                   "channels: gathering %s out of the 18-wide force array is the stored `force6` "
                   "over every row -- and the frozen `select_channels` the live builder calls is "
                   "the same gather, row by row -- while the 14-wide proprio's last six columns "
                   "are bit-for-bit `force6 - zero_load_ref_median` cast once to float32, the "
                   "arithmetic `build_proprio14` performs, and its first eight are exactly the "
                   "8-wide proprio the worker's own `build_proprio` mirrors"
                   % (list(CHANNELS),),
                   proof["gathered_channels_equal_force6"]
                   and proof["gathered_channels_bit_equal_frac"] == 1.0
                   and proof["select_channels_is_a_single_vector_function"]
                   and proof["select_channels_rowwise_equals_vectorised_gather"]
                   and proof["rows_probed_rowwise"] >= 512
                   and proof["state_force_columns_equal_tared_channels"]
                   and proof["state_force_columns_bit_equal_frac"] == 1.0
                   and proof["state_base_columns_equal_stored_8_wide_proprio"]
                   and proof["v1g_proprio_shape"][1] == E.PROPRIO_DIM
                   and proof["base8_width"] == E.PROPRIO_BASE_DIM))
    checks.append(("the recorded normalisation band is the one this run trains on: the stats file "
                   "named by `run40k/config.yaml` describes a 14-wide state, its six force columns "
                   "are banded symmetrically about zero, and that half-width is the residual std in "
                   "`force_calib_v1.json` itself -- which is what makes min/max mode compute "
                   "`residual / std` for them (offset 0, scale 1/std). The +/-std band is the "
                   "dataset's own design, not corruption, and nothing here re-derives it",
                   str(args.stats) == str(E.V1G_STATS)
                   and proof["calib_version"] == E.CALIB_VERSION
                   and proof["stats_state_width"] == E.PROPRIO_DIM
                   and proof["stats_force_band_is_symmetric"]
                   and proof["force_halfwidth_equals_calib_std"]))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    args.step = step
    policy = E.build_policy(args, None, (env_factory, W))
    res["eval_settings"] = dict(E.EVAL_SETTINGS_FACTS)
    ev = res["eval_settings"].get("values", {})
    checks.append(("no inference setting was invented: every knob the policy reads came from the "
                   "deployment config, and the two Hydra interpolations resolved against this run's "
                   "own config (`eval_num_inference_steps`, `default_action_infer_mode`)",
                   res["eval_settings"].get("source") == str(E.DEPLOY_EVAL_CFG)
                   and ev.get("num_inference_steps") == int(policy.cfg.eval_num_inference_steps)
                   and "num_inference_steps" in res["eval_settings"].get(
                        "interpolated_from_run_config", {})))

    # Text conditioning, the part that killed the first smoke attempt. `run40k/config.yaml` turns
    # the T5 off on purpose -- training reads embeddings from the cache -- while the worker that
    # serves these weights runs with it on and hands `infer_action` a prompt. So the harness has to
    # take that one key from the deployment config, and has to prove the sentence it prompts with
    # is the one the 40k steps were conditioned on, not one this file used to invent.
    res["text_encoder"] = dict(E.TEXT_ENCODER_FACTS)
    res["prompt_facts"] = dict(E.TEXT_PROMPT_FACTS)
    res["prompt_conditioning_check"] = dict(E.TEXT_CHECK_FACTS)
    res["weight_paths"] = dict(E.WEIGHT_PATH_FACTS)
    checks.append(("the prompt path exists: `model.load_text_encoder` was taken from the "
                   "deployment config alone (the run config sets it false because training reads "
                   "the cache), the encoder and tokenizer really are attached to the built model, "
                   "the encoder is in eval mode, and every weight file was read from "
                   f"{E.DIFFSYNTH_MODEL_BASE} rather than downloaded",
                   res["text_encoder"].get("source") == str(E.DEPLOY_EVAL_CFG)
                   and res["text_encoder"].get("train_config_value") is False
                   and res["text_encoder"].get("deployment_value") is True
                   and res["text_encoder"].get("applied") is True
                   and policy.model.text_encoder is not None
                   and policy.model.tokenizer is not None
                   and res["prompt_conditioning_check"].get("text_encoder_training") is False
                   and all(v.get("exists") is True
                           for v in res["weight_paths"].values())))
    checks.append(("the instruction is read off the training data, not chosen: the corpus arrays "
                   "carry no `task_str`, so `A3VideoDataset` fell back to its own default for all "
                   f"{res['prompt_facts'].get('rows')} recorded windows, that prompt hashes to the "
                   "single embedding file in the cache directory the training config names, and "
                   "the policy under test is prompted with exactly that task string",
                   res["prompt_facts"].get("rows") == 42681
                   and len(res["prompt_facts"].get("cache_files_present") or ()) == 1
                   and res["prompt_facts"].get("cache_file")
                   == (res["prompt_facts"].get("cache_files_present") or [None])[0]
                   and res["prompt_facts"].get("context_len") == 128
                   and policy.instruction == res["prompt_facts"].get("task")
                   and cross_prompt(policy, W) == res["prompt_facts"].get("prompt")))
    checks.append(("training and inference condition on the same bits: `model.encode_prompt` on "
                   "the training prompt reproduces the cached embedding tensor element for element "
                   "(0 differing bf16 values of %s, max abs diff %s, %s non-zero token rows on both "
                   "sides), so the encoder path in the harness is a re-derivation of what the "
                   "dataset read from disk, not a different conditioning input"
                   % (res["prompt_conditioning_check"].get("n_elements"),
                      res["prompt_conditioning_check"].get("max_abs_diff"),
                      res["prompt_conditioning_check"].get("cached_mask_true")),
                   res["prompt_conditioning_check"].get("n_diff_elements") == 0
                   and res["prompt_conditioning_check"].get("max_abs_diff") == 0.0
                   and res["prompt_conditioning_check"].get("encoded_nonzero_token_rows")
                   == res["prompt_conditioning_check"].get("cached_mask_true")
                   and res["prompt_conditioning_check"].get("encoded_mask_all_true") is True
                   and res["prompt_conditioning_check"].get("encoded_nan") is False))
    # a deviation worth stating rather than hiding: the deployment resolves
    # EVALUATION.action_infer_mode to `one_pass_future_cache` and then calls
    # `model.infer_action_one_pass_future_cache`. This harness always calls `model.infer_action`,
    # because the frozen per-chunk seed table and the recorder are defined against it. Both
    # methods accept the force kwargs (`jointwam.py:127` and `:296`), so this is a video-cache
    # choice, not a missing capability -- but it is a real difference from the worker, so it is
    # recorded and reported instead of quietly normalised away.
    res["deviations"] = [{
        "item": "action_infer_mode",
        "deployment": res["eval_settings"].get("deployment_action_infer_mode"),
        "harness": "joint_rollout (model.infer_action)",
        "why": "the frozen noise-seed protocol and the recording proxy are defined against "
               "infer_action; switching the cache path would change the numbers being compared",
        "model_has_the_other_entry_point": hasattr(policy.model,
                                                   "infer_action_one_pass_future_cache")}, {
        "item": "proprio width (8 vs 14)",
        "deployment": "infer_a3_worker.build_proprio -> 8 columns; the released worker serves a "
                      "model whose shape_meta state raw_shape is 8",
        "harness": "eval40k.build_proprio14 -> the worker's 8 columns + 6 zero-load-tared force "
                   "columns, i.e. the 14 the 40k shape_meta demands",
        "why": "the force checkpoint was trained on `build_v1_dataset.py:130`'s 14-wide state; "
               "no function in the read-only bridge produces it, so the eval toolchain has to, "
               "and it does so by calling the worker's own build_proprio for columns 0-7",
        "wired_via": "policy40k.ForcePolicy(proprio_fn=...), used by every driver that calls "
                     "eval40k.build_policy -- the batch evaluator included"}]
    # The worker's default instruction is not this corpus's instruction, so the harness deviates
    # from the released worker's config on purpose and says so: the model is prompted with the
    # sentence its own training data carries, not with the two-arm sentence the shipped worker
    # hardcodes for a different checkpoint.
    res["deviations"].append({
        "item": "instruction",
        "deployment": str(W.INSTRUCTION),
        "harness": res["prompt_facts"].get("task"),
        "why": "W.INSTRUCTION is the released worker's default for an action-only checkpoint; the "
               "40k weights were trained through A3VideoDataset on the single task string in "
               "a3_fw_v1g, whose sha256 names the only embedding in the training cache. "
               "Conditioning eval episodes on anything else changes the input distribution."})

    cross = preprocess_with_deployment_funcs(env_factory, W, policy, calib_facts, seeds[0], args)
    res["preprocessing_cross_check"] = cross
    checks.append(("the processor this harness builds is usable by the deployment's own "
                   "`build_proprio`/`normalize_proprio`: one finite (1, 14) proprio, and the "
                   "image the model is fed is [1, 3, H, W]",
                   cross["proprio_shape"] == [1, 14] and cross["proprio_finite"]
                   and len(cross["image_shape"]) == 4 and cross["image_shape"][1] == 3))
    checks.append(("same observation, both sides: the proprio and the image this harness hands "
                   "`infer_action` are bit-identical to what `predict_action_chunk` builds with "
                   "the worker's own functions -- which is also the proof that the min/max "
                   "normaliser is applied once, not twice",
                   cross["proprio_identical_to_deployment_call"]
                   and cross["image_identical_to_deployment_call"]))
    checks.append(("the normalised proprio sits where this normaliser puts it: the eight base "
                   "columns land inside the [-1, 1] output band the recorded min/max describe, "
                   "and the six force columns land on `residual / residual_std`, which the "
                   "corpus itself pushes past 1 and which the normaliser only clamps at +-5 "
                   "(a second normalisation would show up as base columns well outside +-1)",
                   -1.02 <= cross["proprio_base_min"] <= 1.02
                   and -1.02 <= cross["proprio_base_max"] <= 1.02
                   and abs(cross["proprio_force_max"]) <= 5.0
                   and abs(cross["proprio_force_min"]) <= 5.0
                   and cross["proprio_force_equals_residual_over_std"]))
    checks.append(("the proprio the model is fed really is 14 wide and really is the training "
                   "layout: the worker's own 8 columns bit-for-bit as the first 8, and the last "
                   "6 built from the worker's own force channels minus the frozen zero-load tare",
                   cross["proprio_shape"] == [1, E.PROPRIO_DIM]
                   and cross["base8_bit_identical_to_14_wide_builder"]
                   and cross["stats_source"] == str(E.V1G_STATS)
                   and max(cross["stats_state_width"].values()) == E.PROPRIO_DIM))
    checks.append(("action horizon, video-frame count and diffusion steps are computed by this "
                   "harness the way the worker computes them from the same config, and the prompt "
                   "uses the worker's own template",
                   cross["harness_action_horizon"] == cross["deployment_action_horizon"]
                   and cross["harness_num_video_frames"] == cross["deployment_num_video_frames"]
                   and cross["harness_num_inference_steps"] == cross["deployment_num_inference_steps"]
                   and cross["prompt_uses_deployment_template"]))
    checks.append(("the force input is the frozen contract at the first replan: history "
                   "(1, %d, %d), mask (1, %d) with exactly the trailing valid frames marked"
                   % (HIST_LEN, len(CHANNELS), HIST_LEN),
                   cross["force_history_shape"] == [1, HIST_LEN, len(CHANNELS)]
                   and cross["force_history_mask_shape"] == [1, HIST_LEN]
                   and cross["force_history_mask_true"] == 1))
    res["prompt_note"] = {"harness": cross["harness_prompt"],
                          "training_prompt": res["prompt_facts"].get("prompt"),
                          "deployment_default": cross["deployment_default_prompt"],
                          "note": "three templates-free strings: the harness prompt is the "
                                  "training prompt (checked above, and proved to reproduce the "
                                  "cached embedding bit for bit); the released worker's own default "
                                  "describes a different task (a two-arm transfer) and has no "
                                  "cached embedding in this corpus, which is why this harness does "
                                  "not inherit it. Until this run the smoke driver prompted with "
                                  "\"Pick up the cookie and place it in the box.\", a sentence the "
                                  "40k steps never conditioned on -- measured against the training "
                                  "embedding it differs on 22.6 percent of the 128x4096 elements "
                                  "and by up to 1.81 where the mean magnitude is 0.011."}
    pfacts = processor_facts(policy)
    res["processor"] = pfacts
    checks.append(("the processor is the deployment's shape: normaliser attached, one merged state "
                   "key of raw_shape 14 and one action key of raw_shape 7, proprio_output_dim 14, "
                   "action_output_dim 7, min/max mode, and built in eval() like the worker",
                   pfacts["normalizer_attached"] and pfacts["proprio_output_dim"] == 14
                   and pfacts["action_output_dim"] == 7
                   and pfacts["norm_default_mode"] == "min/max"
                   and len(pfacts["state_keys"]) == 1 and pfacts["state_raw_shape"] == [14]
                   and len(pfacts["action_keys"]) == 1 and pfacts["action_raw_shape"] == [7]
                   and pfacts["is_train_flag"] is False))
    checks.append(("the harness calls the deployment worker's own functions rather than copies of "
                   "them: `policy.bridge` *is* the `infer_a3_worker` module, so `normalize_proprio`, "
                   "`denormalize_action`, `prepare_fasterwam_images_official` and "
                   "`quat_wxyz_to_matrix` resolve on it directly; and the proprio override is the "
                   "documented one -- `ForcePolicy.proprio_fn` is set to `eval40k.build_proprio14` "
                   "(not the worker's 8-wide builder), and the proprio it produces for the model "
                   "call is bit-identical to the deployment-side computation above",
                   policy.bridge is W
                   and policy.bridge.normalize_proprio is W.normalize_proprio
                   and policy.bridge.denormalize_action is W.denormalize_action
                   and policy.bridge.prepare_fasterwam_images_official
                       is W.prepare_fasterwam_images_official
                   and policy.bridge.quat_wxyz_to_matrix is W.quat_wxyz_to_matrix
                   and policy.proprio_fn is not None
                   and policy.proprio_fn is not W.build_proprio
                   and cross["proprio_identical_to_deployment_call"]))
    res["proprio_source"] = {"bridge_type": type(policy.bridge).__name__,
                             "worker_module": str(W.__name__),
                             "builder": str(E.build_proprio14.__module__) + ".build_proprio14",
                             "why": "the released worker builds an 8-wide proprio for a model "
                                    "that had no force input; the 40k checkpoint's own "
                                    "shape_meta demands 14, so the harness adds the six tared "
                                    "force columns and delegates everything else unchanged",
                             "permanent_home": "policy40k.ForcePolicy.proprio_fn, consulted at the "
                                               "one model call site in _infer_kwargs; every driver "
                                               "that builds a policy through eval40k.build_policy "
                                               "(smoke and batch alike) gets it",
                             "policy_has_override": policy.proprio_fn is not None,
                             "facts": dict(E.PROPRIO14_FACTS)}
    checks.append(("fix 3 holds on the device, not in the source: the loaded model sits on the "
                   "device the policy passes to and is in eval mode",
                   str(next(policy.model.parameters()).device) == str(policy.device)
                   and policy.model.training is False))
    res["model"] = {"device": str(policy.device),
                    "param_device": str(next(policy.model.parameters()).device),
                    "training_flag": bool(policy.model.training),
                    "torch_dtype": str(policy.model.torch_dtype),
                    "cuda_available": bool(torch.cuda.is_available()),
                    "action_horizon": int(policy.action_horizon),
                    "num_video_frames": int(policy.num_video_frames),
                    "hist_len": int(policy.stream.hist_len) if hasattr(policy.stream, "hist_len")
                    else None,
                    "gripper_mode": policy.gripper_mode}

    seen = []
    original, params_before, params_after = install_recorder(policy.model, seen)
    checks.append(("the recording proxy left `infer_action`'s signature as the policy inspects it",
                   params_before == params_after
                   and ("num_video_frames" in params_after) == ("num_video_frames" in params_before)))
    chunk_seen = []
    chunk_original = install_chunk_recorder(policy, chunk_seen)
    checks.append(("the chunk recorder is a pass-through: same callable, and `functools.wraps` "
                   "kept the name and signature the rollout loop relies on",
                   chunk_original.__name__ == "chunk"
                   and str(inspect.signature(chunk_original)) == str(inspect.signature(policy.chunk))))

    def reset(env, seed):
        obs, _info = env_factory.reset(env, seed, randomize_cookies=True,
                                       randomize_boxes=args.randomize_boxes,
                                       batch_offset_m=args.batch_offset)
        sanity = env_factory.scene_sanity(env)
        if not sanity["ok"]:
            raise SystemExit("scene invalid at seed=%s: %s" % (seed, sanity))
        return obs

    account = ForceErrorAccount(horizon=args.force_horizon)
    episodes = []
    t_start = time.time()
    for seed in seeds:
        env = env_factory.make_env(position_noise_m=args.position_noise,
                                   yaw_noise_rad=args.yaw_noise, render_cameras=True)
        base = len(seen)
        cbase = len(chunk_seen)
        rec, measured, chunks = run_episode(env, policy, seed=seed, reset=reset,
                                            max_steps=args.max_steps, replan=args.replan)
        calls = seen[base:]
        clog = chunk_seen[cbase:]
        rec["scene_key"] = E.scene_key(env)
        pred_chunks = [c for c in chunks if c["pred_force"] is not None]
        rec["force_steps_scored"] = sum(
            account.add_chunk(c["frame"], c["pred_force"], measured) for c in pred_chunks)
        summarize_episode(rec, calls, clog, chunks)
        episodes.append(rec)
        print(json.dumps({k: v for k, v in rec.items()
                          if k not in ("seeds_seen_by_model", "expected_seeds",
                                       "meta_noise_seed", "chunk_log",
                                       "mask_true_sequence")}), flush=True)
        if hasattr(env, "close"):
            env.close()
    res["episodes"] = episodes
    res["wall_s"] = round(time.time() - t_start, 1)

    for rec in episodes:
        n = rec["decision_chunks"]
        checks.append(("seed %d: every call the model received is `noise_seed(episode seed, "
                       "chunk index)` for chunk 1..%d, and `meta.noise_seed` agrees"
                       % (rec["seed"], n),
                       rec["seeds_seen_by_model"] == rec["expected_seeds"]
                       and rec["meta_noise_seed"] == rec["expected_seeds"]))
        checks.append(("seed %d: the %d replans of this episode used %d distinct seeds and no "
                       "NaN in the predicted force" % (rec["seed"], n, rec["distinct_seeds"]),
                       rec["distinct_seeds"] == n and rec["nonfinite_pred_force"] == 0))
        checks.append(("seed %d: the closed loop reached the protocol's chunk count with no "
                       "non-finite force abort" % rec["seed"],
                       n == (rec["steps_executed"] + args.replan - 1) // args.replan
                       and rec["nan_abort_frame"] < 0))
        checks.append(("seed %d: the model never saw a gradient-enabled output, the diffusion step "
                       "count and rand device stayed at the deployment values, and the proprio "
                       "handed to it stayed inside this normaliser's own bands" % rec["seed"],
                       not rec["action_requires_grad_any"]
                       and rec["num_inference_steps_seen"]
                       == [cross["deployment_num_inference_steps"]]
                       and rec["rand_device_seen"] == [str(ev.get("rand_device"))]
                       and rec["proprio_widths"] == [E.PROPRIO_DIM]
                       and -1.02 <= rec["proprio_base_min_all"] <= 1.02
                       and -1.02 <= rec["proprio_base_max_all"] <= 1.02
                       and abs(rec["proprio_force_min_all"]) <= 5.05
                       and abs(rec["proprio_force_max_all"]) <= 5.05))
        checks.append(("seed %d: the force history the model was conditioned on is the frozen "
                       "window -- (1, %d, %d) on every call, with the mask filling up over the "
                       "first %d replans and staying full after that"
                       % (rec["seed"], HIST_LEN, len(CHANNELS), HIST_LEN),
                       rec["force_history_shapes"] == [str([1, HIST_LEN, len(CHANNELS)])]
                       and rec["mask_true_sequence"][0] == 1
                       and rec["mask_true_sequence"][-1] == HIST_LEN
                       and rec["mask_true_sequence"] == sorted(rec["mask_true_sequence"])))
        checks.append(("seed %d: every chunk the policy returned is the full %d-step horizon, one "
                       "consistent action width, all finite" % (rec["seed"], policy.action_horizon),
                       rec["chunk_all_finite"] and len(rec["chunk_shapes"]) == 1
                       and rec["chunk_shapes"][0][0] == policy.action_horizon))
    sets = [set(r["seeds_seen_by_model"]) for r in episodes]
    checks.append(("two episodes of the same node share no chunk seed",
                   all(not (a & b) for i, a in enumerate(sets) for b in sets[i + 1:])))
    checks.append(("every episode reports the frozen predicted-force shape (1, %d, %d) out of "
                   "the real model" % (args.force_horizon, len(CHANNELS)),
                   all(r["force_shape_reported"] == [1, args.force_horizon, len(CHANNELS)]
                       for r in episodes if r["force_shape_reported"])))
    distinct_scenes = len({r["scene_key"] for r in episodes})
    checks.append(("the scene is the fixed one: all %d episodes start from the same cookie "
                   "layout hash" % len(episodes), distinct_scenes == 1))
    res["scene"] = {"distinct_scene_keys": distinct_scenes,
                    "keys": [r["scene_key"] for r in episodes],
                    "batch_offset_m": args.batch_offset,
                    "meaning": "one layout, N policy samples: this is the fixed-scene protocol "
                               "item 7 asks for, not a generalisation measure"}

    model_action_shapes = sorted({s for r in episodes for s in r["model_action_shapes"]})
    checks.append(("the raw tensor the model returns as `action` is (1, %d, %d) -- the horizon and "
                   "the processor's action width the denormaliser expects: %s"
                   % (policy.action_horizon, pfacts["action_output_dim"], model_action_shapes),
                   model_action_shapes == [str([1, policy.action_horizon,
                                                pfacts["action_output_dim"]])]))
    widths = sorted({s[1] for r in episodes for s in r["chunk_shapes"]})
    checks.append(("after denormalisation, the gripper rewrite and the adapter, every episode "
                   "hands the simulator chunks of one and the same action width: %s (the same "
                   "width the r4 deployment fed)" % widths, len(widths) == 1))
    gripper_vals = sorted({v for r in episodes for v in r["gripper_pm1_values"]})
    checks.append(("the gripper channel is what the selected mode promises: mode=pm1 gives exactly "
                   "the +/-1 targets the A3 training data use (observed %s)" % gripper_vals,
                   all(r["gripper_is_pm1_binary"] for r in episodes)
                   if args.gripper_mode == "pm1"
                   else all(abs(v) <= 1.0 + 1e-6 for v in gripper_vals)))

    force = account.finalize()
    res["force"] = force
    overall = force.get("overall", {})
    mae = overall.get("mae")
    checks.append(("the force prediction was scored over at least one future step per episode "
                   "and the MAE is finite",
                   force.get("steps_scored", 0) > 0
                   and mae is not None and all(np.isfinite(np.asarray(mae, dtype=float)))))

    res["inference_mode"] = {
        "model_training": bool(policy.model.training),
        "params_seen": int(sum(1 for _ in policy.model.parameters())),
        "params_with_grad": int(sum(1 for p in policy.model.parameters() if p.grad is not None)),
        "outputs_with_grad": int(sum(1 for c in seen if c["action_requires_grad"])),
    }
    checks.append(("inference really ran without gradient tracking: the module is still in eval "
                   "mode after %d real calls, not one of its %d parameters carries a .grad, and no "
                   "action tensor asked for one" % (len(seen), res["inference_mode"]["params_seen"]),
                   res["inference_mode"]["model_training"] is False
                   and res["inference_mode"]["params_with_grad"] == 0
                   and res["inference_mode"]["outputs_with_grad"] == 0))
    if torch.cuda.is_available():
        gib = float(2 ** 30)
        props = torch.cuda.get_device_properties(0)
        res["gpu"] = {"device": str(props.name),
                      "total_gib": round(props.total_memory / gib, 1),
                      "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / gib, 2),
                      "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / gib, 2),
                      "allocated_gib": round(torch.cuda.memory_allocated() / gib, 2)}
        checks.append(("no CUDA OOM and headroom to spare: peak reserved was %.2f GiB against the "
                       "device's %.1f GiB, which is what the eight-node batch is sized against"
                       % (res["gpu"]["peak_reserved_gib"], res["gpu"]["total_gib"]),
                       res["gpu"]["peak_reserved_gib"] < 0.9 * res["gpu"]["total_gib"]))

    if seen and not args.no_noise_probe:
        # a probe failure must not cost the seed evidence: the rollout has already run by this
        # point, so record the exception as a failed check and still write the file
        try:
            res["noise_streams"] = three_stream_noise(policy.model, seen, seeds[0],
                                                      args.force_horizon, len(CHANNELS))
            again = three_stream_noise(policy.model, seen, seeds[0],
                                       args.force_horizon, len(CHANNELS))
            checks.append(("the three-stream reproduction is deterministic: re-running it gives "
                           "the same per-stream digests",
                           all(again[k] == v for k, v in res["noise_streams"].items()
                               if k.endswith("sha256_64"))))
        except Exception as exc:                                  # noqa: BLE001
            res["noise_streams"] = {"error": "%s: %s" % (type(exc).__name__, exc)}
            checks.append(("the three-stream noise probe ran", False))

    # let go of the weights before anything else can hold them: the two recording proxies bind
    # methods of this exact model, so they have to be dropped with the policy or the cycle keeps
    # the whole thing resident for whatever runs next in this process
    del policy, original, chunk_original
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        res["gpu"]["after_release_allocated_gib"] = round(torch.cuda.memory_allocated() / gib, 3)
        res["gpu"]["after_release_reserved_gib"] = round(torch.cuda.max_memory_reserved() / gib, 2)
        checks.append(("the smoke releases what it loaded: after the last inference, dropping the "
                       "policy and both proxies and clearing the cache leaves %.3f GiB allocated "
                       "against a %.2f GiB peak, so the next node starts clean"
                       % (res["gpu"]["after_release_allocated_gib"],
                          res["gpu"]["peak_allocated_gib"]),
                       res["gpu"]["after_release_allocated_gib"] <= 1.0
                       or res["gpu"]["after_release_allocated_gib"]
                       <= 0.05 * res["gpu"]["peak_allocated_gib"]))

    n_ok = sum(1 for _, ok in checks if ok)
    res["summary"] = {"checks_total": len(checks), "checks_passed": n_ok,
                      "episodes": len(episodes),
                      "successes": sum(1 for r in episodes if r["success"]),
                      "mean_frames": round(float(np.mean([r["frames"] for r in episodes])), 1),
                      "infer_s_mean_all": round(float(np.mean([r["infer_s_mean"]
                                                                for r in episodes])), 3),
                      "wall_s": res["wall_s"],
                      "projected_160_episodes_h": round(
                          res["wall_s"] / max(len(episodes), 1) * 160 / 3600.0, 2)}
    res["checks"] = [{"name": name, "ok": bool(ok)} for name, ok in checks]
    path = out_dir / ("smoke_seeds%s.json" % "_".join(str(s) for s in seeds))
    path.write_text(json.dumps(res, indent=2, sort_keys=True, default=str))
    print(json.dumps({"out": str(path), "summary": res["summary"],
                      "failed": [c["name"] for c in res["checks"] if not c["ok"]]}, indent=2))
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
