"""Phase-4 forensics driver: full failure episodes under the RECONSTRUCTED normalizer.

It rebuilds the policy exactly like diag_reconstructed_40k (force-aware protocol, text from the code
constants, reconstructed 14-D stats) and drives the ARCHIVED recording blocks unchanged:

  record40k.ActionSpy        -- per-chunk model_out / denorm / pm1 / adapter_out (passive copies)
  record40k.EnvProbe         -- per-step handed_14d / requested_14d / applied_joint / safety_reason
                                / contact counters / per-frame composed-panel feed (call-through)
  record40k.PanelSink        -- builds the ann dict and feeds panelvideo40k.PanelRecorder
  record40k.chain_for_frame  -- frame -> (chunk, executed row): model_absmax / over5 / denorm / pm1
  panelvideo40k              -- [front | left_wrist | independent right panel], cameras pixel-exact

The archived files are NOT modified. Two runtime overrides are recorded in the run's identity json:

  1. PanelRecorder backend -> cv2/mp4v. The pod's imageio resolves to the PyAV plugin, whose write()
     rejects the compositor's `quality=` kwarg (probed: imageio_default and plugin='FFMPEG' both
     fail, cv2/mp4v writes 5069 B). cv2/mp4v is the compositor's own documented CPU fallback; the
     stats carry backend="cv2/mp4v" and "bitstream parity not".
  2. Nothing else. The policy protocol is the real 40k one; model/Processor/normalizer/Adapter are
     untouched.

Every output carries NORMALIZER_IDENTITY=RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL and
original_stats_sha256=UNKNOWN. The trajectory is reconstructed ONLY from the executed first
`replan` rows (probe.steps / steps.npz), never by concatenating all 32 predicted rows.

Modes
  --pair                     seed 12, short block: A bare vs B forensics -> passive proof (bit-exact)
  --seeds 11 17 --max-steps 600   full failure episodes with composed panel video + steps/actions npz
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path("/workspace/eval_restore_20261117")
OUT_DEFAULT = BASE / "reconstructed_normalizer" / "forensics_out"

import diag_reconstructed_40k as D  # setup_protocol / build_policy_reconstructed / make_reset / artifacts / MARKERS / serialize_result / compare_pair / sha256_file / merged_identity


def _ns(**kw):
    import types
    return types.SimpleNamespace(**kw)


def install_cv2_recorder():
    """Runtime override: PanelRecorder._open -> cv2/mp4v (archived panelvideo40k.py untouched)."""
    import panelvideo40k as PV

    PV.PANEL_W = 960

    class _Cv2PanelRecorder(PV.PanelRecorder):
        def _open(self):
            import cv2
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            size = (2 * PV.CAM_W + PV.PANEL_W, self.canvas_h)
            wr = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, size)
            if not wr.isOpened():
                raise RuntimeError("cv2/mp4v refused to open %s at %s" % (self.path, size))
            self.backend = "cv2/mp4v"
            self._push = lambda rgb: wr.write(rgb[:, :, ::-1])
            self._close = wr.release
            self._wr = wr
            self.notes.append(
                "encoded by cv2/mp4v: the pod imageio resolves to PyAV, whose write() rejects the "
                "compositor's quality= kwarg; layout and synchronisation verified, bitstream (libx264/"
                "yuv420p) parity not")
            return wr

    PV.PanelRecorder = _Cv2PanelRecorder
    return PV


def _opening(obs):
    from a3_dual_arm_sim.contracts import STATE
    return float(np.asarray(obs[STATE])[7])


def run_one_forensics(E, env_factory, policy, W, calib, args, seed, out, PV):
    """One episode with the full forensics stack installed. Returns a D.run_one-shaped dict plus
    panel stats and the written file names; the record/measured/chunks come from the SAME
    run_episode, so compare_pair can be reused verbatim for the passive proof."""
    import record40k as R

    env = env_factory.make_env(position_noise_m=0.0, yaw_noise_rad=0.0, render_cameras=True)
    reset = D.make_reset(E, env_factory, env, args)
    pre = D.rng_snapshot()

    # extra_notes must stay empty: the panel's fixed 300 px text box already fits the four built-in
    # CHAIN_NOTES plus the full per-frame action chain. A fifth note line overflows it (hit live at
    # y=297+12>306). The reconstructed-normalizer marker is carried by the co-located sidecars
    # (seed*_meta.json, steps.npz attrs, forensics_identity.json), not by an overflowing video line.
    panel = R.PanelSink(out / ("seed%d_composed_panel.mp4" % seed), fps=20, seed=seed, step=40000,
                        total_frames=args.max_steps + 1, replan=args.replan, extra_notes=[])
    probe = R.EnvProbe(proprio=lambda o: E.build_proprio14(o, W, calib))
    probe.total_hint = args.max_steps + 1
    probe.panel = panel

    # opening capture: installed BEFORE probe so probe.install_on_env wraps it (probe.step ->
    # open_step -> real step). Reads state[7] (gripper opening) from the returned obs; passive.
    real_step = env.step
    openings = []
    observations = []
    step_info = []

    def snapshot(obs):
        observations.append({key: np.array(value, copy=True) for key, value in obs.items()
                             if isinstance(value, (np.ndarray, float, int, bool, np.number))})

    def open_step(action, **k):
        outv = real_step(action, **k)
        snapshot(outv[0])
        step_info.append(dict(outv[4]))
        openings.append(_opening(outv[0]))
        return outv

    env.step = open_step
    has_contact = probe.install_on_env(env)
    real_bridge, real_convert = policy.bridge, policy.adapter.convert
    spy = R.ActionSpy().install(policy)
    panel.chain_fn = lambda f: R.chain_for_frame(spy, panel.replan, f)
    reset_open = []
    original_panel_add = panel.add_frame

    def panel_add(obs, rec, p14):
        original_rec_add = panel.rec.add

        def annotated_add(front, wrist, ann):
            frame = int(rec["frame"])
            raw = pm1 = "n/a"
            if frame:
                chunk, row = divmod(frame - 1, args.replan)
                stage = spy.rows[chunk]
                raw = "%+.5f" % stage["model_out"].reshape(-1, 7)[row, 6]
                pm1 = "%+d" % np.asarray(stage["pm1"])[row, 6]
            ann["gripper"] = "raw %s pm1 %s actual opening %.5f" % (raw, pm1, _opening(obs))
            return original_rec_add(front, wrist, ann)

        panel.rec.add = annotated_add
        try:
            return original_panel_add(obs, rec, p14)
        finally:
            panel.rec.add = original_rec_add

    panel.add_frame = panel_add

    def wrapped_reset(e, s):
        obs = reset(e, s)
        snapshot(obs)
        probe.record_reset(obs)
        reset_open.append(_opening(obs))
        return obs

    t0 = time.time()
    try:
        result = E.run_episode(env, policy, seed=seed, reset=wrapped_reset,
                               max_steps=args.max_steps, replan=args.replan)
    finally:
        policy.bridge, policy.adapter.convert = real_bridge, real_convert
        try:
            stats = panel.close()
        finally:
            obs_arrays = {key: np.stack([o[key] for o in observations])
                          for key in observations[0]} if observations else {}
            np.savez_compressed(out / ("seed%d_observations.npz" % seed),
                                **obs_arrays, **D.MARKERS)
            original_ranks = {"chunk_%03d_%s" % (i, key): np.asarray(value)
                              for i, row in enumerate(spy.rows)
                              for key, value in row.items()}
            np.savez_compressed(out / ("seed%d_original_chunks.npz" % seed),
                                **original_ranks, **D.MARKERS)
            (out / ("seed%d_step_evidence.json" % seed)).write_text(json.dumps(
                {**D.MARKERS, "step_info": step_info, "probe_steps": probe.steps,
                 "rng_pre": pre, "rng_post": D.rng_snapshot()},
                indent=2, sort_keys=True, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                else str(v)))
            env.close()
    wall = round(time.time() - t0, 2)
    post = D.rng_snapshot()
    rec, measured, chunks = D.serialize_result(result)
    infer_calls = int(policy.calls)

    steps = R._steps_arrays(probe)
    taged = "seed%d" % seed
    steps["opening_reset"] = np.asarray(reset_open, dtype=np.float64)
    steps["opening_after"] = np.asarray(openings, dtype=np.float64)
    steps["pad_forcen"] = np.stack([panel.traces[c] for c in panel.PV.PAD_ORDER], axis=1)
    steps["pad_forcen_frame"] = np.arange(steps["pad_forcen"].shape[0], dtype=np.int64)
    steps["pad_channel_order"] = np.asarray(panel.PV.PAD_ORDER)
    steps["pad_channel_columns"] = np.asarray([panel.PV.PAD_COLS[c] for c in panel.PV.PAD_ORDER])
    steps["executed_chunk"] = np.arange(len(probe.steps), dtype=np.int64) // args.replan
    steps["executed_chunk_row"] = np.arange(len(probe.steps), dtype=np.int64) % args.replan
    steps.update(D.MARKERS)
    np.savez_compressed(out / ("%s_steps.npz" % taged), **steps)

    payload = {"measured": measured, **D.MARKERS}
    payload.update(spy.stack())
    np.savez_compressed(out / ("%s_actions.npz" % taged), **payload)

    per = spy.per_chunk()
    (out / ("%s_meta.json" % taged)).write_text(json.dumps({
        **D.MARKERS, "seed": seed, "max_steps": args.max_steps, "replan": args.replan,
        "gripper_mode": args.gripper_mode, "record": rec, "panel": stats,
        "infer_calls": infer_calls, "wall_s": wall, "contact_counters_recorded": bool(has_contact),
        "pad_channel_columns": dict(panel.PV.PAD_COLS),
        "trajectory_basis": ("steps.npz row j is control step j == video frame j+1; only the executed "
                             "first replan rows are here; the 32-row prediction lives in actions.npz "
                             "stacked per chunk and is never used as the executed trajectory"),
        "per_chunk_action_chain": per, "chunks": chunks,
        "frame_step_note": steps["row_index_note"].item() if "row_index_note" in steps else None,
    }, indent=2, sort_keys=True, default=str))

    return {"record": rec, "measured": measured, "chunks": chunks, "capture": None,
            "wall_s": wall, "rng_pre": pre, "rng_post": post, "infer_calls": infer_calls,
            "panel": stats, "files": [str(out / ("seed%d_composed_panel.mp4" % seed)),
                                      str(out / ("%s_steps.npz" % taged)),
                                      str(out / ("%s_actions.npz" % taged))]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="single seed (used by --pair)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="seeds for full recording")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--replan", type=int, default=10)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--pair", action="store_true", help="short A(bare)/B(forensics) passive proof")
    ap.add_argument("--gripper-mode", default="pm1", choices=["pm1", "libero01"])
    ap.add_argument("--allow-invalid-scene", action="store_true")
    args = ap.parse_args()

    if not args.pair and not args.seeds and args.seed is None:
        raise SystemExit("pass --pair --seed N, or --seeds N [N ...]")
    max_steps = args.max_steps
    if args.pair and max_steps > 60:
        max_steps = 10  # keep the passive proof cheap; the real seeds pass their own --max-steps

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    E, env_factory, W, fw_facts = D.setup_protocol()
    build_args = _ns(cpu=args.cpu, gripper_mode=args.gripper_mode)
    policy, doc, calib, facts = D.build_policy_reconstructed(E, W, build_args)
    PV = install_cv2_recorder()
    mid = D.merged_identity()
    identity = dict(D.MARKERS)
    identity.update({
        "merged": mid, "fw_import": fw_facts,
        "reconstructed_stats_sha256": D.sha256_file(D.RECON),
        "force_calib_sha256": D.sha256_file(D.CALIB),
        "run_config_sha256": D.sha256_file(D.RUN_CFG),
        "sim_config_sha256": D.sha256_file(D.SIM_CFG),
        "protocol": facts,
        "runtime_overrides": {
            "panel_width_px": 960,
            "camera_width_px_each": 512,
            "panel_recorder_backend": "cv2/mp4v (pod imageio->PyAV rejects quality=; archived "
                                      "panelvideo40k.py untouched; compositor documented fallback)",
            "trajectory": "executed first-replan rows only; 32-row prediction never concatenated as "
                          "trajectory"},
        "seed": args.seed, "seeds": args.seeds, "max_steps": max_steps,
        "replan": args.replan, "gripper_mode": args.gripper_mode,
    })
    (out / "forensics_identity.json").write_text(json.dumps(identity, indent=2, sort_keys=True,
                                                            default=str))

    run_args = _ns(cpu=args.cpu, gripper_mode=args.gripper_mode, max_steps=max_steps,
                   replan=args.replan, allow_invalid_scene=args.allow_invalid_scene)

    if args.pair:
        import torch
        seed = args.seed if args.seed is not None else 12
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        A = D.run_one(E, env_factory, policy, run_args, seed, record_chain=False)
        D.save_run(out, seed, "A", A)
        (out / ("seed%d_A_rng.json" % seed)).write_text(json.dumps(
            {"rng_pre": A["rng_pre"], "rng_post": A["rng_post"], **D.MARKERS},
            indent=2, sort_keys=True, default=str))
        # Persist A before B so a B failure cannot lose the bare-run evidence.
        (out / ("forensics_pair_seed%d_A.json" % seed)).write_text(json.dumps(
            {**D.MARKERS, "seed": seed, "A_record": A["record"], "A_infer_calls": A["infer_calls"],
             "A_wall_s": A["wall_s"], "A_chunks": A["chunks"]}, indent=2, sort_keys=True,
            default=str))
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        B = None
        B_error = None
        try:
            B = run_one_forensics(E, env_factory, policy, W, calib, run_args, seed, out, PV)
        except BaseException as exc:
            import traceback
            B_error = "%s: %s" % (type(exc).__name__, exc)
            (out / ("forensics_pair_seed%d_B_traceback.txt" % seed)).write_text(
                traceback.format_exc())
        cmp = D.compare_pair(A, B) if B is not None else {
            "no_side_effect_bit_exact": False, "comparable": False, "B_error": B_error,
            "note": "B failed; A evidence preserved on disk; forensics pair incomparable, STOP"}
        cmp["tolerance_basis"] = (
            "Bit-exact gate covers recorded episode summary, six input-force channels, predicted "
            "force, chunk metadata and call counts. CPU torch/Python/NumPy RNG are checked separately. "
            "Bare A does not record full action layers, full simulator state or opening; equality of "
            "those quantities and universal absence of side effects are NOT established.")
        if B is not None:
            cmp["rng_gate_passed"] = all(
                all(cmp[key].values()) for key in ("rng_pre_equal", "rng_post_equal"))
            cmp["compared_quantities_gate_passed"] = bool(
                cmp["no_side_effect_bit_exact"] and cmp["rng_gate_passed"])
        (out / ("forensics_pair_seed%d.json" % seed)).write_text(json.dumps(
            {**D.MARKERS, "seed": seed, "comparison": cmp, "A_record": A["record"],
             "B_record": None if B is None else B["record"],
             "B_panel": None if B is None else B.get("panel"), "B_error": B_error,
             "recorder": ("record40k.ActionSpy+EnvProbe+PanelSink (archived, untouched); "
                          "PanelRecorder backend=cv2/mp4v (runtime override)")},
            indent=2, sort_keys=True, default=str))
        print(json.dumps({"mode": "pair", "seed": seed,
                          "no_side_effect_bit_exact": cmp.get("no_side_effect_bit_exact"),
                          "B_error": B_error,
                          "force_max_abs_diff": cmp.get("force_trajectory", {}).get("max_abs_diff"),
                          "out": str(out)}, indent=2))
        if B_error is not None:
            return 1
        return 0 if cmp.get("compared_quantities_gate_passed", False) else 2

    results = []
    for seed in args.seeds:
        import torch
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        B = run_one_forensics(E, env_factory, policy, W, calib, run_args, seed, out, PV)
        results.append({"seed": seed, "record": B["record"], "panel": B.get("panel"),
                        "wall_s": B["wall_s"], "infer_calls": B["infer_calls"],
                        "files": B["files"]})
        print("[forensics] " + json.dumps(results[-1], default=str), flush=True)
    (out / "forensics_summary.json").write_text(json.dumps(
        {**D.MARKERS, "seeds": args.seeds, "max_steps": max_steps, "replan": args.replan,
         "results": results}, indent=2, sort_keys=True, default=str))
    print(json.dumps({"mode": "record", "seeds": args.seeds, "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())