"""Prove the recorder records, without loading a model.

`record40k.py` is only worth running on the GPU once every moving part it adds has been shown to
move: the three camera writers, the composed [front | left_wrist | panel] video and the four
fingertip traces behind it, the overlay, the four-stage action chain, the IK/limit captures, the
emergency-stop reason, the contact counters and the proprio witness that the 160-episode batch
recorded as `None` on every single episode. Each of those can fail while the rollout itself looks
fine, and the version of that failure that costs the most is the one found after 25 minutes of
inference.

So this drives the *real* MuJoCo environment, cameras and all, with a stub policy, and asserts the
files and fields a diagnostic run is supposed to produce. It touches no weights and no GPU. The
stub's deliberately wild chunk (586 on `d_ry`, the value seed 17 would have needed) checks that a
saturated command is reported as a saturated command rather than as a 44 m jump, and its injected
`emergency_stop` checks that the reason string now survives into the record.

    python3 recorder_selftest40k.py            # ~1 minute, CPU + EGL only
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval40k as E  # noqa: E402
import record40k as R  # noqa: E402
import smoke40k as S  # noqa: E402
from rollout40k import run_episode  # noqa: E402

CALIB = Path("/workspace/a3_data/a3_fw_force/force_calib_v1.json")
OUT = Path("/tmp/rectest_selftest")
WILD = 586.0
DIM_NAMES = R.DIM_NAMES


class _FakeNormalizer:
    """Just enough of the live `state` normalizer for `live_normalizer_band` to read."""

    def __init__(self, lo, hi):
        lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
        self.stats = {"min": lo, "max": hi}
        self.mode = "min/max"
        self.scale = 2.0 / (hi - lo)
        self.offset = -1.0 - self.scale * lo


def _fake_processor():
    lo = np.linspace(-1.0, -0.4, E.PROPRIO_DIM)
    hi = np.linspace(0.4, 1.0, E.PROPRIO_DIM)
    a_lo, a_hi = np.asarray([-0.3] * 6 + [-1.0], np.float64), np.asarray([0.3] * 6 + [1.0], np.float64)
    return type("P", (), {"normalizer": type("N", (), {"normalizers": {
        "state": {"default": _FakeNormalizer(lo, hi)},
        "action": {"default": _FakeNormalizer(a_lo, a_hi)}}})()})()


class StubPolicy:
    """The shape of `ForcePolicy.chunk`, with a hand-made spike so the chain tables are testable."""

    def __init__(self, horizon=32, n_force=6):
        self.horizon, self.n_force = horizon, n_force
        self.calls = 0
        self.env_ref = None
        self.bridge = self
        self.adapter = self
        self.processor = _fake_processor()
        self.frames = []

    def start_episode(self, obs, seed):
        self.frames = [obs]

    def observe(self, obs, frame):
        self.frames.append(obs)

    def denormalize_action(self, action, processor):
        return (np.asarray(action, dtype=np.float32) * 0.05).astype(np.float32)

    def convert(self, a, rot):
        out = np.zeros(14, dtype=np.float32)
        out[:6] = np.asarray(a, np.float32)[:6] * 2.0
        out[6] = -np.sign(float(a[6]))
        out[7:13] = out[:6]
        out[13] = out[6]
        return np.clip(out, -1.0, 1.0)

    def chunk(self, obs, frame, seed):
        self.calls += 1
        model = np.zeros((1, self.horizon, 7), dtype=np.float32)
        model[0, :, :6] = 0.2 * np.sin(np.arange(self.horizon)[:, None] + self.calls)
        model[0, :, 6] = 0.5
        if self.calls == 2:
            model[0, 3, 4] = WILD
        denorm = self.bridge.denormalize_action(model, self.processor)[0]
        denorm[..., -1] = np.where(denorm[..., -1] >= 0.0, 1.0, -1.0)
        env_actions = [self.adapter.convert(a, np.eye(3)) for a in denorm]
        if self.env_ref is not None and self.calls == 3:
            self.env_ref.emergency_stop("selftest: injected stop")
        meta = {"infer_s": 0.0, "replan_from_frame": int(frame), "noise_seed": seed + self.calls,
                "action_absmax": float(np.abs(denorm).max()), "force_feedback_applied": False,
                "history_mask_true": 16}
        return env_actions, np.zeros((self.horizon, self.n_force), np.float64), meta


RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print("%-62s %s  %s" % (name, "PASS" if ok else "FAIL", detail), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env_factory, W = E.build_env(None)
    R.STATE["W"] = W
    R.STATE["calib"] = E.resolve_calib(str(CALIB), None)
    policy = StubPolicy()

    band = R.live_normalizer_band(policy, "state")
    check("the band read off scale/offset equals the stats band",
          band.get("ok") and band.get("max_disagreement", 1) < 1e-9,
          "mode=%s diff=%.2e" % (band.get("mode"), band.get("max_disagreement", -1)))

    R.patch_witness_band()
    R.install_stash()
    R.stash_logs()
    R.STATE["policy"] = policy
    check("STATS_FACTS really is empty in this process, so the fallback is what is being tested",
          not E.STATS_FACTS.get("bands"), "bands=%r" % (E.STATS_FACTS.get("bands"),))
    witness = S.make_proprio_witness(W, R.STATE["calib"])
    check("the proprio witness builds from the live band (the batch got None)",
          callable(witness), str(S.stats_band_vectors.__name__))

    env = env_factory.make_env(position_noise_m=0.02, yaw_noise_rad=0.05, render_cameras=True)
    policy.env_ref = env
    obs0 = env_factory.reset(env, 7, randomize_cookies=True, randomize_boxes=False,
                             batch_offset_m=0.0)[0]
    w = witness(obs0)
    check("witness(obs) returns 14 raw and 14 unclipped mapped columns",
          w is not None and len(w["raw14"]) == E.PROPRIO_DIM
          and len(w["mapped_no_clamp14"]) == E.PROPRIO_DIM, "" if w else "-> None")
    check("the env hands the recorder all three camera keys",
          all(k in obs0 for _n, k in R.CAM_KEYS),
          str(sorted(k for k in obs0 if str(k).startswith("observation.images"))))

    def reset(e, s):
        return env_factory.reset(e, s, randomize_cookies=True, randomize_boxes=False,
                                 batch_offset_m=0.0)[0]

    R.STATE["chunk_log"], R.STATE["calls"], R.STATE["witness"] = [], [], witness
    ctx = R.Ctx(OUT / "_record", 40000, R.CAM_KEYS, no_physics=False, summarize=False,
                max_steps=45, panel_video=True)
    rec, _measured, chunks = run_episode_probe(ctx, env, policy, reset)
    d = OUT / "_record" / "seed7"
    meta = json.loads((d / "seed7_step40000_meta.json").read_text())

    files = meta["video_files"]
    check("three mp4s written, one per camera, all non-trivial",
          len(files) == 3 and all(v["frames"] > 1 and v["bytes"] > 1000 for v in files.values()),
          json.dumps(files))
    check("one video frame per control step, plus the reset frame",
          all(v["frames"] == rec["steps_executed"] + 1 for v in files.values()),
          "frames=%s steps_executed=%s" % ([v["frames"] for v in files.values()],
                                           rec["steps_executed"]))
    import imageio

    fr = np.asarray(imageio.get_reader(str(d / "seed7_front_step40000.mp4")).get_data(0))
    check("the front video decodes back to an RGB frame", fr.ndim == 3 and fr.shape[2] == 3,
          str(fr.shape))
    check("the overlay drew ink (a blank render would be near-flat)",
          float(fr.std()) > 8.0, "std=%.1f" % float(fr.std()))

    chain = meta["per_chunk_action_chain"]
    wild = [c for c in chain if c.get("model_argmax")
            and abs(c["model_argmax"]["value"]) > 100]
    check("the injected 586 is attributed to one dimension of one chunk",
          len(wild) == 1 and wild[0]["model_argmax"]["dim_name"] == "d_ry",
          json.dumps(wild[0]["model_argmax"]) if wild else "none found")
    check("at the adapter output the same chunk only saturates at +-1",
          len(wild) == 1
          and max(abs(x) for x in wild[0]["adapter_absmax_per_dim"]) <= 1.0 + 1e-6
          and wild[0]["adapter_saturated_rows"] > 0,
          "absmax=%s saturated_rows=%s/%s" % (wild[0]["adapter_absmax_per_dim"],
                                              wild[0]["adapter_saturated_rows"],
                                              wild[0]["adapter_rows"]) if wild else "")
    worst = meta["worst_action_chain"]
    check("worst_action_chain reports the same row at all four stages",
          worst is not None and worst["worst_dim_name"] == "d_ry"
          and abs(worst["model_out_value"] - WILD) < 1e-3 and len(worst["pm1_row"]) == 7
          and len(worst["adapter_out_row"]) == 14,
          json.dumps({k: worst[k] for k in ("worst_dim_name", "model_out_value", "denorm_value")})
          if worst else "None")

    steps = np.load(str(d / "seed7_step40000_steps.npz"))
    check("per-step npz holds the whole executed chain",
          all(k in steps.files for k in ("handed_14d", "requested_14d", "ik_joint_target_16d",
                                         "applied_joint_16d", "proprio14")), str(sorted(steps.files)))
    ik = steps["ik_joint_target_16d"]
    check("IK joint targets captured for the steps that moved",
          ik.ndim == 2 and ik.shape[1] == 16 and ik.shape[0] > 0,
          "shape=%s (16 = 7 arm joints + gripper per arm, not the 14-wide cartesian command)"
          % (ik.shape,))
    check("the applied joint command is the rate-limited version of the same 16 columns",
          steps["applied_joint_16d"].shape == ik.shape and ik.shape[0] > 0,
          "applied=%s" % (steps["applied_joint_16d"].shape,))
    check("requested_14d is the clipped command validate_action handed IK",
          steps["requested_14d"].shape[1] == 14
          and float(np.abs(steps["requested_14d"]).max()) <= 1.0 + 1e-6,
          "absmax=%.3f" % float(np.abs(steps["requested_14d"]).max()))
    check("proprio14 per frame is the 14 columns the model conditions on",
          steps["proprio14"].shape[1] == E.PROPRIO_DIM, str(steps["proprio14"].shape))
    check("joint targets stay finite and inside a sane radian range",
          ik.size and bool(np.all(np.isfinite(ik))) and float(np.abs(ik).max()) < 20.0,
          "absmax=%.2f rad" % (float(np.abs(ik).max()) if ik.size else -1))

    # --- the composed [front | left_wrist | panel] video, which is the deliverable the user asked
    # for and the one thing the three per-camera mp4s cannot substitute for.
    import panelvideo40k as PV

    pv = meta["panel_video"]
    check("the composed panel video comes from this same rollout, one frame per control step",
          pv["frames_written"] == rec["steps_executed"] + 1 and pv["bytes"] > 1000,
          "%d frames, %.1f KB, %s" % (pv["frames_written"], pv["bytes"] / 1024.0, pv["wh"]))
    check("its geometry is the historical result video's: 512 px per camera (2x the env's 256) and "
          "a 576 px canvas",
          pv["wh"] == [1664, 576] == [2 * PV.CAM_W + PV.PANEL_W, PV.PANEL_H], str(pv["wh"]))
    check("the encoder is the one the pod's own recordings use, and is reported rather than "
          "assumed", pv["backend"].startswith("imageio/libx264"), "%s  notes=%s"
          % (pv["backend"], pv["notes"]))
    back = PV.read_frames(OUT / "_record" / "seed7" / "seed7_step40000_composed_panel.mp4")
    check("it decodes back with the frame count and geometry it claims",
          len(back) == pv["frames_written"]
          and back[0].shape == (PV.PANEL_H, 2 * PV.CAM_W + PV.PANEL_W, 3),
          "%d frames, %s" % (len(back), None if not back else list(back[0].shape)))
    pads = steps["pad_forcen"]
    check("the four fingertip traces the panel drew are in the steps npz, one row per video frame",
          pads.shape == (pv["frames_written"], 4)
          and list(steps["pad_forcen_frame"]) == list(range(pads.shape[0]))
          and list(steps["pad_channel_order"]) == list(PV.PAD_ORDER)
          and list(steps["pad_channel_columns"]) == [PV.PAD_COLS[c] for c in PV.PAD_ORDER],
          "shape=%s order=%s columns=%s" % (pads.shape, list(steps["pad_channel_order"]),
                                            list(steps["pad_channel_columns"])))
    zero = np.asarray(R.STATE["calib"]["zero_load_ref_median"], dtype=np.float64)
    p14 = steps["proprio14"].astype(np.float64)
    fr = steps["frame"]
    # The two tables differ by the reset frame on purpose: `pad_forcen` row k is video frame k, while
    # row j of `frame`/`proprio14`/`handed_14d` is control step j, which is video frame j+1. Joining
    # them on their own row index is a one-frame shift, so they are joined on the frame each names.
    check("the steps table starts at frame 1 and the force table at frame 0, one row apart by design",
          p14.shape[0] == fr.shape[0] == len(steps["handed_14d"])
          and int(fr[0]) == 1 and list(fr) == list(range(1, pads.shape[0])),
          "frame[0]=%d, steps rows=%d, pad rows=%d" % (int(fr[0]), fr.shape[0], pads.shape[0]))
    # proprio14 columns 11/12 are raw force channels 12/13 minus the tare (force_stream.CHANNELS =
    # [0,1,2,12,13,16]), so this is the one arithmetic fact the whole panel rests on: the headline
    # number and the number the model conditioned on are the same sensor.
    d_l = np.abs(pads[fr, 0] - zero[3] - p14[:, 11])
    d_r = np.abs(pads[fr, 1] - zero[4] - p14[:, 12])
    check("the panel's L_pad inner/outer are the sensors behind proprio14 columns 11/12",
          float(d_l.max()) < 1e-2 and float(d_r.max()) < 1e-2,
          "max |pad_forcen - tare - proprio14| = %.2e / %.2e N" % (d_l.max(), d_r.max()))
    tared = [int(c) for c in R.STATE["calib"]["channels"]]
    check("the right fingertip has no zero-load tare to subtract, which is why the panel labels it "
          "raw and shows the tared left pair separately",
          PV.PAD_COLS["r_inner"] not in tared and PV.PAD_COLS["r_outer"] not in tared
          and PV.PAD_COLS["l_inner"] in tared and PV.PAD_COLS["l_outer"] in tared,
          "tared env columns=%s, pad columns=%s"
          % (tared, [PV.PAD_COLS[c] for c in PV.PAD_ORDER]))

    es = meta["emergency_stops"]
    check("the injected emergency stop is recorded with its reason and frame",
          len(es) >= 1 and any("selftest" in " ".join(e["args"]) for e in es)
          and all(isinstance(e.get("frame"), int) for e in es), json.dumps(es[:2]))
    reasons = [r for r in meta["step_safety_reasons"] if r]
    check("the env's own info[safety_reason] reached the per-step record",
          any("selftest" in str(r) for r in reasons), json.dumps(reasons[:2]))
    check("the episode ended on that stop, not on the step budget",
          bool(rec.get("terminated")) and rec["steps_executed"] < ctx.max_steps,
          "terminated=%s steps=%s/%s" % (rec.get("terminated"), rec["steps_executed"],
                                         ctx.max_steps))

    ph = meta["physics"]
    check("timing rollup names env.step 总耗时 and its split",
          "env_step_s_total" in ph and "mj_step_s_total" in ph and "ik_convert_s_total" in ph,
          "env_step=%ss unattributed_share=%s" % (ph.get("env_step_s_total"),
                                                 ph.get("unattributed_share")))
    check("every probe hook that the real env exposes was installed",
          set(R.EXPECTED_INSTRUMENTS) <= set(ph["instrumented"]),
          "got=%s" % sorted(ph["instrumented"]))
    check("contact counters recorded (ncon/nefc/solver iterations)",
          ph.get("ncon_max") is not None and ph.get("nefc_max") is not None,
          "ncon max/mean=%s/%s nefc=%s solver_niter=%s" % (ph.get("ncon_max"), ph.get("ncon_mean"),
                                                           ph.get("nefc_max"),
                                                           ph.get("solver_niter_max")))
    check("the band source says where the +-5 interlock read its band",
          bool(meta["band_source_state"]), str(meta["band_source_state"]))

    n_fail = RESULTS.count(False)
    print("\n%s: %d/%d checks passed   artifacts: %s"
          % ("FAIL" if n_fail else "PASS", len(RESULTS) - n_fail, len(RESULTS), d))
    return 1 if n_fail else 0


def run_episode_probe(ctx, env, policy, reset):
    """Call the recorder's capture path with `rollout40k.run_episode` underneath."""
    return R.run_episode_capture(run_episode, ctx, env, policy, 7, reset,
                                 max_steps=ctx.max_steps, replan=10)


if __name__ == "__main__":
    sys.exit(main())
