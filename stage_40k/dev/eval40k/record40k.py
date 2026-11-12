"""A closed-loop recording run: the batch's own protocol, plus everything it never stored.

`eval40k.py` scores an episode and keeps a force curve and one scalar per chunk
(`action_absmax = np.abs(action).max()`). It therefore cannot say *which* of the seven action
dimensions a 44.4 landed in, nor what the robot was doing on screen when a contact spike appeared,
because the per-dimension action, the joint target the simulator was handed, the camera frames and
the emergency-stop reason were never written. The released deployment worker did keep some of that
(`infer_a3_worker` saves `actions_7d` and `executed_14d` next to an `episode_*_front.mp4`); the
independent harness inherited the scoring and not the evidence.

This script re-runs chosen seeds and writes all of it. It wraps rather than edits: `eval40k.main()`
is invoked with the batch's own command line, so the gates, the merged artifact, the scene
randomisation and the inference knobs stay exactly the ones the official numbers came from. Two
properties make that safe to do while a batch is live: the batch launches `eval40k.py` as a fresh
subprocess per node (`batch_eval.py:191`, `:124`), so *editing* a pinned file would change the
protocol of nodes that have not started, whereas this file is in nobody's import graph; and the
toolchain check at step 0a of `RUN_BATCH_STEP6.sh` verifies a fixed list of seven file names, which
a new file cannot disturb.

What is captured, and why each one is the number the current questions turn on:

* the whole action chain, per replan, per dimension -- the tensor the model returned, the same
  rows after `denormalize_action`, after the pm1 gripper rewrite, and the 14-wide array the adapter
  handed back. `action_absmax` alone cannot separate "the sampler emitted 586 on `d_rx`" from
  "the interface multiplied something by 586".
* per control step, on the simulator side: the 14-wide action `run_episode` handed to `env.step`,
  what `validate_action` left of it (clipped to +-1 by `sim/env.py:104-109`), the joint target
  `_ik.convert` solved, and `_limit_joint_action`'s output. That is the last link: a command that
  saturates at 1.0 is a full-rate step, not a 44 m jump, and the joint target is what proves it.
* `info["safety_reason"]` and a wrap of `env.emergency_stop`, with the frame it fired on. The env
  has carried a reason string all along (`sim/env.py:222-224`, `:289`); the batch never read it, so
  an emergency stop was reported as a bare count.
* three 256x256 camera streams at `control_hz` (20 Hz -- one frame per control step, so the force
  read next to a frame *is* that frame's force), with the numbers the model was conditioned on
  burned in: the six tared newton channels, the left EEF position, the commanded step, and a red
  banner the frame the safety stop fired.
* `env.step` split into `mj_step`, `mj_forward`, observation render/assembly and IK, plus
  `ncon`/`nefc`/`solver_niter` per step, so "the simulator was slow" names a line.

Two interlocks, because the expensive failure modes are silent ones:

* a rollout is a closed loop, so the video cannot be reconstructed from stored data -- this needs a
  fresh inference pass on the same GPU as anything else running. The VRAM floor and the allocator
  cap exist for that asymmetry: the failure this script is allowed to cause is its own, never a
  neighbour's.
* `smoke40k.stats_band_vectors()` reads `eval40k.STATS_FACTS`, and the batch's every recorded
  episode has `witnessed == None` / `proprio_band_map_pairs == 0` because `python eval40k.py` binds
  the file to `__main__` while `smoke40k` imports a *second* `eval40k` module whose `STATS_FACTS` is
  empty. That made the +-5 clamp interlock vacuous. Running through this file avoids the double
  module by construction, and on top of that the band here is read off the live processor's own
  normalizer, and the recorder refuses to start a rollout whose witness is `None`.

Usage -- copy the live command line verbatim and point `--out` somewhere else::

    python record40k.py --steps 40000 --seeds 11 17 --backend env --policy merged \\
        --out /workspace/_v1_stage/dev/out40k/record_step040000 \\
        --merged-dir /workspace/_v1_stage/dev/eval40k/merged --batch-offset 0.0 \\
        --rec-panel-video \\
        --calib /workspace/a3_data/a3_fw_force/force_calib_v1.json

`--rec-panel-video` adds, per seed, the composed recording the report asks for: front and left_wrist
side by side in one mp4 with all numbers, both fingertip force channels and the action chain in a
separate panel on the right (`panelvideo40k.py`). It is refused unless a synthetic pass of the whole
panel path writes and decodes first, and the three per-camera files are still written, because the
camera-only videos are what the batch's own evidence trail compares against.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval40k as E  # noqa: E402  (module level only builds paths; all work is in main())

OFFICIAL_OUT = Path("/workspace/_v1_stage/dev/out40k/batch_eval_20seed1x")
CAM_KEYS = (("front", "observation.images.front"),
            ("left_wrist", "observation.images.left_wrist"),
            ("right_wrist", "observation.images.right_wrist"))
DIM_NAMES = ("d_x", "d_y", "d_z", "d_rx", "d_ry", "d_rz", "gripper")
# what `EnvProbe.install_on_env` must find on the real env for the diagnostics to mean anything; a
# rename on the sim side shows up here as a missing name instead of a silently absent column
EXPECTED_INSTRUMENTS = ("_observation", "_ik.convert", "_limit_joint_action", "emergency_stop",
                        "mujoco.mj_step", "mujoco.mj_forward")
FORCE_NAMES = ("L_wrist_fx", "L_wrist_fy", "L_wrist_fz",
               "L_pad_inner", "L_pad_outer", "L_gripper_act")
ADAPTER_NOTE = ("units: the 7-wide action is denormalised to the V1G training band (dims 0-5 are "
                "LIBERO OSC delta units, 1.0 = 0.05 m / 0.5 rad), the adapter doubles 0-2 into A3 "
                "units and rescales 3-5 to rad/0.1, then clips the 14-wide command to [-1, 1]; the "
                "env then clips again in validate_action, so what reaches IK is a rate command in "
                "+-1 and the executed motion is the joint target, not the number in column 2")
# A pad channel is "in contact" here once it leaves the noise floor. It only ever draws a tick
# marking where the curve leaves zero, so the number is a display threshold, not a claim about the
# simulation; the curve itself is what a reader should judge contact from.
CONTACT_ON_N = 20.0
PANEL_CAM_NAMES = ("front", "left_wrist")

# Stashed by the wrapped builders below so the recording code can reach the same objects the
# rollout is using, without editing any of the seven pinned files.
STATE: dict = {}


def _to_uint8(img):
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a
    a = np.nan_to_num(a.astype(np.float64))
    if a.max() <= 2.0:
        a = a * 255.0
    return np.clip(a, 0.0, 255.0).astype(np.uint8)


# --------------------------------------------------------------------------- video

class Overlay:
    """Burn the numbers the model saw into each frame, with PIL."""

    def __init__(self, seed, step):
        self.seed, self.step = int(seed), step
        self.font = self._font(13)
        self.small = self._font(11)

    @staticmethod
    def _font(size):
        from PIL import ImageFont

        for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            if Path(path).is_file():
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    pass
        return ImageFont.load_default()

    def draw(self, frame, ann):
        from PIL import Image, ImageDraw

        img = Image.fromarray(np.ascontiguousarray(frame)).convert("RGB")
        d = ImageDraw.Draw(img, "RGBA")
        w, h = img.size
        d.rectangle([0, 0, w, h], fill=(0, 0, 0, 0))
        head = "seed%d step%s  f%d/%s" % (self.seed, self.step, ann["frame"], ann.get("total", "?"))
        if ann.get("success") is not None:
            head += "  in_target=%s" % ann["success"]
        d.rectangle([0, 0, w, 13], fill=(0, 0, 0, 150))
        d.text((3, 1), head, font=self.font, fill=(255, 255, 255, 255))

        y = 16
        for line in ann["lines"]:
            d.rectangle([0, y, w, y + 12], fill=(0, 0, 0, 120))
            d.text((3, y + 1), line, font=self.small, fill=(180, 255, 180, 255))
            y += 12
        if ann.get("warn"):
            d.rectangle([0, y + 1, w, y + 15], fill=(150, 0, 0, 200))
            d.text((3, y + 2), ann["warn"], font=self.small, fill=(255, 255, 255, 255))
        return np.asarray(img)


class FrameSink:
    """One mp4 per camera, written as the rollout proceeds.

    Streaming rather than accumulating: 600 control steps x three 256x256x3 panels is ~350 MB of
    Python memory per episode, and an episode that dies at frame 590 should still leave two
    watchable files, not a lost rollout.
    """

    def __init__(self, root, seed, step, fps, cams, overlay):
        import imageio

        self.root, self.overlay = Path(root), overlay
        self.fps, self.cams = int(fps), list(cams)
        self.writers = {}
        self.frames = {name: 0 for name, _ in self.cams}
        self.suffix = "step%s" % step if step is not None else "stepNA"
        self._imageio = imageio

    def add(self, obs, ann):
        for name, key in self.cams:
            if key not in obs:
                continue
            img = self.overlay.draw(_to_uint8(obs[key]), ann)
            wr = self.writers.get(name)
            if wr is None:
                self.root.mkdir(parents=True, exist_ok=True)
                wr = self._imageio.get_writer(self.path(name), fps=self.fps, codec="libx264",
                                              quality=8, macro_block_size=16,
                                              ffmpeg_params=["-pix_fmt", "yuv420p"])
                self.writers[name] = wr
            wr.append_data(img)
            self.frames[name] += 1

    def path(self, name):
        return self.root / ("seed%d_%s_%s.mp4" % (self.overlay.seed, name, self.suffix))

    def close(self):
        for wr in self.writers.values():
            wr.close()
        self.writers = {}
        return {n: {"frames": self.frames[n], "bytes": p.stat().st_size if p.is_file() else 0}
                for n, p in ((n, self.path(n)) for n, _ in self.cams)}


# --------------------------------------------------------------------------- composed panel video

CHAIN_NOTES = (
    "force: raw env newtons, cols 0-5 and 12-15 of the 18-wide observation.force",
    "no zero-load tare on R_pad: force_calib_v1 tares only channels 0,1,2,12,13,16",
    "predicted future force is never drawn here; nefc/ncon are counts, not newtons",
    "frame 0 = reset obs (no action yet); chain: model -> denorm -> pm1 -> env.step cmd",
)


class PanelSink:
    """One mp4 of [front | left_wrist | diagnostics], built from the live rollout as it runs.

    It is fed from the same place the per-camera sinks are, so the composed video and the batch's
    own recordings come from the identical frames -- this cannot be a re-render of stored data,
    because a closed-loop rollout only has its observations while it is running.

    Two things are deliberately *not* read here. `pred["force_future_newton"]` is never looked at:
    every newton on the panel comes out of `obs[FORCE_OBS_KEY]`, so the curve is a measurement the
    model cannot influence. And nothing is summed across fingertips: each of the four pad channels
    keeps its own column and its own curve, because the question on the table is which fingertip
    touched when, and a total would answer a different one.
    """

    def __init__(self, path, fps, seed, step, total_frames, replan, extra_notes=None):
        import panelvideo40k as PV

        self.PV = PV
        self.rec = PV.PanelRecorder(path, fps=fps)
        self.replan = int(replan)
        self.total_frames = int(total_frames)
        self.seed, self.step = int(seed), step
        self.traces = {c: [] for c in PV.PAD_ORDER}
        self.emergency = []
        self.notes_extra = list(extra_notes or [])
        self.chain_fn = None          # set by run_episode_capture once the ActionSpy exists
        self.frames_written = 0

    # -- force
    def pads(self, obs):
        from force_stream import FORCE_OBS_KEY

        if FORCE_OBS_KEY not in obs:
            raise SystemExit("the composed panel video needs the env's 18-wide %r; the observation "
                             "has %s" % (FORCE_OBS_KEY, sorted(str(k) for k in obs)[:8]))
        return self.PV.force18_pads(obs[FORCE_OBS_KEY])

    def _ticks(self, frame):
        from panelvideo40k import FINGERTIP, PAD_ORDER

        ticks = []
        for side in ("L", "R"):
            cols = FINGERTIP[side]
            first = min((i for ch in cols for i, v in enumerate(self.traces[ch])
                         if abs(float(v)) >= CONTACT_ON_N), default=None)
            if first is not None and first <= frame:
                ticks.append({"frame": int(first),
                              "text": "%s_pad first contact >=%.0f N" % (side, CONTACT_ON_N)})
        for e in self.emergency:
            if e["frame"] <= frame:
                ticks.append({"frame": int(e["frame"]), "text": "emergency stop"})
        return ticks

    def _chain(self, frame):
        if self.chain_fn is None or frame < 1:
            return {}
        out = self.chain_fn(frame)
        return out or {}

    def add_frame(self, obs, rec, p14):
        """One control step: append this frame's force, then write one composed frame."""
        from panelvideo40k import CAM_SRC

        frame = int(rec["frame"])
        if frame != self.frames_written:
            raise RuntimeError("panel sink asked for frame %d having written %d"
                               % (frame, self.frames_written))
        pads, wrist, torque = self.pads(obs)
        for ch in self.PV.PAD_ORDER:
            self.traces[ch].append(pads[ch])
        cam = {name: obs[key] for name, key in CAM_KEYS if name in PANEL_CAM_NAMES}
        if set(cam) != set(PANEL_CAM_NAMES):
            raise SystemExit("the composed video needs the %s cameras; the observation has %s"
                             % (list(PANEL_CAM_NAMES), sorted(str(k) for k in obs)[:8]))
        for name, img in cam.items():
            a = _to_uint8(img)
            if a.shape[:2] != (CAM_SRC, CAM_SRC):
                raise SystemExit("the composed video needs the %s camera at the env's native "
                                 "%dx%d, got %s" % (name, CAM_SRC, CAM_SRC, a.shape[:2]))
        h = rec.get("handed_14d")
        gripper = "?"
        if h and len(h) >= 14:
            gripper = "L%+d R%+d" % (int(round(h[6])), int(round(h[13])))
        ann = {"frame": frame, "total_frames": self.total_frames, "seed": self.seed,
               "step": self.step, "ctrl_step": max(0, frame - 1),
               "chunk": (frame - 1) // self.replan if frame >= 1 else 0,
               "pad_series": {c: np.asarray(self.traces[c], dtype=np.float64)
                              for c in self.PV.PAD_ORDER},
               "pad_n": dict(pads),
               "wrist_n": list(wrist), "wrist_torque_nm": list(torque),
               "gripper": gripper, "success": rec.get("cookies_in_target"),
               "terminated": bool(rec.get("terminated")),
               "safety_reason": rec.get("safety_reason"),
               "ticks": self._ticks(frame),
               "notes": list(CHAIN_NOTES) + list(self.notes_extra)}
        # Two rows, each read off the variable its label names. `handed_14d` is the array this control
        # step literally passed to `env.step`, so it is the adapter's return value *after* its own
        # `np.clip(output, -1, 1)` (action_adapter_v3.py:53) -- it is never the pre-clip number the
        # report argues about. `requested_14d` is what IK received, i.e. `validate_action(handed,
        # "cartesian_delta")`, which raises rather than clips above +-1 (contracts.py:60-61); printing
        # both makes that equality an observation instead of an assumption. Frame 0 is the reset
        # observation, before any action exists, so it supplies neither and the panel draws no action
        # line rather than a line of zeros.
        if h and len(h) >= 14:
            ann["executed_row"] = [float(x) for x in h]
            ik = rec.get("requested_14d")
            if ik and len(ik) >= 14:
                ann["ik_cmd_row"] = [float(x) for x in ik]
        if p14 is not None and len(p14) >= 14:
            ann["pad_l_tared_n"] = {"l_inner": float(p14[11]), "l_outer": float(p14[12])}
        ann.update(self._chain(frame))
        self.rec.add(_to_uint8(cam["front"]), _to_uint8(cam["left_wrist"]), ann)
        self.frames_written += 1

    def close(self):
        stats = self.rec.close()
        stats["pad_trace_samples"] = {c: len(self.traces[c]) for c in self.PV.PAD_ORDER}
        stats["pad_channel_columns"] = dict(self.PV.PAD_COLS)
        stats["force_columns_note"] = (
            "l_inner/l_outer/r_inner/r_outer are columns 12/13/14/15 of the env's 18-wide "
            "observation.force, raw newtons; the left pair minus force_calib_v1's zero-load "
            "reference is proprio14 columns 11/12, which is what the model conditioned on. The "
            "trace itself is in the steps npz as pad_forcen, one row per video frame.")
        return stats


def chain_for_frame(spy, replan, frame, clamp=5.0):
    """The action-chain rows that produced control step `frame`, in the units each stage uses.

    Chunk c = (frame-1) // replan supplies the row; the policy proposes 32 rows per replan and the
    rollout executes the first `replan` of them, so row r = (frame-1) % replan. Everything here is
    read off the *stored* arrays rather than recomputed, so the panel cannot disagree with the npz
    written beside it.
    """
    rows = getattr(spy, "rows", None)
    if not rows:
        return {}
    c, r = divmod(int(frame) - 1, int(replan))
    if c >= len(rows):
        return {}
    m = np.asarray(rows[c]["model_out"], dtype=np.float64).reshape(-1, 7)
    dn = np.asarray(rows[c]["denorm"], dtype=np.float64).reshape(-1, 7)
    if r >= m.shape[0]:
        return {}
    pm1 = np.asarray(rows[c].get("pm1") or [], dtype=np.float64)
    out = {"model_absmax_dims": [float(x) for x in np.abs(m).max(0)],
           "model_over5_frac": [float(x) for x in np.mean(np.abs(m) > clamp, axis=0)],
           "denorm_row": [float(x) for x in dn[r]]}
    if pm1.size:
        out["pm1_row"] = [float(x) for x in pm1.reshape(-1, 7)[r]]
    return out



# --------------------------------------------------------------------------- action chain

class ActionSpy:
    """Every stage of one replan's action, per dimension.

    `ForcePolicy.chunk` denormalises, rewrites the gripper column, then converts all 32 steps
    through the adapter. `denormalize_action` is reached through `self.bridge` and `convert` through
    `self.adapter`, both instance attributes, so both are replaced on the instance -- the module
    objects a live batch shares are untouched.
    """

    def __init__(self):
        self.rows = []

    def install(self, policy):
        real_bridge = policy.bridge
        spy = self

        class _Bridge:
            def __getattr__(self, name):
                return getattr(real_bridge, name)

            def denormalize_action(self, action, processor):
                out = real_bridge.denormalize_action(action, processor)
                spy.rows.append({"model_out": np.array(np.asarray(action, dtype=np.float32),
                                                       dtype=np.float32, copy=True),
                                 "denorm": np.array(out, dtype=np.float32, copy=True)})
                return out

        policy.bridge = _Bridge()

        real_convert = policy.adapter.convert

        def convert(a, rot):
            out = real_convert(a, rot)
            if spy.rows:
                spy.rows[-1].setdefault("pm1", []).append(np.array(a, dtype=np.float32, copy=True))
                spy.rows[-1].setdefault("adapter_out", []).append(
                    np.array(out, dtype=np.float32, copy=True))
                spy.rows[-1].setdefault("rot_flat", np.array(rot, dtype=np.float32).reshape(-1))
            return out

        policy.adapter.convert = convert
        return self

    def per_chunk(self):
        """Per replan: which dimension each stage put its maximum in, and what IK was told."""
        out = []
        for i, r in enumerate(self.rows):
            m, dn = r["model_out"], r["denorm"]
            m2 = m.reshape(-1, m.shape[-1]) if m.ndim else m
            d2 = dn.reshape(-1, dn.shape[-1]) if dn.ndim else dn
            pm1 = np.asarray(r.get("pm1") or [], dtype=np.float32)
            ao = np.asarray(r.get("adapter_out") or [], dtype=np.float32)
            entry = {"chunk": i,
                     "model_out_shape": list(m.shape),
                     "model_absmax_per_dim": [float(x) for x in np.abs(m2).max(0)] if m2.size else None,
                     "denorm_absmax_per_dim": [float(x) for x in np.abs(d2).max(0)] if d2.size else None}
            flat = np.abs(m2).reshape(-1)
            if flat.size:
                t, d = divmod(int(np.argmax(flat)), m2.shape[-1])
                entry["model_argmax"] = {"row": t, "dim": int(d),
                                         "dim_name": DIM_NAMES[int(d)] if d < 7 else str(d),
                                         "value": float(m2.reshape(-1, m2.shape[-1])[t, d])}
            if ao.size:
                a2 = ao.reshape(-1, ao.shape[-1])
                entry["adapter_absmax_per_dim"] = [float(x) for x in np.abs(a2).max(0)]
                # exactly 1.0 is the adapter's own clip firing, not a value it computed
                entry["adapter_saturated_rows"] = int(np.sum(np.any(np.abs(a2) >= 1.0, axis=1)))
                entry["adapter_rows"] = int(a2.shape[0])
            if pm1.size:
                entry["pm1_absmax_per_dim"] = [float(x) for x in np.abs(pm1).max(0)]
            out.append(entry)
        return out

    def stack(self):
        """Every replan concatenated row-wise, so `N x 7` rows line up with the executed steps."""
        keys = {"model_out": "actions_7d_from_model", "denorm": "actions_7d_denormalised",
                "pm1": "actions_7d_after_gripper_rewrite",
                "adapter_out": "actions_14d_from_adapter"}
        out = {}
        for k, name in keys.items():
            vals = [np.asarray(r[k], dtype=np.float32) for r in self.rows if r.get(k) is not None]
            if not vals:
                out[name] = np.zeros((0, 0), np.float32)
                continue
            out[name] = np.concatenate([v.reshape(-1, v.shape[-1]) for v in vals], axis=0)
        return out


# --------------------------------------------------------------------------- env probe

def _find_data(env):
    """The `mujoco.MjData` behind the wrapper stack, for the contact-solver counters."""
    seen, node = [], env
    while node is not None and node not in seen:
        seen.append(node)
        d = getattr(node, "data", None)
        if d is not None and hasattr(d, "ncon"):
            return d
        node = getattr(node, "unwrapped", None) or getattr(node, "env", None)
    return None


class EnvProbe:
    """One control step, split into the calls it is made of and the arrays it hands each of them.

    `smoke40k.install_env_timer` times `env.step` as a whole, which is why the batch's wall time
    could say 85% is simulation without naming a line inside it. The MuJoCo calls are reached by
    patching the module in *this* process; `_observation` (the three EGL renders plus assembly) and
    `_ik.convert` are wrapped on the env instance. Anything not found is reported as
    uninstrumented rather than guessed at, so a missing number stays visible.
    """

    def __init__(self, sink=None, proprio=None, panel=None):
        self.acc = {}
        self.ncon, self.nefc, self.solver_niter = [], [], []
        self.instrumented = []
        self.steps = []            # per control step: the arrays env.step was handed and produced
        self.emergency = []        # (frame, reason) for every emergency_stop call
        self.sink, self.proprio = sink, proprio
        self.panel = panel         # the composed [front|left_wrist|panel] writer, when asked for
        self.total_hint = None
        self.reset_proprio14 = None
        self.frame = 0

    def add(self, key, dt):
        self.acc[key] = self.acc.get(key, 0.0) + dt
        self.acc[key + "_n"] = self.acc.get(key + "_n", 0) + 1

    def patch_mujoco(self):
        import mujoco

        for name in ("mj_step", "mj_forward"):
            orig = getattr(mujoco, name)
            probe = self

            def timed(*a, _n=name, _o=orig, **k):
                t = time.perf_counter()
                try:
                    return _o(*a, **k)
                finally:
                    probe.add(_n, time.perf_counter() - t)

            setattr(mujoco, name, timed)
        self.instrumented += ["mujoco.mj_step", "mujoco.mj_forward"]

    def install_on_env(self, env):
        probe = self
        data = _find_data(env)

        for host in (env, getattr(env, "unwrapped", None)):
            if host is None or not hasattr(host, "_observation"):
                continue
            orig_obs = host._observation

            def obs_timed(*a, _o=orig_obs, **k):
                t = time.perf_counter()
                out = _o(*a, **k)
                probe.add("observation_render_and_assembly", time.perf_counter() - t)
                return out

            host._observation = obs_timed
            self.instrumented.append("_observation")
            break

        ik = getattr(env, "_ik", None) or getattr(getattr(env, "unwrapped", None), "_ik", None)
        if ik is not None and hasattr(ik, "convert"):
            orig_conv = ik.convert

            def ik_timed(*a, **k):
                t = time.perf_counter()
                out = orig_conv(*a, **k)
                probe.add("ik_convert", time.perf_counter() - t)
                rec = probe.steps[-1] if probe.steps else None
                if rec is not None:
                    rec["requested_14d"] = [float(x) for x in np.asarray(a[1]).reshape(-1)] \
                        if len(a) > 1 else None
                    rec["ik_last_applied_joint_16d"] = [
                        float(x) for x in np.asarray(a[2]).reshape(-1)] if len(a) > 2 else None
                    rec["ik_joint_target_16d"] = [float(x) for x in np.asarray(out).reshape(-1)]
                return out

            ik.convert = ik_timed
            self.instrumented.append("_ik.convert")

        for attr in ("_limit_joint_action",):
            fn = getattr(env, attr, None)
            if fn is None:
                continue

            def limited(*a, _o=fn, **k):
                out = _o(*a, **k)
                rec = probe.steps[-1] if probe.steps else None
                if rec is not None:
                    rec["applied_joint_16d"] = [float(x) for x in np.asarray(out).reshape(-1)]
                return out

            setattr(env, attr, limited)
            self.instrumented.append(attr)

        orig_es = getattr(env, "emergency_stop", None)
        if orig_es is not None:
            def emergency(*a, **k):
                self.emergency.append({"frame": self.frame,
                                       "args": [str(x) for x in a] +
                                               ["%s=%s" % kv for kv in k.items()]})
                return orig_es(*a, **k)

            env.emergency_stop = emergency
            self.instrumented.append("emergency_stop")

        orig_step = env.step

        def step(action, **k):
            t = time.perf_counter()
            handed = np.asarray(action, dtype=np.float64).reshape(-1)
            rec = {"frame": self.frame + 1, "handed_14d": [float(x) for x in handed]}
            self.steps.append(rec)
            out = orig_step(action, **k)
            self.add("env_step", time.perf_counter() - t)
            obs, reward, terminated, truncated, info = out
            self.frame += 1

            if data is not None:
                try:
                    rec["ncon"] = int(data.ncon)
                    rec["nefc"] = int(data.nefc)
                    rec["qvel_max_robot"] = float(np.max(np.abs(data.qvel[robot_dofs]))) \
                        if robot_dofs is not None and len(robot_dofs) else None
                    rec["qvel_max_any"] = float(np.max(np.abs(data.qvel)))
                    sn = np.asarray(getattr(data, "solver_niter", 0)).reshape(-1)
                    rec["solver_niter"] = int(sn.max()) if sn.size else None
                    self.ncon.append(rec["ncon"])
                    self.nefc.append(rec["nefc"])
                    if rec["solver_niter"] is not None:
                        self.solver_niter.append(rec["solver_niter"])
                except Exception as exc:  # a counter the installed MuJoCo does not expose
                    rec["contact_read_error"] = str(exc)[:160]
            rec["reward"] = float(np.asarray(reward).reshape(-1)[0]) if reward is not None else None
            rec["terminated"] = bool(terminated)
            rec["truncated"] = bool(truncated)
            rec["safety_stop"] = bool(info.get("safety_stop", False))
            rec["safety_reason"] = info.get("safety_reason")
            rec["cookies_in_target"] = info.get("cookies_in_target")
            rec["success"] = info.get("success")

            p14 = self.measure(obs)
            rec["proprio14"] = None if p14 is None else [float(x) for x in p14]
            if self.sink is not None:
                self.sink.add(obs, self.annotate(rec, p14))
            if self.panel is not None:
                self.panel.add_frame(obs, rec, p14)
            return out

        robot_dofs = getattr(getattr(env, "unwrapped", env), "_robot_dof_ids", None)
        env.step = step
        return data is not None

    def annotate(self, rec, p14):
        """The overlay text for one frame.

        `p14` is `eval40k.build_proprio14` of *this* observation, i.e. the exact eight pose columns
        plus six zero-tared newton columns the model was conditioned on at the replan that produced
        this motion. Showing that rather than a re-derived number is the point: the video then
        answers "what force was there when the arm did that" without a second, disagreeing sensor
        reading.
        """
        lines = []
        if p14 is not None:
            for j in range(6):
                lines.append("%-13s %9.2f N" % (FORCE_NAMES[j], p14[8 + j]))
            lines.append("L_eef  x%7.3f y%7.3f z%7.3f m" % (p14[0], p14[1], p14[2]))
            h = rec.get("handed_14d")
            if h:
                lines.append("cmd L %6.3f %6.3f %6.3f g%6.2f" % (h[0], h[1], h[2], h[6]))
                lines.append("cmd R %6.3f %6.3f %6.3f g%6.2f" % (h[7], h[8], h[9], h[13]))
        warn = None
        if rec.get("safety_reason"):
            warn = "EMERGENCY STOP: %s" % rec["safety_reason"]
        elif rec.get("ncon") is not None and rec["ncon"] >= 24:
            warn = "contact pairs ncon=%d nefc=%s" % (rec["ncon"], rec.get("nefc"))
        return {"frame": rec["frame"], "lines": lines, "warn": warn,
                "success": rec.get("cookies_in_target"), "total": self.total_hint}

    def measure(self, obs):
        if self.proprio is None:
            return None
        try:
            return np.asarray(self.proprio(obs), dtype=np.float64).reshape(-1)
        except Exception as exc:
            self.acc.setdefault("proprio_read_error", str(exc)[:160])
            return None

    def record_reset(self, obs):
        """Frame 0: what the scene looked like before the policy moved anything.

        `rollout40k.run_episode` treats the reset observation as frame 0 and the observation after
        the k-th `env.step` as frame k, so the sink must be fed once here for the video's frame
        index to match `measured[f]` and `steps[f-1]`.
        """
        self.frame = 0
        if self.sink is None and self.panel is None:
            return
        rec = {"frame": 0, "handed_14d": None, "safety_reason": None, "ncon": None,
               "terminated": False, "cookies_in_target": None}
        p14 = self.measure(obs)
        self.reset_proprio14 = None if p14 is None else [float(x) for x in p14]
        if self.sink is not None:
            self.sink.add(obs, self.annotate(rec, p14))
        if self.panel is not None:
            self.panel.add_frame(obs, rec, p14)

    def rollup(self):
        total = self.acc.get("env_step", 0.0)
        n = max(1, self.acc.get("env_step_n", 0))
        named = sum(self.acc.get(k, 0.0) for k in
                    ("mj_step", "mj_forward", "observation_render_and_assembly", "ik_convert"))
        return {"env_step_s_total": round(total, 2), "steps": self.acc.get("env_step_n", 0),
                "env_step_s_mean": round(total / n, 4),
                "mj_step_s_total": round(self.acc.get("mj_step", 0.0), 2),
                "mj_step_calls": self.acc.get("mj_step_n", 0),
                "mj_forward_s_total": round(self.acc.get("mj_forward", 0.0), 2),
                "observation_s_total": round(self.acc.get("observation_render_and_assembly", 0.0), 2),
                "ik_convert_s_total": round(self.acc.get("ik_convert", 0.0), 2),
                "ik_convert_calls": self.acc.get("ik_convert_n", 0),
                "unattributed_s_total": round(total - named, 2),
                "unattributed_share": (round((total - named) / total, 3) if total else None),
                "instrumented": sorted(self.instrumented),
                "ncon_max": max(self.ncon) if self.ncon else None,
                "ncon_mean": round(float(np.mean(self.ncon)), 2) if self.ncon else None,
                "nefc_max": max(self.nefc) if self.nefc else None,
                "nefc_mean": round(float(np.mean(self.nefc)), 2) if self.nefc else None,
                "solver_niter_max": max(self.solver_niter) if self.solver_niter else None,
                "solver_niter_mean": (round(float(np.mean(self.solver_niter)), 2)
                                      if self.solver_niter else None)}



# --------------------------------------------------------------------------- gates

def vram_preflight(min_free_gib, cap_frac):
    """Refuse to join the GPU unless whatever else is running keeps headroom, then cap this process.

    `set_per_process_memory_fraction` limits the allocator of the process calling it, so the cap
    protects the neighbour rather than the recording run: if the run needs more than the fraction it
    fails here, alone. The batch nodes peaked at 22.64 GiB on a 48 GiB card, which is why the
    default floor is close to the whole card and the default cap is well under it.
    """
    facts = {"min_free_gib": float(min_free_gib), "cap_frac": float(cap_frac)}
    if cap_frac and cap_frac > 1.0:
        raise SystemExit("--rec-vram-cap-frac must be a fraction of the card, got %s" % cap_frac)
    import torch

    if not torch.cuda.is_available():
        facts["ok"] = False
        facts["reason"] = "no CUDA/ROCm device visible; a recording run needs the real model"
        return facts
    free, total = torch.cuda.mem_get_info()
    facts.update(free_gib=round(free / 2**30, 2), total_gib=round(total / 2**30, 2),
                 device=torch.cuda.get_device_name(0))
    facts["ok"] = bool(free >= min_free_gib * 2**30)
    if facts["ok"] and cap_frac and cap_frac > 0:
        torch.cuda.set_per_process_memory_fraction(float(cap_frac))
        facts["allocator_capped_at_gib"] = round(float(cap_frac) * total / 2**30, 2)
    if not facts["ok"]:
        facts["reason"] = ("only %.2f GiB free of %.2f GiB; a full FasterWAM load peaked at 22.64 "
                           "GiB, so a second model instance would risk the neighbour's run"
                           % (free / 2**30, total / 2**30))
    return facts


def video_preflight(rec_root, cams, fps):
    """Prove an mp4 can actually be written, per camera, before the first episode burns the GPU.

    The recorder uses `imageio.get_writer` with libx264 and a 16-pixel macroblock, which is a
    different code path from the `mimsave` a smoke test usually reaches for. Discovering that after a
    25-minute rollout throws away the whole run, and a half-written file is the one failure mode of a
    recording pass that has nothing to do with the model.
    """
    out = {"imageio": None, "cams": {}}
    try:
        import imageio
        from PIL import Image, ImageDraw

        out["imageio"] = getattr(imageio, "__version__", None)
        probe = np.full((64, 64, 3), 40, dtype=np.uint8)
        for name, _key in cams:
            path = rec_root / ("_probe_%s.mp4" % name)
            wr = imageio.get_writer(path, fps=int(fps), codec="libx264", quality=8,
                                    macro_block_size=16, ffmpeg_params=["-pix_fmt", "yuv420p"])
            for i in range(4):
                wr.append_data(np.clip(probe + 12 * i, 0, 255).astype(np.uint8))
            wr.close()
            out["cams"][name] = {"bytes": int(path.stat().st_size) if path.is_file() else 0,
                                 "frames_written": 4}
            path.unlink(missing_ok=True)
        img = Image.fromarray(probe).convert("RGB")
        ImageDraw.Draw(img).text((2, 2), "overlay", font=Overlay(0, None)._font(11),
                                 fill=(0, 255, 0))
        out["overlay_ok"] = True
        out["ok"] = bool(out["cams"]) and all(v["bytes"] > 0 for v in out["cams"].values())
    except Exception as exc:
        out["ok"] = False
        out["error"] = str(exc)[:300]
    return out


def panel_preflight(rec_root, fps, total=8):
    """Drive the real `PanelSink` with a synthetic observation, then decode what it wrote.

    This is the user's "CPU first" rule in pod form: the composed layout, the four-channel force
    trace, the frame-count interlocks and the writer are all proved on eight fake frames before any
    inference is loaded, so a 25-minute rollout cannot be the first time the panel is rendered. It
    exercises `PanelSink.add_frame` rather than `PanelRecorder` directly, so the observation keys and
    column indices are part of the proof.
    """
    import panelvideo40k as PV
    from force_stream import FORCE_OBS_KEY

    out = {"requested": total, "backend": None}
    try:
        path = Path(rec_root) / "_probe_panel.mp4"
        sink = PanelSink(path, fps=fps, seed=0, step="preflight", total_frames=total, replan=4)
        for f in range(total):
            obs = {"observation.images.front": np.full((PV.CAM_SRC, PV.CAM_SRC, 3), 40 + 6 * f,
                                                       dtype=np.uint8),
                   "observation.images.left_wrist":
                       np.full((PV.CAM_SRC, PV.CAM_SRC, 3), 90 + 6 * f, dtype=np.uint8),
                   FORCE_OBS_KEY: np.array([1.0 * f, -0.5 * f, 2.0 * f] + [0.0] * 9
                                           + [10.0 * f, 7.0 * f, -5.0 * f, -3.0 * f]
                                           + [0.0, 0.0], dtype=np.float64)}
            cmd = None if f == 0 else [0.5, -0.2, 1.0, 0.0, 0.0, 0.0, 1.0] * 2
            sink.add_frame(obs, {"frame": f, "handed_14d": cmd, "requested_14d": cmd,
                                 "terminated": False, "cookies_in_target": None,
                                 "safety_reason": None}, None)
        stats = sink.close()
        back = PV.read_frames(path)
        out.update(backend=stats["backend"], frames_written=stats["frames_written"],
                   bytes=stats["bytes"], wh=stats["wh"], frames_decoded=len(back),
                   decoded_shape=None if not back else list(back[0].shape),
                   notes=stats["notes"])
        out["ok"] = bool(stats["frames_written"] == total and len(back) == total
                         and stats["bytes"] > 1000
                         and out["decoded_shape"] == stats["wh"][::-1] + [3])
        path.unlink(missing_ok=True)
    except Exception as exc:
        out["ok"] = False
        out["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:300])
    return out


def _action_band():
    """The 7-wide action band the weights were normalised with, read straight from the stats file."""
    try:
        d = json.loads(Path(E.V1G_STATS).read_text())
        entry = (d.get("action") or {}).get("default") or {}
        if "global_min" in entry and "global_max" in entry:
            return {"source": str(E.V1G_STATS), "min": entry["global_min"],
                    "max": entry["global_max"]}
    except Exception as exc:
        return {"source": str(E.V1G_STATS), "error": str(exc)[:160]}
    return {"source": str(E.V1G_STATS), "error": "no action.default band"}


def live_normalizer_band(policy, which="state"):
    """The band the *installed* normalizer maps with, recovered two ways and compared.

    `stats["min"]/["max"]` is what the stats file said; `((-1-offset)/scale, (1-offset)/scale)` is
    what the live `min/max` map inverts to. Reading the second is what makes this a check on the
    object in use rather than on a file, and the two disagreeing is exactly the interface bug the
    current investigation has to rule out.
    """
    try:
        nrm = policy.processor.normalizer.normalizers[which]["default"]
    except Exception as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}
    out = {"ok": True, "class": type(nrm).__name__, "mode": str(nrm.mode), "field": which}
    try:
        lo = np.asarray(nrm.stats["min"], dtype=np.float64).reshape(-1)
        hi = np.asarray(nrm.stats["max"], dtype=np.float64).reshape(-1)
        scale = np.asarray(nrm.scale, dtype=np.float64).reshape(-1)
        offset = np.asarray(nrm.offset, dtype=np.float64).reshape(-1)
        inv_lo, inv_hi = (-1.0 - offset) / scale, (1.0 - offset) / scale
        out.update({"stats_min": [float(x) for x in lo], "stats_max": [float(x) for x in hi],
                    "inverted_min": [float(x) for x in inv_lo],
                    "inverted_max": [float(x) for x in inv_hi],
                    "max_disagreement": float(max(np.abs(lo - inv_lo).max(),
                                                  np.abs(hi - inv_hi).max()))})
        if lo.size == E.PROPRIO_DIM and hi.size == E.PROPRIO_DIM:
            out["lo_hi"] = [lo, hi]
    except Exception as exc:
        out["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
    return out


def patch_witness_band():
    """Make the +-5 clamp interlock non-vacuous, whatever module state the batch is in.

    `smoke40k.stats_band_vectors()` reads `eval40k.STATS_FACTS`, which is why every episode recorded
    by the batch has `witnessed == None` and `proprio_band_map_pairs == 0`: `python eval40k.py` binds
    the file to `__main__` while `smoke40k` imports a *second* `eval40k` whose `STATS_FACTS` never
    filled. Running through this file already shares one module, but the fallback below reads the
    band off the live processor instead, so the interlock cannot go quiet again without saying so.
    """
    import smoke40k as S

    original = S.stats_band_vectors

    def stats_band_vectors():
        got = original()
        if got is not None:
            STATE["band_source"] = "eval40k.STATS_FACTS"
            return got
        policy = STATE.get("policy")
        if policy is None:
            STATE["band_source"] = "none: no policy stashed yet"
            return None
        band = live_normalizer_band(policy, "state")
        lo_hi = band.get("lo_hi")
        STATE["live_state_band"] = {k: v for k, v in band.items() if k != "lo_hi"}
        STATE["band_source"] = ("live processor normalizer (STATS_FACTS empty); stats vs inverted "
                                "max diff %.2e" % band["max_disagreement"] if lo_hi
                                else "none: %s" % band.get("error"))
        return None if not lo_hi else (lo_hi[0], lo_hi[1])

    S.stats_band_vectors = stats_band_vectors
    return original


def install_stash():
    """Keep a handle on the objects `eval40k.main()` builds, so the recorder uses the same ones."""
    original_build = E.build_policy
    original_calib = E.resolve_calib

    def build_policy(args, model, env_pair):
        policy = original_build(args, model, env_pair)
        STATE["policy"] = policy
        try:
            STATE["W"] = env_pair[1]
        except Exception:
            pass
        return policy

    def resolve_calib(path, node_calib):
        facts = original_calib(path, node_calib)
        STATE["calib"] = facts
        return facts

    E.build_policy = build_policy
    E.resolve_calib = resolve_calib


def stash_logs():
    """Reach the two recording proxies `eval40k.main()` builds, the same read-only way."""
    import smoke40k as S

    orig_icr, orig_ir = S.install_chunk_recorder, S.install_recorder

    def install_chunk_recorder(policy, log, witness=None):
        STATE["chunk_log"] = log
        STATE["witness"] = witness
        return orig_icr(policy, log, witness=witness)

    def install_recorder(model, log):
        STATE["calls"] = log
        return orig_ir(model, log)

    S.install_chunk_recorder = install_chunk_recorder
    S.install_recorder = install_recorder


# --------------------------------------------------------------------------- episode capture

def _worst_chain(spy, per_chunk):
    """The full chain at the replan whose model output was furthest out of band.

    This is the one table that answers 'model or interface' in a single read: the same row of the
    same chunk at every stage, dimension by dimension.
    """
    if not spy.rows or not per_chunk:
        return None
    best, idx = None, None
    for i, r in enumerate(spy.rows):
        m = np.abs(np.asarray(r["model_out"], dtype=np.float64))
        v = float(m.max()) if m.size else 0.0
        if best is None or v > best:
            best, idx = v, i
    r = spy.rows[idx]
    model = np.asarray(r["model_out"], dtype=np.float64).reshape(-1, r["model_out"].shape[-1])
    denorm = np.asarray(r["denorm"], dtype=np.float64).reshape(-1, r["denorm"].shape[-1])
    j, d = np.unravel_index(int(np.argmax(np.abs(model))), model.shape)
    return {"chunk": idx,
            "frame": per_chunk[idx].get("frame") if idx < len(per_chunk) else None,
            "model_absmax": best, "worst_row": int(j), "worst_dim": int(d),
            "worst_dim_name": DIM_NAMES[int(d)] if int(d) < 7 else str(d),
            "model_out_value": float(model[j, d]),
            "denorm_value": float(denorm[j, d]),
            "pm1_row": (None if not r.get("pm1") else
                        [float(x) for x in np.asarray(r["pm1"][j], np.float64).tolist()]),
            "adapter_out_row": (None if not r.get("adapter_out") else
                                [float(x) for x in np.asarray(r["adapter_out"][j],
                                                              np.float64).tolist()]),
            "dim_names_7d": list(DIM_NAMES),
            "denorm_row_absmax_per_dim": [float(x) for x in np.abs(denorm[j]).tolist()]}


def _steps_arrays(probe):
    """One row per executed control step -- which is one fewer row than the video has frames.

    `probe.steps` begins at the first `env.step`, because the reset observation executes nothing. So
    row j of every array below is video frame j+1, and `frame[j]` carries that number. The panel's
    `pad_forcen` is written into the same file with one row per video frame, starting at frame 0.
    Pairing the two by their own row index is exactly what a one-frame shift looks like, so the
    convention is restated inside the npz and not only here.
    """
    keys = ("handed_14d", "requested_14d", "ik_joint_target_16d", "ik_last_applied_joint_16d",
            "applied_joint_16d", "proprio14")
    out = {}
    for k in keys:
        vals = [s[k] for s in probe.steps if s.get(k) is not None]
        out[k] = (np.asarray(vals, dtype=np.float32) if vals else np.zeros((0, 0), np.float32))
    out["frame"] = np.asarray([s["frame"] for s in probe.steps], dtype=np.int64)
    out["ncon"] = np.asarray([s.get("ncon", -1) for s in probe.steps], dtype=np.int64)
    out["nefc"] = np.asarray([s.get("nefc", -1) for s in probe.steps], dtype=np.int64)
    out["row_index_note"] = np.asarray(
        "row j of these arrays is control step j, i.e. video frame j+1, and frame[j] says so; "
        "pad_forcen and pad_forcen_frame have one row per video frame starting at frame 0, the reset "
        "observation, which is why they hold one more row than this table")
    return out


class Ctx:
    """What a recording run needs to know about itself, kept out of the closure so the same code
    can be driven by the self-test with a stub policy and by `eval40k.main()` with the real one."""

    def __init__(self, rec_root, step, cams, fps_arg=0, no_physics=False, summarize=True,
                 allow_missing_witness=False, max_steps=600, panel_video=False):
        self.rec_root, self.step, self.cams = Path(rec_root), step, list(cams)
        self.fps_arg, self.no_physics = int(fps_arg), bool(no_physics)
        self.summarize, self.allow_missing_witness = bool(summarize), bool(allow_missing_witness)
        self.max_steps = int(max_steps)
        self.panel_video = bool(panel_video)
        self.written = []


def run_episode_capture(original_run, ctx, env, policy, seed, reset, **kw):
    """Roll one episode with everything stored, then write the three videos and the evidence files.

    The gate at the top is the reason this function exists: a recording pass that silently loses the
    proprio witness would report `proprio_band_map_pairs == 0` exactly like the batch it is meant to
    diagnose, and 25 minutes of GPU would buy the same blind spot again.
    """
    import smoke40k as S

    STATE["policy"] = policy
    calls, clog = STATE.get("calls"), STATE.get("chunk_log")
    missing = [n for n, v in (("chunk_log", clog), ("model_calls", calls),
                              ("proprio witness", STATE.get("witness"))) if v is None]
    if missing and not ctx.allow_missing_witness:
        raise SystemExit("the recorder never saw %s; recording without them would reproduce the "
                         "blind spot this run exists to close (band source: %s)"
                         % (", ".join(missing), STATE.get("band_source")))

    fps = (ctx.fps_arg or int(getattr(getattr(env, "config", None), "control_hz", 0)) or 20)
    d = ctx.rec_root / ("seed%d" % int(seed))
    d.mkdir(parents=True, exist_ok=True)
    proprio = ((lambda o: E.build_proprio14(o, STATE["W"], STATE["calib"]))
               if ("W" in STATE and "calib" in STATE) else None)
    probe = EnvProbe(sink=FrameSink(d, int(seed), ctx.step, fps, ctx.cams,
                                    Overlay(int(seed), ctx.step)), proprio=proprio)
    probe.total_hint = int(kw.get("max_steps", ctx.max_steps))
    panel = None
    if ctx.panel_video:
        missing = [n for n, _ in CAM_KEYS if n not in [c[0] for c in ctx.cams]]
        if missing:
            raise SystemExit("--rec-panel-video needs the front and left_wrist cameras, but "
                             "--rec-cams left out %s" % missing)
        panel = PanelSink(d / ("seed%d_step%s_composed_panel.mp4"
                               % (int(seed), ctx.step if ctx.step is not None else "NA")),
                          fps=fps, seed=seed, step=ctx.step,
                          total_frames=probe.total_hint, replan=int(kw.get("replan", 10)),
                          extra_notes=["replan=%s steps of %s proposed, %s Hz control"
                                        % (int(kw.get("replan", 10)), 32, fps)])
        panel.emergency = probe.emergency      # one list, shared, so the ticks see every stop
        probe.panel = panel
    if not ctx.no_physics:
        probe.patch_mujoco()
    has_contact = probe.install_on_env(env)
    spy = ActionSpy().install(policy) if hasattr(policy, "bridge") else None
    if panel is not None and spy is not None:
        panel.chain_fn = lambda f: chain_for_frame(spy, panel.replan, f)
    base, cbase = len(calls or []), len(clog or [])

    def wrapped_reset(e, s):
        obs = reset(e, s)
        probe.record_reset(obs)
        return obs

    t0 = time.time()
    try:
        rec, measured, chunks = original_run(env, policy, seed, wrapped_reset, **kw)
    finally:
        files = probe.sink.close()
        panel_stats = panel.close() if panel is not None else None

    per = spy.per_chunk() if spy is not None else []
    tag = "step%s" % ctx.step
    payload = {"measured": np.asarray(measured, dtype=np.float32),
               "reset_proprio14": (np.asarray(probe.reset_proprio14, np.float32)
                                   if probe.reset_proprio14 is not None
                                   else np.zeros((0,), np.float32))}
    if spy is not None:
        payload.update(spy.stack())
    np.savez_compressed(d / ("seed%d_%s_actions.npz" % (int(seed), tag)), **payload)
    step_arrays = _steps_arrays(probe)
    if panel is not None:
        # the same numbers the panel drew, on disk, so the curve can be re-derived from the npz and
        # compared with the video instead of trusted
        step_arrays["pad_forcen"] = np.stack([panel.traces[c] for c in panel.PV.PAD_ORDER], axis=1)
        step_arrays["pad_forcen_frame"] = np.arange(step_arrays["pad_forcen"].shape[0],
                                                    dtype=np.int64)
        step_arrays["pad_channel_order"] = np.asarray(panel.PV.PAD_ORDER)
        step_arrays["pad_channel_columns"] = np.asarray([panel.PV.PAD_COLS[c]
                                                         for c in panel.PV.PAD_ORDER])
    np.savez_compressed(d / ("seed%d_%s_steps.npz" % (int(seed), tag)), **step_arrays)
    rec["record_frames"] = files
    rec["record_emergency_stops"] = probe.emergency
    if ctx.summarize and calls is not None and clog is not None:
        S.summarize_episode(rec, calls[base:], clog[cbase:], chunks)

    meta = {"seed": int(seed), "step": ctx.step, "replan": int(kw.get("replan", 10)),
            "frames_executed": probe.frame, "measured_frames": len(measured),
            "chunks": len(chunks), "video_fps": fps, "video_files": files,
            "panel_video": panel_stats,
            "band_7d": STATE.get("band"), "band_source_state": STATE.get("band_source"),
            "live_state_band": STATE.get("live_state_band"),
            "action_absmax_max": rec.get("action_absmax_max"),
            "proprio_band_map_pairs": rec.get("proprio_band_map_pairs"),
            "proprio_clamped_calls": rec.get("proprio_clamped_calls"),
            "infer_action_calls": rec.get("infer_action_calls"),
            "gripper_pm1_values": rec.get("gripper_pm1_values"),
            "per_chunk_action_chain": per,
            "worst_action_chain": _worst_chain(spy, per) if spy is not None else None,
            "units_note": ADAPTER_NOTE,
            "widths_note": ("the chain changes width at the IK, and the column names say where: "
                            "handed_14d / requested_14d are the cartesian delta command "
                            "(6 per arm + gripper, x2), while ik_joint_target_16d / "
                            "ik_last_applied_joint_16d / applied_joint_16d are joint space "
                            "[L1..L7, L_grip, R1..R7, R_grip] (env.py step() offsets (0,0) and "
                            "(7,8)); a per-dimension comparison therefore pairs requested_14d[d] "
                            "with the model row of the same chunk, never with a joint column"),
            "emergency_stops": probe.emergency,
            "step_safety_reasons": [s.get("safety_reason") for s in probe.steps],
            "step_qvel_max_robot": [s.get("qvel_max_robot") for s in probe.steps],
            "contact_counters_recorded": bool(has_contact),
            "timing_note": ("env.step 总耗时 here carries this process's own sub-timers, so it runs "
                            "a few hundredths of a second above the batch's unsplit sim_s_total; the "
                            "split is the point, and the absolute wall time is not comparable "
                            "seed-to-seed"),
            "physics": probe.rollup(),
            "overlay_note": ("the numbers burned into the video are this observation's own "
                             "eval40k.build_proprio14: columns 8-13 are the six frozen channels in "
                             "newtons after the zero-load tare, columns 0-2 the left EEF position in "
                             "metres. control_hz is %d, so one video frame is one control step and "
                             "the force beside a frame is that frame's force." % fps),
            "recorder_wall_s": round(time.time() - t0, 1),
            "episode_record": rec}
    (d / ("seed%d_%s_meta.json" % (int(seed), tag))).write_text(
        json.dumps(meta, indent=2, sort_keys=True, default=str))
    ctx.written.append(str(d))
    chain = meta["worst_action_chain"]
    print("[record40k] " + json.dumps({
        "seed": int(seed), "frames": probe.frame, "videos": sorted(files),
        "panel": None if panel_stats is None else {
            "frames": panel_stats["frames_written"], "backend": panel_stats["backend"],
            "wh": panel_stats["wh"], "kb": round(panel_stats["bytes"] / 1024.0, 1)},
        "action_absmax_max": rec.get("action_absmax_max"),
        "witness_pairs": rec.get("proprio_band_map_pairs"),
        "worst": None if chain is None else {
            "dim": chain["worst_dim_name"], "model_out": round(chain["model_out_value"], 2),
            "denorm": round(chain["denorm_value"], 2)},
        "emergency": [e["args"] for e in probe.emergency][:3]}, default=str), flush=True)
    return rec, measured, chunks


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--rec-cams", nargs="+", default=[n for n, _ in CAM_KEYS])
    ap.add_argument("--rec-fps", type=int, default=0)
    ap.add_argument("--rec-min-free-gib", type=float, default=25.0)
    ap.add_argument("--rec-vram-cap-frac", type=float, default=0.35)
    ap.add_argument("--rec-no-vram-check", action="store_true")
    ap.add_argument("--rec-no-physics", action="store_true")
    ap.add_argument("--rec-allow-missing-witness", action="store_true")
    ap.add_argument("--rec-panel-video", action="store_true",
                    help="also write one composed [front | left_wrist | diagnostics] mp4 per seed")
    known, rest = ap.parse_known_args()

    out_arg = None
    for i, tok in enumerate(rest):
        if tok == "--out" and i + 1 < len(rest):
            out_arg = Path(rest[i + 1]).resolve()
        elif tok.startswith("--out="):
            out_arg = Path(tok.split("=", 1)[1]).resolve()
    if out_arg is None:
        raise SystemExit("pass --out for the recording run (it is forwarded to eval40k.py unchanged)")
    if OFFICIAL_OUT in out_arg.parents or out_arg == OFFICIAL_OUT:
        raise SystemExit("refusing to write into the official batch tree: %s" % out_arg)
    step = None
    for i, tok in enumerate(rest):
        if tok == "--steps" and i + 1 < len(rest):
            step = int(rest[i + 1])
    cams = [(n, k) for n, k in CAM_KEYS if n in known.rec_cams]
    if not cams:
        raise SystemExit("--rec-cams named none of %s" % [n for n, _ in CAM_KEYS])

    rec_root = out_arg / "_record"
    rec_root.mkdir(parents=True, exist_ok=True)

    pre = ({"skipped": True, "ok": True} if known.rec_no_vram_check
           else vram_preflight(known.rec_min_free_gib, known.rec_vram_cap_frac))
    (rec_root / "preflight.json").write_text(json.dumps(pre, indent=2, sort_keys=True))
    if not pre.get("ok"):
        raise SystemExit("VRAM gate refused this recording run: %s" % pre.get("reason"))
    vp = video_preflight(rec_root, cams, known.rec_fps or 20)
    (rec_root / "video_preflight.json").write_text(json.dumps(vp, indent=2, sort_keys=True))
    if not vp.get("ok"):
        raise SystemExit("no mp4 writer available (%s); refusing to burn a rollout it cannot save"
                         % vp.get("error"))
    if known.rec_panel_video:
        pp = panel_preflight(rec_root, known.rec_fps or 20)
        (rec_root / "panel_preflight.json").write_text(json.dumps(pp, indent=2, sort_keys=True))
        if not pp.get("ok"):
            raise SystemExit("the composed panel video does not work here (%s); refusing to record "
                             "rollouts whose panel was never rendered" % pp.get("error"))

    STATE["band"] = _action_band()
    install_stash()
    patch_witness_band()
    stash_logs()

    ctx = Ctx(rec_root, step, cams, fps_arg=known.rec_fps, no_physics=known.rec_no_physics,
              allow_missing_witness=known.rec_allow_missing_witness,
              panel_video=known.rec_panel_video)
    original_run = E.run_episode
    E.run_episode = functools.partial(run_episode_capture, original_run, ctx)
    print("[record40k] forwarding to eval40k.main(): %s" % " ".join(rest), flush=True)
    sys.argv = ["eval40k.py"] + rest
    E.main()
    print("[record40k] recordings under %s: %s" % (rec_root, ctx.written), flush=True)
    return ctx


if __name__ == "__main__":
    main()
