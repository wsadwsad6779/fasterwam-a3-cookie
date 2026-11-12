"""Policy backends for the 40k evaluation.

`ForcePolicy` is the deployment path for a force-conditioned checkpoint: it does what
`infer_a3_worker.predict_action_chunk` does for the released model -- official image
preparation, proprio normalisation, action denormalisation, gripper rewrite, A3 action
adapter -- and adds the one thing that model never had to do, feed the six frozen force
channels and take the predicted force trajectory back out.

`StubPolicy` exists so the loop, the metrics and the report can be executed on CPU in
seconds, with no model and no simulator weights in sight. Its force prediction is a
closed-form function of the frame it was called on, which is what makes the metric test in
`flow_test.py` able to assert an exact MAE instead of a plausible one. It is a test fixture
and is named like one; nothing about it should be read as a policy result.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from force_stream import HIST_LEN, ForceStream, select_channels

INT32_MAX = 2 ** 31 - 1


def noise_seed(seed, call):
    """The diffusion seed for chunk `call` of the episode seeded `seed`.

    `infer_action` builds its noise generators fresh on every call --
    `torch.Generator(...).manual_seed(seed)` at `jointwam.py:183-184` for the video and action
    latents and again at `fastwam.py:935` for the force latents -- so an episode that passes its
    own seed unchanged hands every replan the *identical* initial latents: same shape every call
    (`[1, action_horizon, action_dim]`), same seed, therefore the same draw. Training never did
    that (`torch.randn_like` per window), so the fixed draw is both a train/infer mismatch and a
    single point of failure: one unlucky noise realisation repeats its bias on all ~60 replans.

    Deriving the seed from `(episode seed, chunk index)` rather than from a counter on the object
    keeps it a pure function, so a rerun -- or a node evaluated after a restart -- replays exactly
    the same sequence of chunk noises, and two different episodes never share one.
    """
    return int(np.random.default_rng([int(seed), int(call)]).integers(0, INT32_MAX))


class ForcePolicy:
    """A loaded force-conditioned model, driven one action chunk at a time.

    Each chunk gets its own diffusion seed, `noise_seed(episode seed, chunk index)`, because
    `infer_action` re-seeds its generators per call and would otherwise replay one noise draw for
    the whole episode.

    `proprio_fn` replaces the worker's proprio source for the model call. The released
    `infer_a3_worker.build_proprio` returns 8 columns; these checkpoints were trained on the 14
    columns `build_v1_dataset.py` assembles (8 + 6 force-tared), and the processor's normalizer
    asserts the 14-wide meta shape. The driver must therefore pass the shared 14-wide builder --
    leaving it `None` keeps the 8-wide worker path, which only the CPU fixtures with their own
    fake bridge can use.
    """

    def __init__(self, model, processor, cfg, adapter, device, *, bridge,
                 instruction: str, gripper_mode: str = "libero01",
                 hist_len: int = HIST_LEN, num_inference_steps=None,
                 proprio_fn=None, name: str = "force40k"):
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.adapter = adapter
        self.device = device
        self.bridge = bridge          # the infer_a3_worker module, passed in by the driver
        self.proprio_fn = proprio_fn
        self.instruction = instruction
        self.gripper_mode = gripper_mode
        self.num_inference_steps = num_inference_steps
        self.name = name
        self.stream = ForceStream(hist_len=hist_len)
        self.calls = 0
        self.action_horizon = int(cfg.data.train.num_frames) - 1
        self.num_video_frames = (int(cfg.data.train.num_frames) - 1) \
            // int(cfg.data.train.action_video_freq_ratio) + 1

    def start_episode(self, obs, seed):
        self.stream.reset()
        self.calls = 0

    def observe(self, obs, frame):
        if self.stream.n_frames != frame:
            raise RuntimeError(
                f"force stream holds {self.stream.n_frames} frames but the loop is at "
                f"frame {frame}; every executed step must reach policy.observe() or the "
                "history the model sees is not the 16 consecutive frames it trained on")
        self.stream.push(obs["observation.force"])

    def _infer_kwargs(self, obs, history, mask, ns):
        import inspect

        _rgb, x_np = self.bridge.prepare_fasterwam_images_official(obs)
        image = torch.from_numpy(x_np).to(device=self.device, dtype=self.model.torch_dtype)
        build = self.proprio_fn or self.bridge.build_proprio
        proprio = self.bridge.normalize_proprio(build(obs), self.processor)
        cfg = self.cfg
        steps = int(self.num_inference_steps
                    if self.num_inference_steps is not None
                    else cfg.EVALUATION.get("num_inference_steps", None)
                    or cfg.get("eval_num_inference_steps", 20))
        kwargs = {
            "prompt": self.bridge.DEFAULT_PROMPT.format(task=self.instruction),
            "input_image": image,
            "action_horizon": self.action_horizon,
            "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
            "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
            "num_inference_steps": steps,
            "proprio": proprio,
            "sigma_shift": (None if cfg.EVALUATION.get("sigma_shift") is None
                            else float(cfg.EVALUATION.get("sigma_shift"))),
            "seed": int(ns),
            "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
            "tiled": bool(cfg.EVALUATION.get("tiled", False)),
            "force_history": torch.from_numpy(np.asarray(history)).unsqueeze(0),
            "force_history_mask": torch.from_numpy(np.asarray(mask)).unsqueeze(0),
        }
        if "num_video_frames" in inspect.signature(self.model.infer_action).parameters:
            kwargs["num_video_frames"] = self.num_video_frames
        return kwargs

    def chunk(self, obs, frame, seed):
        if self.stream.n_frames != frame + 1:
            raise RuntimeError(
                f"chunk requested at frame {frame} with {self.stream.n_frames} observed "
                "frames; the current frame must be observed before it is conditioned on")
        history, mask = self.stream.snapshot()
        self.calls += 1
        ns = noise_seed(seed, self.calls)
        t0 = time.time()
        with torch.no_grad():
            pred = self.model.infer_action(**self._infer_kwargs(obs, history, mask, ns))
        infer_s = time.time() - t0
        if "force_future_newton" not in pred:
            raise RuntimeError(
                "the model returned no `force_future_newton`; this checkpoint is supposed to "
                "be force-conditioned, and evaluating it without the force prediction would "
                "report a success rate with the force half of the architecture unmeasured")
        action = self.bridge.denormalize_action(pred["action"], self.processor)[0]
        if self.gripper_mode == "pm1":
            action[..., -1] = np.where(action[..., -1] >= 0.0, 1.0, -1.0)
        else:
            action[..., -1] = action[..., -1] * 2 - 1
            action[..., -1] = action[..., -1] * -1.0
            if bool(self.cfg.EVALUATION.get("binarize_gripper", False)):
                action[..., -1] = np.sign(action[..., -1])
        rot = self.bridge.quat_wxyz_to_matrix(
            np.asarray(obs["observation.eef_pose"][3:7], dtype=np.float64))
        env_actions = [self.adapter.convert(a, rot) for a in action]
        pf = np.asarray(pred["force_future_newton"], dtype=np.float64)
        if pf.ndim == 3:
            pf = pf[0]
        if pf.shape != (self.action_horizon, self.stream._buf.shape[1]):
            raise RuntimeError(
                f"predicted force chunk is {tuple(pf.shape)}, expected "
                f"{(self.action_horizon, self.stream._buf.shape[1])}")
        meta = {"infer_s": round(infer_s, 3), "replan_from_frame": int(frame),
                "noise_seed": ns,
                "action_absmax": float(np.abs(action).max()),
                "force_feedback_applied": bool(pred.get("force_feedback_applied", False)),
                "history_mask_true": int(mask.sum())}
        return env_actions, pf, meta


class StubPolicy:
    """Deterministic stand-in used by the CPU flow test.

    `pred_mode="offset"` returns persistence + `offset`, which makes the scored MAE exactly
    `|offset|` on every channel except where the truth is clipped by the episode ending;
    that is the property `flow_test.py` asserts.

    Its own noise (`pred_mode="noise"`) is drawn from an rng built once per episode, so it
    advances across replans on its own. That is why the stub can never catch a policy that
    replays a single noise draw every chunk -- the check for that is on `noise_seed`, not here.
    """

    def __init__(self, action_horizon=32, n_actions=10, action_dim=14, hist_len=HIST_LEN,
                 pred_mode="offset", offset=0.5, name="stub", rng_seed=0):
        self.action_horizon = int(action_horizon)
        self.n_actions = int(n_actions)
        self.action_dim = int(action_dim)
        self.stream = ForceStream(hist_len=hist_len)
        self.pred_mode = pred_mode
        self.offset = float(offset)
        self.name = name
        self.calls = 0
        self.rng = np.random.default_rng(int(rng_seed))

    def start_episode(self, obs, seed):
        self.stream.reset()
        self.calls = 0
        self.rng = np.random.default_rng(int(seed))

    def observe(self, obs, frame):
        self.stream.push(obs["observation.force"])

    def chunk(self, obs, frame, seed):
        self.calls += 1
        history, _mask = self.stream.snapshot()
        last = history[-1]
        if self.pred_mode == "offset":
            pf = last[None, :] + self.offset
        elif self.pred_mode == "zero":
            pf = np.zeros_like(history[None, :])
        elif self.pred_mode == "noise":
            pf = last[None, :] + self.rng.normal(0.0, 1.0, (self.action_horizon, last.size))
        else:
            raise ValueError(f"unknown pred_mode {self.pred_mode!r}")
        pf = np.repeat(pf, self.action_horizon, axis=0) if pf.shape[0] == 1 else pf
        actions = [self.rng.normal(0.0, 0.05, self.action_dim) for _ in range(self.n_actions)]
        return actions, pf.astype(np.float64), {"stub": True, "frame": int(frame)}


def measured_to_list(force18_stream):
    """Small helper for tests that build a `measured` list straight from raw vectors."""
    return [select_channels(v) for v in force18_stream]
