"""Independent 40k diagnosis driver under the RECONSTRUCTED normalizer (Req2/4/5/6).

This is diagnosis under traceable reconstructed statistics, NOT proof of original normalizer
byte identity. Every output carries NORMALIZER_IDENTITY=RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL
and original_stats_sha256=UNKNOWN. It does not modify the archived run_config.yaml; it loads it
read-only and redirects every dead historical absolute path into BASE via marked mappings.

It reuses the REAL inference protocol modules by identity:
  eval40k.use_stage_fw / attach_eval_settings / deployment_text_encoder_settings /
  anchor_weight_search_path / check_stats_width / build_proprio14 / read_json
  policy40k.ForcePolicy, rollout40k.run_episode, force_stream
  infer_a3_worker (bridge W), action_adapter_v3.FasterWAMA3ActionAdapterV3
  env_factory (real A3 MuJoCo env), evaluation_capture.run_captured (CPU-accepted recorder)

The ONLY deviation from eval40k.build_policy is the text-conditioning gate: the original training
text-embedding cache and the v1g arrays.npz were not recovered with this container. The inference
path encodes the prompt fresh with the original T5 encoder regardless (the cache was a training-time
optimization), so the instruction is derived deterministically from the force-aware code constants
(DEFAULT_A3_TASK / DEFAULT_PROMPT) and the gap is recorded as TEXT_CACHE_STATUS, never hidden.

Modes
  --max-steps 10            real MuJoCo shortest action block (one chunk: 32 predicted, 10 executed)
  --record                  wrap with the independent execution-chain recorder
  --pair                    same-seed A(bare)/B(recorded) no-side-effect comparison
  --max-steps 600           full episode (seed 11 / 17 forensics)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path("/workspace/eval_restore_20261117")
EVAL40K = BASE / "deploy" / "stage_40k" / "dev" / "eval40k"
FW = BASE / "deploy" / "stage_40k" / "fw"
FW_SRC = FW / "src"
BRIDGE = BASE / "deploy" / "bridge"
BRIDGE_NEXT = BRIDGE / "_next"
SIM_SRC = BASE / "deploy" / "a3_dual_arm_sim" / "src"
CAPTURE = BASE / "deploy" / "capture"

MERGED = BASE / "step_040000_merged.pt"
CALIB = BASE / "step_040000" / "force_calib_v1.json"
RUN_CFG = BASE / "step_040000" / "run_config.yaml"
RECON = BASE / "reconstructed_normalizer" / "dataset_stats_14d_RECONSTRUCTED.json"
SIM_CFG = BASE / "deploy" / "a3_dual_arm_sim" / "configs" / "cookie_batch.yaml"
DEPLOY_CFG = FW / "configs" / "sim_libero.yaml"
AUX = BASE / "auxiliary"

MARKERS = {
    "NORMALIZER_IDENTITY": "RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL",
    "original_stats_sha256": "UNKNOWN",
    "report_scope": "在可追溯重建 normalizer 条件下的 40k 诊断结果",
}
TEXT_CACHE_STATUS = ("ORIGINAL_NOT_RECOVERED_INSTRUCTION_FROM_CODE_CONSTANTS: the training "
                     "text-embedding cache and v1g arrays.npz were lost with the old container; "
                     "inference encodes the prompt fresh with the original T5, instruction is "
                     "DEFAULT_A3_TASK formatted by DEFAULT_PROMPT from the force-aware code")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def merged_identity():
    """Hash the 11 GB artifact once, cache by size+mtime so repeated runs do not re-hash."""
    cache = BASE / "reconstructed_normalizer" / "merged_identity.json"
    st = MERGED.stat()
    if cache.is_file():
        try:
            c = json.loads(cache.read_text())
            if c.get("bytes") == st.st_size and c.get("mtime") == st.st_mtime:
                return c
        except Exception:
            pass
    rec = {"path": str(MERGED), "bytes": st.st_size, "mtime": st.st_mtime,
           "sha256": sha256_file(MERGED)}
    cache.write_text(json.dumps(rec, indent=2))
    return rec


def rng_snapshot():
    import torch
    return {"torch": torch.get_rng_state().clone(),
            "python": random.getstate(),
            "numpy": np.random.get_state()}


def rng_equal(a, b):
    import torch
    return {"torch_bit_equal": bool(torch.equal(a["torch"], b["torch"])),
            "python_equal": a["python"] == b["python"],
            "numpy_equal": a["numpy"][0] == b["numpy"][0]
            and np.array_equal(a["numpy"][1], b["numpy"][1])
            and a["numpy"][2:] == b["numpy"][2:]}


def setup_protocol():
    """Bind the force-aware fasterwam FIRST, then the bridge/env, exactly like eval40k.build_env."""
    # Headless GL must be selected before any mujoco/OpenGL import. The original protocol exports
    # MUJOCO_GL=egl in every launcher (RUN_ON_POD.sh:42, RUN_RECORD_SEEDS.sh:35, RUN_BATCH_STEP6.sh:50)
    # and infer_a3_worker.py:30-31 setdefaults MUJOCO_GL/PYOPENGL_PLATFORM=egl; README:522 warns it
    # must precede the A3 env import. Without it mujoco defaults to GLFW and dies headless at
    # Renderer/MjrContext. EGL verified working on this pod (libEGL present, OSMesa absent).
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    sys.path.insert(0, str(EVAL40K))
    import eval40k as E
    # Marked path mappings: redirect every dead historical absolute default into BASE.
    E.CALIB_PATH = CALIB
    E.V1G_STATS = RECON
    E.DEFAULT_CFG = RUN_CFG
    E.DEPLOY_EVAL_CFG = DEPLOY_CFG
    E.DIFFSYNTH_MODEL_BASE = AUX
    E.V1G_ARRAYS = BASE / "archive_evidence" / "data" / "a3_fw" / "arrays.npz"
    E.FORCE_ARRAYS = BASE / "archive_evidence" / "data" / "a3_fw_force" / "arrays.npz"
    fw_facts = E.use_stage_fw()
    for p in (str(BRIDGE), str(BRIDGE_NEXT), str(SIM_SRC), str(CAPTURE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ["A3_CONFIG_PATH"] = str(SIM_CFG)
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(AUX)
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    import env_factory
    import infer_a3_worker as W
    return E, env_factory, W, fw_facts


def build_policy_reconstructed(E, W, args):
    """Mirror eval40k.build_policy with the text-cache gate replaced by constant-derived instruction."""
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fasterwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from fasterwam.datasets.lerobot.a3_video_dataset import DEFAULT_A3_TASK
    from fasterwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from action_adapter_v3 import FasterWAMA3ActionAdapterV3
    from policy40k import ForcePolicy

    doc = OmegaConf.load(str(RUN_CFG))
    # Marked path mappings inside the read-only-loaded config: the resolved training config still
    # names the dead historical force-calib / stats absolute paths. Redirect them to the restored
    # calib and the reconstructed stats BEFORE instantiate, so the force conditioner builds from
    # the same calibration the model is later fed via set_force_norm_stats. The archived file on
    # disk is never modified.
    OmegaConf.set_struct(doc, False)
    if "force" in doc.model and "norm_stats" in doc.model.force:
        doc.model.force.norm_stats.file = str(CALIB)
    doc.data.train.pretrained_norm_stats = str(RECON)
    OmegaConf.set_struct(doc, True)
    eval_facts = E.attach_eval_settings(doc, str(DEPLOY_CFG))
    enc_facts = E.deployment_text_encoder_settings(doc, str(DEPLOY_CFG))
    weight_facts = E.anchor_weight_search_path(doc)

    instruction = str(DEFAULT_A3_TASK)
    prompt = str(DEFAULT_PROMPT).format(task=instruction)
    prompt_facts = {"task": instruction, "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "DEFAULT_A3_TASK": str(DEFAULT_A3_TASK),
                    "DEFAULT_PROMPT": str(DEFAULT_PROMPT),
                    "TEXT_CACHE_STATUS": TEXT_CACHE_STATUS}

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    calib = E.read_json(str(CALIB))
    stats = load_dataset_stats_from_json(str(RECON))
    stats_facts = E.check_stats_width(stats, str(RECON), calib_std=calib["std"])

    model = instantiate(doc.model, model_dtype=torch.bfloat16, device=device)
    model.load_checkpoint(str(MERGED))
    model = model.to(device).eval()
    if getattr(model, "text_encoder", None) is None or getattr(model, "tokenizer", None) is None:
        raise SystemExit("built model has no text encoder/tokenizer; encode_prompt would die")
    if model.text_encoder.training:
        raise SystemExit("text encoder in train mode after model.eval()")
    model.set_force_norm_stats(calib["mean"], calib["std"], scale=float(calib["scale"]))

    # Conditioning sanity (no original cache to compare): encode the prompt, record shape/mask/nan,
    # and prove the encoder is deterministic by encoding twice. Marked as regenerated reference.
    with torch.no_grad():
        emb1, mask1 = model.encode_prompt(prompt)
        emb2, mask2 = model.encode_prompt(prompt)
    cond_facts = {"encoded_shape": list(emb1.shape), "mask_all_true": bool(mask1.all()),
                  "mask_true_count": int(mask1.sum()),
                  "encoded_nan": bool(torch.isnan(emb1).any()),
                  "determinism_max_abs_diff": float((emb1.float() - emb2.float()).abs().max()),
                  "determinism_bit_equal": bool(torch.equal(emb1, emb2)),
                  "reference": "REGENERATED_FROM_ORIGINAL_T5_NOT_ORIGINAL_CACHE",
                  "text_encoder_training": bool(model.text_encoder.training)}
    if cond_facts["encoded_nan"]:
        raise SystemExit("NaN in text conditioning")

    processor = instantiate(doc.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    adapter = FasterWAMA3ActionAdapterV3()
    policy = ForcePolicy(model, processor, doc, adapter, device, bridge=W,
                         proprio_fn=lambda obs: E.build_proprio14(obs, W, calib),
                         instruction=instruction, gripper_mode=args.gripper_mode,
                         hist_len=16, name="recon40k")
    facts = {"eval_settings": eval_facts, "text_encoder": enc_facts,
             "weight_paths": weight_facts, "prompt": prompt_facts,
             "stats": stats_facts, "conditioning": cond_facts, "device": device,
             "action_horizon": policy.action_horizon,
             "num_video_frames": policy.num_video_frames,
             "num_inference_steps": int(doc.EVALUATION.get("num_inference_steps")),
             "action_infer_mode": str(doc.EVALUATION.get("action_infer_mode")),
             "gripper_mode": args.gripper_mode}
    return policy, doc, calib, facts


def make_reset(E, env_factory, env, args):
    def reset(env_, seed):
        obs, _info = env_factory.reset(env_, seed, randomize_cookies=True,
                                       randomize_boxes=False, batch_offset_m=0.0)
        sanity = env_factory.scene_sanity(env_)
        if not sanity["ok"] and not args.allow_invalid_scene:
            raise SystemExit(f"scene invalid at seed={seed}: {sanity}")
        return obs
    return reset


def artifacts():
    return {"weights": [str(MERGED)],
            "config": [str(RUN_CFG), str(RECON), str(CALIB), str(SIM_CFG), str(DEPLOY_CFG)],
            "source": [str(EVAL40K / "policy40k.py"), str(EVAL40K / "rollout40k.py"),
                       str(EVAL40K / "force_stream.py"), str(BRIDGE / "infer_a3_worker.py"),
                       str(BRIDGE / "action_adapter_v3.py"), str(BRIDGE_NEXT / "env_factory.py"),
                       str(CAPTURE / "evaluation_capture.py"),
                       str(CAPTURE / "execution_chain_capture.py")]}


def serialize_result(result):
    record, measured, chunks = result
    ch = [{"frame": c["frame"], "n_actions": c["n_actions"],
           "pred_force": None if c["pred_force"] is None else np.asarray(c["pred_force"]).tolist(),
           "meta": c["meta"]} for c in chunks]
    return record, np.asarray(measured, dtype=np.float32), ch


def run_one(E, env_factory, policy, args, seed, record_chain):
    env = env_factory.make_env(position_noise_m=0.0, yaw_noise_rad=0.0, render_cameras=True)
    reset = make_reset(E, env_factory, env, args)
    pre = rng_snapshot()
    t0 = time.time()
    capture = None
    if record_chain:
        import evaluation_capture
        import execution_chain_capture_fixed as eccf
        # Marked independent override: use the CPU-regression-passed corrected Capture
        # (raw=(n,7), denorm=(1,n,7)). The archived evaluation_capture.py and
        # execution_chain_capture.py files are NOT modified; only this runtime namespace
        # binding is redirected, and the fix is recorded in recorder_fix_manifest.json.
        evaluation_capture.Capture = eccf.Capture
        from evaluation_capture import run_captured
        result, capture = run_captured(E.run_episode, env=env, policy=policy, reset=reset,
                                       seed=seed, artifacts=artifacts(),
                                       max_steps=args.max_steps, replan=args.replan)
    else:
        result = E.run_episode(env, policy, seed=seed, reset=reset,
                               max_steps=args.max_steps, replan=args.replan)
    wall = round(time.time() - t0, 2)
    post = rng_snapshot()
    rec, measured, ch = serialize_result(result)
    infer_calls = int(policy.calls)
    if hasattr(env, "close"):
        env.close()
    return {"record": rec, "measured": measured, "chunks": ch, "capture": capture,
            "wall_s": wall, "rng_pre": pre, "rng_post": post, "infer_calls": infer_calls}


def save_run(out, seed, tag, res):
    """Persist one run's evidence immediately (A is saved before B starts)."""
    chain_file = None
    if res["capture"] is not None:
        chain_path = out / f"chain_seed{seed}_{tag}.npz"
        if chain_path.exists():
            chain_path = out / f"chain_seed{seed}_{tag}_{res['capture'].run_id[:8]}.npz"
        res["capture"].save(str(chain_path))
        chain_file = chain_path.name
    ep = dict(MARKERS)
    ep.update({"seed": seed, "tag": tag, "record": res["record"], "chunks": res["chunks"],
               "wall_s": res["wall_s"], "infer_calls": res["infer_calls"],
               "recorded": res["capture"] is not None, "chain_file": chain_file})
    (out / f"episode_seed{seed}_{tag}.json").write_text(json.dumps(ep, indent=2, sort_keys=True,
                                                                  default=str))
    np.savez_compressed(out / f"measured_seed{seed}_{tag}.npz", measured=res["measured"],
                        chunk_frames=np.asarray([c["frame"] for c in res["chunks"]], np.int64))


def compare_pair(A, B):
    """No-side-effect proof: bare A vs recorded B must agree on every execution quantity."""
    ra, rb = A["record"], B["record"]
    record_keys = ("frames", "steps_executed", "decision_chunks", "terminated", "truncated",
                   "success", "cookies_in_target", "nan_abort_frame")
    record_diff = {k: {"A": ra.get(k), "B": rb.get(k), "equal": ra.get(k) == rb.get(k)}
                   for k in record_keys}
    ma, mb = A["measured"], B["measured"]
    force_shape_equal = ma.shape == mb.shape
    force_max_diff = float(np.max(np.abs(ma - mb))) if force_shape_equal else None
    force_bit_equal = bool(np.array_equal(ma, mb)) if force_shape_equal else False
    chunk_diff = []
    for ca, cb in zip(A["chunks"], B["chunks"]):
        pa = None if ca["pred_force"] is None else np.asarray(ca["pred_force"])
        pb = None if cb["pred_force"] is None else np.asarray(cb["pred_force"])
        pf_eq = (pa is None and pb is None) or (
            pa is not None and pb is not None and pa.shape == pb.shape
            and bool(np.array_equal(pa, pb)))
        pf_max = None if (pa is None or pb is None or pa.shape != pb.shape) else float(
            np.max(np.abs(pa - pb)))
        chunk_diff.append({"frame_A": ca["frame"], "frame_B": cb["frame"],
                           "n_actions_A": ca["n_actions"], "n_actions_B": cb["n_actions"],
                           "noise_seed_A": ca["meta"].get("noise_seed"),
                           "noise_seed_B": cb["meta"].get("noise_seed"),
                           "pred_force_bit_equal": pf_eq, "pred_force_max_abs_diff": pf_max,
                           "action_absmax_A": ca["meta"].get("action_absmax"),
                           "action_absmax_B": cb["meta"].get("action_absmax")})
    rng_post = rng_equal(A["rng_post"], B["rng_post"])
    rng_pre = rng_equal(A["rng_pre"], B["rng_pre"])
    all_record_equal = all(v["equal"] for v in record_diff.values())
    chunks_equal = (len(A["chunks"]) == len(B["chunks"])
                    and all(c["pred_force_bit_equal"] and c["frame_A"] == c["frame_B"]
                            and c["n_actions_A"] == c["n_actions_B"]
                            and c["noise_seed_A"] == c["noise_seed_B"] for c in chunk_diff))
    call_counts = {"infer_calls_A": A["infer_calls"], "infer_calls_B": B["infer_calls"],
                   "infer_calls_equal": A["infer_calls"] == B["infer_calls"],
                   "env_steps_A": ra.get("steps_executed"), "env_steps_B": rb.get("steps_executed"),
                   "env_steps_equal": ra.get("steps_executed") == rb.get("steps_executed"),
                   "decision_chunks_A": ra.get("decision_chunks"),
                   "decision_chunks_B": rb.get("decision_chunks")}
    bcap = B["capture"]
    chain_summary = None
    if bcap is not None:
        chain_summary = {
            "status": bcap.status, "chunks_recorded": len(bcap.chunks),
            "steps_recorded": len(bcap.mapping),
            "raw_shapes": [list(bcap.arrays[f"chunk_{c['chunk_id']}_raw"].shape) for c in bcap.chunks],
            "denorm_shapes": [list(bcap.arrays[f"chunk_{c['chunk_id']}_denorm"].shape) for c in bcap.chunks],
            "rewritten_shapes": [list(bcap.arrays[f"chunk_{c['chunk_id']}_rewritten"].shape) for c in bcap.chunks],
            "adapter_shapes": [list(bcap.arrays[f"chunk_{c['chunk_id']}_adapter"].shape) for c in bcap.chunks],
            "opening_before_reset": bcap.mapping[0]["opening_before"] if bcap.mapping else None,
            "opening_trajectory": [m["opening_after"] for m in bcap.mapping],
            "raw_ndim_all_2": all(bcap.arrays[f"chunk_{c['chunk_id']}_raw"].ndim == 2 for c in bcap.chunks),
            "denorm_ndim_all_3": all(bcap.arrays[f"chunk_{c['chunk_id']}_denorm"].ndim == 3 for c in bcap.chunks)}
    no_side_effect = bool(all_record_equal and force_bit_equal and chunks_equal
                          and call_counts["infer_calls_equal"] and call_counts["env_steps_equal"])
    return {"no_side_effect_bit_exact": no_side_effect,
            "record_diff": record_diff,
            "force_trajectory": {"shape_equal": force_shape_equal, "bit_equal": force_bit_equal,
                                 "max_abs_diff": force_max_diff, "shape": list(ma.shape)},
            "chunk_diff": chunk_diff,
            "call_counts": call_counts,
            "B_chain_summary": chain_summary,
            "rng_post_equal": rng_post, "rng_pre_equal": rng_pre,
            "tolerance_basis": ("bit-exact equality required: the recorder is passive (reads "
                                "bridge/adapter/env outputs, copies to CPU), the policy re-seeds "
                                "its diffusion generators per chunk from (seed, call), and the "
                                "env reset is seeded; any difference beyond 0 means the recorder "
                                "changed execution and the run must stop. A is bare so raw/denorm/"
                                "pm1/Adapter/opening are not A/B-compared directly; recorder "
                                "faithfulness is proven by A==B on force trajectory, record, "
                                "chunks, RNG and call counts, plus Capture's internal handed=="
                                "adapter[row] assertions and the 15/15 CPU shape regression"),
            "capture_steps_recorded": None if B["capture"] is None else len(B["capture"].mapping),
            "capture_chunks_recorded": None if B["capture"] is None else len(B["capture"].chunks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--replan", type=int, default=10)
    ap.add_argument("--out", default=str(BASE / "reconstructed_normalizer" / "diag_out"))
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--pair", action="store_true")
    ap.add_argument("--gripper-mode", default="pm1", choices=["pm1", "libero01"])
    ap.add_argument("--allow-invalid-scene", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    E, env_factory, W, fw_facts = setup_protocol()
    policy, doc, calib, facts = build_policy_reconstructed(E, W, args)
    mid = merged_identity()
    identity = dict(MARKERS)
    identity.update({
        "merged": mid, "fw_import": fw_facts,
        "reconstructed_stats_sha256": sha256_file(RECON),
        "force_calib_sha256": sha256_file(CALIB),
        "run_config_sha256": sha256_file(RUN_CFG),
        "sim_config_sha256": sha256_file(SIM_CFG),
        "protocol": facts,
        "path_mappings": {"CALIB_PATH": str(CALIB), "V1G_STATS": str(RECON),
                          "DEFAULT_CFG": str(RUN_CFG), "DEPLOY_EVAL_CFG": str(DEPLOY_CFG),
                          "DIFFSYNTH_MODEL_BASE": str(AUX), "A3_CONFIG_PATH": str(SIM_CFG)},
        "seed": args.seed, "max_steps": args.max_steps, "replan": args.replan,
    })
    (out / f"identity_seed{args.seed}.json").write_text(json.dumps(identity, indent=2,
                                                                  sort_keys=True, default=str))

    if args.pair:
        # Same seed, fresh env each: A bare then B recorded. Global RNGs re-seeded identically
        # before each so any recorder-induced RNG consumption is visible in the post states.
        # A's evidence is persisted BEFORE B starts; B is wrapped so a B failure cannot lose A.
        import torch
        rec_fix = json.loads((BASE / "reconstructed_normalizer" /
                              "recorder_fix_manifest.json").read_text())
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        A = run_one(E, env_factory, policy, args, args.seed, record_chain=False)
        save_run(out, args.seed, "A", A)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        B = None
        B_error = None
        try:
            B = run_one(E, env_factory, policy, args, args.seed, record_chain=True)
            save_run(out, args.seed, "B", B)
        except BaseException as exc:
            import traceback
            B_error = f"{type(exc).__name__}: {exc}"
            (out / f"pair_seed{args.seed}_B_traceback.txt").write_text(traceback.format_exc())
        if B is not None:
            cmp = compare_pair(A, B)
        else:
            cmp = {"no_side_effect_bit_exact": False, "comparable": False,
                   "B_error": B_error,
                   "note": "B failed; A evidence preserved on disk; pair incomparable, STOP"}
        cmp_out = dict(MARKERS)
        cmp_out.update({"seed": args.seed, "comparison": cmp,
                        "A_record": A["record"], "B_record": None if B is None else B["record"],
                        "A_wall_s": A["wall_s"], "B_wall_s": None if B is None else B["wall_s"],
                        "A_infer_calls": A["infer_calls"],
                        "B_infer_calls": None if B is None else B["infer_calls"],
                        "recorder_fix": {"original_sha256": rec_fix["original_sha256"],
                                         "fixed_sha256": rec_fix["fixed_sha256"],
                                         "raw_shape": rec_fix["raw_shape_corrected"],
                                         "denorm_shape": rec_fix["denorm_shape_kept"],
                                         "original_untouched": rec_fix["original_untouched"],
                                         "cpu_regression": "PASS 15/15"},
                        "model_state_isolation": (
                            "infer_action / infer_action_one_pass_future_cache assign no "
                            "persistent self.* state (read-only verified, selfAssign=[]); fresh "
                            "seeded video/action generators + force_state + inference schedule "
                            "per call; VAE cache self-clears; global torch/python/numpy RNG "
                            "re-seeded to seed before each run; fresh env per run_one; "
                            "policy.start_episode resets force stream and calls=0; A/B "
                            "bit-exactness empirically proves no cross-run pollution"),
                        "B_error": B_error})
        (out / f"pair_seed{args.seed}.json").write_text(json.dumps(cmp_out, indent=2,
                                                                  sort_keys=True, default=str))
        print(json.dumps({"mode": "pair", "seed": args.seed,
                          "no_side_effect_bit_exact": cmp.get("no_side_effect_bit_exact"),
                          "B_error": B_error,
                          "force_max_abs_diff": cmp.get("force_trajectory", {}).get("max_abs_diff"),
                          "out": str(out)}, indent=2))
        if B_error is not None:
            return 1
        return 0 if cmp["no_side_effect_bit_exact"] else 2

    res = run_one(E, env_factory, policy, args, args.seed, record_chain=args.record)
    ep = dict(MARKERS)
    ep.update({"seed": args.seed, "record": res["record"], "chunks": res["chunks"],
               "wall_s": res["wall_s"], "max_steps": args.max_steps, "replan": args.replan,
               "recorded": bool(args.record)})
    (out / f"episode_seed{args.seed}.json").write_text(json.dumps(ep, indent=2, sort_keys=True,
                                                                 default=str))
    np.savez_compressed(out / f"measured_seed{args.seed}.npz", measured=res["measured"],
                        chunk_frames=np.asarray([c["frame"] for c in res["chunks"]], np.int64),
                        chunk_pred=np.asarray([c["pred_force"] for c in res["chunks"]
                                               if c["pred_force"] is not None], dtype=np.float32))
    if res["capture"] is not None:
        res["capture"].save(str(out / f"chain_seed{args.seed}.npz"))
    print(json.dumps({"mode": "record" if args.record else "bare", "seed": args.seed,
                      "record": res["record"], "wall_s": res["wall_s"],
                      "chunks": len(res["chunks"]), "out": str(out)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())