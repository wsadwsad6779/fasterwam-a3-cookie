"""Evaluate the eight 5,000-step nodes of the 40k run with one script and one seed list.

Why this file exists separately from `run_matrix.py`: that harness scores a released
action-only model. The 40k checkpoints are force-conditioned, so an eval of them has to
feed the same 16-frame raw-newton history the trainer fed (`force_stream`), execute the
chunk, and then score the force trajectory the model predicted against the force the
simulator went on to measure (`force_metrics`). Neither of those exists in the old harness,
and neither can be retrofitted into it without changing what the r4 numbers mean.

Discipline this script keeps to:

* one model per process, and the merged artifact is resolved, never assumed: a node without
  a merged checkpoint is reported as `missing_artifact` rather than silently evaluated from
  the LoRA adapter, because the release path is `base + adapter` and that is what a
  deployment loads;
* the same 20 screened seeds for every node, exactly one episode per seed (20 episodes per
  node, 160 for the eight), so nodes are paired seed by seed;
* `--dry-run` answers "what would run, what is on disk, is there room" without loading a
  single tensor, which is what made it safe to write while segment 1 is still training;
* nothing here deletes anything. Merged artifacts pile up at ~11 GB each on a volume that
  holds three, so `--disk-policy report` prints the cleanup candidate list and stops.

    cd /workspace/_v1_stage/dev/eval40k
    PYTHONDONTWRITEBYTECODE=1 python eval40k.py --dry-run
    PYTHONDONTWRITEBYTECODE=1 python eval40k.py --backend fixture --policy stub --steps 5000 10000
    PYTHONDONTWRITEBYTECODE=1 python eval40k.py --backend fixture --policy stub \
        --fixture-force-mode const --steps 5000     # flow test: MAE must equal --stub-offset
    PYTHONDONTWRITEBYTECODE=1 python eval40k.py --steps 5000            # real env + merged node
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent.parent                     # /workspace/_v1_stage
FW = STAGE / "fw"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(FW / "scripts"))
sys.path.insert(0, str(FW / "src"))

from force_metrics import ForceErrorAccount      # noqa: E402
from force_stream import (CALIB_VERSION, CHANNELS, CHANNEL_NAMES, FORCE_DIM, HIST_LEN,  # noqa: E402
                          ForceContractError, check_calib, check_node_calibration)
from rollout40k import run_episode               # noqa: E402

RUN = STAGE / "run40k"
LIGHT = RUN / "checkpoints" / "light"
NODE_INTERVAL = 5000
TOTAL_STEPS = 40000
N_SCREENED_SEEDS = 20
SEED_SCREEN = STAGE / "dev" / "out40k" / "seed_screen.json"
CALIB_PATH = Path("/workspace/a3_data/a3_fw_force/force_calib_v1.json")
MERGED_DEFAULT = HERE / "merged"
DEFAULT_CFG = RUN / "config.yaml"
# the released deployment config, i.e. the file whose `EVALUATION:` block the worker that serves
# these weights actually reads; used only to look inference knobs up, never written to
DEPLOY_EVAL_CFG = Path("/workspace/fasterwam_a3_bridge/FasterWAM/configs/sim_libero.yaml")
# The stats the 40k weights were normalised with. `run40k/config.yaml` names this file as
# `pretrained_norm_stats`; the released libero stats describe an 8-wide state and a different
# action set, so pointing the eval at them would normalise the observation with numbers the model
# never saw. It is a data-processing input, read-only here.
V1G_STATS = Path("/workspace/a3_data/a3_fw_v1g/dataset_stats.json")
V1G_ARRAYS = Path("/workspace/a3_data/a3_fw_v1g/arrays.npz")
FORCE_ARRAYS = Path("/workspace/a3_data/a3_fw_force/arrays.npz")

# The 14 columns the trained `proprio_encoder` expects, and how they are assembled.
PROPRIO_BASE_DIM = 8      # what `infer_a3_worker.build_proprio` returns, unchanged
PROPRIO_DIM = 14          # `shape_meta.state[0].raw_shape`, i.e. what the model was trained on
PROPRIO14_FACTS: dict = {}    # what the last `build_proprio14` call saw, for the report

# Where the upstream Wan2.2 weight files live. `ModelConfig.reset_local_model_path`
# (`fw/src/fasterwam/models/wan22/helpers/io.py:48`) falls back to `./checkpoints/` relative to the
# *process* cwd, so a harness started from anywhere else resolves the text encoder to a path that
# does not exist and then reaches for the network. The env var is process-local; nothing is written.
DIFFSYNTH_MODEL_BASE = Path("/workspace/FasterWAM/checkpoints")
# The encoder tag inside the cache file name `robot_video_dataset.py:252` opens. It is derived from
# `model_id` by the precompute script, so a different encoder would have written a different name
# and the training loader would have raised `FileNotFoundError` instead of running.
PROMPT_CACHE_ENC_ID = "wan22ti2v5b"
TEXT_PROMPT_FACTS: dict = {}      # the instruction these weights were conditioned on, for the report
TEXT_CHECK_FACTS: dict = {}       # the last encode-prompt-against-cache comparison, for the report
TEXT_ENCODER_FACTS: dict = {}     # where `load_text_encoder` came from, for the report
WEIGHT_PATH_FACTS: dict = {}      # the resolved on-disk paths of the encoder/tokenizer/VAE


def node_steps():
    return [NODE_INTERVAL * i for i in range(1, TOTAL_STEPS // NODE_INTERVAL + 1)]


def read_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.is_file() else None


def screened_seeds(path=SEED_SCREEN, want=N_SCREENED_SEEDS):
    """The seeds `seed_screen.py` proved legal and reproducible, read from its own report."""
    doc = read_json(path)
    if doc is None:
        raise SystemExit("no seed screen at %s; run seed_screen.py --want %d first -- "
                         "guessing a seed list would score on scenes nobody checked"
                         % (path, want))
    seeds = [int(s) for s in doc["valid_seeds"]]
    if len(seeds) != want or len(set(seeds)) != want:
        raise SystemExit("seed screen at %s holds %d seeds, want %d distinct ones"
                         % (path, len(seeds), want))
    return seeds


def check_one_episode_per_seed(seeds):
    """One episode per seed is the protocol; a repeated seed would silently overwrite."""
    dupes = sorted({s for s in seeds if seeds.count(s) > 1})
    if dupes:
        raise SystemExit("seed%s repeated: the protocol is exactly one episode per seed, and "
                         "episode_seed<N>.json / measured_seed<N>.npz are keyed by seed, so the "
                         "later repeat would overwrite the earlier file while both still count "
                         "as episodes" % dupes)
    return seeds


def node_state(step):
    st = read_json(LIGHT / ("step_%06d" % step) / "trainer_state.json")
    if st is None:
        return None
    return {"step": step, "dir": str(LIGHT / ("step_%06d" % step)),
            "global_step": st.get("global_step"), "format": st.get("format"),
            "force_calibration": st.get("force_calibration")}


def disk_state(path):
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    u = shutil.disk_usage(path)
    return {"path": str(path), "free_gib": round(u.free / 2**30, 2),
            "total_gib": round(u.total / 2**30, 2)}


def plan_doc(steps, seeds, merged_dir, calib, args):
    nodes = []
    for s in steps:
        st = node_state(s)
        merged = Path(merged_dir) / ("step_%06d_merged.pt" % s)
        nodes.append({
            "step": s,
            "light_node": st["dir"] if st else None,
            "light_global_step": None if st is None else st["global_step"],
            "light_force_calibration": None if st is None else st["force_calibration"],
            "merged_artifact": str(merged) if merged.is_file() else None,
            "merged_bytes": merged.stat().st_size if merged.is_file() else None,
            "status": ("ready" if st and merged.is_file()
                       else ("no_light_node" if not st else "no_merged_artifact")),
        })
    return {"generated": time.strftime("%FT%TZ"), "backend": args.backend,
            "policy": args.policy, "steps": list(steps), "seeds": list(seeds),
            "episodes_per_node": len(seeds), "max_steps": args.max_steps,
            "replan": args.replan, "config": str(args.config),
            "instruction": args.instruction,
            "instruction_source": TEXT_PROMPT_FACTS.get("task_source"),
            "prompt_cache_file": TEXT_PROMPT_FACTS.get("cache_file"),
            "calibration": calib, "nodes": nodes,
            "disk": disk_state(merged_dir), "out": str(args.out),
            "torch_device_note": ("this run loads on CPU; use it as a load smoke, "
                                  "not as a result" if args.cpu else "cuda if available")}


def resolve_calib(path, node_calib):
    calib = read_json(path)
    if calib is None:
        raise ForceContractError(f"{path} is not readable; every force number would be meaningless")
    facts = check_calib(calib, where=str(path))
    facts.update({"zero_load_ref_median": _zero_load_reference(calib, str(path)),
                  "source": str(path),
                  "channel_names": list(calib.get("channel_names") or []),
                  "units": calib.get("units"), "calib_version": calib.get("version")})
    if node_calib is not None:
        check_node_calibration(node_calib, where="checkpoint node")
        for key in ("force_mean_sha256", "force_std_sha256"):
            if key in node_calib:
                facts.setdefault("node_" + key, node_calib[key])
    return facts


STATS_FACTS: dict = {}


def check_stats_width(stats, path, calib_std=None):
    """Prove the stats file describes the 14-wide state, and record what it says.

    `SingleFieldLinearNormalizer` in `min/max` mode maps `stats["min"]..stats["max"]` onto
    `[-1, 1]` (`normalizer.py:115-120`, `output_min=-1`), and `build_v1_dataset.py:144-145`
    deliberately writes the six force columns' band as `+/- residual_std`. Together those two
    choices are the reason the file cannot be substituted: for columns 8-13 the min/max mode
    therefore computes exactly `residual / residual_std`, a z-score by the *builder's own* std,
    and for columns 0-7 it computes the released base dataset's own band. A stats file whose
    state is 8 wide has no force columns at all, and the shape assertion in the processor then
    fires -- or worse, an 8-wide file with a 14-wide `shape_meta` normalises the force columns
    against numbers that were never computed for them.

    Nothing here re-derives a normalisation range; the widths and the two force bands are read
    from the file and reported, so a mismatch stops the run with the file named. With `calib_std`
    the report becomes a proof: the force half-widths must be the calibration file's own residual
    std, which is the only reason `min/max` mode is a z-score for those columns.
    """
    import torch

    state = stats.get("state") if isinstance(stats, dict) else None
    if not isinstance(state, dict) or not state:
        raise SystemExit(
            f"{path} has no `state` block, so it cannot describe the 14-wide proprio this model "
            f"was trained with; expected {V1G_STATS}")
    facts = {"source": str(path), "keys": sorted(state), "widths": {}, "bands": {}}
    for key, entry in sorted(state.items()):
        if not isinstance(entry, dict):
            continue
        for kind in ("global_min", "global_max", "global_std"):
            v = entry.get(kind)
            if v is None:
                continue
            t = torch.as_tensor(v).reshape(-1)
            facts["widths"][f"{key}/{kind}"] = int(t.shape[0])
        if "global_min" not in entry or "global_max" not in entry:
            continue
        w = int(torch.as_tensor(entry["global_min"]).reshape(-1).shape[0])
        if w != PROPRIO_DIM:
            raise SystemExit(
                "%s describes a %d-wide state, but `run40k/config.yaml` trains a %d-wide proprio "
                "(8 base columns + 6 zero-load-tared force columns); the state normaliser would "
                "either fail on shape or scale the force columns with numbers taken from another "
                "dataset. Use --stats %s" % (path, w, PROPRIO_DIM, V1G_STATS))
        lo = torch.as_tensor(entry["global_min"]).reshape(-1).double()
        hi = torch.as_tensor(entry["global_max"]).reshape(-1).double()
        half = ((hi - lo) / 2.0)[PROPRIO_BASE_DIM:]
        facts["bands"][key] = {
            "min": lo.tolist(), "max": hi.tolist(),
            "force_band_is_symmetric_std": bool(
                w >= PROPRIO_DIM and torch.allclose(hi[PROPRIO_BASE_DIM:],
                                                    -lo[PROPRIO_BASE_DIM:], atol=1e-6)),
            "half_width_force_columns": [float(x) for x in half.tolist()]}
        if calib_std is not None:
            cs = torch.as_tensor(calib_std, dtype=torch.float64).reshape(-1)
            if cs.shape != half.shape:
                raise SystemExit(
                    "%s: the calibration lists %d force std values, the stats file bands %d "
                    "force columns -- the two files do not describe the same channels"
                    % (path, int(cs.shape[0]), int(half.shape[0])))
            if not torch.allclose(half, cs, rtol=1e-6, atol=1e-6):
                raise SystemExit(
                    "%s: the state's force columns are banded %s, but %s says the residual std is "
                    "%s. min/max mode turns that band into `residual / std` only while the two "
                    "agree; a file that re-derives or rounds the band would silently change what "
                    "the model was trained to read, so the run stops here."
                    % (path, [round(float(x), 6) for x in half.tolist()],
                       CALIB_PATH, [round(float(x), 6) for x in cs.tolist()]))
            facts["bands"][key]["force_halfwidth_equals_calib_std"] = True
    if not facts["bands"]:
        raise SystemExit(
            "%s has a `state` block (%s) but no entry carrying both `global_min` and `global_max`, "
            "so the 14-wide width was never actually checked; expected %s"
            % (path, facts["keys"], V1G_STATS))
    STATS_FACTS.clear()
    STATS_FACTS.update(facts)
    return facts


def _zero_load_reference(calib, where):
    """The `zero_load_ref_median` the corpus was tared against, as six float64 numbers.

    `force_stream.check_calib` guards the mean/std/scale/channels the *force conditioner* uses;
    this is the other number in the same file, the one the *proprio* force columns need. The
    dataset builder subtracts it from the raw channels before they become state columns 8-13
    (`build_v1_dataset.py:112-130`), so evaluating without the identical subtraction would feed
    the model a proprio offset by the sensor's own preload -- up to 10.7 N on the left wrist fy.
    """
    v = calib.get("zero_load_ref_median")
    if not isinstance(v, list) or len(v) != FORCE_DIM:
        raise ForceContractError(
            f"{where}: `zero_load_ref_median` must list {FORCE_DIM} numbers, got {v!r}; "
            "state columns 8-13 are the raw channels minus this, so without it the 14-wide "
            "proprio cannot be rebuilt the way it was trained")
    arr = np.asarray(v, dtype=np.float64)
    if arr.shape != (FORCE_DIM,) or not np.all(np.isfinite(arr)):
        raise ForceContractError(f"{where}: non-finite `zero_load_ref_median`: {v!r}")
    return arr.tolist()


def build_proprio14(obs, W, calib, where="obs"):
    """The 14-wide proprio the 40k model was trained on, from one live observation.

    `infer_a3_worker.build_proprio` returns eight columns (left EEF pose 3 + axis-angle 3 +
    the two finger joints) and that was right for the model it was written against. The force
    checkpoint's `shape_meta` declares `raw_shape: 14`, and `build_v1_dataset.py:130` assembled
    the extra six as `gathered raw newtons - zero_load_ref_median` for channels
    `[0, 1, 2, 12, 13, 16]` -- a zero-load tare, not a mean/std scaling: the mean/std calibration
    belongs to the force *conditioner*, which applies it to `force_history` inside the model.
    So this function supplies only what the deployment worker does not have, and calls the
    worker's own function for the eight columns it does.

    Arithmetic is done in float64 and cast once to float32, the way the builder computed it
    (`raw6` float32 promoted by a float64 reference, then `.astype(np.float32)`), so a replayed
    observation reproduces the stored column bit-for-bit. `PROPRIO14_FACTS` records the last call.
    """
    from force_stream import FORCE_OBS_KEY, select_channels

    base = np.asarray(W.build_proprio(obs), dtype=np.float64).reshape(-1)
    if base.shape != (PROPRIO_BASE_DIM,):
        raise ForceContractError(
            f"`{where}`: the deployment worker's `build_proprio` returned {base.shape[0]} "
            f"columns, expected {PROPRIO_BASE_DIM}; the state layout it builds is no longer the "
            "one this module appends the force tare to, so the 14-wide proprio would be wrong "
            "in its first eight columns too")
    raw6 = select_channels(obs[FORCE_OBS_KEY], where=where).astype(np.float64)
    zero = _zero_load_reference(calib, "calib")
    resid = raw6 - np.asarray(zero, dtype=np.float64)
    p14 = np.concatenate([base, resid]).astype(np.float32)
    if p14.shape != (PROPRIO_DIM,) or not np.all(np.isfinite(p14)):
        raise ForceContractError(f"`{where}`: 14-wide proprio is {p14.shape} / non-finite")
    PROPRIO14_FACTS.clear()
    PROPRIO14_FACTS.update({"dim": int(p14.shape[0]), "base_dim": int(base.shape[0]),
                            "force_dim": int(resid.shape[0]),
                            "channels": list(CHANNELS),
                            "zero_load_ref_median": list(zero),
                            "base": [float(x) for x in base],
                            "raw6": [float(x) for x in raw6],
                            "residual6": [float(x) for x in resid],
                            "proprio14": [float(x) for x in p14.tolist()]})
    return p14


FW_IMPORT_FACTS = {}


def use_stage_fw():
    """Bind the name `fasterwam` to the force-aware copy under `fw/src`, and prove it.

    The order matters more than the path list does: `bridge/_next/env_factory.py` front-loads
    `/workspace/fasterwam_a3_bridge/FasterWAM/src` -- a released copy whose `create_fasterwam` has
    no `force` parameter -- and `infer_a3_worker` imports `fasterwam.datasets...` at module level, so
    once the bridge has been imported the package is bound there and no later `sys.path` edit can
    undo it (the modules are already in `sys.modules`). Importing first wins by a different rule:
    submodule lookup follows the parent package's own directory, so anchoring `fasterwam` here pins
    every `fasterwam.*` import that the bridge and `build_policy` do afterwards.

    Nothing here trusts the path order: the module `__file__`, the factory's own source file and the
    factory's signature are all read off the objects that were actually imported.
    """
    import inspect

    fw_src = str(FW / "src")
    try:
        import fasterwam
        from fasterwam import runtime as R
    except ImportError as exc:
        raise SystemExit(f"`fasterwam` cannot be imported with {fw_src} on sys.path: {exc}")
    facts = {"fw_src": fw_src,
             "package_file": str(getattr(fasterwam, "__file__", "") or ""),
             "runtime_file": str(R.__file__),
             "create_fasterwam_file": str(inspect.getsourcefile(R.create_fasterwam)),
             "create_fasterwam_accepts_force":
                 "force" in inspect.signature(R.create_fasterwam).parameters,
             "already_imported_submodules": len([k for k in sys.modules
                                                 if k.startswith("fasterwam.")])}
    for key in ("package_file", "runtime_file", "create_fasterwam_file"):
        if not facts[key].startswith(fw_src + os.sep):
            raise SystemExit(
                "`fasterwam` resolved to %s (%s=%s), which is not the force-aware copy under %s; "
                "the merged nodes carry a force conditioner that this copy cannot build, so the "
                "checkpoint would load without the half of the architecture being evaluated"
                % (facts["package_file"], key, facts[key], fw_src))
    if not facts["create_fasterwam_accepts_force"]:
        raise SystemExit("%s's create_fasterwam takes no `force` parameter: this is an action-only "
                         "copy and `run40k/config.yaml` cannot build a force model from it"
                         % facts["create_fasterwam_file"])
    return facts


def build_env(args):
    """Real A3 environment, built the way the r4 harness builds it, in this process.

    `use_stage_fw()` runs first because importing the bridge is what binds the `fasterwam` package.
    """
    FW_IMPORT_FACTS.clear()
    FW_IMPORT_FACTS.update(use_stage_fw())
    for p in (str(Path("/workspace/fasterwam_a3_bridge/bridge")),
              str(Path("/workspace/fasterwam_a3_bridge/bridge/_next")),
              str(Path("/workspace/a3_dual_arm_sim/src"))):
        if p not in sys.path:
            sys.path.insert(0, p)
    import env_factory
    import infer_a3_worker as W
    return env_factory, W


EVAL_SETTINGS_FACTS = {}

# exactly the keys `policy40k.ForcePolicy._infer_kwargs` reads off `cfg.EVALUATION`, plus the two
# the deployment worker reads there too. Anything else in that block is a training-time or
# task-suite setting this harness must not inherit.
EVAL_KEYS = ("num_inference_steps", "negative_prompt", "text_cfg_scale", "sigma_shift",
             "rand_device", "tiled", "action_horizon", "action_infer_mode", "binarize_gripper")


def attach_eval_settings(doc, deploy_cfg=DEPLOY_EVAL_CFG):
    """Give the training config the `EVALUATION` node the inference path reads, and log where
    every value came from.

    `run40k/config.yaml` is the *training* config: it carries no `EVALUATION:` block, while the
    deployment worker reads its inference knobs from one. So `cfg.EVALUATION.get(...)` raised
    `ConfigAttributeError: Missing key EVALUATION` on the first replan of the first real episode
    -- invisible to the CPU flow test, which never touches this config.

    The fix is to copy the deployment's own block, key by key, not to invent defaults: a value
    written as `${some_key}` is resolved against *this* run's config, which defines both
    `eval_num_inference_steps` and `default_action_infer_mode`, so the numbers stay the ones this
    model trained with. If a copied key interpolates something this config does not define, or a
    key the policy reads is missing from the deployment block, this raises instead of guessing --
    an inference setting chosen silently is exactly the kind of change that would make two nodes
    incomparable.

    Read with `yaml`, not `OmegaConf.load`: the deployment file is a Hydra package whose
    interpolations point at group configs this harness does not compose, so resolving them through
    OmegaConf would raise on keys that are perfectly readable as raw text here.
    """
    import yaml
    from omegaconf import OmegaConf

    if "EVALUATION" in doc:
        return {"source": "run config already defines EVALUATION",
                "values": {k: doc.EVALUATION.get(k) for k in EVAL_KEYS}}
    if not Path(deploy_cfg).is_file():
        raise SystemExit("%s is not readable; the inference settings have no deployment-side "
                         "source left to copy from" % deploy_cfg)
    dep = (yaml.safe_load(Path(deploy_cfg).read_text()) or {}).get("EVALUATION")
    if not isinstance(dep, dict):
        raise SystemExit("%s has no EVALUATION block; there is no deployment-side source for the "
                         "inference settings, so they cannot come from anywhere but a guess"
                         % deploy_cfg)
    values, interpolations = {}, {}
    for key in EVAL_KEYS:
        if key not in dep:
            raise SystemExit("the deployment config does not define EVALUATION.%s, which the "
                             "inference path reads; refusing to supply a default" % key)
        raw = dep[key]
        text = str(raw)
        if isinstance(raw, str) and text.startswith("${") and text.endswith("}"):
            ref = text[2:-1].strip()
            if ref in doc:
                values[key] = doc.get(ref)
                interpolations[key] = str(ref)
                continue
            raise SystemExit("EVALUATION.%s interpolates %s, which this run config does not "
                             "define; refusing to guess" % (key, ref))
        else:
            values[key] = raw
    OmegaConf.set_struct(doc, False)
    doc.EVALUATION = OmegaConf.create(values)
    OmegaConf.set_struct(doc, True)
    return {"source": str(deploy_cfg), "values": dict(values),
            "interpolated_from_run_config": interpolations,
            "deployment_action_infer_mode": values.get("action_infer_mode"),
            "harness_calls": "model.infer_action"}


def deployment_text_encoder_settings(doc, deploy_cfg=DEPLOY_EVAL_CFG):
    """Give the model block the one inference-side key the training config cannot carry.

    `run40k/config.yaml` sets `model.load_text_encoder: false` because training reads text
    conditioning from the precomputed embedding cache and never wants the 11 GB T5 in memory. The
    deployment worker runs with `load_text_encoder: true` (`sim_libero.yaml`), because
    `infer_a3_worker` passes a `prompt` and `infer_action` then calls `encode_prompt`
    (`jointwam.py:211`), which raises when no encoder is loaded (`fastwam.py:451-455`). Building the
    evaluation model from the training config alone therefore cannot serve a prompt at all -- that
    is what killed the first real smoke run.

    Exactly one key is taken, and only from the deployment file: `skip_dit_load_from_pretrain`,
    `action_dit_pretrained_path` and `model_id` decide which weights the architecture is built
    around, and those belong to this run's config, not to the released checkpoint's defaults.
    """
    import yaml
    from omegaconf import OmegaConf

    dep = (yaml.safe_load(Path(deploy_cfg).read_text()) or {}).get("model")
    if not isinstance(dep, dict) or "load_text_encoder" not in dep:
        raise SystemExit("%s has no model.load_text_encoder; the harness has no deployment-side "
                         "source for that key and will not guess it" % deploy_cfg)
    facts = {"source": str(deploy_cfg),
             "train_config_value": bool(doc.model.get("load_text_encoder", False)),
             "deployment_value": bool(dep["load_text_encoder"])}
    if not facts["deployment_value"]:
        raise SystemExit("%s sets model.load_text_encoder=false, but the harness conditions on a "
                         "`prompt` the way the worker does; with no encoder there is nothing to "
                         "encode it with" % deploy_cfg)
    OmegaConf.set_struct(doc, False)
    doc.model.load_text_encoder = True
    OmegaConf.set_struct(doc, True)
    facts["applied"] = bool(doc.model.load_text_encoder)
    facts["keys_left_on_run_config"] = {
        k: doc.model.get(k) for k in ("model_id", "tokenizer_model_id", "tokenizer_max_len",
                                      "redirect_common_files", "skip_dit_load_from_pretrain",
                                      "action_dit_pretrained_path")}
    return facts


def anchor_weight_search_path(doc):
    """Point the component loader at the weight files this pod already has, and prove they exist.

    `ModelConfig.reset_local_model_path` resolves relative to the process cwd, so the same config
    loads from `/workspace/FasterWAM/checkpoints` when the worker is started there and reaches for
    a download when the harness is started anywhere else. The env var is process-local, and
    `DIFFSYNTH_SKIP_DOWNLOAD` is set so a missing file fails here instead of quietly pulling from
    the network inside an eval run.
    """
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(DIFFSYNTH_MODEL_BASE)
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    from omegaconf import OmegaConf

    from fasterwam.models.wan22.helpers.loader import _resolve_configs

    mb = OmegaConf.to_container(doc.model, resolve=True)
    _, text_cfg, vae_cfg, tok_cfg = _resolve_configs(
        model_id=str(mb["model_id"]), tokenizer_model_id=str(mb["tokenizer_model_id"]),
        redirect_common_files=bool(mb["redirect_common_files"]))
    facts = {}
    for key, cfg in (("text_encoder", text_cfg), ("vae", vae_cfg), ("tokenizer", tok_cfg)):
        cfg.download_if_necessary()
        path = str(cfg.path)
        exists = os.path.isfile(path) if key != "tokenizer" else os.path.isdir(path)
        facts[key] = {"path": path, "exists": bool(exists)}
        if not exists:
            raise SystemExit("%s weights are not under %s (%s); the prompt path needs them and "
                             "this harness does not download model weights during an eval"
                             % (key, DIFFSYNTH_MODEL_BASE, path))
    return facts


def hashed_prompt(task, template):
    """The prompt a task string becomes at train time, and the hash its embedding is stored under.

    Both halves are the deployed code's own recipe, copied rather than re-derived:
    `robot_video_dataset.py:211-235` formats `DEFAULT_PROMPT` with the task string, and
    `:251-252` opens `f"{sha256(prompt)}.t5_len{context_len}.{enc_id}.pt"`.
    """
    prompt = str(template).format(task=task)
    return prompt, hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def prompt_cache_name(sha256_hex, context_len, enc_id=PROMPT_CACHE_ENC_ID):
    return "%s.t5_len%d.%s.pt" % (sha256_hex, int(context_len), enc_id)


def training_prompt_facts(doc, default_task, template, arrays_path=None):
    """The instruction these weights were conditioned on, read off the training data.

    `data.train._target_` names `A3VideoDataset`, whose constructor takes the task from
    `arrays.npz["task_str"]` when that key exists and from `DEFAULT_A3_TASK` when it does not
    (`a3_video_dataset.py:115`). This corpus has no `task_str` -- measured below, not assumed -- so
    every recorded window was conditioned on one prompt, and `text_embedding_cache_dir` holds one
    embedding file for it.

    The instruction is therefore not a free knob at eval time. A prompt the model never saw is not
    a harder version of the task, it is a different conditioning input, and a low success rate
    could not be told apart from the model ignoring its text. So every step that fails raises:
    more than one task string, no cached embedding for the prompt, or more cache files than the
    one this prompt names.
    """
    target = str(doc.data.train.get("_target_") or "")
    if not target.endswith("a3_video_dataset.A3VideoDataset"):
        raise SystemExit("data.train._target_ is %r, not the A3 dataset whose task-string rule "
                         "this function implements; the training instruction has to be read from "
                         "the dataset class that was actually used" % target)
    dirs = [str(x) for x in (doc.data.train.get("dataset_dirs") or [])]
    if len(dirs) != 1:
        raise SystemExit("expected exactly one dataset_dirs entry in the training config, got %r; "
                         "one eval run conditions on one task string" % (dirs,))
    arrays = Path(str(arrays_path or (Path(dirs[0]) / "arrays.npz")))
    if not arrays.is_file():
        raise SystemExit("%s is not readable, so the training task string cannot be established"
                         % arrays)
    with np.load(str(arrays), allow_pickle=True) as a:
        has = "task_str" in a.files
        tasks = (sorted(set(str(x) for x in np.asarray(a["task_str"]).reshape(-1)))
                 if has else [str(default_task)])
        rows = int(np.asarray(a["action"]).shape[0])
    if len(tasks) != 1:
        raise SystemExit("the training corpus carries %d distinct task strings (%s); this harness "
                         "conditions every episode on one instruction, so it cannot evaluate these "
                         "weights without a per-window task" % (len(tasks), tasks))
    task = tasks[0]
    context_len = int(doc.data.train.context_len)
    cache_dir = Path(str(doc.data.train.text_embedding_cache_dir))
    prompt, sha = hashed_prompt(task, template)
    name = prompt_cache_name(sha, context_len)
    if not (cache_dir / name).is_file():
        raise SystemExit("no cached training embedding for the training prompt in %s (expected %s); "
                         "the loader that trained these weights reads exactly that file name and "
                         "would have raised FileNotFoundError, so something here is out of step "
                         "with the corpus" % (cache_dir, name))
    present = sorted(p.name for p in Path(cache_dir).glob("*.pt"))
    if len(present) != 1:
        raise SystemExit("%s holds %d embedding files (%s); the corpus and the cache no longer "
                         "describe one instruction, so which conditioning is being evaluated would "
                         "have to be re-derived before any of it is scored"
                         % (cache_dir, len(present), present))
    return {"task": task,
            "task_source": ("arrays.npz:task_str" if has
                            else "a3_video_dataset.DEFAULT_A3_TASK, because %s carries no task_str "
                                 "key (measured)" % arrays.name),
            "prompt": prompt, "prompt_sha256": sha, "context_len": context_len,
            "cache_dir": str(cache_dir), "cache_file": name, "cache_files_present": present,
            "dataset_target": target, "arrays": str(arrays), "rows": rows,
            "encoder_tag": PROMPT_CACHE_ENC_ID}


def resolve_prompt_facts(config):
    """`training_prompt_facts` with the two constants imported from the copy that trained them.

    Kept apart from the pure function above so that logic stays testable without `fasterwam`
    importable, and so the import happens after `use_stage_fw()` has pinned the package.
    """
    from omegaconf import OmegaConf

    from fasterwam.datasets.lerobot.a3_video_dataset import DEFAULT_A3_TASK
    from fasterwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    doc = config if hasattr(config, "data") else OmegaConf.load(str(config))
    facts = training_prompt_facts(doc, DEFAULT_A3_TASK, DEFAULT_PROMPT)
    import fasterwam
    facts["constants_from"] = {"package_file": str(getattr(fasterwam, "__file__", "") or ""),
                               "DEFAULT_A3_TASK": DEFAULT_A3_TASK,
                               "DEFAULT_PROMPT": DEFAULT_PROMPT}
    return facts


def check_instruction(requested, facts):
    """Accept only the instruction the weights were trained on; `None` means "use that one"."""
    if requested is None:
        return facts["task"]
    if str(requested) != str(facts["task"]):
        raise SystemExit(
            "--instruction %r is not the instruction these weights were trained on, which is %r "
            "(%s). It has no cached embedding either: %s holds %s. Refusing to evaluate a prompt "
            "the model never saw -- a low success rate under it would not be a policy result."
            % (requested, facts["task"], facts["task_source"], facts["cache_dir"],
               facts["cache_files_present"]))
    return facts["task"]


def verify_prompt_conditioning(model, facts):
    """Prove the inference-time encoder reproduces, bit for bit, what training conditioned on.

    Training never ran the T5: the dataset read the cached embedding and applied
    `context[~context_mask] = 0.0; context_mask = ones` (`robot_video_dataset.py:218-221`). The
    harness goes the other way round: `infer_action` calls `encode_prompt(prompt)`
    (`jointwam.py:211`), which zeroes past the last real token and returns an all-True mask
    (`fastwam.py:456-465`). Those two tensors are all the cross-attention ever sees, so they have to
    be the same bits.

    Measured on the pod 2026-11-12 against the real encoder: `max_abs_diff` 0.0 over the whole
    (128, 4096) bf16 tensor, `bit_equal_frac` 1.0, 26 real tokens in a contiguous prefix on both
    sides. The assertion is therefore exact equality with no tolerance, because anything else means
    the encoder, the tokenizer length or the device differs from what wrote the cache.
    """
    import torch

    cache_path = Path(facts["cache_dir"]) / facts["cache_file"]
    payload = torch.load(str(cache_path), map_location="cpu", weights_only=False)
    cached, cached_mask = payload["context"], payload["mask"].bool()
    training_context = cached.clone()
    training_context[~cached_mask] = 0.0
    with torch.no_grad():
        emb, mask = model.encode_prompt(facts["prompt"])
    if emb.ndim != 3 or emb.shape[0] != 1 or tuple(emb.shape[1:]) != tuple(cached.shape):
        raise SystemExit("encode_prompt returned %s where the training cache holds %s; the two are "
                         "not comparable, so the conditioning cannot be checked"
                         % (tuple(emb.shape), tuple(cached.shape)))
    encoded = emb[0].to(device=training_context.device)
    out = {"cache_path": str(cache_path), "cached_shape": list(cached.shape),
           "encoded_shape": list(emb.shape), "cache_context_len": int(cached.shape[0]),
           "cached_mask_true": int(cached_mask.sum()),
           "encoded_mask_all_true": bool(mask.all()),
           "encoded_nonzero_token_rows": int((encoded != 0).any(dim=-1).sum()),
           "training_context_nan": bool(torch.isnan(training_context).any()),
           "encoded_nan": bool(torch.isnan(encoded).any()),
           "max_abs_diff": float((encoded.float() - training_context.float()).abs().max()),
           "n_diff_elements": int((encoded != training_context).sum()),
           "n_elements": int(training_context.numel()),
           "dtype": str(encoded.dtype)}
    if out["encoded_nan"] or out["training_context_nan"]:
        raise SystemExit("NaN in the text conditioning (cached=%s encoded=%s); nothing downstream "
                         "of this can be trusted"
                         % (out["training_context_nan"], out["encoded_nan"]))
    if out["n_diff_elements"] != 0:
        raise SystemExit(
            "the prompt the harness conditions on does not reproduce the cached training embedding: "
            "%d of %d elements differ (max abs %s). The raw cache and the encoder output can only "
            "disagree through the padding the dataset zeroed, so a difference here means the "
            "encoder, tokenizer length or device is not what produced %s, and evaluating through it "
            "would change the text conditioning rather than measure the policy under it."
            % (out["n_diff_elements"], out["n_elements"], out["max_abs_diff"], facts["cache_file"]))
    if out["encoded_nonzero_token_rows"] != out["cached_mask_true"]:
        raise SystemExit("the encoder put %d non-zero token rows where training had %d; the prompt "
                         "is not the one the cache was written for"
                         % (out["encoded_nonzero_token_rows"], out["cached_mask_true"]))
    return out


def build_policy(args, cfg, bridge_mods):
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf


    env_factory, W = bridge_mods
    doc = OmegaConf.load(str(args.config))
    EVAL_SETTINGS_FACTS.clear()
    EVAL_SETTINGS_FACTS.update(attach_eval_settings(doc))
    # The text conditioning is settled before anything expensive opens: it decides both whether
    # the model can be served a prompt at all (the encoder) and which sentence every chunk of all
    # 160 episodes is conditioned on, and a wrong answer there invalidates the whole batch.
    TEXT_ENCODER_FACTS.clear()
    TEXT_ENCODER_FACTS.update(deployment_text_encoder_settings(doc))
    TEXT_PROMPT_FACTS.clear()
    TEXT_PROMPT_FACTS.update(resolve_prompt_facts(doc))
    instruction = check_instruction(getattr(args, "instruction", None), TEXT_PROMPT_FACTS)
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    calib = read_json(args.calib)
    from fasterwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    stats = load_dataset_stats_from_json(str(args.stats))
    # A stats file whose state columns are 8 wide loads happily and then fails inside the
    # normaliser with a shape assertion, or worse, normalises the six force columns against the
    # wrong band. Both the width and the force band are checked against the files themselves --
    # before the 11 GB artifact is opened, so a wrong stats path costs no GPU time.
    check_stats_width(stats, args.stats, calib_std=calib["std"])
    WEIGHT_PATH_FACTS.clear()
    WEIGHT_PATH_FACTS.update(anchor_weight_search_path(doc))
    model = instantiate(doc.model, model_dtype=torch.bfloat16, device=device)
    merged = Path(args.merged) if args.merged else \
        Path(args.merged_dir) / ("step_%06d_merged.pt" % args.step)
    if not merged.is_file():
        raise SystemExit(f"merged artifact {merged} does not exist; run "
                         "scripts/export_merged_v3.py for that node first (see --dry-run)")
    model.load_checkpoint(str(merged))
    # the deployment worker does `model.to(device).eval()` after loading; without it a parameter
    # that `load_checkpoint` replaced wholesale could still sit on the CPU, and the module would
    # stay in train mode
    model = model.to(device).eval()
    if getattr(model, "text_encoder", None) is None or getattr(model, "tokenizer", None) is None:
        raise SystemExit("the built model has no text encoder/tokenizer even though the deployment "
                         "config asks for one (text_encoder=%s tokenizer=%s); every chunk would die "
                         "inside `encode_prompt` on the first replan"
                         % (type(getattr(model, "text_encoder", None)).__name__,
                            type(getattr(model, "tokenizer", None)).__name__))
    # `model.eval()` above has to have reached the encoder: `WanTextEncoder` is loaded by
    # `_load_registered_model`, which leaves a fresh module in train mode, and dropout there would
    # make the conditioning both random and different from the cache this check compares against.
    TEXT_CHECK_FACTS.clear()
    TEXT_CHECK_FACTS.update({"text_encoder_training": bool(model.text_encoder.training),
                             "device": str(device)})
    if model.text_encoder.training:
        raise SystemExit("the text encoder is in train mode after model.eval(); its output would not "
                         "be the deterministic tensor training read from the cache")
    TEXT_CHECK_FACTS.update(verify_prompt_conditioning(model, TEXT_PROMPT_FACTS))
    model.set_force_norm_stats(calib["mean"], calib["std"], scale=float(calib["scale"]))
    # built the way `infer_a3_worker.main` builds it -- the whole processor from the config, then
    # the stats attached to its normalizer. `FastWAMProcessor(stats=...)` is not a call that
    # exists: the constructor has no `stats` parameter in either copy on this pod.
    processor = instantiate(doc.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    from action_adapter_v3 import FasterWAMA3ActionAdapterV3
    adapter = FasterWAMA3ActionAdapterV3()
    from policy40k import ForcePolicy
    # `W` stays the deployment module itself -- every bridge call is the worker's own function by
    # identity -- and only the proprio source is overridden, with the shared 14-wide builder the
    # checkpoints were trained against. The override lives at the model call site in
    # `policy40k.ForcePolicy`, so the batch evaluator uses it too, not just the smoke test.
    return ForcePolicy(model, processor, doc, adapter, device, bridge=W,
                       proprio_fn=lambda obs: build_proprio14(obs, W, calib),
                       instruction=instruction, gripper_mode=args.gripper_mode,
                       hist_len=HIST_LEN, name="node%06d" % args.step)


def scene_key(env):
    """Hash of the cookies' start pose.

    Two eval episodes are two different *scenes* only if this differs; otherwise they are two
    samples from one scene and the spread between them is the policy's own sampling noise. The
    env's `position_noise_m`/`yaw_noise_rad`/`randomize_boxes` knobs decide that, and `reset`
    draws nothing random when they are all at zero, so this is measured rather than assumed.
    """
    import mujoco

    uw = getattr(env, "unwrapped", env)
    m = uw.model
    adr = sorted(int(m.jnt_qposadr[i]) for i in range(m.njnt)
                 if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or "")
                 .startswith("cookie_"))
    if not adr:
        return None
    q = np.round(np.asarray(uw.data.qpos, dtype=np.float64)[adr], 7)
    return hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest()[:16]


def run_node(args, step, seeds, calib_facts):
    """Run every seed of one node. Returns the per-node summary dict."""
    from policy40k import StubPolicy

    out_dir = Path(args.out) / ("step_%06d" % step)
    out_dir.mkdir(parents=True, exist_ok=True)
    st = node_state(step)
    if st is None:
        return {"step": step, "status": "no_light_node"}
    resolve_calib(args.calib, st["force_calibration"])

    account = ForceErrorAccount(horizon=args.force_horizon)
    records = []
    scene_keys = {}
    model_calls, chunk_log = [], []
    if args.backend == "fixture":
        from flow_test import FixtureEnv
        reset = lambda env, seed: env.reset(seed=seed)  # noqa: E731
        make_policy = lambda: StubPolicy(  # noqa: E731
            action_horizon=args.force_horizon, pred_mode=args.stub_pred_mode,
            offset=args.stub_offset)
    else:
        env_factory, W = build_env(args)
        args.step = step
        policy = build_policy(args, None, (env_factory, W))
        # The batch is instrumented by the smoke's own helpers, not by a copy of them: if the
        # 160-episode numbers came from different recording code than the two-episode live check,
        # passing the live check would prove nothing about the batch. Imported inside the branch
        # because `smoke40k` imports this module, and a top-level import would be a cycle.
        from smoke40k import (install_chunk_recorder, install_env_timer,  # noqa: E402
                              install_recorder, make_proprio_witness, proprio_band_rollup,
                              cpu_throttle_facts, record_time_split, summarize_episode,
                              time_rollup)
        install_recorder(policy.model, model_calls)
        install_chunk_recorder(policy, chunk_log, witness=make_proprio_witness(W, calib_facts))

        def reset(env, seed):
            obs, _info = env_factory.reset(env, seed, randomize_cookies=True,
                                           randomize_boxes=args.randomize_boxes,
                                           batch_offset_m=args.batch_offset)
            sanity = env_factory.scene_sanity(env)
            if not sanity["ok"] and not args.allow_invalid_scene:
                raise SystemExit(f"scene invalid at seed={seed}: {sanity}")
            scene_keys[int(seed)] = scene_key(env)
            return obs

        def make_policy():
            return policy
    shadow = None
    if args.shadow:
        sys.path.insert(0, str(STAGE / "dev" / "p4_shadow"))
        from shadow import ShadowMonitor  # noqa: E402
        shadow = ShadowMonitor(calib=calib_facts, log_path=out_dir / "shadow.jsonl",
                               horizon=args.force_horizon, hist_len=HIST_LEN,
                               channel_names=list(CHANNEL_NAMES))

    for seed in seeds:
        base, cbase = len(model_calls), len(chunk_log)
        env = FixtureEnv(seed=seed, frames=args.fixture_frames,
                         force_mode=args.fixture_force_mode) if args.backend == "fixture" \
            else env_factory.make_env(position_noise_m=args.position_noise,
                                      yaw_noise_rad=args.yaw_noise, render_cameras=True)
        pol = make_policy() if args.backend == "fixture" else policy
        # The wall-time split is measured, not inferred from the total: an episode that took 8x
        # longer than its sibling was CPU starvation of the simulator, not of the GPU, and the only
        # way to say that about 160 episodes is to have the two numbers per episode.
        meter = None if args.backend == "fixture" else install_env_timer(env)
        th0 = {} if meter is None else cpu_throttle_facts()
        rec, measured, chunks = run_episode(
            env, pol, seed=seed, reset=reset, max_steps=args.max_steps, replan=args.replan,
            shadow=shadow)
        scored = 0
        pred_chunks = [c for c in chunks if c["pred_force"] is not None]
        for c in pred_chunks:
            scored += account.add_chunk(c["frame"], c["pred_force"], measured)
        rec["force_steps_scored"] = scored
        rec["step"] = step
        if args.backend != "fixture":
            summarize_episode(rec, model_calls[base:], chunk_log[cbase:], chunks)
            record_time_split(rec, meter, th0, cpu_throttle_facts())
        records.append(rec)
        # only chunks that actually carry a prediction are stored, in both arrays, so the two
        # columns stay one-per-row and a reader can pair frame f with its predicted force
        np.savez_compressed(out_dir / ("measured_seed%d.npz" % seed),
                            measured=np.asarray(measured, dtype=np.float32),
                            chunk_frames=np.asarray([c["frame"] for c in pred_chunks], np.int64),
                            chunk_pred=np.asarray([c["pred_force"] for c in pred_chunks],
                                                  dtype=np.float32))
        (out_dir / ("episode_seed%d.json" % seed)).write_text(json.dumps(rec, indent=2))
        if hasattr(env, "close"):
            env.close()
        print(json.dumps(rec), flush=True)

    n = len(records)
    k = sum(1 for r in records if r["success"])
    lo, hi = wilson_ci(k, n)
    summary = {"step": step, "status": "ran", "backend": args.backend,
               "policy": args.policy, "seeds": list(seeds), "episodes": n,
               "successes": k, "success_rate": round(k / n, 4) if n else None,
               "success_ci95": [round(lo, 4), round(hi, 4)],
               "mean_frames": round(float(np.mean([r["frames"] for r in records])), 1) if n else None,
               "nan_aborts": sum(1 for r in records if r["nan_abort_frame"] >= 0),
               "force": account.finalize(), "records": records,
               "calibration": calib_facts,
               "fw_import": dict(FW_IMPORT_FACTS),
               "eval_settings": dict(EVAL_SETTINGS_FACTS),
               "light_node": st["dir"], "force_contract_channels": list(CHANNELS)}
    if args.backend != "fixture":
        roll = proprio_band_rollup(records)
        paired = [r["proprio_band_map_maxdiff"] for r in records
                  if r.get("proprio_band_map_maxdiff") is not None]
        summary["proprio"] = {
            "band": roll["band"], "clamped_calls": roll["clamped_calls"],
            "clamped_columns": roll["clamped_columns"], "exceedances": roll["exceedances"],
            "unexplained": roll["unexplained"],
            "band_map_worst_diff": None if not paired else max(paired),
            "episodes_paired": len(paired), "episodes": n,
            "episodes_without_witness": roll["episodes_without_witness"],
            "meaning": ("what the model was handed, per replan, is the recorded band's own map of "
                        "the raw 14 columns of the observation the same replan measured; every "
                        "column that left [-1, 1] is listed with the raw value and frame that put "
                        "it there, so a value at the +-5 clamp is attributable to the robot "
                        "leaving the 42,681 training rows and not to this harness")}
    if args.backend != "fixture":
        distinct = len({v for v in scene_keys.values() if v})
        summary["scene"] = {
            "position_noise_m": args.position_noise, "yaw_noise_rad": args.yaw_noise,
            "randomize_boxes": args.randomize_boxes, "batch_offset_m": args.batch_offset,
            "distinct_scene_keys": int(distinct), "episodes": n,
            "meaning": ("each episode starts from a different scene" if distinct > 1 else
                        "every episode starts from the SAME scene: the seed only re-draws the "
                        "policy's diffusion noise, so this success rate measures repeatability "
                        "on one layout, not generalisation across layouts"),
        }
        # rolled up by the smoke's own function, on the batch's own records: the eight-node run is
        # where the 160-episode time estimate is actually made, so the estimate cannot be left in
        # the two-episode script. Mean and median both, because one CPU-starved episode doubles a
        # mean without making any other episode slower.
        timing = time_rollup(records)
        timing["projected_160_episodes_h_median"] = round(
            ((timing["wall"] or {}).get("median") or 0.0) * 160 / 3600.0, 2)
        timing["projected_160_episodes_h_mean"] = round(
            ((timing["wall"] or {}).get("mean") or 0.0) * 160 / 3600.0, 2)
        timing["this_run_h"] = round(sum(float(r.get("wall_s") or 0.0) for r in records) / 3600.0, 3)
        summary["timing"] = timing
    if shadow is not None:
        summary["shadow"] = shadow.finalize()
    if args.backend == "fixture":
        summary["fixture"] = {"force_mode": args.fixture_force_mode,
                              "stub_offset": args.stub_offset,
                              "frames_cap": args.fixture_frames}
        if args.fixture_force_mode == "const":
            # a constant-force world makes persistence exact, so the error left over is the offset
            # the stub adds: the MAE this run must report for the accounting to be right
            summary["expected_mae"] = abs(float(args.stub_offset))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True,
                                                     default=str))
    return summary


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - s) / d, (c + s) / d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=node_steps())
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="one episode per seed; the default is the %d seeds in %s"
                         % (N_SCREENED_SEEDS, SEED_SCREEN))
    ap.add_argument("--backend", choices=["fixture", "env"], default="env")
    ap.add_argument("--policy", choices=["stub", "merged"], default="merged")
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--merged-dir", default=str(MERGED_DEFAULT))
    ap.add_argument("--merged", default=None, help="explicit artifact for a single --step")
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--stats", default=str(V1G_STATS),
                    help="normalisation stats; the default is the file "
                         "`run40k/config.yaml` names as `pretrained_norm_stats`, i.e. the stats "
                         "these weights were trained with (14-wide state, force tare included)")
    ap.add_argument("--calib", default=str(CALIB_PATH))
    ap.add_argument("--instruction", default=None,
                    help="the task string every chunk is conditioned on. The default is not a "
                         "default at all: it is read off the training corpus by "
                         "`resolve_prompt_facts`, and any other string is refused, because these "
                         "weights saw exactly one prompt in 40k steps (see README, text "
                         "conditioning).")
    ap.add_argument("--gripper-mode", default="pm1", choices=["pm1", "libero01"])
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--replan", type=int, default=10)
    ap.add_argument("--force-horizon", type=int, default=32)
    ap.add_argument("--fixture-frames", type=int, default=80)
    ap.add_argument("--fixture-force-mode", choices=["const", "ramp"], default="ramp",
                    help="const makes the stub's MAE exactly --stub-offset, which is how the "
                         "flow test checks the accounting; ramp only exercises the plumbing")
    ap.add_argument("--stub-offset", type=float, default=0.5,
                    help="newtons the stub predictor adds to persistence in the fixture world")
    ap.add_argument("--stub-pred-mode", default="offset")
    ap.add_argument("--shadow", action="store_true", help="attach the P4 shadow observer")
    ap.add_argument("--cpu", action="store_true", help="load on CPU: a smoke, not a result")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--position-noise", type=float, default=0.0)
    ap.add_argument("--yaw-noise", type=float, default=0.0)
    ap.add_argument("--batch-offset", type=float, default=0.0)
    ap.add_argument("--randomize-boxes", action="store_true")
    ap.add_argument("--allow-invalid-scene", action="store_true")
    ap.add_argument("--disk-policy", choices=["report", "refuse"], default="report")
    args = ap.parse_args()
    args.seeds = check_one_episode_per_seed(screened_seeds() if args.seeds is None
                                            else args.seeds)

    if args.backend == "env" and args.policy != "merged":
        raise SystemExit("--backend env needs --policy merged: a stub in the real env "
                         "measures the env, not the checkpoint")
    calib_facts = resolve_calib(args.calib, None)
    if args.backend == "env":
        # Resolved before the first merge so a corpus/config disagreement stops the batch here
        # rather than 20 episodes into it. `build_policy` checks it again against the model it
        # actually built; this call only puts the same facts into `plan.json`.
        TEXT_PROMPT_FACTS.clear()
        TEXT_PROMPT_FACTS.update(resolve_prompt_facts(args.config))
        args.instruction = check_instruction(args.instruction, TEXT_PROMPT_FACTS)
    doc = plan_doc(args.steps, args.seeds, args.merged_dir, calib_facts, args)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "plan.json").write_text(json.dumps(doc, indent=2, sort_keys=True,
                                                         default=str))
    if args.dry_run:
        print(json.dumps(doc, indent=2, sort_keys=True, default=str))
        return 0
    not_ready = [n["step"] for n in doc["nodes"] if n["status"] != "ready"
                 and args.backend == "env"]
    if not_ready:
        raise SystemExit("refusing to start: these nodes have no merged artifact: %s "
                         "(see %s)" % (not_ready, Path(args.out) / "plan.json"))
    merged_bytes = 11_117_679_929
    if args.backend == "env" and doc["disk"]["free_gib"] * 2**30 < 1.5 * merged_bytes:
        msg = ("only %.2f GiB free under %s; one merged artifact is %.2f GiB"
               % (doc["disk"]["free_gib"], doc["disk"]["path"], merged_bytes / 2**30))
        if args.disk_policy == "refuse":
            raise SystemExit(msg)
        print("WARNING " + msg + " -- report only, nothing here deletes anything", flush=True)

    summaries = []
    for step in args.steps:
        t0 = time.time()
        s = run_node(args, step, args.seeds, calib_facts)
        s["wall_s"] = round(time.time() - t0, 1)
        summaries.append(s)
        (Path(args.out) / "summaries.jsonl").open("a").write(json.dumps(s, default=str) + "\n")
    (Path(args.out) / "all_nodes.json").write_text(json.dumps(
        {"plan": doc, "nodes": summaries}, indent=2, sort_keys=True, default=str))
    print(json.dumps({"out": str(args.out), "nodes": [
        {"step": s["step"], "successes": s.get("successes"),
         "episodes": s.get("episodes"),
         "force_mae_mean": (None if not s.get("force") or not s["force"]["overall"]["mae"]
                            else round(float(np.mean(s["force"]["overall"]["mae"])), 4))}
        for s in summaries]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
