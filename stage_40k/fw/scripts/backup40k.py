"""Back up one 40k node to Hugging Face and prove the remote copy can be restored.

The instance is rented, so a node that only lives on /workspace is a node that does not
survive the month. Uploading is the easy half. The three claims that actually matter,
none of which an HTTP 200 establishes, are:

  V  the remote bytes *are* the local bytes -- per-file sha256 read back from the repo
     tree (V2), plus a re-download and byte-for-byte compare under --deep (W1);
  W  the remote set is the whole restore, not just the weights: adapter, optimizer
     moments, scheduler, RNG, counters, sampler offset, calibration JSON, the pinned
     base-checkpoint version + SHA256, and a copy of the code that produced it, so the
     eight nodes can be resumed from and merged from independently of this machine;
  R  a fresh process can resume from the re-downloaded files. The saved scheduler state
     must be byte-for-byte the state an untouched run has at that step, and replaying the
     trainer's own resume order (optimizer groups, then scheduler state) on the archived
     bytes must put the frozen 40k curve's lr back in the optimizer and keep the next
     steps on it (R2, and the whole set as R7).

A backup nobody can restore is a scary-named file, so --deep on node 1 is not optional:
no local file is proposed for cleanup before it passes.

The token is read from a file (--token-file / HF_TOKEN_FILE), never printed, never
written into a report or an uploaded document, and never allowed to live under a tree
this project syncs to GitHub or to the Hub. U9 checks that claim mechanically by grepping
the staged files for the credential before uploading, and _write() refuses to persist a
report or a ledger line that contains it.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive40k as plan  # noqa: E402  single source of truth for paths and the frozen contract

STAGE = plan.STAGE
BACKUP = STAGE / "backup40k"
DEFAULT_REPO = "wsadwsad6779/fasterwam-a3-force-40k"
ARCHIVE_REPO = "wsadwsad6779/fasterwam-a3-cookie-lora"
# The 11 GB base was never uploaded to our archive repo -- it arrived with the upstream
# release -- so "can someone without this instance restore the eight nodes" is a question
# about the upstream repo, and W2 has to search there. Order matters only for speed.
UPSTREAM_BASE_REPO = "hustvl/FasterWAM"
BASE_DIR_IN_REPO = "libero"
BASE_SOURCES = (UPSTREAM_BASE_REPO, ARCHIVE_REPO)
DEFAULT_PREFIX = "v1_force_40k"
TOKEN_FILE_DEFAULT = "/root/.cache/hf_token"
VERIFY_ROOT = Path("/root/hfverify")
# Anything under these paths gets pushed to GitHub or tarred up by the archive scripts,
# so a credential must never be parked there.
FORBIDDEN_DIRS = ("/workspace/_v1_stage", "/workspace/FasterWAM",
                  "/workspace/fasterwam_a3_bridge", str(STAGE))
STATE_FILES = ("adapter.pt", "optimizer.pt", "scheduler.pt", "rng.pt", "trainer_state.json")
CODE_FILES = (
    "fw/scripts/train_seg.py", "fw/scripts/drive40k.py", "fw/scripts/export_merged_v3.py",
    "fw/src/fasterwam/light_state_v3.py", "fw/src/fasterwam/r4_patch.py",
    "fw/src/fasterwam/seg_patch.py", "fw/src/fasterwam/base_load_guard.py",
    "fw/src/fasterwam/models/wan22/force_condition.py",
    "fw/src/fasterwam/models/wan22/fastwam.py",
)
CALIB_FALLBACK = STAGE / "force_calib_v1.json"
FORCE_TENSORS = 42
ADAPTER_TENSORS = 828
SECRET = ""
# Which rounding produced the buffers is not decoration: the calibration JSON holds
# float64-looking decimals, the model holds whatever dtype it was built with, and the
# fingerprint hashes the *rounded* values. Deciding that from the repo alone is the
# difference between "the units match" and "we hope they match".
ROUND_CANDIDATES = (("float32", torch.float32), ("bfloat16", torch.bfloat16),
                    ("float16", torch.float16))


def mask(text):
    """No log line, report field or exception may carry the credential.

    Presigned upload URLs are redacted too: they are short-lived write credentials,
    and an exception message from the transport is exactly where one would surface.
    """
    out = str(text)
    if SECRET:
        out = out.replace(SECRET, "<HF_TOKEN>")
    return re.sub(r"(https?://[^\s'\"]*)\?[^\s'\"]*", r"\1?<query-redacted>", out)


@functools.lru_cache(maxsize=None)
def _lrs(steps):
    """The LR the *trainer's own* scheduler reports after N steps, for arbitrary N.

    `drive40k.boundary_lrs` only samples the 5000-step boundaries, but a rehearsal node
    at step 12 needs its neighbours, and writing the cosine out again would create a
    second source of truth free to disagree with the first.
    """
    import torch
    from fasterwam.trainer import Wan22Trainer

    want = set(int(s) for s in steps)
    upto = max(want)

    class Shim:
        def __init__(self, optimizer):
            self.optimizer = optimizer
            self.learning_rate = plan.LR

    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=plan.LR, betas=(0.9, 0.95), weight_decay=1e-2)
    sch = Wan22Trainer._build_scheduler(Shim(opt), "cosine", plan.TOTAL, int(plan.TOTAL * 0.05))
    out = {0: float(opt.param_groups[0]["lr"])}
    for i in range(1, upto + 1):
        sch.step()
        if i in want:
            out[i] = float(opt.param_groups[0]["lr"])
    return out


def lr_at_local(step):
    # lru_cache needs a hashable key, hence the tuple
    return float(_lrs((int(step),))[int(step)])


def boundary_cross_check(step):
    """On a 5000 boundary the driver's own table must agree, digit for digit."""
    if step % plan.NODE:
        return {"boundary": False, "local": lr_at_local(step)}
    return {"boundary": True, "drive40k_lr_at": plan.lr_at(step), "local": lr_at_local(step),
            "equal": plan.lr_at(step) == lr_at_local(step)}


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_blob_sha1(path, chunk=1 << 22):
    """What the Hub reports as `oid` for a file stored in git rather than LFS."""
    h = hashlib.sha1()
    h.update(b"blob %d\x00" % os.path.getsize(path))
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_token(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit("token file %s missing" % p)
    if any(str(p.resolve()).startswith(d) for d in FORBIDDEN_DIRS):
        raise SystemExit("refusing to use a credential stored inside a synced tree: %s" % p)
    mode = stat.S_IMODE(p.stat().st_mode)
    if mode & 0o077:
        raise SystemExit("refusing to use a credential readable beyond its owner: %s is %s"
                         % (p, oct(mode)))
    text = p.read_text().strip()
    if len(text) < 20 or not text.startswith("hf_"):
        raise SystemExit("token file %s does not look like a HF token (len=%d)" % (p, len(text)))
    return text


def _hub(endpoint):
    """Import huggingface_hub AFTER pinning the endpoint and the transfer format.

    `constants.py` resolves the endpoint at import time, so setting the env var later
    would silently send uploads wherever the container pointed them.

    Xet is deliberately forced ON here, and that is a measurement rather than a preference:
    this instance's egress proxy answers `CONNECT tunnel failed: 403` for
    `*.s3-accelerate.amazonaws.com`, so the LFS/S3 leg cannot be used at all, while
    `cas-server.xethub.hf.co` goes through (70 MB probe: 39.2 s = 1.87 MB/s). The same probe
    showed the server forces LFS for any binary payload regardless of extension, so there is
    no "regular commit" fallback to fall back on either. Leaving the variable untouched is
    not an option -- the container exports HF_HUB_DISABLE_XET=1, which silently selects the
    dead route.

    What Xet costs is the dedup stall seen on the archive instance: a stalled transfer looks
    like a slow one, so the watcher wraps every attempt in a wall-clock timeout and retries.
    """
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ["HF_HUB_DISABLE_XET"] = "0"
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    from huggingface_hub import HfApi, hf_hub_download
    return HfApi, hf_hub_download


def _entry_info(entry):
    """Normalise a repo-tree entry to the fields the verifier needs.

    Written against what this client actually returns, measured on the rehearsal repo rather
    than the API docs: `RepoFile.type` is None in huggingface_hub 0.36 (filtering on it made
    a fully populated repo look empty), the git blob hash is `blob_id` and not `oid`, and an
    LFS object is a `BlobLfsInfo` carrying `sha256`, not the `oid` the JSON calls it.
    """
    get = entry.get if isinstance(entry, dict) else (lambda k: getattr(entry, k, None))
    lfs = get("lfs") or {}
    lget = lfs.get if isinstance(lfs, dict) else (lambda k: getattr(lfs, k, None))
    cls = type(entry).__name__
    kind = get("type") or {"RepoFile": "file", "RepoFolder": "directory"}.get(cls)
    return {"path": get("path") or get("rfilename"), "size": get("size"), "type": kind,
            "oid": get("oid") or get("blob_id") or get("sha"),
            "lfs_oid": lget("oid") or lget("sha256"), "lfs_size": lget("size")}


def calibration_source():
    """The one calibration JSON the live config actually hands the model.

    Hardcoding a path here would be the bug this whole function exists to avoid: a
    backup that ships a calibration file the training never read would make the repo
    look self-consistent while the real units stayed unpinned.
    """
    cfg_path = plan.RUN / "config.yaml"
    found = []
    if cfg_path.is_file():
        import yaml
        text = cfg_path.read_text()
        try:
            cfg = yaml.safe_load(text)
        except Exception:  # noqa: BLE001
            cfg = None
        if isinstance(cfg, dict):
            _walk_norm_stats(cfg, found)
    unique = sorted({p for p in found if p})
    if len(unique) > 1:
        raise SystemExit("config declares %d different force norm_stats files: %s"
                         % (len(unique), unique))
    path = Path(unique[0]) if unique else CALIB_FALLBACK
    if not path.is_file():
        raise SystemExit("calibration JSON %s does not exist; the fingerprint would be "
                         "unverifiable and the units unbacked" % path)
    return path


def _walk_norm_stats(obj, out):
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key == "norm_stats" and isinstance(val, dict) and "file" in val:
                out.append(str(val["file"]))
            else:
                _walk_norm_stats(val, out)
    elif isinstance(obj, list):
        for item in obj:
            _walk_norm_stats(item, out)


def calibration_closure(calib_path, fingerprint):
    """Prove the uploaded JSON, the uploaded fingerprint and one dtype explain each other.

    Exactly one dtype rounding may reproduce both sha256s; zero matches means the
    shipped file is not the one that produced the weights, and two would mean the
    evidence does not identify anything.
    """
    if not fingerprint:
        return False, {"error": "node declares no force_calibration fingerprint"}
    doc = json.loads(Path(calib_path).read_text())
    matches = []
    for label, dtype in ROUND_CANDIDATES:
        digests = {}
        for key in ("force_mean_sha256", "force_std_sha256"):
            field = "mean" if "mean" in key else "std"
            if field not in doc:
                digests[key] = None
                continue
            widened = torch.tensor(doc[field], dtype=torch.float32).to(dtype).to(torch.float32)
            digests[key] = hashlib.sha256(widened.contiguous().numpy().tobytes()).hexdigest()
        if digests.get("force_mean_sha256") == fingerprint.get("force_mean_sha256") and \
           digests.get("force_std_sha256") == fingerprint.get("force_std_sha256"):
            matches.append(label)
    evidence = {"calib_path": str(calib_path), "calib_sha256": sha256_file(calib_path),
                "calib_version": doc.get("version"), "channels": doc.get("channels"),
                "fingerprint": fingerprint, "dtype_roundings_that_match": matches,
                "json_mean": doc.get("mean"), "json_std": doc.get("std")}
    return len(matches) == 1, evidence


def base_weights_info():
    path = plan.BASE
    return {"file": path.name, "path_on_instance": str(path),
            "bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256_file(path) if path.exists() else None,
            "upstream_release": "fasterwam_release/libero",
            "hub_sources": [{"repo": UPSTREAM_BASE_REPO,
                             "path": "%s/%s" % (BASE_DIR_IN_REPO, path.name),
                             "revision": "main",
                             "note": "W2 checks this repo and BASE_SOURCES by sha256+bytes; "
                                     "this field is the claim being checked"}],
            "role": "the frozen 40k run resumed from this file at global_step 0; each node "
                    "is this base plus that node's adapter, so pinning it is what makes the "
                    "eight merged models reproducible elsewhere",
            "merge_recipe": "python scripts/export_merged_v3.py <node light dir> "
                            "--base <this file> --out merged_stepNNNNNN.pt"}


def frozen_contract():
    state = plan._expected_membership()[0]
    if state is None:
        raise SystemExit("no rehearsal light state to take the frozen contract from")
    return state


def local_hashes(node_dir):
    rows = []
    for p in sorted(node_dir.iterdir()):
        if not p.is_file() or p.name.endswith(".tmp"):
            continue
        rows.append({"path": p.name, "bytes": p.stat().st_size,
                     "sha256": sha256_file(p), "git_blob": git_blob_sha1(p)})
    names = sorted(r["path"] for r in rows)
    missing = [n for n in STATE_FILES if n not in names]
    if missing:
        raise SystemExit("%s is not a complete light state; missing %s" % (node_dir, missing))
    return rows


README = """---
license: other
license_name: upstream-fasterwam-terms-unstated
license_details: >-
  These files are LoRA adapters, optimizer state and configuration produced by training on
  top of the FasterWAM `fasterwam_release/libero` base checkpoint (step_021700.pt). The
  upstream model card states no license for those base weights, so this repo claims none: the
  merged models that the adapters produce inherit that ambiguity and are not offered under
  any explicit terms. Private backup of one's own training run, pending clarification.
tags: [robotics, vision-language-action, lora, force-aware]
---

# FasterWAM V1 force-conditioned 40k run

Eight checkpoints from one continuous 40,000-optimizer-step run on a single cosine
schedule (2,000-step linear warmup to 1e-4, then cosine to 1e-6 at step 40,000; segments
only ever lower the loop bound, never the schedule horizon). The run resumes from the
original FasterWAM `fasterwam_release/libero` base -- NOT from any earlier force
experiment -- so every node is that base plus the node's adapter.

Layout

    nodes/step_NNNNNN/          one directory per 5,000-step node
      adapter.pt                v3 adapter: LoRA + every non-LoRA tensor the optimizer stepped
      optimizer.pt              Adam moments for all 828 optimizer members
      scheduler.pt              the trainer's own SequentialLR state (horizon 40000)
      rng.pt                    captured RNG, so the data order replays
      trainer_state.json        counters, sampler offset, coverage, force calibration fingerprint
      manifest.json             sha256 + bytes for every file above, LR table, code hashes
      base_weights.json         base checkpoint version, path, bytes, SHA256
      force_calib_v1.json       the calibration the weights were trained under
      frozen_contract.json      the rehearsal state that froze membership and fingerprint
      RESTORE.md                how to resume or merge from here
    code/                       the training code that produced these nodes, sha256-pinned

The repo is private. The base weights inherit FasterWAM / Wan-AI terms that still need
the owner's confirmation, so nothing here should be made public before that is settled.
"""

RESTORE_MD = """# Resuming or merging from a node

Restore the instance first (see the code archive's `RESTORE_NEW_INSTANCE.md`), then:

    accelerate launch --num_processes 1 scripts/train_seg.py \
        data=a3_2cam_force task=a3_force_40k batch_size=6 max_steps=40000 \
        learning_rate=0.0001 save_every=<next boundary> \
        output_dir=<run> resume=<directory holding this node>

`max_steps` stays 40000 and `FW_STOP_STEP` alone says where the segment ends. Never set
`FW_LR_RESTART`: rebuilding a short cosine per segment is the defect that made the r5
extension meaningless.

The 11 GB base is deliberately not in this repo -- it is the upstream release file, and W2
in each node's report proves the pinned bytes are fetchable from exactly one address:

    huggingface-cli download hustvl/FasterWAM libero/step_021700.pt

Size and SHA256 are in `base_weights.json` next to this file; a copy that differs in either
is a different model (the archive repo's `step_002000_merged.pt` is 11,117,977,659 B, within
300 KB of the base, and is not the base). Point `FW_BASE_CKPT` at the verified file, because
a v3 light state holds only the 70 MB adapter and the 140 MB of optimizer moments -- the
frozen backbone is not in the node.

On the learning rate, since it is the thing people try to "fix": the lr the run had at the
node's step is stored inside `optimizer.pt`'s param groups, and `scheduler.pt` carries the
continuation. Loading the scheduler without the optimizer leaves the optimizer at the fresh
warmup value, and the next step then recomputes something near the floor -- that is a
restore-order mistake, not a broken checkpoint. Load both, in `load_light_state`'s order,
and do not step the scheduler by hand to catch up.

To publish a usable model rather than a resumable state:

    python scripts/export_merged_v3.py <node dir> --base <base_weights.json path> \
        --out merged_stepNNNNNN.pt

The export refuses when `force_calibration` in `trainer_state.json` does not match the
calibration JSON handed to the model, so `force_calib_v1.json` next to each node is part
of the artifact, not documentation.
"""


def build_upload_tree(node_dir, step, work_root, endpoint, repo):
    """Materialise a directory whose tree IS the remote tree; return (work, pairs, manifest)."""
    contract = frozen_contract()
    state_files = local_hashes(node_dir)
    state = json.loads((node_dir / "trainer_state.json").read_text())
    calib = calibration_source()
    manifest = {
        "node_step": int(step),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo": repo,
        "endpoint": endpoint,
        "schedule": {"total_steps": plan.TOTAL, "node_every": plan.NODE,
                     "warmup_steps": int(plan.TOTAL * 0.05), "base_lr": plan.LR,
                     "lr_at_this_step_by_trainer_scheduler": lr_at_local(step),
                     "boundary_cross_check": boundary_cross_check(step)},
        "files": state_files,
        "trainer_state": state,
        "contract_equals_node": {
            "force_calibration": state.get("force_calibration") == contract.get("force_calibration"),
            "optimizer_member_order": state.get("optimizer_member_order") == contract.get("optimizer_member_order"),
            "adapter_bytes": state.get("adapter_bytes") == contract.get("adapter_bytes"),
        },
        "calibration_source": str(calib),
        "base_weights": base_weights_info(),
        "run_config_sha256": (sha256_file(plan.RUN / "config.yaml")
                              if (plan.RUN / "config.yaml").is_file() else None),
        "code_sha256": {rel: sha256_file(STAGE / rel) for rel in CODE_FILES
                        if (STAGE / rel).exists()},
    }

    if work_root.exists():
        shutil.rmtree(work_root)
    node_dir_remote = work_root / "nodes" / ("step_%06d" % step)
    node_dir_remote.mkdir(parents=True)
    for row in state_files:
        shutil.copy2(node_dir / row["path"], node_dir_remote / row["path"])
    (node_dir_remote / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (node_dir_remote / "base_weights.json").write_text(
        json.dumps(manifest["base_weights"], indent=2, sort_keys=True))
    (node_dir_remote / "frozen_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True))
    shutil.copy2(calib, node_dir_remote / calib.name)
    (node_dir_remote / "RESTORE.md").write_text(RESTORE_MD)
    # The resolved Hydra config and the driver's frozen plan: without them a remote node
    # cannot be told apart from a node trained with different channels or a different
    # warmup, which is the whole thing §四 asks a restore to prove.
    for src, dst in ((plan.RUN / "config.yaml", "run_config.yaml"),
                     (plan.RUN / "plan.json", "driver_plan.json")):
        if src.is_file():
            shutil.copy2(src, node_dir_remote / dst)
    (work_root / "README.md").write_text(README)
    (work_root / "code").mkdir()
    for rel in manifest["code_sha256"]:
        shutil.copy2(STAGE / rel, work_root / "code" / Path(rel).name)

    pairs = []
    for p in sorted(work_root.rglob("*")):
        if p.is_file():
            pairs.append((p, str(p.relative_to(work_root)).replace("\\", "/")))
    return work_root, pairs, manifest


def _parent_dir(rel):
    """Parent component of a repo-relative path, '' for a top-level file.

    `rel.split("/")[-2]` raises for README.md, which sits at the root of the upload tree;
    the node payload is the only thing with a step_ parent.
    """
    parts = str(rel).split("/")
    return parts[-2] if len(parts) >= 2 else ""


def _is_node_payload(rel):
    return _parent_dir(rel).startswith("step_")


def remote_tree(api, repo, prefix):
    """expand=True is not cosmetic: without it the payload has no type/size/lfs fields,
    every entry is then skipped as "not a file", and a perfectly good repo looks empty."""
    entries = {}
    for e in api.list_repo_tree(repo_id=repo, repo_type="model", path_in_repo=prefix,
                                recursive=True, expand=True):
        info = _entry_info(e)
        if info["type"] != "file":
            continue
        rel = info["path"]
        if rel.startswith(prefix.rstrip("/") + "/"):
            rel = rel[len(prefix) + 1:]
        entries[rel] = info
    return entries


def verify_remote(entries, pairs, manifest, out):
    """Every staged file must be present remotely with matching size AND a matching hash.

    A file stored through LFS reports its sha256 as `lfs.oid`; one committed into git
    reports the git blob sha1 as `oid`. Both are checked, and a file whose hash cannot be
    matched either way is a failure -- an absent size match is not "probably fine".
    """
    by_name = {f["path"]: f for f in manifest["files"]}
    rows, bad = [], []
    for local, rel in pairs:
        entry = entries.get(rel)
        if entry is None:
            bad.append({"path": rel, "why": "absent from the repo tree"})
            continue
        local_bytes = os.path.getsize(local)
        name = rel.rsplit("/", 1)[-1]
        in_node_dir = _is_node_payload(rel)
        # the five state files were hashed once already; the docs are small enough to hash here
        expect_sha = by_name[name]["sha256"] if in_node_dir and name in by_name \
            else sha256_file(local)
        expect_blob = git_blob_sha1(local)
        rec = {"path": rel, "bytes_local": local_bytes, "bytes_remote": entry["size"],
               "sha256_local": expect_sha}
        if entry["lfs_oid"]:
            rec["store"] = "lfs"
            rec["sha_remote"] = entry["lfs_oid"]
            rec["ok"] = (str(entry["lfs_oid"]).lower() == expect_sha
                         and entry["lfs_size"] == local_bytes)
        else:
            rec["store"] = "git"
            rec["sha_remote"] = entry["oid"]
            rec["ok"] = (entry["size"] == local_bytes
                         and (entry["oid"] in (expect_blob, expect_sha)))
        rows.append(rec)
        if not rec["ok"]:
            bad.append(rec)
    out["remote_tree_files"] = len(entries)
    out["files_verified"] = [{"path": r["path"], "store": r["store"], "ok": r["ok"]}
                             for r in rows]
    out["files_mismatched"] = bad
    return not bad and len(rows) == len(pairs)


def download_and_compare(api_download, repo, prefix, pairs, token, step):
    """Re-download the node into a fresh dir and hash it: proof it comes back over the net.

    `local_dir` keeps the repo prefix in the on-disk path (`dest/v1_force_40k/nodes/...`), so
    the caller has to use the path the client returned; joining `dest / rel` yourself finds
    nothing and the resume test then fails on a file that is sitting two directories away.
    """
    dest = VERIFY_ROOT / prefix.strip("/") / ("step_%06d" % step)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rows, bad, paths = [], [], {}
    for local, rel in pairs:
        got = api_download(repo_id=repo, repo_type="model",
                           filename="%s/%s" % (prefix.strip("/"), rel),
                           revision="main", token=token, local_dir=str(dest))
        want = sha256_file(local)
        have = sha256_file(got)
        rec = {"path": rel, "bytes": os.path.getsize(got), "sha256_match": want == have}
        rows.append(rec)
        paths[rel] = Path(got)
        if not rec["sha256_match"] or os.path.getsize(got) != os.path.getsize(local):
            bad.append(rec)
    out_rows = {"files": len(rows), "dest": str(dest), "mismatches": bad}
    return not bad and len(rows) == len(pairs), paths, out_rows


def _fresh_scheduler():
    """A scheduler built exactly the way the trainer builds it, on a throwaway optimizer.

    The lr ladder in drive40k and this builder are the only definition of "the frozen 40k
    curve"; the run itself uses Wan22Trainer._build_scheduler, so a resume is on the curve
    when the state it produces is indistinguishable from this one at the same step.
    """
    from fasterwam.trainer import Wan22Trainer

    class Shim:
        def __init__(self, optimizer):
            self.optimizer = optimizer
            self.learning_rate = plan.LR

    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=plan.LR,
                            betas=(0.9, 0.95), weight_decay=1e-2)
    return opt, Wan22Trainer._build_scheduler(Shim(opt), "cosine", plan.TOTAL,
                                              int(plan.TOTAL * 0.05))


SCHED_FIELDS = ("last_epoch", "base_lrs", "_milestones", "_step_count", "_gammas", "_last_lr")
INNER_FIELDS = ("last_epoch", "base_lrs", "start_factor", "end_factor", "total_iters",
                "T_max", "eta_min", "_step_count")


def _sched_fields(sd):
    """Scalar fields of a SequentialLR state_dict, inner schedulers flattened in."""
    out = {k: sd[k] for k in SCHED_FIELDS if k in sd}
    for i, sub in enumerate(sd.get("_schedulers") or []):
        if isinstance(sub, dict):
            for k in INNER_FIELDS:
                if k in sub:
                    out["sched%d.%s" % (i, k)] = sub[k]
    return out


def resume_test(paths, step, out):
    """Load the re-downloaded scheduler into the trainer's own builder.

    This is the resume test that costs 200 MB instead of 11 GB: it needs no base weights,
    and it is exactly the check that a segment restart lands on the same curve.

    It follows the trainer's order, which is the only order that means anything here:
    `load_light_state` restores `optimizer.pt` *first*, and `Optimizer.load_state_dict`
    copies every non-`params` key of the saved param groups -- `lr` included -- onto the
    live ones, then hands `scheduler.pt` to the scheduler. So the optimizer arrives
    already carrying the curve's value at the node's step and the scheduler only has to
    keep the continuation going. Replaying `step()` on a loaded scheduler -- what this
    function used to do -- measures neither: with no optimizer lr behind it, torch 2.9
    recomputed 5.38e-08 at step 13 where the curve says 6.997e-07, and that artifact of
    the harness was briefly misread as a blocker on the run.
    """
    files = {}
    for rel, p in paths.items():
        if _is_node_payload(rel):
            files[rel.rsplit("/", 1)[-1]] = p
    missing = [name for name in STATE_FILES if name not in files]
    if missing:
        raise AssertionError("re-download produced no %s -- nothing was resumed" % missing)
    state = json.loads(files["trainer_state.json"].read_text())

    step = int(step)
    adapter = torch.load(files["adapter.pt"], map_location="cpu", weights_only=False)
    lora, extra = adapter.get("lora", {}), adapter.get("extra_trainable", {})
    union = dict(lora)
    union.update(extra)
    force_keys = sorted(k for k in union if "force_conditioner" in k)
    opt_state = torch.load(files["optimizer.pt"], map_location="cpu", weights_only=False)
    sched_state = torch.load(files["scheduler.pt"], map_location="cpu", weights_only=False)
    members = sum(len(g["params"]) for g in opt_state["param_groups"])
    with_moments = sum(1 for s in opt_state["state"].values()
                       if s.get("exp_avg") is not None and s.get("exp_avg_sq") is not None)
    order = state["optimizer_member_order"]
    force_with_grad_signal = 0
    for idx, s in opt_state["state"].items():
        if s.get("exp_avg") is None or idx >= len(order):
            continue
        if "force_conditioner" in order[idx] and float(s["exp_avg"].detach().abs().sum()) > 0.0:
            force_with_grad_signal += 1
    # R2, in the trainer's own order. First: is the saved scheduler state the state an
    # untouched run would have at this step? Identical state means identical continuation,
    # which is a stronger claim than any lr number computed from a formula.
    ref_opt, ref = _fresh_scheduler()
    for _ in range(step):
        ref.step()
    saved_fields = _sched_fields(sched_state)
    ref_fields = _sched_fields(ref.state_dict())
    state_diff = {k: [saved_fields.get(k, "<absent>"), ref_fields.get(k, "<absent>")]
                  for k in sorted(set(saved_fields) | set(ref_fields))
                  if saved_fields.get(k, "<absent>") != ref_fields.get(k, "<absent>")}

    # Then: resume exactly as load_light_state does and watch the next few steps.
    cont_opt, cont = _fresh_scheduler()
    for live, saved_group in zip(cont_opt.param_groups, opt_state["param_groups"]):
        for key, val in saved_group.items():
            if key != "params":
                live[key] = val
    cont.load_state_dict(sched_state)
    lr_on_load = float(cont_opt.param_groups[0]["lr"])
    rows = []
    for k in range(1, 4):
        # The loop is optimizer.step(); scheduler.step(); global_step += 1, so step n runs
        # with the lr the curve left in the optimizer after step n-1.
        want = lr_at_local(step + k - 1)
        got = float(cont_opt.param_groups[0]["lr"])
        rows.append({"global_step": step + k, "lr_used": got, "want": want,
                     "ok": abs(got - want) <= 1e-12 * max(want, 1e-12)})
        cont_opt.step()
        cont.step()
    lr_in_optimizer_pt = float(opt_state["param_groups"][0].get("lr", -1.0))
    curve_here = lr_at_local(step)
    on_curve = (state_diff == {}
                and abs(lr_in_optimizer_pt - curve_here) <= 1e-12 * max(curve_here, 1e-12)
                and abs(lr_on_load - curve_here) <= 1e-12 * max(curve_here, 1e-12)
                and all(r["ok"] for r in rows))
    out["resume_rows"] = rows

    checks = [
        ("R1 node is a v3 light state for this step",
         state.get("format") == "fasterwam-lora-light-state-v3"
         and int(state.get("global_step", -1)) == int(step),
         {"format": state.get("format"), "global_step": state.get("global_step")}),
        ("R2 scheduler restores onto the frozen 40k curve", on_curve,
         {"scheduler_state_field_diff_vs_untouched_run_at_this_step": state_diff,
          "lr_saved_in_optimizer_pt": lr_in_optimizer_pt,
          "lr_in_optimizer_after_resume": lr_on_load,
          "curve_at_step": curve_here, "continued_steps": rows}),
        ("R3 adapter carries every trainable tensor incl. the force modules",
         adapter.get("format") == "fasterwam-lora-adapter-v3" and len(union) == members
         and len(force_keys) == FORCE_TENSORS,
         {"lora": len(lora), "extra_trainable": len(extra), "union": len(union),
          "optimizer_members": members, "force_tensors": len(force_keys)}),
        ("R4 optimizer moments came back for all members", with_moments == members,
         {"members": members, "with_moments": with_moments,
          "force_members_with_nonzero_exp_avg": force_with_grad_signal}),
        ("R5 counters and sampler offset came back",
         all(k in state for k in ("epoch", "batch_in_epoch", "sampler"))
         and isinstance(state.get("sampler"), dict) and bool(state["sampler"]),
         {"epoch": state.get("epoch"), "batch_in_epoch": state.get("batch_in_epoch"),
          "sampler_keys": sorted(state.get("sampler", {}))}),
        ("R6 force calibration fingerprint survived the round trip",
         state.get("force_calibration") == frozen_contract().get("force_calibration"),
         state.get("force_calibration")),
    ]
    out["resume_checks"] = [{"name": n, "ok": bool(ok), "evidence": e} for n, ok, e in checks]
    return all(bool(ok) for _, ok, _ in checks)


def scan_for_secret(pairs, token):
    """Text files only: a .pt legitimately contains weight bytes that could match nothing."""
    hits = []
    for local, rel in pairs:
        if local.suffix == ".pt":
            continue
        try:
            if token in local.read_text(errors="ignore"):
                hits.append(rel)
        except Exception as exc:  # noqa: BLE001
            hits.append("%s (%s)" % (rel, type(exc).__name__))
    return hits


def search_base(HfApi, token, endpoint, sha, want_bytes=0):
    """Can the base checkpoint be fetched by someone who does not have this instance?

    A match must agree on sha256 *and* byte length. The Hub's tree projection is not
    consistent about hashes across storage backends: an LFS file reports `lfs.sha256`,
    while the same field came back `null` for a file that went up through Xet, so a
    hash-only comparison quietly misses copies that exist, and a size-only comparison
    accepts a coincidentally-similar file (`step_002000_merged.pt` in our own archive is
    within 300 KB of the base and is a completely different model).
    """
    api = HfApi(endpoint=endpoint, token=token)
    scanned, errors = [], []
    for repo in BASE_SOURCES:
        try:
            rows = [_entry_info(e) for e in api.list_repo_tree(repo_id=repo, repo_type="model",
                                                               recursive=True, expand=True)]
        except Exception as exc:  # noqa: BLE001
            errors.append({"repo": repo, "error": type(exc).__name__,
                           "msg": mask(str(exc))[:160]})
            continue
        hits = [r for r in rows if r["lfs_oid"] and str(r["lfs_oid"]).lower() == sha.lower()
                and (not want_bytes or (r["lfs_size"] or r["size"]) == want_bytes)]
        scanned.append({"repo": repo, "entries": len(rows), "hit": bool(hits)})
        if hits:
            top = hits[0]
            return {"found": True, "repo": repo, "at": top["path"], "sha256": sha,
                    "bytes": top["lfs_size"] or top["size"],
                    "hf_download": "huggingface-cli download %s %s" % (repo, top["path"]),
                    "sources_scanned": scanned, "errors": errors}
    return {"found": False, "sha256": sha, "sources_scanned": scanned, "errors": errors}


def write_report(report_path, out, token):
    text = json.dumps(out, indent=2, sort_keys=True, default=str)
    if token and token in text:
        raise SystemExit("refusing to write a report that contains the credential")
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(text)
    line = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "step": out["step"],
            "repo": out.get("repo"), "deep": out.get("deep"),
            "verify_only": bool(out.get("upload_skipped")),
            "failed": [c["name"] for c in out["checks"] if not c["ok"]]}
    if token and token in json.dumps(line):
        raise SystemExit("refusing to ledger a line that contains the credential")
    with open(BACKUP / "ledger.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, sort_keys=True) + "\n")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--node-dir", default="")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    ap.add_argument("--token-file", default=os.environ.get("HF_TOKEN_FILE", TOKEN_FILE_DEFAULT))
    ap.add_argument("--deep", action="store_true", help="re-download and run the resume test")
    ap.add_argument("--verify-only", action="store_true",
                    help="do not upload; prove the hashes and the resume of what is already on the Hub")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--check-base", action="store_true")
    ap.add_argument("--report", default="")
    a = ap.parse_args()

    node_dir = Path(a.node_dir or plan.light_dir(a.step))
    if not node_dir.is_dir():
        raise SystemExit("node dir %s does not exist" % node_dir)
    BACKUP.mkdir(parents=True, exist_ok=True)
    report_path = Path(a.report or (BACKUP / ("report_step%06d_%s.json"
                                             % (a.step, "deep" if a.deep else "shallow"))))
    token = load_token(a.token_file)
    global SECRET
    SECRET = token
    out = {"step": a.step, "node_dir": str(node_dir), "repo": a.repo, "prefix": a.prefix,
           "endpoint": a.endpoint, "deep": bool(a.deep), "checks": [],
           "token_source": {"path": str(Path(a.token_file).resolve()), "length": len(token),
                            "mode": oct(Path(a.token_file).stat().st_mode)}}

    def check(name, ok, evidence):
        out["checks"].append({"name": name, "ok": bool(ok), "evidence": evidence})
        print(("PASS " if ok else "FAIL ") + name + " :: "
              + json.dumps(evidence, sort_keys=True, default=str)[:500], flush=True)

    work, pairs, manifest = build_upload_tree(node_dir, a.step,
                                              BACKUP / "stage" / ("step_%06d" % a.step),
                                              a.endpoint, a.repo)
    state = manifest["trainer_state"]
    node_bytes = sum(f["bytes"] for f in manifest["files"])
    check("U1 node complete and hashed", len(manifest["files"]) >= len(STATE_FILES),
          {"state_files": len([f for f in manifest["files"] if f["path"] in STATE_FILES]),
           "node_bytes": node_bytes, "adapter_bytes": state.get("adapter_bytes")})
    check("U2 node agrees with the frozen contract",
          all(manifest["contract_equals_node"].values()), manifest["contract_equals_node"])
    check("U3 recorded LR equals the trainer's own 40k curve",
          abs(float(state["extra"].get("lr", -1)) - lr_at_local(a.step))
          <= 1e-12 * lr_at_local(a.step)
          and boundary_cross_check(a.step).get("equal", True),
          {"recorded": state["extra"].get("lr"), "expected": lr_at_local(a.step),
           "boundary": boundary_cross_check(a.step)})
    check("U4 base weights pinned", bool(manifest["base_weights"]["sha256"]),
          {k: manifest["base_weights"][k] for k in ("file", "bytes", "sha256")})
    calib_ok, calib_ev = calibration_closure(calibration_source(), state.get("force_calibration"))
    check("U7 calibration file, fingerprint and dtype rounding close on each other",
          calib_ok, calib_ev)
    check("U5 upload tree staged", len(pairs) >= len(STATE_FILES) + 5,
          {"files": len(pairs), "bytes": sum(os.path.getsize(p) for p, _ in pairs),
           "paths": [rel for _, rel in pairs][:4]})

    secrets = scan_for_secret(pairs, token)
    check("U9 no credential in any staged file", secrets == [], secrets)
    if secrets:
        write_report(report_path, out, token)
        return 1

    if a.no_upload:
        write_report(report_path, out, token)
        failed = [c["name"] for c in out["checks"] if not c["ok"]]
        print(json.dumps({"report": str(report_path), "uploaded": False,
                          "failed": failed}, indent=2), flush=True)
        return 0 if not failed else 1

    HfApi, hf_hub_download = _hub(a.endpoint)
    api = HfApi(endpoint=a.endpoint, token=token)
    try:
        api.create_repo(a.repo, repo_type="model", private=True, exist_ok=True)
        info = api.repo_info(a.repo, repo_type="model")
        check("U6 repo exists and is private", bool(info),
              {"id": getattr(info, "id", None), "private": getattr(info, "private", None)})
    except Exception as exc:  # noqa: BLE001
        check("U6 repo exists and is private", False,
              {"error": type(exc).__name__, "msg": mask(str(exc))[:300]})
        write_report(report_path, out, token)
        return 1

    t0 = time.time()
    up_err = None
    if a.verify_only:
        out["upload_skipped"] = True
    else:
        try:
            api.upload_folder(folder_path=str(work), repo_id=a.repo, repo_type="model",
                              path_in_repo=a.prefix.strip("/"), revision="main",
                              commit_message="node step_%06d (%s)" % (a.step, "deep" if a.deep else "shallow"))
        except Exception as exc:  # noqa: BLE001
            up_err = {"error": type(exc).__name__, "msg": mask(str(exc))[:300]}
    out["upload_seconds"] = round(time.time() - t0, 1)
    check("V1 upload_folder accepted", up_err is None,
          up_err or {"seconds": out["upload_seconds"], "skipped": bool(a.verify_only),
                     "bytes_per_second": int(sum(os.path.getsize(p) for p, _ in pairs)
                                             / max(out["upload_seconds"], 1))})

    entries = remote_tree(api, a.repo, a.prefix)
    check("V2 remote tree matches local hashes", verify_remote(entries, pairs, manifest, out),
          {"staged": len(pairs), "verified": len(out.get("files_verified", [])),
           "mismatched": out.get("files_mismatched")})

    if a.deep:
        ok_dl, paths, dl_rows = download_and_compare(hf_hub_download, a.repo, a.prefix, pairs,
                                                     token, a.step)
        out["download"] = dl_rows
        check("W1 re-downloaded every file byte-for-byte", ok_dl,
              {"files": dl_rows["files"], "mismatches": dl_rows["mismatches"]})
        try:
            check("R7 remote node resumes on the frozen curve",
                  resume_test(paths, a.step, out), out.get("resume_checks"))
        except Exception as exc:  # noqa: BLE001
            check("R7 remote node resumes on the frozen curve", False,
                  {"error": type(exc).__name__, "msg": mask(str(exc))[:400]})
        shutil.rmtree(VERIFY_ROOT / a.prefix.strip("/") / ("step_%06d" % a.step),
                      ignore_errors=True)

    if a.check_base:
        out["base_search"] = search_base(HfApi, token, a.endpoint,
                                         manifest["base_weights"]["sha256"],
                                         manifest["base_weights"].get("bytes"))
        check("W2 base checkpoint independently restorable", out["base_search"]["found"],
              out["base_search"])

    write_report(report_path, out, token)
    failed = [c["name"] for c in out["checks"] if not c["ok"]]
    print(json.dumps({"report": str(report_path), "passed": len(out["checks"]) - len(failed),
                      "failed": failed}, indent=2), flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
