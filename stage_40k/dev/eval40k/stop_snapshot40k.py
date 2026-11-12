"""Evidence snapshot taken before the 160-episode batch is stopped.

The run is interrupted on purpose, so the record of what it had already produced has to be written
down by something other than the run itself: `eval40k.py` only emits `summary.json` when a node
finishes all 20 seeds, so an interrupted node would otherwise leave its 11 completed episodes
described by nothing but a directory listing. Every file hash here is of the file as it sits on disk,
and nothing in this script writes to or deletes anything inside the results tree.
"""

import hashlib
import json
import os
import time
from pathlib import Path

BASE = Path("/workspace/_v1_stage/dev/out40k/batch_eval_20seed1x")
STEP_DIR = BASE / "fixed" / "step_040000"
DST = Path("/workspace/_v1_stage/dev/out40k/stop_20261112")
MERGED = Path("/workspace/_v1_stage/dev/eval40k/merged/step_040000_merged.pt")


def sha256_file(p):
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def utc(p):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p))


def main():
    DST.mkdir(parents=True, exist_ok=True)
    files, episodes = [], []
    for p in sorted(STEP_DIR.iterdir()):
        if not p.is_file():
            continue
        files.append({"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p),
                      "mtime_utc": utc(p.stat().st_mtime)})
        if p.name.startswith("episode_seed"):
            d = json.loads(p.read_text())
            episodes.append({"seed": d.get("seed"), "success": d.get("success"),
                             "cookies_in_target": d.get("cookies_in_target"),
                             "frames": d.get("frames"), "terminated": d.get("terminated"),
                             "truncated": d.get("truncated"),
                             "nan_abort_frame": d.get("nan_abort_frame"),
                             "wall_s": d.get("wall_s"), "sim_s_total": d.get("sim_s_total"),
                             "infer_s_total": d.get("infer_s_total"),
                             "action_absmax_max": d.get("action_absmax_max"),
                             "emergency_stop": d.get("emergency_stop"),
                             "decision_chunks": d.get("decision_chunks")})

    log = BASE.parent / "batch_eval_20seed1x.log"
    node_log = STEP_DIR.parent / "eval_step040000.log"
    head = MERGED.open("rb").read(1 << 20) if MERGED.is_file() else b""
    procs = [l.rstrip() for l in os.popen("ps -eo pid,ppid,etime,stat,args").readlines()
             if any(k in l for k in ("eval40k", "batch_eval", "RUN_BATCH", "train_seg"))]

    doc = {"snapshot_utc": utc(time.time()),
           "reason": ("stopped by decision of 2026-11-12: no per-dimension action or video evidence "
                      "was being kept, and the action overshoot has to be attributed before any "
                      "official 160-episode number is spent"),
           "node": "step_040000, seeds 11..30 (fixed scene, batch-offset 0.0)",
           "episodes_completed": len(episodes),
           "episodes_expected": 20,
           "node_summary_json_present": (STEP_DIR / "summary.json").is_file(),
           "episodes": episodes,
           "files": files,
           "merged_artifact": {"path": str(MERGED),
                              "bytes": MERGED.stat().st_size if MERGED.is_file() else None,
                              "head_sha256_1MiB": hashlib.sha256(head).hexdigest()},
           "logs": {"driver": {"path": str(log),
                               "bytes": log.stat().st_size if log.is_file() else None,
                               "sha256": sha256_file(log) if log.is_file() else None},
                    "node": {"path": str(node_log),
                             "bytes": node_log.stat().st_size if node_log.is_file() else None,
                             "sha256": sha256_file(node_log) if node_log.is_file() else None}},
           "procs_at_snapshot": procs,
           "vram_used_bytes": None,
           "successes_so_far": sum(1 for e in episodes if e["success"])}
    try:
        out = os.popen("rocm-smi --showmeminfo vram 2>/dev/null").read()
        doc["vram_raw"] = [l.strip() for l in out.splitlines() if "Used" in l]
    except Exception as exc:
        doc["vram_raw"] = ["unreadable: %s" % exc]

    p = DST / "STOP_SNAPSHOT_step040000.json"
    p.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str))
    print(json.dumps({"snapshot": str(p), "episodes_completed": doc["episodes_completed"],
                      "successes_so_far": doc["successes_so_far"],
                      "files_hashed": len(files),
                      "seeds": [e["seed"] for e in episodes],
                      "vram": doc["vram_raw"]}, indent=2))


if __name__ == "__main__":
    main()
