"""merge -> evaluate -> delete the merged file, one node at a time, for all eight 5k checkpoints.

`eval40k.py` resolves an artifact and scores it, and it deliberately never deletes anything: a merged
checkpoint is ~11 GiB, the volume held 44 GiB free during training, and a hand-run loop at the end of
a 40,000-step run is exactly where an 11 GiB mistake gets made. So the loop that has to delete is
written down here, with the rules stated as code:

* **one merged file on disk at a time.** The pre-flight refuses if more than one is already in the
  artifact directory, and the node loop stops rather than merging on top of a leftover from an
  earlier node;
* **an artifact that is already there is reused, not re-merged**, when its publish record is still
  on disk next to it -- a failed eval should cost the eval again, not another 11 GiB merge;
* **a merged file is deleted only after every split it was meant to score has a complete
  `summary.json`** -- episodes equal to the seed list, a success rate, a force block -- and the freed
  bytes go into the ledger;
* **nothing starts while the trainer is still running.** An 11 GiB load on the same card slows the
  40k run and changes no result; the interlock reads the segment log's mtime and the process table;
* **calibration is checked before the merge is trusted**: the publish record's
  `published_force_calibration` must equal the fingerprint the node's own `trainer_state.json` says
  it trained under, and `published_step` must be the step being merged.

Two modes make the flow testable while the GPU belongs to training:

    PYTHONDONTWRITEBYTECODE=1 nice -n 19 python batch_eval.py --dry-run
    PYTHONDONTWRITEBYTECODE=1 nice -n 19 python batch_eval.py --fake-merged --steps 5000 10000

`--fake-merged` writes a 2 MiB placeholder into the artifact slot and forces the fixture backend with
the stub policy, so no weight is loaded and no GPU is touched; what it exercises is the part that
matters here -- slot, order, completeness check, delete, ledger, report. Those ledger lines carry
`"flow_test": true` and `report40k.py` already labels the runs `(test run)`, so they can never be read
as model numbers. The real pass is the same command without the flag, after step 40,000.

`--diversity` adds a second split per node at `--batch-offset 0.008`, the one setting measured in
`knob_probe.py` that is both legal and a different layout; it writes to its own output directory so
the fixed-layout and the out-of-distribution numbers are never averaged into one table.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval40k as E                                     # noqa: E402

STAGE = E.STAGE
FW = E.FW
RUN = E.RUN
MERGE_SCRIPT = FW / "scripts" / "export_merged_v3.py"
BASE_CKPT = ("/workspace/FasterWAM/checkpoints/fasterwam_release/libero/step_021700.pt")
MERGED_BYTES = 11_117_679_929                            # measured, task #64
DEFAULT_OVERRIDES = ["data=a3_2cam_force", "task=a3_force_40k", "batch_size=6",
                     "max_steps=40000", "learning_rate=1e-4"]
DIVERSITY_BATCH_OFFSET_M = 0.008                          # measured in knob_probe.py


def sha256_file(path, chunk=1 << 23):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def free_gib(path):
    p = Path(path)
    while not p.exists() and p != p.parent:      # the artifact dir is usually not created yet
        p = p.parent
    return shutil.disk_usage(str(p)).free / 2**30


def disk_blocker(free_gib_value, merged_bytes, margin):
    """The disk gate in one unit, both sides GiB.

    This call site used to compare `free_gib()`'s GiB reading against a byte threshold, which is true
    for every volume that can hold the run, and it sat inside `if not args.dry_run`, so the step-6
    dry-run interlock reported `blockers: []` and the real launch refused anyway.
    """
    need_gib = merged_bytes * margin / 2**30
    if free_gib_value < need_gib:
        return ("only %.2f GiB free, merging needs %.2f GiB at margin %.1fx"
                % (free_gib_value, need_gib, margin))
    return None


def merged_slot(merged_dir, step):
    return Path(merged_dir) / ("step_%06d_merged.pt" % step)


def stray_merged(merged_dir):
    return sorted(glob.glob(str(Path(merged_dir) / "step_*_merged.pt*")))


def training_active(stale_s):
    """True if the 40k trainer looks alive: a fresh segment log or a live train_seg process."""
    logs = sorted(glob.glob(str(RUN / "logs" / "*.log")), key=os.path.getmtime)
    fresh = bool(logs) and (time.time() - os.path.getmtime(logs[-1])) < stale_s
    try:
        proc = subprocess.run(["pgrep", "-f", "scripts/train_seg.py"],
                              capture_output=True, text=True)
        live = proc.returncode == 0 and proc.stdout.strip() != ""
    except OSError:
        live = False
    return {"active": bool(fresh or live), "log": logs[-1] if logs else None,
            "log_age_s": round(time.time() - os.path.getmtime(logs[-1]), 1) if logs else None,
            "train_seg_pids": (subprocess.run(["pgrep", "-f", "scripts/train_seg.py"],
                                              capture_output=True, text=True).stdout
                               .split() if live else [])}


def run_cmd(cmd, cwd, log_path, env=None, nice=True):
    full = (["nice", "-n", "19"] + cmd) if nice else cmd
    started = time.time()
    with open(log_path, "wb") as out:
        rc = subprocess.call(full, cwd=str(cwd), stdout=out, stderr=subprocess.STDOUT,
                             env=env or os.environ)
    return {"cmd": full, "cwd": str(cwd), "rc": int(rc), "secs": round(time.time() - started, 1),
            "log": str(log_path), "log_tail": Path(log_path).read_text(errors="replace")[-1500:]}


def merge_node(step, args, work):
    """Produce the deployment artifact for one node, or a labelled placeholder in fake mode."""
    merged = merged_slot(args.merged_dir, step)
    record_path = work / ("publish_record_step%06d.json" % step)
    if merged.is_file() and record_path.is_file() and not args.no_reuse:
        record = json.loads(record_path.read_text())
        record["reused_existing"] = True
        return record, merged, None
    if args.fake_merged:
        merged.parent.mkdir(parents=True, exist_ok=True)
        with open(merged, "wb") as fh:
            fh.write(b"\x00" * (2 << 20))
        # the node's own fingerprint, not a version string: `check_merge` compares shapes the real
        # publish record produces, and a fake record that always agrees would test nothing
        state = E.node_state(step) or {}
        record = {"event": "publish_merged", "flow_test": True, "output": str(merged),
                  "published_step": step,
                  "published_force_calibration": state.get("force_calibration"),
                  "output_bytes": merged.stat().st_size}
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True))
        return record, merged, None

    st = E.node_state(step)
    light_dir = Path(st["dir"])
    cmd = [args.py, "-u", str(MERGE_SCRIPT)] + list(args.overrides) + [
        "output_dir=%s" % work,
        "+base_ckpt=%s" % args.base_ckpt, "+light_dir=%s" % light_dir,
        "+output=%s" % merged, "+record=%s" % record_path]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(FW / "src")
    env["FW_BASE_CKPT"] = args.base_ckpt
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    res = run_cmd(cmd, FW, work / ("merge_step%06d.log" % step), env=env, nice=not args.no_nice)
    record = json.loads(record_path.read_text()) if record_path.is_file() else None
    if res["rc"] != 0 or record is None:
        raise SystemExit("merge failed for step %d: rc=%s record=%s\n%s"
                         % (step, res["rc"], record, res["log_tail"]))
    return record, merged, res


def check_merge(record, merged, step, state, fake=False):
    """The merge is only usable if it is the step it claims, byte-exact, and the units it claims."""
    problems = []
    if int(record.get("published_step", -1)) != step:
        problems.append("published_step=%s while merging %d" % (record.get("published_step"), step))
    if record.get("published_force_calibration") != state.get("force_calibration"):
        problems.append("calibration %s does not match the node's trained units %s"
                        % (record.get("published_force_calibration"),
                           state.get("force_calibration")))
    actual = merged.stat().st_size if merged.is_file() else None
    if actual != record.get("output_bytes"):
        problems.append("merged file is %s B, the publish record says %s B"
                        % (actual, record.get("output_bytes")))
    if not fake and actual is not None and actual < MERGED_BYTES * 0.9:
        problems.append("merged file is %.2f GiB, expected ~%.2f GiB"
                        % (actual / 2**30, MERGED_BYTES / 2**30))
    return problems


def eval_node(step, args, out_dir, seeds, batch_offset):
    Path(out_dir).mkdir(parents=True, exist_ok=True)     # the child writes its log before it can
    cmd = [args.py, str(HERE / "eval40k.py"), "--steps", str(step),
           "--seeds"] + [str(s) for s in seeds] + [
           "--backend", args.backend, "--policy", args.policy,
           "--out", str(out_dir), "--merged-dir", str(args.merged_dir),
           "--batch-offset", str(batch_offset), "--calib", args.calib]
    if args.fake_merged:
        cmd += ["--fixture-force-mode", "const"]
    if args.shadow:
        cmd.append("--shadow")
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    res = run_cmd(cmd, HERE, out_dir / ("eval_step%06d.log" % step), env=env,
                  nice=not args.no_nice)
    summary_path = out_dir / ("step_%06d" % step) / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
    return res, summary


def summary_complete(summary, seeds):
    if summary is None:
        return ["no summary.json written"]
    problems = []
    if summary.get("status") not in (None, "ran"):
        problems.append("summary status=%s" % summary.get("status"))
    if summary.get("episodes") != len(seeds):
        problems.append("summary has %s episodes for %d seeds"
                        % (summary.get("episodes"), len(seeds)))
    if summary.get("success_rate") is None:
        problems.append("summary has no success_rate")
    if not summary.get("force"):
        problems.append("summary has no force block")
    return problems


def ledger_append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, indent=None, sort_keys=True, default=str) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=E.node_steps())
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="one episode per seed; the default is the 20 screened seeds")
    ap.add_argument("--out", default=str(STAGE / "dev" / "out40k" / "batch_eval"))
    ap.add_argument("--merged-dir", default=str(HERE / "merged"))
    ap.add_argument("--backend", choices=["env", "fixture"], default="env")
    ap.add_argument("--policy", choices=["merged", "stub"], default="merged")
    ap.add_argument("--py", default=sys.executable)
    ap.add_argument("--base-ckpt", default=BASE_CKPT)
    ap.add_argument("--calib", default=str(E.CALIB_PATH))
    ap.add_argument("--overrides", nargs="*", default=DEFAULT_OVERRIDES,
                    help="hydra group overrides, which must be the set the training ran with")
    ap.add_argument("--diversity", action="store_true",
                    help="also run each node at --batch-offset %.3f in its own output tree"
                         % DIVERSITY_BATCH_OFFSET_M)
    ap.add_argument("--shadow", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fake-merged", action="store_true",
                    help="flow test: a placeholder artifact, fixture backend, stub policy")
    ap.add_argument("--allow-during-training", action="store_true")
    ap.add_argument("--stale-log-seconds", type=int, default=300)
    ap.add_argument("--disk-margin", type=float, default=1.5,
                    help="required multiple of one merged artifact in free bytes before merging")
    ap.add_argument("--keep-merged", action="store_true")
    ap.add_argument("--no-reuse", action="store_true",
                    help="merge again even when a publish record for that node is already on disk")
    ap.add_argument("--no-nice", action="store_true")
    ap.add_argument("--no-report", action="store_true")
    args = ap.parse_args()
    # the protocol is the 20 screened seeds, one episode each, identical list for all eight nodes
    args.seeds = E.check_one_episode_per_seed(E.screened_seeds() if args.seeds is None
                                              else args.seeds)

    if args.fake_merged:
        args.backend, args.policy = "fixture", "stub"
    if args.backend == "env" and args.policy != "merged":
        raise SystemExit("--backend env needs --policy merged (the same refusal eval40k.py makes)")

    out = Path(args.out)
    work = out / "_work"
    merged_dir = Path(args.merged_dir)
    led = out / "batch_eval_ledger.jsonl"
    train = training_active(args.stale_log_seconds)
    strays = stray_merged(merged_dir)
    preflight = {
        "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": args.steps, "seeds": len(args.seeds), "backend": args.backend,
        "policy": args.policy, "fake_merged": args.fake_merged, "overrides": args.overrides,
        "training": train, "stray_merged_files": strays,
        "merged_dir": str(merged_dir), "free_gib_before": round(free_gib(merged_dir), 2),
        "merge_needs_gib": round(MERGED_BYTES * args.disk_margin / 2**30, 2),
        "diversity": bool(args.diversity),
    }
    if not args.dry_run:
        print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)

    blockers = []
    if train["active"] and not (args.allow_during_training or args.fake_merged):        blockers.append("trainer looks alive (%s last written %.0f s ago, pids %s); the batch eval "
                        "is scheduled after step 40,000 -- re-run with --allow-during-training only "
                        "if the 40k run is finished"
                        % (train["log"], train["log_age_s"] or -1, train["train_seg_pids"]))
    if strays and not args.keep_merged and len(strays) > 1:
        blockers.append("more than one merged artifact on disk in %s: %s -- one at a time is the "
                        "rule; clear them or pass --keep-merged only if you know why they are there"
                        % (merged_dir, strays))
    if not args.fake_merged:
        # arithmetic on the volume, no side effects, so the dry-run answers it too: that is the only
        # way an interlock that reads `blockers` from --dry-run can refuse a run the real launch would
        # refuse. Before this the disk gate lived behind `not args.dry_run`, which is how a gate that
        # fired on every volume reached the live run untested.
        gate = disk_blocker(free_gib(merged_dir), MERGED_BYTES, args.disk_margin)
        if gate:
            blockers.append(gate)
    if not args.fake_merged and not args.dry_run:
        if not Path(args.base_ckpt).is_file():
            blockers.append("base checkpoint %s is missing; the merge needs it" % args.base_ckpt)

    if args.dry_run:
        rows = []
        for step in args.steps:
            st = E.node_state(step)
            rows.append({"step": step,
                         "light_node": st["dir"] if st else None,
                         "trainer_state_global_step": st and st["global_step"],
                         "node_calib": st and st["force_calibration"],
                         "merged_would_be": str(merged_slot(merged_dir, step)),
                         "merge_cmd": [args.py, "-u", str(MERGE_SCRIPT)] + list(args.overrides) + [
                             "output_dir=%s" % work, "+base_ckpt=%s" % args.base_ckpt,
                             "+light_dir=%s" % (st["dir"] if st else "<no light node>"),
                             "+output=%s" % merged_slot(merged_dir, step),
                             "+record=%s" % (work / ("publish_record_step%06d.json" % step))],
                         "eval_cmd": [args.py, "eval40k.py", "--steps", str(step), "--backend",
                                      args.backend, "--seeds"] + [str(s) for s in args.seeds]})
        print(json.dumps({"dry_run": True, "preflight": preflight, "blockers": blockers,
                          "nodes": rows}, indent=2, sort_keys=True))
        return 1 if blockers else 0
    if blockers:
        raise SystemExit("refusing to start:\n  - " + "\n  - ".join(blockers))

    work.mkdir(parents=True, exist_ok=True)
    splits = {"fixed": 0.0}
    if args.diversity:
        splits["diversity"] = DIVERSITY_BATCH_OFFSET_M
    failures = []
    for step in args.steps:
        st = E.node_state(step)
        if st is None:
            print("step %6d: no light node on disk, skipped" % step, flush=True)
            failures.append({"step": step, "why": "no_light_node"})
            continue
        merged = merged_slot(merged_dir, step)
        others = [p for p in stray_merged(merged_dir)
                  if os.path.realpath(p) != os.path.realpath(str(merged))]
        if others:
            # a leftover from an earlier node means the volume is already 11 GiB lighter than the
            # plan assumed; merging a second node on top of it is how a full disk gets discovered
            # late, so the loop stops rather than adding to the pile
            why = ("another node's merged artifact is still on disk: %s -- either a retry "
                   "candidate or residue from a failed node, both of which need a look first"
                   % others)
            failures.append({"step": step, "why": why})
            print("step %6d ABORTED THE LOOP: %s" % (step, why), file=sys.stderr, flush=True)
            break
        record = None
        row = {"t": preflight["t"], "step": step, "flow_test": bool(args.fake_merged),
               "node_dir": st["dir"], "seeds": list(args.seeds), "backend": args.backend}
        try:
            free_at_merge = round(free_gib(merged_dir), 2)
            record, merged, merge_res = merge_node(step, args, work)
            problems = check_merge(record, merged, step, st, fake=args.fake_merged)
            if problems:
                raise SystemExit("merged artifact rejected for step %d: %s" % (step, problems))
            row.update({"merged_bytes": merged.stat().st_size,
                        "merged_sha256": sha256_file(merged),
                        "merge_secs": merge_res and merge_res["secs"],
                        "free_gib_at_merge": free_at_merge,
                        "publish_record": str(work / ("publish_record_step%06d.json" % step))})
            for name, offset in splits.items():
                split_out = out / name
                res, summary = eval_node(step, args, split_out, args.seeds, offset)
                problems = summary_complete(summary, args.seeds)
                if res["rc"] != 0:
                    problems.insert(0, "eval40k rc=%d, tail: %s" % (res["rc"], res["log_tail"][-400:]))
                if problems:
                    raise SystemExit("step %d %s split incomplete: %s" % (step, name, problems))
                sc = summary.get("scene") or {}
                mae = (summary["force"].get("overall") or {}).get("mae")
                row["split_%s" % name] = {
                    "out": str(Path(split_out) / ("step_%06d" % step)),
                    "success_rate": summary["success_rate"],
                    "successes": summary["successes"], "episodes": summary["episodes"],
                    "force_mae_per_channel_n": mae,
                    "force_mae_max_n": round(max(mae), 4) if mae else None,
                    "force_coverage": summary["force"].get("coverage"),
                    "mae_vs_zero": summary["force"].get("mae_vs_zero"),
                    "distinct_scenes": sc.get("distinct_scene_keys"),
                    "batch_offset_m": offset, "nan_aborts": summary["nan_aborts"]}
                print("step %6d [%s] success %s/%s MAEmax %s distinct_scenes %s"
                      % (step, name, summary["successes"], summary["episodes"],
                         row["split_%s" % name]["force_mae_max_n"],
                         row["split_%s" % name]["distinct_scenes"]), flush=True)
        except SystemExit as exc:
            failures.append({"step": step, "why": str(exc)})
            print("step %6d FAILED: %s" % (step, exc), file=sys.stderr, flush=True)
        finally:
            done_splits = len([k for k in row if k.startswith("split_")])
            if merged.is_file() and not args.keep_merged and record is not None:
                if done_splits == len(splits):
                    before = free_gib(merged_dir)
                    size = merged.stat().st_size
                    merged.unlink()
                    row["deleted_merged"] = {"bytes": size,
                                             "free_gib_after": round(before + size / 2**30, 2)}
                else:
                    row["merged_kept_after_failure"] = str(merged)
            row["free_gib_end"] = round(free_gib(merged_dir), 2)
            ledger_append(led, row)

    if not args.no_report:
        for name in splits:
            d = out / name
            if d.is_dir():
                res = run_cmd([args.py, str(HERE / "report40k.py"), "--out", str(d)],
                              HERE, out / ("report_%s.log" % name),
                              nice=not args.no_nice)
                print("--- report %s (rc=%s) ---" % (name, res["rc"]))
                print(res["log_tail"])
    print(json.dumps({"ledger": str(led), "failures": failures,
                      "stray_merged_after": stray_merged(merged_dir)}, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
