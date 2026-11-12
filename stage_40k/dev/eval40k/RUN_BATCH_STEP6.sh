#!/bin/bash
# Step 6 of RUN_ON_POD.sh: 8 nodes x 20 screened seeds = 160 episodes on the fixed scene.
#
# This exists as a script rather than a command line because the volume cannot hold the eight merged
# artifacts at once (27.44 GiB free, ~11.13 GiB each, and batch_eval.py demands 1.5x one artifact free
# before it merges), while the batch driver also refuses to start a node while *another* node's
# merged file is still sitting in merged/ -- by design, so a full disk is never discovered at 03:00.
# With only step_040000 merged today, the loop would abort on its very first iteration.
#
# ORDER. The 40k node goes first, because it is the only node whose merged artifact is already on
# disk: `--keep-merged` reuses it from its publish record in _work/, so that node costs no merge, no
# move and no delete risk, and the headline number is on disk ~5 h in rather than ~40 h in. It also
# means the one node the live smoke already scored is the one node scored before anything else is
# rearranged. The reuse only happens if the record sits in THIS run's _work/, so step 0b copies the
# record the merge step itself wrote for that artifact (byte-identical copy, original left in place)
# and refuses to start if the copy or the file it describes does not match -- because the silent
# alternative is re-merging over the bytes the smoke validated. After the node passes its own
# interlock, the artifact moves to merged_hold/ on the same filesystem (mv, instant, reversible),
# merged/ is empty, and the seven unmerged nodes run one at a time -- each merges, evaluates, and
# unlinks only the merged file its own merge step wrote. The 40k file then moves back and is proved to
# be the same bytes the smoke scored (first-MiB sha16 f9fe3c9407c89169, 11,126,493,579 B) before the
# report reads all eight summaries.
#
# The light checkpoints, adapters, training artifacts and Hugging Face backups are never touched by
# anything here; batch_eval.py only ever unlinks a path of the form step_NNNNNN_merged.pt that its
# own merge step wrote in this run.
#
# It is safe to stop and restart, with one caveat. Every node appends exactly one row to
# batch_eval_ledger.jsonl when it finishes or fails, and a node whose merged artifact and publish
# record are still on disk is REUSED rather than re-merged -- so an interrupted merge costs a merge
# again only if you pass --no-reuse. Nothing else is skipped: `batch_eval.py` re-runs the rollouts for
# whatever appears in --steps, so resume means listing the remaining nodes (and, if the 40k artifact
# is still in merged_hold/, re-running move-back before the last command). A node killed mid-episode
# is scored from the top, which is the honest cost of not inventing an episode-level checkpoint format
# the report would have to trust.
#
# COST, measured on the machine this runs on: the smoke put one 601-frame episode at 215.55 s wall
# (111.5 s simulator + 89.8 s inference, share 0.482) and another at 1676.89 s (1592.4 s simulator,
# one MuJoCo step alone at 10.774 s, share 0.05). The median-of-two projection for 160 episodes is
# 42.06 h; if one regime had dominated the other, the same arithmetic gives ~9.6 h or ~74.5 h. The
# run is simulator-bound and shares the box, so treat 42 h as a floor, not a ceiling.
#
# STOP CONDITIONS inherited from the directive, all of which the driver enforces per node and this
# script forwards: NaN abort, severe action anomaly, persistent OOM, model-load error, protocol
# mismatch. Read the ledger line before restarting anything: `merged_kept_after_failure` in
# batch_eval_ledger.jsonl means the driver stopped and left the evidence on disk on purpose.
set -e
set -o pipefail
cd /workspace/_v1_stage/dev/eval40k
export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1
PY=/opt/venv/bin/python
OUT=/workspace/_v1_stage/dev/out40k/batch_eval_20seed1x
HOLD=/workspace/_v1_stage/dev/eval40k/merged_hold
MERGED=/workspace/_v1_stage/dev/eval40k/merged
LOG=/workspace/_v1_stage/dev/out40k/batch_eval_20seed1x.log

echo "== 0a. the toolchain this run is scored with =="
# Refuse to start on a stale file rather than produce 160 episodes the report cannot grade. The
# prefixes are the hashes the round-10 sync read back: `batch_eval.py` and `flow_test.py` are the
# disk-gate fix (round 10, pre-images 3d6292ab / 215458cc), `report40k.py` is the round-9 one, and the
# rest are the round-8 toolchain the live smoke ran on. Every one of those pre-images is archived under
# _diffrecord/20261112_iface_fix/pre/ with its round's diff and post-manifest beside it.
$PY - <<'PYX'
import hashlib
import json
import sys
from pathlib import Path

want = {"eval40k.py": "fb723254", "policy40k.py": "f7d8ec3c", "smoke40k.py": "d69a038d",
        "flow_test.py": "ee45c272", "proprio14_test.py": "6b0fa29c", "batch_eval.py": "7dc2a0c5",
        "report40k.py": "b48b7a33"}
bad = []
for f, w in sorted(want.items()):
    b = Path(f).read_bytes()
    got = hashlib.sha256(b).hexdigest()
    print("  %s %s %d B" % (f, got[:8], len(b)))
    if not got.startswith(w):
        bad.append({"file": f, "expected8": w, "actual8": got[:8], "bytes": len(b)})
print(json.dumps({"toolchain_files_checked": len(want), "mismatches": bad}, indent=2))
sys.exit(1 if bad else 0)
PYX
echo "== 0b. the 40k node has to REUSE its artifact, not re-merge it =="
# batch_eval.py's reuse rule (batch_eval.py:120) needs both the merged file and
# $OUT/_work/publish_record_step040000.json, and $OUT is a fresh directory. The record for this exact
# artifact was written by the merge step itself into dev/out40k/batch_eval/_work/; it is copied, not
# moved, so the earlier rounds' evidence stays where their records point, and check_merge still runs
# on it (published_step, the node's trained force calibration, and the file's byte count).
# If the copy does not describe the file on disk, this script refuses to start. The alternative is
# spending the first hour of the run re-merging 11,126,493,579 B over the bytes the live smoke scored,
# and then citing the result as though it were scored on the artifact the smoke validated.
mkdir -p "$OUT/_work"
cp -n /workspace/_v1_stage/dev/out40k/batch_eval/_work/publish_record_step040000.json "$OUT/_work/"
$PY - <<'PYX'
import hashlib, json, sys
from pathlib import Path
SRC = Path("/workspace/_v1_stage/dev/out40k/batch_eval/_work/publish_record_step040000.json")
DST = Path("/workspace/_v1_stage/dev/out40k/batch_eval_20seed1x/_work/publish_record_step040000.json")
MERGED = Path("/workspace/_v1_stage/dev/eval40k/merged/step_040000_merged.pt")
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rec = json.loads(DST.read_text()) if DST.is_file() else {}
first = None
if MERGED.is_file():
    with MERGED.open("rb") as fh:
        first = hashlib.sha256(fh.read(1 << 20)).hexdigest()[:16]
out = {"source_record_sha256": (h(SRC) if SRC.is_file() else None),
       "copied_record_sha256": (h(DST) if DST.is_file() else None),
       "published_step": rec.get("published_step"), "record_output": rec.get("output"),
       "record_output_bytes": rec.get("output_bytes"),
       "record_calibration": str(rec.get("published_force_calibration"))[:80],
       "merged_exists": MERGED.is_file(),
       "merged_bytes": (MERGED.stat().st_size if MERGED.is_file() else None),
       "merged_first_mib_sha16": first}
out["copy_is_byte_identical"] = (out["source_record_sha256"] is not None
                                 and out["copied_record_sha256"] == out["source_record_sha256"])
out["record_describes_this_file"] = (out["published_step"] == 40000
                                     and out["merged_bytes"] is not None
                                     and out["merged_bytes"] == out["record_output_bytes"]
                                     and first == "f9fe3c9407c89169")
out["ok"] = bool(out["copy_is_byte_identical"] and out["record_describes_this_file"])
print(json.dumps(out, indent=2, sort_keys=True))
sys.exit(0 if out["ok"] else 1)
PYX
echo "  reuse gate passed: step 1 costs no merge"

echo "== 0c. the state this script is allowed to start from =="
df -h /workspace | tail -1
ls -l "$MERGED" "$HOLD" 2>/dev/null || true
$PY batch_eval.py --dry-run --out "$OUT" > "$OUT.dry_run_before_step6.json" 2>&1 || {
    echo "dry-run refused; read $OUT.dry_run_before_step6.json before going further"; exit 1; }
echo "  dry-run blockers: $($PY -c "import json;print(json.load(open('$OUT.dry_run_before_step6.json'))['blockers'])")"

echo "== 1. the 40k node first: the artifact is already merged, so nothing is built or moved =="
$PY -u batch_eval.py --out "$OUT" --steps 40000 --keep-merged --no-report 2>&1 | tee -a "$LOG"

echo "== 1b. interlock on the node that just ran =="
$PY - "$OUT" <<'PYX'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
bad = []
for p in sorted(out.glob("*/step_040000/summary.json")):
    s = json.loads(p.read_text())
    prop = s.get("proprio") or {}
    if prop.get("unexplained"):
        bad.append({"summary": str(p), "unexplained": prop["unexplained"],
                    "clamped_columns": prop.get("clamped_columns")})
print(json.dumps({"nodes_with_unexplained_clamps": bad}, indent=2))
sys.exit(1 if bad else 0)
PYX

echo "== 2. move the 40k artifact aside (same filesystem, nothing deleted) =="
mkdir -p "$HOLD"
if [ -f "$MERGED/step_040000_merged.pt" ]; then
    mv -n "$MERGED/step_040000_merged.pt" "$HOLD/step_040000_merged.pt"
fi
ls -l "$HOLD" "$MERGED"
echo "  merged/ must now be empty -- that is what unblocks the node loop"

echo "== 3. the seven unmerged nodes: merge -> evaluate -> delete the merged file, one at a time =="
$PY -u batch_eval.py --out "$OUT" --steps 5000 10000 15000 20000 25000 30000 35000 \
    --no-report 2>&1 | tee -a "$LOG"

echo "== 3b. interlock: no node may report a proprio clamp the harness cannot explain =="
# `unexplained` is the one number in a node's summary that would mean this toolchain, not the robot,
# produced the out-of-band value -- and it is the same number the live smoke had to zero before this
# run was allowed to start. If any node reports a non-zero count, stop here rather than spend the
# remaining hours scoring a harness that normalised the wrong columns.
$PY - "$OUT" <<'PYX'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
bad = []
for p in sorted(out.glob("*/step_*/summary.json")):
    s = json.loads(p.read_text())
    prop = s.get("proprio") or {}
    if prop.get("unexplained"):
        bad.append({"summary": str(p), "unexplained": prop["unexplained"],
                    "clamped_columns": prop.get("clamped_columns")})
print(json.dumps({"summaries_scanned": len(list(out.glob('*/step_*/summary.json'))),
                  "nodes_with_unexplained_clamps": bad}, indent=2))
sys.exit(1 if bad else 0)
PYX

echo "== 4. move the 40k artifact back and prove it is the file the smoke scored =="
mv -n "$HOLD/step_040000_merged.pt" "$MERGED/step_040000_merged.pt"
stat -c '%n %s bytes' "$MERGED/step_040000_merged.pt"
echo "  first-MiB sha16 must be f9fe3c9407c89169:"
head -c 1048576 "$MERGED/step_040000_merged.pt" | sha256sum | cut -c1-16

echo "== 5. the report over all eight nodes =="
# Exits non-zero if any node left the training band with no raw excursion behind it, so the batch's
# own evidence is re-checked by the same arithmetic the interlock used.
$PY report40k.py --out "$OUT/fixed" --md "$OUT/FIXED_SCENE_SAMPLING_REPORT.md" 2>&1 | tee -a "$LOG"
echo "  ledger: $OUT/batch_eval_ledger.jsonl (one row per node; failures carry a reason and the"
echo "  merged file they deliberately left behind)"
