#!/bin/bash
# Recording pass: the diagnostic seeds, with three cameras, the per-dimension action chain, the
# proprio witness and -- above all -- the composed 前视+左腕+右侧数据面板 video the report asks for.
#
# Why a script and not a command line: the batch's own protocol lives in RUN_BATCH_STEP6.sh
# (`--steps 40000 --backend env --policy merged --batch-offset 0.008`, `--calib` at its default), and
# a hand-typed variant is how a recording ends up incomparable to the 160 episodes it is supposed to
# explain. Everything that changes the scored behaviour is therefore copied from the batch, and the
# only additions are the recorder's own `--rec-*` flags.
#
# What is deliberately NOT here: no `--position-noise`, no `--randomize-boxes` (the batch ran the
# fixed scene, and item 5 of the work order says the videos have to be comparable to it), no
# concurrency (one process loads one FasterWAM and rolls the seeds in sequence -- 48 GiB cannot hold
# two, and the work order forbids trying), and no write into batch_eval_20seed1x (record40k.py
# refuses it; this tree is a sibling).
#
# `--rec-vram-cap-frac 0` leaves the allocator uncapped on purpose. The default cap (0.35 = 16.8 GiB)
# sits under the 22.64 GiB a full load measured in the batch, so capping here would OOM a run that has
# the whole card to itself. The gate that still protects the card is `--rec-min-free-gib 25`, checked
# before the model is loaded.
#
# Why the CPU checks come before the GPU: the composed panel video has never been encoded by this
# pod's libx264, and the local machine that proved the layout has no imageio. So the two panel
# selftests and `recorder_selftest40k.py` are run here first, on CPU + EGL with no weights loaded --
# they cost a couple of minutes and are the only place where the encoder, the frame count and the
# four force traces can be shown to work before 25 minutes of inference are spent. If any of them
# fails, nothing is rolled and the failure is on a laptop-readable artifact instead of a half-recorded
# seed. The rollout's own `video_preflight` still runs inside `record40k.py` as the last gate.
#
# Usage:  bash RUN_RECORD_SEEDS.sh 11 17      # the two seeds the report wants first
#         bash RUN_RECORD_SEEDS.sh 12 19      # only after 11/17 have been watched
set -e
set -o pipefail
cd /workspace/_v1_stage/dev/eval40k
export MUJOCO_GL=egl
export PYTHONDONTWRITEBYTECODE=1

PY=/opt/venv/bin/python
OUT=/workspace/_v1_stage/dev/out40k/record_diag
MERGED=/workspace/_v1_stage/dev/eval40k/merged
CALIB=/workspace/a3_data/a3_fw_force/force_calib_v1.json
SEEDS="$*"

if [ -z "$SEEDS" ]; then
    echo "usage: bash RUN_RECORD_SEEDS.sh <seed> [seed ...]"
    exit 2
fi

echo "== 0a. the scored protocol must be the batch's own bytes =="
# The same seven files, with the same prefixes RUN_BATCH_STEP6.sh pinned. If any of these differs, the
# recording shows a policy that is not the one the 160 episodes scored, and the video is decoration.
$PY - <<'PYX'
import hashlib, sys
from pathlib import Path
want = {"eval40k.py": "fb723254", "policy40k.py": "f7d8ec3c", "smoke40k.py": "d69a038d",
        "flow_test.py": "ee45c272", "proprio14_test.py": "6b0fa29c", "batch_eval.py": "7dc2a0c5",
        "report40k.py": "b48b7a33"}
bad = []
for f, w in sorted(want.items()):
    got = hashlib.sha256(Path(f).read_bytes()).hexdigest()
    print("  %-26s %s" % (f, got[:8]))
    if not got.startswith(w):
        bad.append("%s: expected %s got %s" % (f, w, got[:8]))
if bad:
    raise SystemExit("the scored protocol moved: " + "; ".join(bad))
PYX

echo "== 0b. the recorder and the panel composer this pass runs with =="
# None of these are among the seven files above, so the gate is stated here instead: a selftest is only
# evidence for the bytes that actually ran. These ten are the 2026-11-13 set, cut after the three
# pre-recording confirmations: the panel now draws the executed command it really passed to env.step
# and the IK input beside it, the count next to them says "sitting on the +-1 rail" rather than
# "clipped" because it is measured on the post-clip row, the sample replays logged stage numbers
# instead of inventing them, the steps npz carries the frame-index convention that keeps 601
# frames and 600 actions aligned, and --require-right-press now grades a real press recording (C6)
# instead of refusing unconditionally -- which the tamper harness proves by feeding it four fake ones.
$PY - <<'PYX'
import hashlib
from pathlib import Path
want = {"record40k.py": "a036ad81", "recorder_selftest40k.py": "0515eb5d",
        "panelvideo40k.py": "1fec32db", "panelvideo_selftest40k.py": "ff599a99",
        "panelsink_selftest40k.py": "53e4adb6", "sensor_identity40k.py": "b05b510f",
        "sensor_identity_tamper40k.py": "969964f3",
        "panel_sample40k.py": "14d7aeaf", "force_stream.py": "49c22eb8",
        "rollout40k.py": "70d9adc9"}
bad = []
for f, w in sorted(want.items()):
    got = hashlib.sha256(Path(f).read_bytes()).hexdigest()
    print("  %-30s %s" % (f, got[:8]))
    if not got.startswith(w):
        bad.append("%s: expected %s got %s" % (f, w, got[:8]))
if bad:
    raise SystemExit("stale recorder: " + "; ".join(bad))
PYX

echo "== 1. CPU-only gates: no weights, no GPU, before the card is touched =="
# The composed video's layout, its per-frame sync with the four pad channels and the refusal paths are
# proved here on this pod's own encoder. A failure must be fixed before any rollout, never papered over.
# sensor_identity40k.py is first because everything downstream labels columns 12-15 as fingertip
# newtons; it proves that from the model file and the env's own assembly order (legs A+B), then tries
# the corpus (leg C). Exit 3 means A+B passed and the data leg had no files to read -- the recording
# may still proceed, but the report must not call the mapping three-legged for this run.
$PY -u sensor_identity40k.py; SI=$?
if [ "$SI" -eq 3 ]; then
    echo "  NOTE: leg C (corpus) had no files at the default globs -- mapping proved by A+B only"
elif [ "$SI" -ne 0 ]; then
    echo "  the channel map itself failed; refusing to record"; exit "$SI"
fi
# ... and the gate is then shown to bite: seven ways the model file could be wrong about columns 12-17,
# each one expected to make sensor_identity40k.py exit non-zero and name the check that caught it.
# A PASS line is only evidence if the corresponding FAIL is reachable.
$PY -u sensor_identity_tamper40k.py
$PY -u panelvideo_selftest40k.py
$PY -u panelsink_selftest40k.py
$PY -u recorder_selftest40k.py

echo "== 2. the GPU must be alone before a full load =="
/opt/rocm/bin/rocm-smi --showmeminfo vram 2>/dev/null | sed -n '4,7p'
pgrep -af "eval40k.py|batch_eval.py|train" | grep -v RUN_RECORD || echo "  no other eval/train process"
stat -c '%n %s bytes' "$MERGED/step_040000_merged.pt"
df -h /workspace | tail -1

echo "== 3. rollout =="
mkdir -p "$OUT"
$PY -u record40k.py --steps 40000 --seeds $SEEDS --backend env --policy merged \
    --out "$OUT" --merged-dir "$MERGED" --batch-offset 0.008 --calib "$CALIB" \
    --rec-vram-cap-frac 0 --rec-min-free-gib 25 --rec-panel-video

echo "== 4. what landed, and whether the composed video is the deliverable =="
find "$OUT/_record" -name '*.mp4' -printf '%p %s B\n' | sort
find "$OUT/_record" -name '*_meta.json' -printf '%p %s B\n' | sort
# The three checks the format request is actually graded on, re-read from the artifacts rather than
# from what the run claimed while writing them: one frame per control step, the historical geometry,
# and the four fingertip columns present with the sensor values the episode really measured.
$PY - "$OUT" $SEEDS <<'PYX'
import json, sys
from pathlib import Path

import numpy as np

import panelvideo40k as PV

out, seeds = Path(sys.argv[1]), [int(s) for s in sys.argv[2:]]
want_geom = [2 * PV.CAM_W + PV.PANEL_W, PV.PANEL_H]
want_cols = [PV.PAD_COLS[c] for c in PV.PAD_ORDER]
bad = []
for seed in seeds:
    d = out / "_record" / ("seed%d" % seed)
    meta = json.loads((d / ("seed%d_step40000_meta.json" % seed)).read_text())
    pv, rec = meta["panel_video"], meta["episode_record"]
    steps = np.load(str(d / ("seed%d_step40000_steps.npz" % seed)))
    if pv is None or "pad_forcen" not in steps.files:
        bad.append("seed%d: ran without --rec-panel-video (panel_video=%s, npz has %s)"
                   % (seed, pv, sorted(steps.files)[:6]))
        continue
    found = sorted(d.glob("*_composed_panel.mp4"))
    if not found:
        bad.append("seed%d: no composed panel video at all" % seed)
        continue
    for f in found:
        back = PV.read_frames(f)
        cols = [int(c) for c in steps["pad_channel_columns"]]
        order = [str(c) for c in steps["pad_channel_order"]]
        print("  seed%-3d %s\n           %d decoded frames (sink wrote %s, steps_executed %s) "
              "%dx%d %d B at %s fps\n           encoder %s  notes %s\n"
              "           force order %s columns %s, %d rows x %d traces, absmax %s"
              % (seed, f.name, len(back), pv["frames_written"], rec["steps_executed"],
                 back[0].shape[1], back[0].shape[0], f.stat().st_size, pv["fps"],
                 pv["backend"], pv["notes"],
                 order, cols, steps["pad_forcen"].shape[0], steps["pad_forcen"].shape[1],
                 np.round(steps["pad_forcen"].max(axis=0), 1).tolist()))
        if not len(back) == pv["frames_written"] == rec["steps_executed"] + 1:
            bad.append("seed%d: frame count %s/%s/%s is not one per control step"
                       % (seed, len(back), pv["frames_written"], rec["steps_executed"]))
        # `PanelRecorder.close` reports only what it wrote; this is the second, independent reader, so
        # the file's real decoded shape has to agree with the claim beside it.
        if list(back[0].shape) != pv["wh"][::-1] + [3]:
            bad.append("seed%d: the file decodes to %s but the sink claims it wrote %s"
                       % (seed, list(back[0].shape), pv["wh"]))
        if pv["wh"] != want_geom:
            bad.append("seed%d: geometry %s != %s" % (seed, pv["wh"], want_geom))
        if pv["fps"] != meta["video_fps"]:
            bad.append("seed%d: the video is tagged %s fps while the env ran at %s Hz, so the force "
                       "curve and the frames would drift apart"
                       % (seed, pv["fps"], meta["video_fps"]))
        if f.stat().st_size < 1000:
            bad.append("seed%d: %d B is not a recording" % (seed, f.stat().st_size))
        if cols != want_cols or order != list(PV.PAD_ORDER):
            bad.append("seed%d: pad identity %s %s" % (seed, order, cols))
        if steps["pad_forcen"].shape[0] != pv["frames_written"]:
            bad.append("seed%d: %d force rows for %d frames -- the curve and the video would "
                       "drift apart" % (seed, steps["pad_forcen"].shape[0], pv["frames_written"]))
print(json.dumps({"ok": not bad, "problems": bad}, indent=2))
sys.exit(1 if bad else 0)
PYX
