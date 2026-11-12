#!/bin/sh
# Resume script for the 40k evaluation, to run on the pod.
# Written 2026-11-12 while the browser/kernel bridge was blocked by Qoder's permission classifier
# (see dev/_diffrecord/20261112_iface_fix/RECORD.md); the bridge works again and steps 1-5 below have
# been executed through it. The hashes are the round-8 identities, read back with sha256sum on the
# pod after dev/_diffrecord/20261112_iface_fix/post_manifest_round8.json was written -- earlier
# revisions of this file listed the round-7 images, before that the round-5 ones, and before that the
# pre-fix ones, which is the kind of stale record that gets a sync wrong.
#
# Check BEFORE and AFTER each write and stop on mismatch; never overwrite a file whose pre-image is
# not what the record says. The gate that enforces this is dev/_probe_1112/install_gate.py.
#   dev/eval40k/eval40k.py              61094  fb723254032813c7bc5d187fa98bea1b6ce39d46d9389366bdccce8b4ede4ff8
#   dev/eval40k/policy40k.py            10865  f7d8ec3c9936aa34d8a7776781c539d4bd5312e0ee21135f117d5a4eee87a8d8
#   dev/eval40k/smoke40k.py             92302  d69a038d512a3cb37b4ac7e3c3478c14cb7a4cc38a69bc472214767f58beaf72
#   dev/eval40k/flow_test.py           100599  215458cc1476241f61e993b9249c66093354952ea7652b062fb4fdf81f155929
#   dev/eval40k/proprio14_test.py       18628  6b0fa29cad45841df3ee048cad0965f304ab48d569fb1109c996582d5429398f
#   dev/eval40k/batch_eval.py           21698  3d6292ab014626ad664235a20a70dd4fb006dd9bdccd2315d020e2c8f8438c78
#   dev/eval40k/report40k.py            33924  b48b7a33f5f638381bcf6c68cc57f3e4838ef8b3ed16247ca611b868d83646ea
#       ^ round 9. This line is the POST-install identity. Until install_gate.py runs against
#         dev/_probe_1112/install_spec.json (round 9), the pod's live copy is still the round-8
#         image 14145 5b78b5d41756b84f455f13c45b1a34b26e0f54ca2b2b45245edde5a7c6ab8d8c, and the
#         staged bytes sit at dev/_probe_1112/stage_report40k.py with the sha above, read back and
#         verified. Do not cite this row as the pod's current state.
#   dev/eval40k/RUN_BATCH_STEP6.sh       8212  b2799dd28c9fa96278b22169100af2b58ef4881b1dea2a51e181275a1ae534b0
#   dev/_probe_1112/r9_report_test.py   13724  33d7d00abc621311cba36c9bdd35a416ba0b403781769fcb6f19570634dfc157
#   dev/_probe_1112/r8_r7_recheck.py     8202  612733e3290665370cd38e4f50d7b12ad8f647c528fe40eb6d1aa773cb858e1f
#   dev/_probe_1112/r3_plan_compare.py   4567  1f4b8d0f33ab128c95c0d04cf6d5af186ce32a6d8503082d1dc98cc16d99dfac
# Round 8 wrote two files: smoke40k.py (the proprio band assertion split into an interface check and
# a per-episode attribution check) and flow_test.py (two checks for the new episode_band_finding, plus
# a fix to the report-consumption regex that read a write as a read). eval40k.py, policy40k.py,
# proprio14_test.py, batch_eval.py and report40k.py were not written by round 8; their hashes are
# recorded too because steps 0-2 run them, and the only honest statement about a file this table does
# not cover is "unknown", not "unchanged". Round 9 wrote report40k.py only -- the two report sections
# step 6 depends on (proprio against the recorded band, and measured wall/simulation/inference time
# with the 160-episode projection), plus the fix to the per-column collapse that was dropping the
# seeds already recorded for a column whenever a wider episode arrived.
# pod pre-images that must be gone after the sync: eval40k.py 20862 a04a1255..., smoke40k.py
# 21092 a1470ba5...  (snapshotted 2026-11-11 in dev/_diffrecord/20261111_three_fixes/pre/).

set -e
cd /workspace/_v1_stage/dev/eval40k
export MUJOCO_GL=egl                 # without it the A3 env dies on headless import
export PYTHONDONTWRITEBYTECODE=1
PY=/opt/venv/bin/python

echo "== 0. the files that will run, hashed =="
sha256sum eval40k.py policy40k.py smoke40k.py flow_test.py proprio14_test.py

echo "== 1. CPU: the 14-wide proprio, arithmetic + call-site routing (30 checks) =="
$PY proprio14_test.py

echo "== 2. CPU: the eval loop flow test (99 checks on the pod; 91 under the local stub) =="
# the eight extra pod checks come from check_text_conditioning, which only appends them when the
# real arrays.npz, the real config and a non-stub omegaconf are present (flow_test.py:1117).
# The local shim is a wiring pre-check; this command is the acceptance run. Round 8 ran it as
# out/flow_test_20261112_r8.json so the round-7 evidence stays where the record points to it.
$PY flow_test.py --json out/flow_test_20261112.json

echo "== 3. CPU: whole-corpus equivalence proof, 42,681 rows, no model, no GPU =="
# the six force columns of `a3_fw_v1g/arrays.npz` state must equal channels [0,1,2,12,13,16] of
# `a3_fw_force/arrays.npz` minus zero_load_ref_median, and the first eight must equal the stored
# 8-wide proprio -- max abs difference required to be exactly 0. The smoke re-runs this itself;
# running it standalone first keeps a data problem away from the GPU.
$PY -c "import json, smoke40k as S; print(json.dumps(S.corpus_proprio14_proof(), indent=2, default=str))"

echo "== 4. CPU: stats width/values, and the frozen protocol dry-run =="
$PY eval40k.py --dry-run
$PY batch_eval.py --dry-run
# the protocol diff against the frozen plan; it lives outside this directory on purpose
# NOTE on paths: r3_plan_compare.py reads dev/eval40k/out/dryrun_20261112/plan.json as the frozen
# side and dev/eval40k/out/dryrun_round3/plan.json as the current side. The bare --dry-run above
# writes dev/eval40k/out/plan.json (the default --out is this directory's out/), so if this round's
# bytes are what you want compared, re-run it with --out out/dryrun_round3; otherwise the verdict
# describes the plan that was frozen on 2026-11-11, not whatever is on disk now.
PYTHONPATH=/workspace/_v1_stage/dev/eval40k $PY /workspace/_v1_stage/dev/_probe_1112/r3_plan_compare.py

echo "== 5. GPU: real step_040000, two complete MuJoCo rollouts on the fixed scene =="
# merged artifact already on the pod: step_040000_merged.pt 11,126,493,579 B, first-MiB f9fe3c9407c89169
# the fixed original cookie scene is the default: no position noise, no yaw noise, no box
# randomisation, and the smoke asserts one scene key across the seeds it runs
$PY -u smoke40k.py --steps 40000 --seeds 11 12 \
    --out out/smoke_20261112_r8 --max-steps 600 --replan 10 | tee out/smoke_20261112_r8.log

echo "  READ the log before step 6: the eleven smoke confirmations in the directive have to be"
echo "  in the JSON (no OOM, image path, 14-wide proprio, 16-frame history, 32 actions,"
echo "  future force 32x6 finite, per-chunk noise seeds, actions executed, episodes closed,"
echo "  identical scene key across seeds, normalise/denormalise/control/gripper contract),"
echo "  plus the round-7 additions: the per-column proprio exceedance table and the measured"
echo "  simulation/inference split behind any wall-clock projection; plus the round-8 split, where"
echo "  the interface check asserts only what an interface can prove (no gradient, deployment"
echo "  diffusion-step count and rand device, 14-wide proprio, the normaliser's own +-5 force clamp)"
echo "  and a separate per-episode check asserts the band evidence instead: every column handed to"
echo "  the model outside the recorded band must be attributed to that replan's raw value and frame"
echo "  (episodes[i].proprio_out_of_band: base/force columns out of band, worst excursion in band"
echo "  widths, clamped calls, unexplained == 0, witness present). An out-of-band number is a result"
echo "  about the robot leaving the 42,681 training rows; an unexplained one would be a defect here."

echo "== 5b. CPU: would round 8's split check have passed round 7's failing episodes? =="
# The only way to show that re-scoping an assertion raised the standard instead of lowering it is to
# run the new standard against the evidence that failed the old one. Pure arithmetic over
# out/smoke_20261112_r7's own JSON -- no model, no GPU, no simulator -- and it exits non-zero unless
# round 8's attribution explains every out-of-band column AND still reports the episode as
# out-of-band (a check that passed because the columns looked in-band would be the weakening).
#   dev/_probe_1112/r8_r7_recheck.py   5260  fb8be4a7baeba90406b829545400b62f7f23c078d294f5bdc1cfe151d4396983
# Run it only after step 5 has released the CPU: the smoke's simulation/inference split is worthless
# if something else competes for the cores while it is being measured -- that is how seed 12 in the
# round-7 run got a single 10.9 s MuJoCo step.
PYTHONPATH=/workspace/_v1_stage/dev/eval40k $PY /workspace/_v1_stage/dev/_probe_1112/r8_r7_recheck.py

echo "== 6. GPU: 8 nodes x 20 screened seeds = 160 episodes, one model at a time =="
# This step is RUN_BATCH_STEP6.sh in this directory -- read that file first, it is the four-move
# order (40k merged artifact moved aside, seven nodes merge->evaluate->delete, moved back, 40k run
# with --keep-merged) and it states the stop conditions the driver enforces per node.
# CHECK FIRST: at 2026-11-12T04:39Z the pod had 27.44 GiB free on the merged volume and exactly one
# merged artifact (step_040000, 11.13 GB). Seven of the eight nodes still need merging, so a run that
# keeps every merged .pt on disk cannot finish; read batch_eval.py --dry-run blockers and its
# merge/keep behaviour before starting, and never delete one of the eight original checkpoints.
# COST, measured rather than projected: the round-7 smoke put two 601-frame episodes at 201.41 s and
# 1676.89 s wall, of which 111.5 s and 1593.0 s were inside env.step -- the run is simulator-bound,
# and 160 episodes is a ~42 h commitment (median 41.74 h, mean 41.77 h). Say so before starting it.
$PY batch_eval.py --out /workspace/_v1_stage/dev/out40k/batch_eval_20seed1x \
    2>&1 | tee /workspace/_v1_stage/dev/out40k/batch_eval_20seed1x.log
$PY report40k.py --out /workspace/_v1_stage/dev/out40k/batch_eval_20seed1x \
    --md /workspace/_v1_stage/dev/out40k/batch_eval_20seed1x/FIXED_SCENE_SAMPLING_REPORT.md
