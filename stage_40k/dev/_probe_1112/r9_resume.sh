#!/bin/bash
# Round-9 resume on the pod: the three CPU actions that needed a shell -- the hash-gated install of
# report40k.py, the 33-check report test, and a render of the new report sections over trees that
# already exist. No GPU, no rollouts, nothing deleted, nothing overwritten except eval40k/report40k.py
# (and its pre-image is verified byte-identical to _diffrecord/20261112_iface_fix/pre/ first).
#
# Each step records its own rc and the chain stops at the first non-zero, so r9_resume.done exists
# only when every step before it returned 0. The report render is the one step allowed to exit 1 --
# that is the unexplained-clamp path firing -- so its rc is captured into a file, not chained.
export PYTHONDONTWRITEBYTECODE=1
ST=/workspace/_v1_stage/dev/_probe_1112
D=/workspace/_v1_stage/dev/eval40k
R=/workspace/_v1_stage/dev/_diffrecord/20261112_iface_fix
B=/workspace/_v1_stage/dev/out40k
PY=/opt/venv/bin/python
cd "$ST" || exit 9

echo "== 1. the state the gate is allowed to start from =="
{ date -u +%FT%TZ
  sha256sum $D/eval40k.py $D/policy40k.py $D/smoke40k.py $D/flow_test.py $D/proprio14_test.py \
      $D/batch_eval.py $D/report40k.py $ST/stage_report40k.py $ST/r9_report_test.py
  ls -l $D/report40k.py $ST/stage_report40k.py $ST/install_spec.json $R/pre/report40k.py.pre
  df -h /workspace | tail -1
} > r9_pre.out 2>&1
cat r9_pre.out

# Before writing anything: the staged copy must equal the spec, the live copy must equal the archived
# pre-image, and there must be no post_manifest_round9.json yet (a second gate run would diff the
# round-9 record against a target that is no longer the pre-image).
$PY - <<'PYX' > r9_precheck.json 2>&1
import hashlib, json
from pathlib import Path
D = Path('/workspace/_v1_stage/dev/eval40k')
ST = Path('/workspace/_v1_stage/dev/_probe_1112')
R = Path('/workspace/_v1_stage/dev/_diffrecord/20261112_iface_fix')
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
spec = json.loads((ST / 'install_spec.json').read_text())
want = spec['expect_sha256']['report40k.py']
out = {'spec_round': spec['round'], 'spec_want': want,
       'staged': h(ST / 'stage_report40k.py'), 'staged_bytes': (ST / 'stage_report40k.py').stat().st_size,
       'live_before': h(D / 'report40k.py'), 'live_bytes': (D / 'report40k.py').stat().st_size,
       'pre_image': h(R / 'pre' / 'report40k.py.pre'),
       'pre_image_bytes': (R / 'pre' / 'report40k.py.pre').stat().st_size,
       'round9_manifest_already_exists': (R / 'post_manifest_round9.json').is_file(),
       'r9_test_sha256': h(ST / 'r9_report_test.py')}
out['staged_matches_spec'] = out['staged'] == want
out['live_matches_pre_image'] = out['live_before'] == out['pre_image']
out['ok'] = bool(out['staged_matches_spec'] and out['live_matches_pre_image']
                 and not out['round9_manifest_already_exists'])
print(json.dumps(out, indent=2, sort_keys=True))
raise SystemExit(0 if out['ok'] else 1)
PYX
rc=$?
cat r9_precheck.json
echo "  precheck rc=$rc"
[ $rc -eq 0 ] || { echo "STOP: precheck refused the install"; echo $rc > r9_resume.rc; exit $rc; }

echo "== 2. the gate: one os.replace, read-back sha, pod py_compile, round-9 diff + manifest =="
$PY install_gate.py > r9_gate.out 2>&1
rc=$?
echo $rc > r9_gate.rc
cat r9_gate.out
[ $rc -eq 0 ] || { echo "STOP: install_gate.py returned $rc"; echo $rc > r9_resume.rc; exit $rc; }

echo "== 3. verify the install is what the spec says before anything uses it =="
$PY - <<'PYX' > r9_verify.json 2>&1
import hashlib, json
from pathlib import Path
D = Path('/workspace/_v1_stage/dev/eval40k')
ST = Path('/workspace/_v1_stage/dev/_probe_1112')
R = Path('/workspace/_v1_stage/dev/_diffrecord/20261112_iface_fix')
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
want = json.loads((ST / 'install_spec.json').read_text())['expect_sha256']['report40k.py']
man = json.loads((R / 'post_manifest_round9.json').read_text())
f = man['files']['report40k.py']
out = {'round': man.get('round'), 'live_sha256': h(D / 'report40k.py'),
       'live_bytes': (D / 'report40k.py').stat().st_size, 'spec_want': want,
       'manifest_status': f.get('status'), 'manifest_py_compile': f.get('py_compile'),
       'manifest_read_back': f.get('pod_read_back_sha256'),
       'manifest_was_installed': f.get('was_installed_sha256'),
       'manifest_pre_pod_sha256': f.get('pre_pod_sha256'),
       'diff_lines_vs_original': f.get('diff_lines_vs_original'),
       'diff_file': f.get('diff_file'), 'post_copy': str(R / 'post' / 'report40k.py.round9.post'),
       'manifest': str(R / 'post_manifest_round9.json')}
out['ok'] = bool(out['live_sha256'] == want == out['manifest_read_back']
                 and out['manifest_status'] == 'INSTALLED' and out['manifest_py_compile'] is True)
print(json.dumps(out, indent=2, sort_keys=True))
raise SystemExit(0 if out['ok'] else 1)
PYX
rc=$?
cat r9_verify.json
echo "  verify rc=$rc"
[ $rc -eq 0 ] || { echo "STOP: post-install verification failed ($rc)"; echo $rc > r9_resume.rc; exit $rc; }

echo "== 4. the 33 report checks on the pod interpreter =="
PYTHONPATH=$D $PY r9_report_test.py > r9_test.out 2>&1
rc=$?
echo $rc > r9_test.rc
cat r9_test.out
[ $rc -eq 0 ] || { echo "STOP: r9_report_test.py returned $rc on the pod"; echo $rc > r9_resume.rc; exit $rc; }

echo "== 5. render the new sections over the trees that already exist =="
# report40k.py only ever opens out*/step_*/summary.json and writes to --md/--json, both of which point
# into this directory: the two batch trees are read, never touched.
$PY $D/report40k.py --out $B/batch_eval/fixed \
    --md $ST/r9_render_batcheval.md --json $ST/r9_render_batcheval.json > r9_render1.out 2>&1
rc1=$?
echo $rc1 > r9_render1.rc
$PY $D/report40k.py --out $B/batch_eval_20seed_check/fixed \
    --md $ST/r9_render_20seedcheck.md > r9_render2.out 2>&1
rc2=$?
echo $rc2 > r9_render2.rc
{ echo "=== render1 ($B/batch_eval/fixed) rc=$rc1 ==="; cat r9_render1.out
  echo "=== render2 ($B/batch_eval_20seed_check/fixed) rc=$rc2 ==="; cat r9_render2.out; } > r9_render.out 2>&1
echo "  render rc: batch_eval/fixed=$rc1  batch_eval_20seed_check/fixed=$rc2"
echo "  (exit 1 from a render is the unexplained-clamp path, not a crash -- read r9_render.out)"
tail -n 40 r9_render.out
echo DONE > r9_resume.done
echo 0 > r9_resume.rc
