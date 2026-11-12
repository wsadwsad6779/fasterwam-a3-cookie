#!/bin/bash
# Round-10 gate on the pod: install the batch_eval.py disk-gate fix and the four flow_test.py checks
# that cover it, through the same hash-gated installer as rounds 1-9, then prove from the manifest and
# a read-back that the two live files are exactly the bytes the local working copy has.
#
# Steps are chained with explicit rc checks, so a failing step stops the chain and leaves its .out/.rc
# behind as the evidence of where it stopped; r10_gate.done is only written if every step returned 0.
# Nothing here deletes or moves an artifact: the merge directory, the eight checkpoints and every
# publish record stay exactly where the refused launch left them.
export PYTHONDONTWRITEBYTECODE=1
ST=/workspace/_v1_stage/dev/_probe_1112
D=/workspace/_v1_stage/dev/eval40k
R=/workspace/_v1_stage/dev/_diffrecord/20261112_iface_fix
PY=/opt/venv/bin/python
cd "$ST" || exit 9

echo "== 1. the state the gate is allowed to start from =="
{ date -u +%FT%TZ
  sha256sum $D/batch_eval.py $D/flow_test.py \
      $ST/stage_batch_eval.py $ST/stage_flow_test.py \
      $R/pre/batch_eval.py.pre $R/pre/flow_test.py.pre \
      $ST/install_spec.json $ST/install_spec_round10.json
  ls -l $D/batch_eval.py $D/flow_test.py $ST/stage_batch_eval.py $ST/stage_flow_test.py
  ls -l $R/post_manifest_round10.json 2>/dev/null || echo "  no round-10 manifest yet -- correct"
  df -h /workspace | tail -1
} > r10_pre.out 2>&1
cat r10_pre.out

# Before writing anything: the two staged copies must equal the spec, the spec must be round 10, the
# live copies must not already be the target, and there must be no round-10 manifest (a second gate run
# would otherwise re-diff round 10 against a target that is no longer its pre-image).
$PY - <<'PYX' > r10_precheck.json 2>&1
import hashlib
import json
import sys
from pathlib import Path

D = Path("/workspace/_v1_stage/dev/eval40k")
ST = Path("/workspace/_v1_stage/dev/_probe_1112")
R = Path("/workspace/_v1_stage/dev/_diffrecord/20261112_iface_fix")
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
spec = json.loads((ST / "install_spec.json").read_text())
want = spec["expect_sha256"]
out = {"spec_round": spec["round"], "spec_files": sorted(want), "files": {}}
for f, w in sorted(want.items()):
    staged = ST / ("stage_" + f)
    live = D / f
    pre = R / "pre" / (f + ".pre")
    out["files"][f] = {
        "want": w,
        "staged": h(staged) if staged.is_file() else None,
        "staged_bytes": staged.stat().st_size if staged.is_file() else None,
        "live": h(live) if live.is_file() else None,
        "live_bytes": live.stat().st_size if live.is_file() else None,
        "pre_image": h(pre) if pre.is_file() else None,
        "pre_image_bytes": pre.stat().st_size if pre.is_file() else None,
        "staged_matches_spec": staged.is_file() and h(staged) == w,
        "live_is_already_target": live.is_file() and h(live) == w,
        "live_matches_pre_image": live.is_file() and pre.is_file() and h(live) == h(pre),
    }
out["round10_manifest_already_exists"] = (R / "post_manifest_round10.json").is_file()
out["spec_is_round_10"] = spec["round"] == 10
out["ok"] = (out["spec_is_round_10"]
             and all(v["staged_matches_spec"] and not v["live_is_already_target"]
                     for v in out["files"].values())
             and not out["round10_manifest_already_exists"])
print(json.dumps(out, indent=2, sort_keys=True))
sys.exit(0 if out["ok"] else 1)
PYX
rc=$?
echo "  precheck rc=$rc"
cat r10_precheck.json
[ $rc -eq 0 ] || { echo "REFUSED at precheck (rc=$rc) -- nothing was installed"; exit $rc; }

echo "== 2. the gate: two os.replace calls, read-back shas, pod py_compile, round-10 diffs + manifest =="
$PY install_gate.py > r10_gate.out 2>&1
rc=$?
echo "  gate rc=$rc"
cat r10_gate.out
[ $rc -eq 0 ] || { echo "REFUSED at gate (rc=$rc)"; exit $rc; }

echo "== 3. verify the installed files are what the spec says before using them =="
$PY - <<'PYX' > r10_verify.json 2>&1
import hashlib
import json
import sys
from pathlib import Path

D = Path("/workspace/_v1_stage/dev/eval40k")
ST = Path("/workspace/_v1_stage/dev/_probe_1112")
R = Path("/workspace/_v1_stage/dev/_diffrecord/20261112_iface_fix")
h = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
want = json.loads((ST / "install_spec.json").read_text())["expect_sha256"]
man = json.loads((R / "post_manifest_round10.json").read_text())
out = {"manifest": str(R / "post_manifest_round10.json"), "round": man.get("round"), "files": {}}
for f, w in sorted(want.items()):
    ent = man["files"].get(f, {})
    out["files"][f] = {
        "live_sha256": h(D / f), "live_bytes": (D / f).stat().st_size, "spec_want": w,
        "manifest_status": ent.get("status"), "manifest_py_compile": ent.get("py_compile"),
        "manifest_read_back": ent.get("pod_read_back_sha256"),
        "manifest_was_installed": ent.get("was_installed_sha256"),
        "pre_pod_sha256": ent.get("pre_pod_sha256"),
        "diff_file": ent.get("diff_file"),
        "diff_lines_vs_original": ent.get("diff_lines_vs_original"),
        "py_compile_error": ent.get("py_compile_error"),
    }
    out["files"][f]["ok"] = (out["files"][f]["live_sha256"] == w
                             == out["files"][f]["manifest_read_back"]
                             and ent.get("status") == "INSTALLED"
                             and ent.get("py_compile") is True)
out["ok"] = all(v["ok"] for v in out["files"].values()) and man.get("round") == 10
print(json.dumps(out, indent=2, sort_keys=True))
sys.exit(0 if out["ok"] else 1)
PYX
rc=$?
echo "  verify rc=$rc"
cat r10_verify.json
[ $rc -eq 0 ] || { echo "REFUSED at verify (rc=$rc) -- do not run flow_test or the batch"; exit $rc; }

echo "== 4. the two files now on the pod, in the form the batch driver pins =="
sha256sum $D/batch_eval.py $D/flow_test.py | cut -c1-16
$PY -c "import ast,sys;[ast.parse(open(p,encoding='utf-8').read()) for p in ['$D/batch_eval.py','$D/flow_test.py']];print('  ast parse ok')"
echo DONE > r10_gate.done
echo "chain finished"
