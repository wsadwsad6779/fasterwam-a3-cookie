"""Hash-gated installer for one eval-toolchain file at a time.

Round 1 (`install_three.py`) and round 2 (`install_round2.py`) hard-coded their expectations, which
meant writing a new script for every correction. This one reads the same facts from
`install_spec.json`, so a follow-up round is a two-line spec plus the staged file -- the gate itself
never changes:

  * the staged copy must hash to `expect_sha256` (the local file's hash, recorded before the sync);
  * the pre-edit backup is read from `pre/<file>.pre`, which round 1 made and nothing overwrites;
  * the install is one `os.replace`, so the target is either the old bytes or the new ones;
  * the read-back must match, `py_compile` must pass on the pod, and the unified diff against the
    original pre-edit copy plus a post-write copy are archived per round;
  * each round writes its own `post_manifest_round<N>.json` -- earlier rounds' records are never
    rewritten, so the correction history is auditable rather than squashed.
"""
import difflib
import hashlib
import json
import os
import py_compile
from pathlib import Path

D = Path("/workspace/_v1_stage/dev/eval40k")
ST = Path("/workspace/_v1_stage/dev/_probe_1112")
R = Path("/workspace/_v1_stage/dev/_diffrecord/20261112_iface_fix")


def sha(b):
    return hashlib.sha256(b).hexdigest()


spec = json.loads((ST / "install_spec.json").read_text())
rep = {"round": spec["round"], "reason": spec["reason"], "files": {}}
for f, want in sorted(spec["expect_sha256"].items()):
    src = ST / ("stage_" + f)
    dst = D / f
    pre_path = R / "pre" / (f + ".pre")
    if not src.is_file() or not pre_path.is_file():
        rep["files"][f] = {"status": "STAGING_OR_BACKUP_MISSING", "staging": str(src),
                           "pre": str(pre_path)}
        continue
    b = src.read_bytes()
    got = sha(b)
    if got != want:
        rep["files"][f] = {"status": "STAGING_MISMATCH", "staged": got, "expected": want}
        continue
    pre = pre_path.read_bytes()
    before = dst.read_bytes() if dst.is_file() else None
    tmp = D / (f + ".new")
    tmp.write_bytes(b)
    os.replace(str(tmp), str(dst))
    back = dst.read_bytes()
    try:
        py_compile.compile(str(dst), doraise=True, cfile=str(D / (f + ".pycheck")))
        compiles, err = True, None
    except Exception as exc:
        compiles, err = False, str(exc)[:400]
    d = "\n".join(difflib.unified_diff(pre.decode("utf-8").splitlines(),
                                       back.decode("utf-8").splitlines(),
                                       fromfile="pre/" + f, tofile="pod_round%s/" % spec["round"] + f,
                                       lineterm="", n=3))
    diff_name = ("%s.diff" % f) if int(spec["round"]) == 1 else ("%s.round%s.diff" % (f, spec["round"]))
    (R / diff_name).write_text(d + "\n", encoding="utf-8")
    post_name = ("%s.post" % f) if int(spec["round"]) == 1 else ("%s.round%s.post" % (f, spec["round"]))
    (R / "post" / post_name).write_bytes(back)
    rep["files"][f] = {"expected_local_sha256": want, "pod_read_back_sha256": sha(back),
                       "pod_bytes": len(back), "pre_pod_sha256": sha(pre), "pre_pod_bytes": len(pre),
                       "was_installed_sha256": (sha(before) if before is not None else None),
                       "was_installed_bytes": (len(before) if before is not None else None),
                       "status": "INSTALLED" if sha(back) == want else "READBACK_MISMATCH",
                       "py_compile": compiles, "py_compile_error": err,
                       "diff_file": str(R / diff_name), "diff_lines_vs_original": d.count("\n") + 1}
    p = D / (f + ".pycheck")
    if p.is_file():
        p.unlink()
manifest = R / ("post_manifest.json" if int(spec["round"]) == 1
                else "post_manifest_round%s.json" % spec["round"])
manifest.write_text(json.dumps(rep, indent=2, sort_keys=True), encoding="utf-8")
rep["manifest"] = str(manifest)
print(json.dumps(rep, indent=1, sort_keys=True))
