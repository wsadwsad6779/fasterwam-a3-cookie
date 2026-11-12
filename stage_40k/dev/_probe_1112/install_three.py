import difflib
import hashlib
import json
import os
import py_compile
from pathlib import Path

D = Path("/workspace/_v1_stage/dev/eval40k")
ST = Path("/workspace/_v1_stage/dev/_probe_1112")
R = Path("/workspace/_v1_stage/dev/_diffrecord/20261112_iface_fix")
EXPECT = {"eval40k.py": "0e0661643da0aeb428d88c88cb7291bb025c8fdefc6d1fbf2df618c5e9b4fce0",
          "smoke40k.py": "3fe31764a4fe8c8353587f19004ffad09e12acd5dc8f0012e3882a42fe124499",
          "flow_test.py": "355513901734476c5e266caed40b07b6093601be8eac5acd76943bda9ea92d8c"}


def sha(b):
    return hashlib.sha256(b).hexdigest()


rep = {}
for f in sorted(EXPECT):
    want = EXPECT[f]
    src = ST / ("stage_" + f)
    dst = D / f
    pre_path = R / "pre" / (f + ".pre")
    if not src.is_file():
        rep[f] = {"status": "STAGING_MISSING", "staging": str(src)}
        continue
    b = src.read_bytes()
    got = sha(b)
    if got != want:
        rep[f] = {"status": "STAGING_MISMATCH", "staged": got, "expected": want}
        continue
    pre = pre_path.read_bytes()
    tmp = D / (f + ".new")
    tmp.write_bytes(b)
    os.replace(str(tmp), str(dst))
    back = dst.read_bytes()
    try:
        py_compile.compile(str(dst), doraise=True, cfile=str(D / (f + ".pycheck")))
        compiles = True
        err = None
    except Exception as exc:
        compiles = False
        err = str(exc)[:400]
    d = "\n".join(difflib.unified_diff(pre.decode("utf-8").splitlines(),
                                       back.decode("utf-8").splitlines(),
                                       fromfile="pre/" + f, tofile="pod/" + f,
                                       lineterm="", n=3))
    (R / (f + ".diff")).write_text(d + "\n", encoding="utf-8")
    (R / "post" / (f + ".post")).write_bytes(back)
    rep[f] = {"expected_local_sha256": want, "pod_read_back_sha256": sha(back),
              "pod_bytes": len(back), "pre_pod_sha256": sha(pre), "pre_pod_bytes": len(pre),
              "status": "INSTALLED" if sha(back) == want else "READBACK_MISMATCH",
              "py_compile": compiles, "py_compile_error": err,
              "diff_file": str(R / (f + ".diff")), "diff_lines": d.count("\n") + 1}
(R / "post_manifest.json").write_text(json.dumps(rep, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(rep, indent=1, sort_keys=True))
for f in sorted(EXPECT):
    p = D / (f + ".pycheck")
    if p.is_file():
        p.unlink()
