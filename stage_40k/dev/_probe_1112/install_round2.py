"""Round 2 of the interface-fix sync: `flow_test.py` only.

Why a second round exists: the first pod run of `flow_test.py` returned rc=1 with
`AttributeError: _set_flag`. The synthetic training config the text-conditioning checks build was an
attribute-getting dict, but `eval40k.deployment_text_encoder_settings` locks and unlocks the struct
flag through `OmegaConf.set_struct`, which only a real config node answers. The fixture now builds a
genuine `DictConfig` wherever `omegaconf` is installed and keeps the dict stand-in only under this
file's own stub, so the laptop pre-check and the pod acceptance run test the same type.

Same gate as round 1, against the same untouched pre-edit backup: staged file must hash to
`EXPECT`, the install is one atomic `os.replace`, the read-back must match, `py_compile` must pass,
and the unified diff versus the original pod copy plus the post-write copy are archived here. The
round-1 manifest and diff are left in place; this writes `post_manifest_round2.json` and
`flow_test.py.round2.diff`.
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
EXPECT = {"flow_test.py": "b06ad135aea86871a81ab7f5e79cdb948f78d6381463ac402917d4999653f8d0"}
ROUND1 = {"flow_test.py": "355513901734476c5e266caed40b07b6093601be8eac5acd76943bda9ea92d8c"}
REASON = ("pod flow_test rc=1: AttributeError _set_flag -- the synthetic config has to be a real "
          "DictConfig because deployment_text_encoder_settings calls OmegaConf.set_struct")


def sha(b):
    return hashlib.sha256(b).hexdigest()


rep = {"round": 2, "reason": REASON, "files": {}}
for f in sorted(EXPECT):
    want = EXPECT[f]
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
                                       fromfile="pre/" + f, tofile="pod_round2/" + f,
                                       lineterm="", n=3))
    (R / (f + ".round2.diff")).write_text(d + "\n", encoding="utf-8")
    (R / "post" / (f + ".round2.post")).write_bytes(back)
    rep["files"][f] = {"expected_local_sha256": want, "pod_read_back_sha256": sha(back),
                       "pod_bytes": len(back), "pre_pod_sha256": sha(pre),
                       "pre_pod_bytes": len(pre), "round1_installed_sha256": ROUND1[f],
                       "status": "INSTALLED" if sha(back) == want else "READBACK_MISMATCH",
                       "py_compile": compiles, "py_compile_error": err,
                       "diff_file": str(R / (f + ".round2.diff")),
                       "diff_lines_vs_original": d.count("\n") + 1}
    p = D / (f + ".pycheck")
    if p.is_file():
        p.unlink()
(R / "post_manifest_round2.json").write_text(json.dumps(rep, indent=2, sort_keys=True),
                                             encoding="utf-8")
print(json.dumps(rep, indent=1, sort_keys=True))
