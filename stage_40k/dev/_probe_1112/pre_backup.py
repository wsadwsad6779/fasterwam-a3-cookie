import hashlib
import json
import shutil
from pathlib import Path

D = Path("/workspace/_v1_stage/dev/eval40k")
R = Path("/workspace/_v1_stage/dev/_diffrecord/20261112_iface_fix")
TARGETS = ["eval40k.py", "smoke40k.py", "flow_test.py"]
print(json.dumps({"record_dir_existed": R.is_dir(),
                  "record_contents": sorted([p.name for p in R.rglob("*")])[:40]
                  if R.is_dir() else []}, indent=1))
(R / "pre").mkdir(parents=True, exist_ok=True)
(R / "post").mkdir(parents=True, exist_ok=True)
man = {}
for f in TARGETS + ["batch_eval.py", "report40k.py", "README.md"]:
    src = D / f
    if not src.is_file():
        man[f] = "MISSING"
        continue
    b = src.read_bytes()
    sha = hashlib.sha256(b).hexdigest()
    dst = R / "pre" / (f + ".pre")
    if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest() != sha:
        dst = R / "pre" / (f + ".pre.dup")
    shutil.copyfile(str(src), str(dst))
    man[f] = {"bytes": len(b), "sha256": sha,
              "backup": str(dst), "backup_verified":
                  hashlib.sha256(dst.read_bytes()).hexdigest() == sha}
(R / "pre_manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(man, indent=1, sort_keys=True))
