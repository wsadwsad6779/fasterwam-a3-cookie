import os, sys, zipfile
from pathlib import Path

ZIP = Path(r"C:\Users\ASUS\Desktop\FasterWAM_code.zip")
SRC = Path(r"E:\Innovative_practice\WAM\FasterWAM")
TOP_EXCL = {".venv", ".venvs", "checkpoints", "third_party", ".runtime",
            "evaluate_results", "experiment_backups", "runs", "results",
            "data", "code", "cache", "assets"}

in_zip = {n for n in zipfile.ZipFile(ZIP).namelist()}
missing, extra = [], []
for root, dirs, files in os.walk(SRC):
    rel_root = Path(root).relative_to(SRC)
    if rel_root.parts and rel_root.parts[0] in TOP_EXCL:
        dirs[:] = []
        continue
    for f in files:
        rel = (rel_root / f).as_posix()
        arc = f"FasterWAM/{rel}"
        if arc not in in_zip:
            missing.append((arc, (Path(root) / f).stat().st_size))

print("zip entries:", len(in_zip))
print("missing:", len(missing))
for a, s in missing[:80]:
    print(f"  {s:>10}  {a}")
