import os
import zipfile
from pathlib import Path

SRC_ROOT = Path(r"E:\Innovative_practice\WAM")
OUT = Path(r"C:\Users\ASUS\Desktop")

JOBS = {
    "FasterWAM_code.zip": ("FasterWAM", {
        ".venv", ".venvs", "checkpoints", "third_party", ".runtime",
        "evaluate_results", "experiment_backups", "runs", "results",
        "data", "code", "cache", "assets",
    }),
    "a3_dual_arm_sim.zip": ("a3_dual_arm_sim", {".venv"}),
    "fasterwam_a3_bridge.zip": ("fasterwam_a3_bridge", {"outputs", ".venv"}),
}

SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache"}

for name, (src, top_excl) in JOBS.items():
    src_dir = SRC_ROOT / src
    out_path = OUT / name
    count = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(src_dir):
            rel = Path(root).relative_to(SRC_ROOT)
            if rel.parts and rel.parts[0] in top_excl:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                if f.endswith(".pyc"):
                    continue
                full = Path(root) / f
                arc = (Path(src) / full.relative_to(src_dir)).as_posix()
                z.write(full, arc)
                count += 1
    print(f"{name}: {count} files, {out_path.stat().st_size} bytes")
