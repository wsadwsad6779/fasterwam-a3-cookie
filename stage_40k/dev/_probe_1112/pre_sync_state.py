import hashlib
import json
import subprocess
from pathlib import Path

D = Path("/workspace/_v1_stage/dev/eval40k")
files = ["eval40k.py", "smoke40k.py", "flow_test.py", "policy40k.py", "batch_eval.py",
         "report40k.py", "proprio14_test.py", "README.md", "force_metrics.py",
         "force_stream.py", "rollout40k.py"]
rec = {}
for f in files:
    p = D / f
    if p.is_file():
        b = p.read_bytes()
        rec[f] = {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
    else:
        rec[f] = None
out = {"files": rec}
du = subprocess.run(["df", "-B1", "/workspace"], capture_output=True, text=True)
out["df_bytes_line"] = du.stdout.strip().splitlines()[-1]
out["free_gib"] = round(int(du.stdout.strip().splitlines()[-1].split()[3]) / 2 ** 30, 2)
r = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True, timeout=40)
out["vram"] = "\n".join(r.stdout.strip().splitlines()[:8])
out["train_seg"] = subprocess.run(["pgrep", "-af", "train_seg.py"],
                                  capture_output=True, text=True).stdout.strip()[:300]
out["python_procs"] = subprocess.run(["pgrep", "-af", "python"],
                                     capture_output=True,
                                     text=True).stdout.strip()[:900]
M = D / "merged"
out["merged"] = sorted([p.name for p in M.iterdir()]) if M.is_dir() else "absent"
L = Path("/workspace/_v1_stage/run40k/checkpoints/light")
out["light"] = sorted([p.name for p in L.iterdir()]) if L.is_dir() else "absent"
S = Path("/workspace/_v1_stage/dev/out40k")
out["out40k"] = sorted([p.name for p in S.iterdir()])[:40] if S.is_dir() else "absent"
out["smoke_logs"] = sorted([p.name for p in (S / "smoke_logs").iterdir()])[:20] \
    if (S / "smoke_logs").is_dir() else "absent"
print(json.dumps(out, indent=1, sort_keys=True))
