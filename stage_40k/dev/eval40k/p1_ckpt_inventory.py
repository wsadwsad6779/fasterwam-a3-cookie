#!/usr/bin/env python3
"""P1 pre-work: what checkpoints actually exist, and which of them are forceless.

Reads only. Lists every .pt reachable under the usual roots, prints size + mtime, and for each one
that sits next to a config.yaml prints the fields that decide whether it saw force: the dataset, the
proprio width, and the force switches.
"""
import time
import os
from pathlib import Path


ROOTS = ["/workspace/a3_train_runs", "/workspace/_v1_stage", "/workspace/a3_ckpt",
         "/workspace/checkpoints", "/workspace/models", "/workspace", "/workspace/a3_data"]


def walk_pt(root, maxdepth=4):
    out = []
    rootp = Path(root)
    if not rootp.is_dir():
        return out
    base = len(rootp.parts)
    for dirpath, dirnames, filenames in os.walk(rootp):
        d = Path(dirpath)
        if len(d.parts) - base > maxdepth:
            dirnames[:] = []
            continue
        dirnames[:] = [x for x in dirnames if not x.startswith((".", "__"))]
        for fn in filenames:
            if fn.endswith((".pt", ".safetensors", ".bin")):
                p = d / fn
                try:
                    st = p.stat()
                except OSError:
                    continue
                out.append((str(p), st.st_size, int(st.st_mtime)))
    return out


def cfg_near(pt_path):
    d = Path(pt_path).parent
    for _ in range(4):
        for cand in (d / "config.yaml", d / "config.yml"):
            if cand.is_file():
                return cand
        if d.parent == d:
            break
        d = d.parent
    return None


def fields(cand):
    txt = cand.read_text(errors="replace")
    keep = ("dataset", "proprio", "force", "max_steps", "save_every", "norm_default_mode",
            "pretrained_norm_stats", "horizon", "run_dir", "output", "exp")
    lines = [ln.strip() for ln in txt.splitlines()
             if any(k in ln.lower() for k in keep) and not ln.strip().startswith("#")]
    return lines[:22]


seen = set()
rows = []
for root in ROOTS:
    for p, sz, mt in walk_pt(root):
        if p in seen:
            continue
        seen.add(p)
        rows.append((p, sz, mt))
rows.sort(key=lambda x: (-x[1], x[0]))
print("== .pt/.safetensors/.bin under the usual roots (%d hits) ==" % len(rows))
for p, sz, mt in rows[:120]:
    print("  %14d  %s  %s" % (sz, time.strftime("%Y-%m-%d %H:%M", time.localtime(mt)), p))

print("== configs next to the interesting ones ==")
for p, sz, mt in rows[:120]:
    if sz < 1_000_000:
        continue
    c = cfg_near(p)
    print("  %s -> config=%s" % (p, c))
    if c is not None:
        for ln in fields(c):
            print("      %s" % ln)

print("== top-level of /workspace ==")
try:
    for e in sorted(Path("/workspace").iterdir()):
        print("  %s %s" % ("d" if e.is_dir() else "f", e.name))
except Exception as exc:
    print("  err %s" % exc)
print("== /workspace/a3_eval* and any older eval result dirs ==")
for pat in ("a3_eval*", "a3_*", "*12k*", "*free*"):
    try:
        for e in sorted(Path("/workspace").glob(pat)):
            print("  %s %s" % ("d" if e.is_dir() else "f", e))
    except Exception as exc:
        print("  glob %s err %s" % (pat, exc))
