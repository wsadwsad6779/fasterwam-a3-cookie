"""Check 3: does a real interruption and resume restore the training state?

Run A trains N steps uninterrupted; run B trains to the first light-state save,
is killed, and resumes from that directory to N.  Both runs share every hydra
key except `save_every`, so the cosine schedule horizon is identical and any LR
difference is a restore defect rather than a different schedule.

Compare, for the two step-N light-state directories:
  adapter.pt / optimizer.pt / scheduler.pt / rng.pt   structurally, tensorwise
  trainer_state.json                                  global step, epoch,
                                                      batch_in_epoch, sampler
And the per-step `[train] ... loss= lr=` lines from the logs.

    python3 chk3_cmp.py <A_state_dir> <B_state_dir> <A.log> <B1.log> <B2.log>
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

MAX_DIFFS = 12


def describe(value):
    if torch.is_tensor(value):
        return f"tensor{tuple(value.shape)}:{value.dtype}"
    if isinstance(value, np.ndarray):
        return f"ndarray{value.shape}:{value.dtype}"
    if isinstance(value, (dict, list, tuple)):
        return f"{type(value).__name__}({len(value)})"
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def walk(a, b, path, diffs):
    """Structural comparison; appends human-readable differences to `diffs`."""
    if len(diffs) >= MAX_DIFFS:
        return
    if torch.is_tensor(a) and torch.is_tensor(b):
        if a.shape != b.shape:
            diffs.append(f"{path}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
            return
        if not torch.equal(a, b):
            md = (a.float() - b.float()).abs().max().item()
            nz = int((a != b).sum().item())
            diffs.append(f"{path}: tensors differ, {nz} elements, max|d|={md!r}")
        return
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if a.shape != b.shape:
            diffs.append(f"{path}: ndarray shape differs")
            return
        if not np.array_equal(a, b):
            diffs.append(f"{path}: ndarray differs at {int((a != b).sum())} entries")
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            if key not in a:
                diffs.append(f"{path}.{key}: missing in first")
            elif key not in b:
                diffs.append(f"{path}.{key}: missing in second")
            else:
                walk(a[key], b[key], f"{path}.{key}", diffs)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", diffs)
        return
    if isinstance(a, float) and isinstance(b, float):
        if a != b:
            diffs.append(f"{path}: {a!r} vs {b!r}")
        return
    if a != b:
        diffs.append(f"{path}: {describe(a)} vs {describe(b)}")


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def compare_state(dir_a, dir_b):
    print(f"### light state: {dir_a}  vs  {dir_b}")
    for name in ("adapter.pt", "optimizer.pt", "scheduler.pt", "rng.pt"):
        pa, pb = Path(dir_a) / name, Path(dir_b) / name
        if not pa.exists() or not pb.exists():
            print(f"  {name:<14} MISSING (a={pa.exists()} b={pb.exists()})")
            continue
        diffs = []
        walk(load(pa), load(pb), name, diffs)
        verdict = "bitwise_equal" if not diffs else f"differs ({len(diffs)} shown)"
        print(f"  {name:<14} {verdict}")
        for line in diffs:
            print(f"      {line}")
    sa = json.loads((Path(dir_a) / "trainer_state.json").read_text())
    sb = json.loads((Path(dir_b) / "trainer_state.json").read_text())
    diffs = []
    walk(sa, sb, "trainer_state.json", diffs)
    print(f"  trainer_state {'bitwise_equal' if not diffs else 'differs'}")
    for line in diffs:
        print(f"      {line}")
    for key in ("global_step", "epoch", "batch_in_epoch"):
        print(f"      {key}: a={sa.get(key)} b={sb.get(key)}")
    print(f"      sampler: a={sa.get('sampler')} b={sb.get('sampler')}")
    print(f"      extra:   a={sa.get('extra')}")


STEP_RE = re.compile(r"step=(\d+)/\d+ loss=([0-9.eE+-]+).*?lr=([0-9.eE+-]+)")


def parse_log(path):
    rows = {}
    if not Path(path).exists():
        return rows
    for text in Path(path).read_text(errors="replace").splitlines():
        m = STEP_RE.search(text)
        if m:
            rows[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return rows


def compare_logs(log_paths):
    a = parse_log(log_paths[0])
    b = {}
    for path in log_paths[1:]:
        for step, values in parse_log(path).items():
            b.setdefault(step, values)
    print("### per-step logged metrics")
    print("  step   A.loss       B.loss       d(loss)     A.lr         B.lr")
    worst_loss = worst_lr = 0.0
    for step in sorted(set(a) | set(b)):
        la, ra = a.get(step, (float("nan"), float("nan")))
        lb, rb = b.get(step, (float("nan"), float("nan")))
        dl = abs(la - lb)
        dr = abs(ra - rb)
        worst_loss = max(worst_loss, 0.0 if dl != dl else dl)
        worst_lr = max(worst_lr, 0.0 if dr != dr else dr)
        print(f"  {step:>4}  {la:<13.6f}  {lb:<13.6f}  {dl:<11.6f}  {ra:<12.3e}  {rb:<12.3e}")
    print(f"  max_abs_loss_diff={worst_loss:.6f} (loss is logged at 4 decimals)")
    print(f"  max_abs_lr_diff={worst_lr:.3e}")
    missing = sorted(set(a) ^ set(b))
    print(f"  steps_present_in_only_one_run={missing}")


if __name__ == "__main__":
    dir_a, dir_b = sys.argv[1], sys.argv[2]
    compare_state(dir_a, dir_b)
    compare_logs(sys.argv[3:])
