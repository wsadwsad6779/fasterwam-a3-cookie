"""P0 static cross-validation, part 1: read the DEPLOYMENT code as it is, not through my own tools.

The question this answers before any GPU time is spent: does the 40k evaluation toolchain hand the
simulator the same per-dimension command that the deployment path -- the one that produced the
earlier forceless successes -- handed it? That can only be answered from the files that were actually
deployed, so this prints them: which adapter version each entry point imports, the full body of every
`convert`, the proprio construction, and the action/state bands each dataset's own stats file carries.

No model, no GPU, no MuJoCo, no writes outside /tmp. CPU stdlib + json only.

    python3 p0_audit40k.py            # writes /tmp/p0.out
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

BRIDGE = Path("/workspace/fasterwam_a3_bridge/bridge")
DATA = Path("/workspace/a3_data")
STAGE = Path("/workspace/_v1_stage")
OUT = Path("/tmp/p0.out")

ENTRY_POINTS = ["a3_wrap.py", "infer_a3_worker.py", "run_wrapped.py", "run_oracle.py",
                "run_guarded.py", "observation_adapter.py", "gripper_proprio_adapter.py",
                "export_fasterwam_a3_input.py", "vision_adapter_official.py"]
ADAPTERS = ["action_adapter.py", "action_adapter_v2.py", "action_adapter_v3.py"]
INTEREST = re.compile(
    r"(import |from |Adapter|adapter|normaliz|denormal|clip|gripper|proprio|state\b|"
    r"dataset_stats|action_horizon|chunk|rotvec|axis_angle|quaternion|euler|deg|rad|"
    r"0\.05|0\.025|0\.1|\* *2|/ *2|def |class |step_0|\bscale\b|offset)")
FUNC_HEAD = re.compile(r"^\s*def ([A-Za-z_][A-Za-z0-9_]*)\(")


def sha8(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def head(path: Path, n: int = 4) -> list:
    out = []
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    hits = 0
    for i, ln in enumerate(lines):
        if INTEREST.search(ln) and hits < n:
            out.append("%s:%d: %s" % (path.name, i + 1, ln.strip()[:150]))
            hits += 1
    return out


def bodies(path: Path, names=None) -> list:
    """Print whole function bodies, with line numbers, for the named functions (or all)."""
    if not path.is_file():
        return ["%s: MISSING" % path]
    lines = path.read_text(errors="replace").splitlines()
    out, i = [], 0
    while i < len(lines):
        m = FUNC_HEAD.match(lines[i])
        if not m:
            i += 1
            continue
        fname = m.group(1)
        if names is None or any(n in fname for n in names):
            indent = len(lines[i]) - len(lines[i].lstrip())
            j = i + 1
            while j < len(lines):
                cur = lines[j]
                if cur.strip() and (len(cur) - len(cur.lstrip())) <= indent:
                    break
                j += 1
            out.append("--- %s:def %s  (%d..%d) ---" % (path.name, fname, i + 1, j))
            out += ["%s:%d: %s" % (path.name, k + 1, lines[k]) for k in range(i, min(j, i + 90))]
            i = j
            continue
        i += 1
    return out or ["%s: no function matched %s" % (path.name, names)]


def band_of(stats_path: Path) -> list:
    """Per-column min/max band for action and state, straight from the file the run shipped."""
    try:
        d = json.loads(stats_path.read_text())
    except Exception as exc:
        return ["  %s -> unreadable: %s" % (stats_path, str(exc)[:120])]
    out = ["  %s  sha=%s" % (stats_path, sha8(stats_path))]
    for field in ("action", "state", "proprio"):
        if field not in d:
            continue
        found = _num_lists(d[field], "")
        if not found:
            out.append("    %s: no numeric vector (keys=%s)" % (field, sorted(d[field])[:8]))
        for name, vec in sorted(found.items()):
            tag = "MIN" if "min" in name.lower() else ("MAX" if "max" in name.lower() else "")
            out.append("    %s.%s%s n=%d first8=%s"
                       % (field, name, (" <-" + tag) if tag else "", len(vec),
                          [round(float(x), 6) for x in vec[:8]]))
    return out


def _num_lists(node, prefix, depth=0, acc=None):
    """Collect every numeric vector in a stats blob, whatever nesting the writer used."""
    acc = {} if acc is None else acc
    if depth > 4:
        return acc
    if isinstance(node, list) and len(node) >= 2 and all(isinstance(x, (int, float)) for x in node):
        acc[prefix] = node
    elif isinstance(node, dict):
        for k, v in sorted(node.items(), key=lambda kv: str(kv[0])):
            _num_lists(v, "%s.%s" % (prefix, k) if prefix else str(k), depth + 1, acc)
    return acc


def main() -> None:
    L = []
    w = L.append
    w("== A. the deployment tree, with the bytes this report describes ==")
    if BRIDGE.is_dir():
        for p in sorted(BRIDGE.iterdir()):
            if p.is_file() and p.suffix == ".py":
                st = p.stat()
                w("  %-40s %9d B  mtime=%d  sha=%s"
                  % (p.name, st.st_size, int(st.st_mtime), sha8(p)))
        nxt = BRIDGE / "_next"
        w("  -- _next/ --")
        if nxt.is_dir():
            for p in sorted(nxt.iterdir()):
                w("     %-36s %s %d B sha=%s"
                  % (p.name, "d" if p.is_dir() else "f",
                     p.stat().st_size if p.is_file() else -1,
                     sha8(p) if p.is_file() and p.suffix == ".py" else "-"))
    else:
        w("  MISSING %s" % BRIDGE)

    w("\n== B. what each entry point imports and builds ==")
    for name in ENTRY_POINTS + ADAPTERS:
        p = BRIDGE / name
        if not p.is_file():
            w("  %s: MISSING" % name)
            continue
        w("--- %s (%d B, sha=%s) ---" % (name, p.stat().st_size, sha8(p)))
        for line in head(p, 26):
            w("  " + line)

    w("\n== C. the action math: whole convert/__call__ bodies ==")
    for name in ADAPTERS + ["a3_wrap.py", "infer_a3_worker.py", "observation_adapter.py"]:
        p = BRIDGE / name
        if p.is_file():
            for line in bodies(p, ["convert", "to_env", "__call__", "denormal", "transform"]):
                w(line[:200])

    w("\n== D. the observation/proprio math: whole bodies ==")
    for name in ["observation_adapter.py", "gripper_proprio_adapter.py", "a3_wrap.py",
                 "infer_a3_worker.py", "export_fasterwam_a3_input.py"]:
        p = BRIDGE / name
        if p.is_file():
            for line in bodies(p, ["proprio", "observation", "state", "gripper", "force"]):
                w(line[:200])

    w("\n== E. every dataset_stats.json under a3_data, per-column band ==")
    for p in sorted(DATA.glob("*/dataset_stats.json")):
        for line in band_of(p):
            w(line)

    w("\n== F. the run40k config this evaluation loads (action/proprio shape + norm) ==")
    cfg = STAGE / "run40k" / "config.yaml"
    if cfg.is_file():
        keep = re.compile(r"(norm|action|proprio|state|shape|raw_shape|min|max|transform|"
                          r"task|data|batch_size|max_steps|horizon|gripper)")
        for i, ln in enumerate(cfg.read_text(errors="replace").splitlines()):
            if keep.search(ln):
                w("  config.yaml:%d: %s" % (i + 1, ln.rstrip()[:160]))
    else:
        w("  MISSING %s" % cfg)

    OUT.write_text("\n".join(L) + "\n")
    print("wrote %s (%d B, %d lines)" % (OUT, OUT.stat().st_size, len(L)))


if __name__ == "__main__":
    main()
