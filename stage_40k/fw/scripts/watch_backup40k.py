"""Upload every 40k node to Hugging Face as soon as it lands, and prove it came back.

Why a watcher instead of a hook in the driver: `drive40k.py --all` is already running and
Python holds its module in memory, so editing the driver now would change nothing about
the live run and everything about the next accident. This loop is a separate process, so
adding a node to the backup plan costs the training nothing -- it never touches the GPU,
never imports the model, and spawns one fresh `backup40k.py` per node so a segfault in
torch cannot take the watcher down with it.

The rules it exists to enforce:

* only a *stable* node is uploaded. The trainer writes each file through a `.tmp` and
  renames, but the directory as a whole is not atomic, so a node must show identical
  (name, size) pairs on two polls a minute apart before anything leaves the box.
* a failed backup is recorded and retried, never worked around. Nothing here deletes,
  moves, or rewrites a local checkpoint: the local copy is the thing training resumes
  from, and the remote copy only ever earns cleanup rights by passing --deep.
* it does not exit. A watcher that quits on the first network hiccup turns a 30-second
  blip into an unbacked 5,000 steps.

    cd /workspace/_v1_stage/fw
    mkdir -p /workspace/_v1_stage/backup40k
    setsid nohup env PYTHONPATH=/workspace/_v1_stage/fw/src HF_TOKEN_FILE=/root/.cache/hf_token \
        /opt/venv/bin/python -u scripts/watch_backup40k.py --loop > \
        /workspace/_v1_stage/backup40k/watch.log 2>&1 &

    # -u matters: the log is the only live view of a loop that is meant to run for two days,
    # and a block-buffered one looks exactly like a dead one.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive40k as plan  # noqa: E402
import backup40k as bk  # noqa: E402

FW = plan.FW
POLL_SECONDS = 60
STABLE_POLLS = 2
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 300
# A Xet dedup stall looks exactly like a slow upload, and this transfer path is the one the
# proxy leaves usable. Without a wall clock the watcher would sit on node 1 forever while
# nodes 2..8 accumulated on a disk that holds three. 211 MB measures ~2-3 min at 1.9 MB/s,
# so an hour is fifteen times the observed cost and still far shorter than a lost segment.
ATTEMPT_TIMEOUT_SECONDS = 3600


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def heartbeat(payload):
    bk.BACKUP.mkdir(parents=True, exist_ok=True)
    (bk.BACKUP / "watch_heartbeat.json").write_text(
        json.dumps(dict(payload, t=_now()), indent=2, sort_keys=True, default=str))


def ledger_lines():
    path = bk.BACKUP / "ledger.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:  # noqa: BLE001
            rows.append({"unparsed": line[:120]})
    return rows


def verified_deep_steps():
    """Steps whose report says every check passed, deep included.

    A --verify-only run says nothing about whether the bytes ever left the box, so it can
    never mark a node as backed up.
    """
    good = set()
    for line in ledger_lines():
        if line.get("deep") and not line.get("verify_only") and not line.get("failed"):
            good.add(int(line["step"]))
    return good


def node_dirs():
    root = plan.RUN / "checkpoints" / "light"
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and p.name.startswith("step_"):
            out.append((int(p.name.split("_")[1]), p))
    return out


def signature(node_dir):
    return sorted((p.name, p.stat().st_size) for p in node_dir.iterdir()
                  if p.is_file() and not p.name.endswith(".tmp"))


def wait_stable(step, node_dir, log):
    """A node is only uploadable when two polls a minute apart disagree about nothing."""
    prev = None
    stable = 0
    while stable < STABLE_POLLS:
        if not (node_dir / "trainer_state.json").exists():
            log("step=%d waiting for trainer_state.json" % step)
            prev = None
            stable = 0
            time.sleep(POLL_SECONDS)
            continue
        sig = signature(node_dir)
        if prev is not None and sig == prev:
            stable += 1
            log("step=%d signature stable %d/%d (%d files, %d B)"
                % (step, stable, STABLE_POLLS, len(sig), sum(n[1] for n in sig)))
        else:
            if prev is not None:
                log("step=%d signature changed, restarting the stability count" % step)
            stable = 0
        prev = sig
        if stable < STABLE_POLLS:
            time.sleep(POLL_SECONDS)
    return prev


def run_backup(step, log):
    cmd = [sys.executable, str(FW / "scripts" / "backup40k.py"), "--step", str(step),
           "--deep"]
    if step == plan.NODE:
        # The task only demands the first-node proof, and it is the one place where it pays:
        # if the 11 GB base is not independently fetchable, every node in the repo is
        # unmergeable off this instance and we want to know at node 1, not at node 8.
        cmd.append("--check-base")
    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(FW / "src"))
    env.setdefault("HF_TOKEN_FILE", bk.TOKEN_FILE_DEFAULT)
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(FW), env=env, capture_output=True, text=True,
                              timeout=ATTEMPT_TIMEOUT_SECONDS)
        rc = proc.returncode
        out_tail = (proc.stdout or "")[-1500:]
        err_tail = (proc.stderr or "")[-800:]
    except subprocess.TimeoutExpired as exc:
        rc = 124
        out_tail = ((exc.stdout or b"") if isinstance(exc.stdout, bytes)
                    else (exc.stdout or ""))[-1500:]
        err_tail = "TIMEOUT after %ds -- treated as a stalled transfer, attempt will retry" % (
            ATTEMPT_TIMEOUT_SECONDS)
    secs = round(time.time() - started, 1)
    log("step=%d backup rc=%d in %.1fs" % (step, rc, secs))
    with open(bk.BACKUP / ("watch_step%06d.log" % step), "a", encoding="utf-8") as fh:
        fh.write("\n=== %s rc=%d secs=%.1f ===\n" % (_now(), rc, secs))
        fh.write(out_tail + "\n--- stderr ---\n" + err_tail + "\n")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--once-step", type=int, default=0)
    ap.add_argument("--max-iterations", type=int, default=0, help="0 means forever")
    a = ap.parse_args()
    bk.BACKUP.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = "%s %s" % (_now(), msg)
        print(line, flush=True)
        with open(bk.BACKUP / "watch.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": _now(), "msg": msg}) + "\n")

    if a.once_step:
        return run_backup(a.once_step, log)

    lock = bk.BACKUP / "watch.lock"
    if lock.exists():
        try:
            other = int(lock.read_text().strip())
            os.kill(other, 0)
            raise SystemExit("another watcher is alive (pid %d); refusing to double-upload"
                             % other)
        except (ValueError, ProcessLookupError, PermissionError):
            lock.write_text(str(os.getpid()))
    else:
        lock.write_text(str(os.getpid()))

    iterations = 0
    while True:
        iterations += 1
        done = verified_deep_steps()
        pending = [(s, d) for s, d in node_dirs() if s not in done
                   and s % plan.NODE == 0 and 0 < s <= plan.TOTAL]
        heartbeat({"state": "idle" if not pending else "pending",
                   "pending": [s for s, _ in pending], "deep_verified": sorted(done),
                   "nodes_seen": [s for s, _ in node_dirs()], "pid": os.getpid()})
        if not pending:
            log("idle: deep-verified=%s nodes_seen=%s" % (sorted(done), [s for s, _ in node_dirs()]))
        for step, node_dir in pending:
            log("node step=%d queued for backup" % step)
            wait_stable(step, node_dir, log)
            ok = False
            for attempt in range(1, MAX_ATTEMPTS + 1):
                rc = run_backup(step, log)
                if rc == 0:
                    ok = True
                    break
                log("step=%d attempt %d/%d failed (rc=%d); local checkpoint untouched"
                    % (step, attempt, MAX_ATTEMPTS, rc))
                time.sleep(BACKOFF_SECONDS)
            log("step=%d backup %s" % (step, "VERIFIED" if ok else "STILL FAILING"))
        if a.max_iterations and iterations >= a.max_iterations:
            return 0
        if not a.loop:
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
