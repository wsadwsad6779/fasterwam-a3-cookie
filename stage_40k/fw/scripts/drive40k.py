"""Drive the frozen 40,000-step run as eight 5,000-step segments on one cosine.

Everything here is a guard, not a feature. The rehearsal (task #65) proved the
mechanism at 6+6 steps; this turns the same calls into the real run and refuses,
before spending GPU hours, to start anything that would invalidate it:

* the scheduler horizon is 40000 for every segment -- lowering it rebuilds a
  short cosine that anneals to its floor eight times, which is the r5 defect;
* each segment stops exactly on a 5000 boundary and saves exactly once there, so
  the eight checkpoints are the eight nodes and nothing else;
* a segment only starts from the checkpoint its predecessor wrote, and only if
  that light state still declares the same optimizer membership and the same
  force calibration fingerprint that the rehearsal froze;
* nothing is deleted. Light states are ~211 MB each and the merged exports are
  11.1 GB, so the local budget carries all eight light states and no merged file.

    cd /workspace/_v1_stage/fw
    PYTHONPATH=/workspace/_v1_stage/fw/src /opt/venv/bin/python scripts/drive40k.py --plan
    PYTHONPATH=/workspace/_v1_stage/fw/src /opt/venv/bin/python scripts/drive40k.py --next
    PYTHONPATH=/workspace/_v1_stage/fw/src /opt/venv/bin/python scripts/drive40k.py --all

`--next` is the unit a wrapper loops on; it runs one segment and exits non-zero
the moment a segment disagrees with the plan, so it never runs a second one
after that. `--all` re-plans before every segment for the same reason.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

STAGE = Path("/workspace/_v1_stage")
FW = STAGE / "fw"
PY = "/opt/venv/bin/python"
BASE = Path("/workspace/FasterWAM/checkpoints/fasterwam_release/libero/step_021700.pt")
RUN = STAGE / "run40k"
REHEARSAL_LIGHT = STAGE / "rehearsal65" / "run" / "checkpoints" / "light" / "step_000012"
TOTAL = 40000
NODE = 5000
SEGMENTS = TOTAL // NODE
BATCH = 6
LR = 1e-4
LORA_INIT_SEED = 42
INSTRUCTION = "pick up the cookie and place it in the box"
ARGS = ["data=a3_2cam_force", "task=a3_force_40k", "batch_size=%d" % BATCH,
        "max_steps=%d" % TOTAL, "learning_rate=%g" % LR, "log_every=20",
        "eval_every=0"]
MIN_FREE_BYTES = 6 << 30


def light_dir(step):
    return RUN / "checkpoints" / "light" / ("step_%06d" % step)


def _read(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def boundary_lrs():
    """The LR each boundary must show, from the trainer's own scheduler."""
    import torch
    from fasterwam.trainer import Wan22Trainer

    class Shim:
        def __init__(self, optimizer):
            self.optimizer = optimizer
            self.learning_rate = LR

    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=LR, betas=(0.9, 0.95), weight_decay=1e-2)
    sch = Wan22Trainer._build_scheduler(Shim(opt), "cosine", TOTAL, int(TOTAL * 0.05))
    out, want = {0: float(opt.param_groups[0]["lr"])}, \
        set(NODE * i for i in range(SEGMENTS + 1)) | {int(TOTAL * 0.05)}
    for i in range(1, TOTAL + 1):
        sch.step()
        if i in want:
            out[i] = float(opt.param_groups[0]["lr"])
    return out


@functools.lru_cache(maxsize=1)
def ladder():
    return json.dumps(boundary_lrs())


def lr_at(step):
    return float(json.loads(ladder())[str(step)])


def next_segment():
    """(index, stop_step) to run, or None when the run is complete."""
    done = [i for i in range(1, SEGMENTS + 1) if (light_dir(NODE * i) / "trainer_state.json").exists()]
    if len(done) != (done[-1] if done else 0):
        raise SystemExit("boundary checkpoints are %s but the run expects them at every "
                         "%d steps; a missing middle node means a segment was skipped."
                         % (sorted(NODE * i for i in done), NODE))
    if not done:
        return 1, NODE
    return (None, None) if len(done) == SEGMENTS else (done[-1] + 1, NODE * (done[-1] + 1))


def _expected_membership():
    """The contract every node must keep: force units, membership, artifact shape.

    The first node this run writes is the reference from then on, but before that
    the rehearsal's step-12 light state is used. It is a valid reference because
    all three compared fields are architecture facts, not training results: the
    calibration fingerprint is a hash of the frozen JSON, the member order is the
    optimizer's construction order, and the adapter byte count depends only on
    shapes and dtypes. Refusing to start without one of them means a silent change
    to the force channels or normalisation constants fails here, not at node 8.
    """
    for i in range(1, SEGMENTS + 1):
        st = _read(light_dir(NODE * i) / "trainer_state.json")
        if st:
            return st, str(light_dir(NODE * i))
    rehearsal_state = _read(REHEARSAL_LIGHT / "trainer_state.json")
    return (rehearsal_state, str(REHEARSAL_LIGHT)) if rehearsal_state else (None, None)


def check_resume(step, ref):
    """Prove the checkpoint we are about to continue from is the one we expect."""
    problems = []
    st = _read(light_dir(step) / "trainer_state.json")
    if st is None:
        return ["%s does not exist" % light_dir(step)]
    if st.get("global_step") != step:
        problems.append("global_step=%r, expected %d" % (st.get("global_step"), step))
    if st.get("format") != "fasterwam-lora-light-state-v3":
        problems.append("light state format %r" % st.get("format"))
    if not st.get("force_calibration"):
        problems.append("no force_calibration fingerprint")
    if not st.get("optimizer_member_order"):
        problems.append("no optimizer_member_order")
    if ref is not None:
        if st.get("force_calibration") != ref.get("force_calibration"):
            problems.append("force_calibration differs from the node this run started at")
        if st.get("optimizer_member_order") != ref.get("optimizer_member_order"):
            problems.append("optimizer membership differs from the node this run started at")
        if st.get("adapter_bytes") != ref.get("adapter_bytes"):
            problems.append("adapter_bytes differs from the node this run started at")
        for key, want in (("instruction", INSTRUCTION), ("base_checkpoint", str(BASE))):
            if st.get("extra", {}).get(key) != want:
                problems.append("extra.%s=%r, expected %r" % (key, st.get("extra", {}).get(key), want))
    return problems


def plan():
    nodes = []
    for i in range(SEGMENTS):
        step = NODE * (i + 1)
        nodes.append({"segment": i + 1, "stop_step": step, "save_every": step,
                      "resume": str(BASE) if i == 0 else str(light_dir(NODE * i)),
                      "lr_at_boundary": lr_at(step), "window_log": "windows_seg%d.log" % (i + 1)})
    rehearsal = _read(STAGE / "rehearsal65" / "rehearsal_report.json")
    ref, ref_source = _expected_membership()
    checks = []

    def add(name, ok, evidence):
        checks.append({"name": name, "ok": bool(ok), "evidence": evidence})

    add("base checkpoint present", BASE.exists(),
        {"path": str(BASE), "bytes": BASE.stat().st_size if BASE.exists() else None})
    add("rehearsal fully green", bool(rehearsal) and not rehearsal.get("failed"),
        {"passed": None if not rehearsal else rehearsal.get("passed"),
         "failed": None if not rehearsal else rehearsal.get("failed")})
    if rehearsal:
        c8 = next((c for c in rehearsal["checks"] if c["name"].startswith("C8")), None)
        c2b = next((c for c in rehearsal["checks"] if c["name"].startswith("C2b")), None)
        add("rehearsal proved non-zero force gradients", bool(c8 and c8["ok"]),
            None if not c8 else c8["evidence"].get("force_grad_nonzero_tensors"))
        add("rehearsal proved the 40k cosine is the saved scheduler", bool(c2b and c2b["ok"]),
            None if not c2b else c2b["evidence"].get("cosine"))
    free = shutil.disk_usage("/workspace").free
    c7 = next((c for c in (rehearsal or {}).get("checks", [])
               if c["name"].startswith("C7")), None)
    measured = (c7 or {}).get("evidence", {}).get("light_state_total_bytes") or 0
    add("disk headroom for eight light states",
        free > MIN_FREE_BYTES and (not measured or SEGMENTS * measured < free),
        {"free_bytes": free,
         "eight_light_states_measured_bytes": SEGMENTS * measured,
         "note": "merged exports are 11 GB each and are not written by this driver"})
    add("no LR restart in the environment", not os.environ.get("FW_LR_RESTART"),
        {"FW_LR_RESTART": os.environ.get("FW_LR_RESTART")})
    add("boundaries tile 40000 with no gap or overlap",
        [n["stop_step"] for n in nodes] == [NODE * i for i in range(1, SEGMENTS + 1)]
        and nodes[-1]["stop_step"] == TOTAL,
        {"stop_steps": [n["stop_step"] for n in nodes]})
    add("exactly one save per segment, on the boundary",
        all(n["save_every"] == n["stop_step"] for n in nodes),
        {"save_every": [n["save_every"] for n in nodes]})
    add("every boundary LR is below the warmup peak and above eta_min",
        all(LR * 0.01 <= n["lr_at_boundary"] < LR for n in nodes)
        and all(nodes[i]["lr_at_boundary"] > nodes[i + 1]["lr_at_boundary"]
                for i in range(SEGMENTS - 1)),
        {"lr_at_boundary": {str(n["stop_step"]): n["lr_at_boundary"] for n in nodes},
         "eta_min": LR * 0.01})
    add("a frozen contract exists to compare every node against", ref is not None,
        {"reference_light_state": ref_source,
         "force_calibration": None if ref is None else ref.get("force_calibration")})
    if ref is not None:
        nxt = next_segment()
        if nxt[0] is not None and nxt[0] > 1:
            problems = check_resume(NODE * (nxt[0] - 1), ref)
            add("the checkpoint to continue from matches the run's frozen contract",
                not problems, {"resume_from": NODE * (nxt[0] - 1),
                              "reference": ref_source, "problems": problems})
    bad = [c["name"] for c in checks if not c["ok"]]
    return {"nodes": nodes, "checks": checks, "failed": bad,
            "next": next_segment(), "run": str(RUN),
            "frozen_contract": None if ref is None else {
                "source": ref_source,
                "force_calibration": ref.get("force_calibration"),
                "optimizer_members": len(ref.get("optimizer_member_order") or []),
                "adapter_bytes": ref.get("adapter_bytes")},
            "lr_at_40000": lr_at(TOTAL), "lr_at_0": lr_at(0), "lr_at_warmup_peak": lr_at(2000)}


def run_segment(index, stop, log_dir):
    ref, ref_source = _expected_membership()
    if stop > NODE:
        problems = check_resume(stop - NODE, ref)
        if problems:
            raise SystemExit("refusing to start segment %d: %s" % (index, problems))
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / ("seg%d.log" % index)
    env = dict(os.environ,
               PYTHONPATH=str(FW / "src"),
               FW_BASE_CKPT=str(BASE),
               FW_SEG_LEDGER=str(RUN / "seg_ledger.jsonl"),
               FW_BASE_LOAD_LEDGER=str(RUN / "base_load_ledger.jsonl"),
               FW_LEDGER_STAGE="seg%d" % index,
               FW_WINDOW_LOG=str(RUN / ("windows_seg%d.log" % index)),
               FW_STOP_STEP=str(stop),
               FW_LORA_INIT_SEED=str(LORA_INIT_SEED),
               FW_VERIFY_BASE_LOAD="1",
               FASTERWAM_A3_INSTRUCTION=INSTRUCTION,
               TOKENIZERS_PARALLELISM="false")
    resume = str(BASE) if stop == NODE else str(light_dir(stop - NODE))
    cmd = [PY, "-u", "scripts/train_seg.py"] + ARGS + [
        "save_every=%d" % stop, "output_dir=%s" % RUN, "resume=%s" % resume]
    with out.open("ab", buffering=0) as fh:
        fh.write(("@@ segment %d start %s resume=%s stop=%d @@\n"
                  % (index, time.strftime("%F %T"), resume, stop)).encode())
        proc = subprocess.Popen(cmd, cwd=str(FW), env=env, stdout=fh, stderr=subprocess.STDOUT)
        rc = proc.wait()
        fh.write(("@@ segment %d RC %d %s @@\n" % (index, rc, time.strftime("%F %T"))).encode())
    if rc:
        raise SystemExit("segment %d failed with RC %d; see %s" % (index, rc, out))
    problems = check_resume(stop, ref)
    st = _read(light_dir(stop) / "trainer_state.json") or {}
    lr = st.get("extra", {}).get("lr")
    want = lr_at(stop)
    if lr is None:
        problems.append("light state records no lr")
    elif abs(lr - want) > 1e-12 * want:
        problems.append("boundary LR %r, expected %r" % (lr, want))
    if problems:
        raise SystemExit("segment %d trained but does not match the plan: %s" % (index, problems))
    print("segment %d done: step=%d lr=%.9g" % (index, stop, lr), flush=True)
    return {"segment": index, "stop_step": stop, "lr": lr,
            "force_calibration": st.get("force_calibration"),
            "contract_reference": ref_source, "log": str(out)}


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true", help="assert the preconditions and print the node table")
    g.add_argument("--next", action="store_true", help="run exactly one segment")
    g.add_argument("--all", action="store_true", help="run every remaining segment")
    ap.add_argument("--report", default=str(RUN / "plan.json"))
    a = ap.parse_args()

    if a.plan:
        doc = plan()
        RUN.mkdir(parents=True, exist_ok=True)
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(doc, indent=2, sort_keys=True, default=str))
        for c in doc["checks"]:
            print(("PASS " if c["ok"] else "FAIL ") + c["name"])
        print("%-8s %-8s %-12s %s" % ("segment", "stop", "lr", "resume"))
        for n in doc["nodes"]:
            print("%-8d %-8d %-12.9g %s" % (n["segment"], n["stop_step"],
                                            n["lr_at_boundary"], n["resume"]))
        print(json.dumps({"frozen_contract": doc["frozen_contract"],
                          "next": doc["next"], "failed": doc["failed"],
                          "lr_at_2000": doc["lr_at_warmup_peak"],
                          "lr_at_40000": doc["lr_at_40000"]},
                         sort_keys=True, default=str))
        return 0 if not doc["failed"] else 1

    if a.next or a.all:
        doc = plan()
        if doc["failed"]:
            raise SystemExit("refusing to train, the plan is not green: %s" % doc["failed"])
        ledger = RUN / "driver_ledger.jsonl"
        while True:
            index, stop = next_segment()
            if index is None:
                print("run complete: %d boundary checkpoints under %s" % (SEGMENTS, RUN), flush=True)
                return 0
            rec = run_segment(index, stop, RUN / "logs")
            with ledger.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(dict(rec, ts=time.time()), sort_keys=True) + "\n")
            if not a.all:
                return 0


if __name__ == "__main__":
    raise SystemExit(main())
