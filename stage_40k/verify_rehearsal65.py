"""Task #65 stage 6: read the real trainer's own records and decide pass/fail.

Every number here comes from a process that ran `scripts/train_seg.py` against
the real 11 GB base, the real corpus and the real 30-layer action expert. Nothing
is re-simulated: the reference learning-rate ladder is produced by the trainer's
own `_build_scheduler`, and the sampler order is read out of the log the patched
sampler writes by itself.

The harness has three rules it will not break, because each of them hid a real
defect once:

* A check never compares two racy tails. The sampler log records windows as they
  are *emitted*, which runs ahead of what was trained, so continuity is asserted
  over the overlap and only over an overlap long enough to cover every trained
  window.
* A check never counts events that other stages also write. The base-load ledger
  is appended to by the training segments, the exporter and the deployment
  reload, so events are scoped by the stage that recorded them.
* A check never relies on a field being merely present. `loss_force` has to be
  found in the trainer's own log records, which rich wraps across lines, so the
  records are reassembled before they are parsed.

Checks
  C1  segment chain        -- stop steps 6 then 12, one 40k horizon, the resumed
                              segment starts exactly where its predecessor ended
  C2  LR continuity        -- ledger and trainer-reported LR match an uninterrupted
                              cosine; bitwise for the ledger, to log precision for
                              the trainer
  C2b scheduler state      -- the saved state is the 40k curve at step 6, not a
                              rebuilt short one (the r5 defect)
  C3  counters             -- global_step / epoch / batch_in_epoch / provenance
  C3b force calibration    -- fingerprint written by the real run, stable across saves
  C4  base load            -- proprio 8->14 expansion and every base tensor verified
                              elementwise, in both training segments
  C5  sampler continuity   -- segment 2 resumes inside segment 1's permutation and
                              no trained window repeats
  C6  loss wiring          -- loss_force is present, finite and non-constant in every
                              real step of both segments
  C7  disk budget          -- measured size of both artifact classes x 8 nodes
  C8  gradient signal      -- the force modules are optimizer members, the stored
                              Adam moments prove their gradients were non-zero, and
                              both saves carry them
  C9  force seam           -- the dataset-to-trainer force report is green
  C10 deployment reload    -- the publication verify report is green
  C11 cosine shape         -- one curve: warmup peak, the eight 5k boundaries, and
                              eta_min at 40000
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

BATCH = 6
SEG1_STOP = 6
SEG2_STOP = 12
TOTAL = 40000
WARMUP = 2000
BASE_LR = 1e-4
FORCE_NAME_MARK = "force_conditioner"
TRAIN_STAGES = ("seg1", "seg2")
LOG_KEYS = ("loss", "loss_action", "loss_video", "loss_force", "lr", "grad_norm",
            "force_valid_step_frac", "force_weight_mean")


class _Shim:
    """Just enough trainer for the real `_build_scheduler` to run."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.learning_rate = BASE_LR


def _reference_ladder():
    """The LR the trainer would use at every step of one uninterrupted 40k run."""
    import torch
    from fasterwam.trainer import Wan22Trainer

    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=BASE_LR, betas=(0.9, 0.95), weight_decay=1e-2)
    sch = Wan22Trainer._build_scheduler(_Shim(opt), "cosine", TOTAL, int(TOTAL * 0.05))
    ladder = [float(opt.param_groups[0]["lr"])]
    for _ in range(TOTAL):
        sch.step()
        ladder.append(float(opt.param_groups[0]["lr"]))
    return ladder


def _read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _windows(path: Path):
    out = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        out.append((int(parts[0]), int(parts[1])))
    return out


def _records(path: Path):
    """Reassemble rich's wrapped log lines into one string per log record.

    rich prints `| >>` once, at the head of a record, and folds the rest onto
    following blank-indented lines, so a per-line scan loses every field that
    did not fit the first 60 columns -- including `loss_force` and `lr`.
    """
    if not path.exists():
        return []
    out, cur = [], None
    for raw in path.read_text(errors="replace").splitlines():
        if "| >>" in raw:
            if cur is not None:
                out.append(cur)
            cur = raw
        elif cur is not None:
            cur = cur + " " + raw
    if cur is not None:
        out.append(cur)
    return [re.sub(r"\s+", " ", rec) for rec in out]


def _train_lines(path: Path):
    kv = re.compile(r"\b([a-z_][a-z0-9_]*)=(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
    rows = []
    for rec in _records(path):
        head = re.search(r"\bstep=(\d+)/(\d+)\b", rec)
        if head is None or "loss=" not in rec or "trainer.py" not in rec:
            continue
        values = {k: float(v) for k, v in kv.findall(rec) if k in LOG_KEYS}
        if "loss" not in values:
            continue
        rows.append({"step": int(head.group(1)), "of": int(head.group(2)),
                     "values": values})
    return rows


def _force_names(payload):
    blob = dict(payload.get("lora", {}))
    blob.update(payload.get("extra_trainable", {}))
    return blob


def _maxabs(a, b):
    import torch

    diff = (a.to(torch.float32) - b.to(torch.float32)).abs()
    return float(diff.max()) if diff.numel() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--led", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    led = Path(a.led)
    run = led / "run"
    ckpt = {step: run / "checkpoints" / "light" / ("step_%06d" % step)
            for step in (SEG1_STOP, SEG2_STOP)}

    seg = _read_jsonl(led / "seg_ledger.jsonl")
    base_events = _read_jsonl(led / "base_load_ledger.jsonl")
    w1, w2 = _windows(led / "windows_seg1.log"), _windows(led / "windows_seg2.log")
    s1, s2 = _train_lines(led / "s1.log"), _train_lines(led / "s2.log")
    states = {}
    for step in (SEG1_STOP, SEG2_STOP):
        f = ckpt[step] / "trainer_state.json"
        states[step] = json.loads(f.read_text()) if f.exists() else None
    ladder = _reference_ladder()

    checks = []

    def check(name, ok, evidence):
        checks.append({"name": name, "ok": bool(ok), "evidence": evidence})
        print(("PASS " if ok else "FAIL ") + name + " :: "
              + json.dumps(evidence, sort_keys=True, default=str)[:1400])
        return ok

    starts = [e for e in seg if e.get("event") == "segment_start"]
    ends = [e for e in seg if e.get("event") == "segment_end"]

    # ---- C1 -----------------------------------------------------------------
    want = [(SEG1_STOP, 0), (SEG1_STOP, SEG1_STOP), (SEG2_STOP, SEG1_STOP), (SEG2_STOP, SEG2_STOP)]
    got = [(e.get("stop_step"), e.get("global_step")) for e in seg]
    # Within one segment the LR may legitimately climb (everything before step
    # 2000 is warmup), so direction is only meaningful against the reference curve.
    moved = []
    for st, en in zip(starts, ends):
        g0, g1 = st.get("global_step"), en.get("global_step")
        if None in (g0, g1) or not (g1 < len(ladder)):
            moved.append(None)
            continue
        ref_climbs = ladder[g1] > ladder[g0]
        moved.append(bool((en.get("lr") or 0.0) > (st.get("lr") or 0.0)) == ref_climbs)
    handoff = None
    if len(starts) > 1 and starts[1].get("global_step") == SEG1_STOP:
        handoff = (starts[1].get("prev_segment_end_step") == SEG1_STOP
                   and starts[1].get("prev_segment_end_lr") == ends[0].get("lr"))
    check("C1 segment chain on one 40k horizon, resumed exactly where it stopped",
          got == want and len(starts) == 2 and len(ends) == 2
          and all(e.get("hit_stop") for e in ends)
          and all(e.get("scheduler_total") == TOTAL for e in seg)
          and all(m is not False for m in moved)
          and handoff is True,
          {"events": got, "hit_stop": [e.get("hit_stop") for e in ends],
           "scheduler_total": sorted({e.get("scheduler_total") for e in seg}),
           "lr_direction_matches_reference": moved,
           "resume_handoff_verified": handoff,
           "seg2_start": {k: starts[1].get(k) for k in
                          ("global_step", "lr", "prev_segment_end_step",
                           "prev_segment_end_lr")} if len(starts) > 1 else None})

    # ---- C2 -----------------------------------------------------------------
    lrs = [e.get("lr") for e in seg]
    steps = [e.get("global_step") for e in seg]
    ref = [ladder[s] for s in steps if s is not None and s < len(ladder)]
    trained_logs = s1 + s2
    log_lrs = [r["values"].get("lr") for r in trained_logs]
    log_ref = []
    for r in trained_logs:
        want_lr = ladder[r["step"]]
        have = r["values"].get("lr")
        log_ref.append(have is not None
                       and abs(have - want_lr) <= max(1e-3 * want_lr, 1e-12))
    ev = {"ledger_step": steps, "ledger_lr": lrs, "reference_lr": ref,
          "ledger_matches_reference_bitwise": lrs == ref and len(lrs) == 4,
          "trainer_reported_lr": log_lrs, "trainer_log_matches_reference": log_ref,
          "note": "the log prints 3 significant figures, so it is compared to "
                  "within 0.1%; the ledger is compared to the bit"}
    check("C2 LR continuity (resumed optimizer == uninterrupted cosine)",
          lrs == ref and len(lrs) == 4 and len(log_ref) == SEG2_STOP
          and all(log_ref), ev)

    sched_state = ckpt[SEG1_STOP] / "scheduler.pt"
    import torch
    ss = torch.load(sched_state, map_location="cpu", weights_only=False) \
        if sched_state.exists() else None
    sub = (ss.get("_schedulers") or [{}])[1] if isinstance(ss, dict) and len(ss.get("_schedulers") or []) > 1 else {}
    warm = (ss.get("_schedulers") or [{}])[0] if isinstance(ss, dict) else {}
    ok2b = bool(ss) and ss.get("last_epoch") == SEG1_STOP \
        and ss.get("_milestones") == [WARMUP] \
        and sub.get("T_max") == TOTAL - WARMUP \
        and abs(float(sub.get("eta_min", -1)) - BASE_LR * 0.01) < 1e-12 \
        and warm.get("total_iters") == WARMUP \
        and abs(float((ss.get("_last_lr") or [-1])[0]) - ladder[SEG1_STOP]) < 1e-18
    check("C2b saved scheduler state is the 40k cosine advanced to step 6",
          ok2b,
          {"exists": sched_state.exists(), "last_epoch": ss.get("last_epoch"),
           "milestones": ss.get("_milestones"), "last_lr": ss.get("_last_lr"),
           "reference_lr_at_6": ladder[SEG1_STOP],
           "warmup": {k: warm.get(k) for k in ("start_factor", "end_factor", "total_iters")},
           "cosine": {k: sub.get(k) for k in ("T_max", "eta_min", "base_lrs")}})

    # ---- C3 -----------------------------------------------------------------
    st6, st12 = states[SEG1_STOP], states[SEG2_STOP]
    ev = {}
    if st6 and st12:
        ev = {"step6": {k: st6[k] for k in ("global_step", "epoch", "batch_in_epoch")},
              "step12": {k: st12[k] for k in ("global_step", "epoch", "batch_in_epoch")},
              "extra6": st6.get("extra"), "format": st6.get("format")}
    check("C3 counters continue",
          st6 and st12 and st6["global_step"] == SEG1_STOP and st6["batch_in_epoch"] == SEG1_STOP
          and st12["global_step"] == SEG2_STOP and st12["batch_in_epoch"] == SEG2_STOP
          and st6["epoch"] == st12["epoch"] == 0, ev)

    fp6 = (st6 or {}).get("force_calibration")
    fp12 = (st12 or {}).get("force_calibration")
    check("C3b force calibration fingerprint written by the real run and stable across saves",
          fp6 is not None and fp6 == fp12,
          {"step6": fp6, "step12": fp12})

    # ---- C4 -----------------------------------------------------------------
    by_stage = {}
    for e in base_events:
        by_stage.setdefault(str(e.get("stage", "unlabelled")), []).append(e)
    train = {st: by_stage.get(st, []) for st in TRAIN_STAGES}
    per_stage = {}
    for st, events in train.items():
        proprio = [e for e in events if e.get("event") == "proprio_load"]
        mot = [e for e in events if e.get("event") == "mot_load"]
        per_stage[st] = {
            "proprio": proprio, "mot": mot,
            "proprio_expanded_verified":
                len(proprio) == 1 and all(
                    e.get("action") == "expanded_zero_filled"
                    and e.get("src_in") == 8 and e.get("target_in") == 14
                    and e.get("zero_columns_verified") and e.get("copied_columns_verified")
                    and e.get("bias_verified") for e in proprio),
            "mot_verified":
                len(mot) == 1 and all(
                    e.get("absent") == 0 and e.get("shape_mismatch") == 0
                    and e.get("value_verified") and e.get("verified_elements", 0) > 0
                    for e in mot),
        }
    later = {st: [e for e in by_stage.get(st, []) if e.get("event") == "mot_load"]
             for st in ("export", "publish_verify")}
    check("C4a each training segment expanded proprio 8->14 and verified it",
          all(v["proprio_expanded_verified"] for v in per_stage.values())
          and len(per_stage) == len(TRAIN_STAGES),
          {"stages": sorted(by_stage),
           "events_per_stage": {st: len(evs) for st, evs in sorted(by_stage.items())},
           "detail": {st: v["proprio"] for st, v in sorted(per_stage.items())}})
    check("C4b every base tensor verified elementwise in each training segment",
          all(v["mot_verified"] for v in per_stage.values()),
          {"detail": {st: [{"path": e.get("path"),
                            "checkpoint_tensors": e.get("checkpoint_tensors"),
                            "model_tensors": e.get("model_tensors"),
                            "verified_elements": e.get("verified_elements")}
                           for e in v["mot"]] for st, v in sorted(per_stage.items())},
           "reload_stages": {st: len(evs) for st, evs in sorted(later.items())}})
    check("C4c the exporter and the deployment reload also verified every tensor they read",
          all(len(evs) >= 1 and all(e.get("absent") == 0 and e.get("shape_mismatch") == 0
                                    and e.get("value_verified")
                                    and e.get("verified_elements", 0) > 0 for e in evs)
              for evs in later.values()),
          {st: [{"path": e.get("path"), "checkpoint_tensors": e.get("checkpoint_tensors"),
                 "model_tensors": e.get("model_tensors"),
                 "verified_elements": e.get("verified_elements")} for e in evs]
           for st, evs in sorted(later.items())})

    # ---- C5 -----------------------------------------------------------------
    n1 = BATCH * SEG1_STOP
    n2 = BATCH * (SEG2_STOP - SEG1_STOP)
    trained1 = [i for _, i in w1[:n1]]
    trained2 = [i for _, i in w2[:n2]]
    tail = w1[n1:]
    n_cmp = min(len(tail), len(w2))
    cont = n_cmp >= n2 and w2[:n_cmp] == tail[:n_cmp]
    check("C5a sampler pass is 0 in both segments",
          all(p == 0 for p, _ in w1) and all(p == 0 for p, _ in w2),
          {"seg1_passes": sorted({p for p, _ in w1}), "seg2_passes": sorted({p for p, _ in w2}),
           "seg1_emitted": len(w1), "seg2_emitted": len(w2)})
    check("C5b segment 2 resumes inside segment 1's permutation at position 36",
          cont,
          {"compared": n_cmp, "must_cover_at_least": n2,
           "seg2_head_equals_seg1_from_36": w2[:n_cmp] == tail[:n_cmp],
           "seg2_head": w2[:6], "seg1_at_36": tail[:6],
           "note": "the log records emission order, which runs ahead of training, so "
                   "only the overlap is compared -- and only if it covers every window "
                   "segment 2 actually trained"})
    check("C5c no trained window repeats across the resume",
          len(set(trained1) & set(trained2)) == 0
          and len(trained1) == n1 and len(set(trained1)) == n1
          and len(trained2) == n2 and len(set(trained2)) == n2,
          {"trained_seg1": len(trained1), "trained_seg2": len(trained2),
           "intersection": sorted(set(trained1) & set(trained2))[:10],
           "note": "prefetch overlap is allowed; trained batches are what must not repeat"})

    # ---- C6 -----------------------------------------------------------------
    forces = [r["values"].get("loss_force") for r in (s1 + s2)]
    finite = all(f is not None and f == f and abs(f) != float("inf") for f in forces)
    per_seg = {"seg1": [r["values"].get("loss_force") for r in s1],
               "seg2": [r["values"].get("loss_force") for r in s2]}
    check("C6 loss_force present, finite and non-constant in every real step",
          len(s1) == SEG1_STOP and len(s2) == SEG2_STOP - SEG1_STOP
          and len(forces) == SEG2_STOP and finite and len(set(forces)) > 1
          and all(r["values"].get("loss_action") is not None for r in s1 + s2)
          and all(r["values"].get("force_valid_step_frac") == 1.0 for r in s1 + s2),
          {"n_train_records": {"seg1": len(s1), "seg2": len(s2)},
           "loss_force_per_step": forces,
           "keys_seen": sorted({k for r in (s1 + s2) for k in r["values"]}),
           "first_record_values": s1[0]["values"] if s1 else None,
           "per_segment": per_seg,
           "loss_action_first": s1[0]["values"].get("loss_action") if s1 else None})

    # ---- C7 -----------------------------------------------------------------
    sizes = {}
    sp = led / "sizes.txt"
    if sp.exists():
        for line in sp.read_text(errors="replace").splitlines():
            m = re.match(r"^(\d+)\t(/.*)$", line)
            if m:
                sizes[m.group(2)] = int(m.group(1))
    per_light = {}
    for step in (SEG1_STOP, SEG2_STOP):
        d = ckpt[step]
        per_light["step_%06d" % step] = {p.name: sizes.get(str(p), None)
                                         for p in sorted(d.glob("*")) if p.is_file()}
    light_total = max([sum(v for v in d.values() if v) for d in per_light.values()] or [0])
    merged_total = 0
    pub = led / "publish_record.json"
    if pub.exists():
        merged_total = int(json.loads(pub.read_text()).get("output_bytes", 0))
    free = shutil.disk_usage("/workspace").free
    check("C7 artifact budget measured", light_total > 0 and merged_total > 0,
          {"light_state_files": per_light,
           "light_state_total_bytes": light_total,
           "merged_bytes": merged_total,
           "eight_nodes_light_bytes": light_total * 8,
           "eight_nodes_merged_bytes": merged_total * 8,
           "both_classes_bytes": (light_total + merged_total) * 8,
           "workspace_free_bytes": free,
           "fits_both_classes_locally": (light_total + merged_total) * 8 <= free,
           "fits_light_only_locally": light_total * 8 <= free})

    # ---- C8 -----------------------------------------------------------------
    import torch as _t

    order = (st6 or {}).get("optimizer_member_order")
    cov = (st6 or {}).get("coverage", {})
    adapters = {}
    for step in (SEG1_STOP, SEG2_STOP):
        f = ckpt[step] / "adapter.pt"
        adapters[step] = _t.load(f, map_location="cpu", weights_only=False) if f.exists() else None
    blob6 = _force_names(adapters[SEG1_STOP]) if adapters[SEG1_STOP] else {}
    blob12 = _force_names(adapters[SEG2_STOP]) if adapters[SEG2_STOP] else {}
    force6 = sorted(n for n in blob6 if FORCE_NAME_MARK in n)
    force12 = sorted(n for n in blob12 if FORCE_NAME_MARK in n)
    expect_extra = sorted(n for n in cov.get("saved_nonlora_names", [])
                          if FORCE_NAME_MARK in n)
    opt_state = None
    of = ckpt[SEG1_STOP] / "optimizer.pt"
    if of.exists():
        opt_state = _t.load(of, map_location="cpu", weights_only=False).get("state", {})
    # Index -> name is the only mapping `load_state_dict` uses, so prove it by
    # shape before using it to read the moments.
    shape_ok, shape_bad = 0, []
    if order and blob6:
        first_at = {}
        for i, n in enumerate(order):
            first_at.setdefault(n, i)
        for n, tensor in blob6.items():
            i = first_at.get(n)
            st = opt_state.get(i) if i is not None else None
            mom = (st or {}).get("exp_avg")
            if mom is not None and tuple(mom.shape) == tuple(tensor.shape):
                shape_ok += 1
            else:
                shape_bad.append((n, i, tuple(tensor.shape),
                                  None if mom is None else tuple(mom.shape)))
    first_at = {}
    if order:
        for i, n in enumerate(order):
            first_at.setdefault(n, i)
    force_slots = sorted((n, first_at.get(n)) for n in force12)
    moments = {}
    for n, i in force_slots:
        st = opt_state.get(i) if i is not None else None
        m, v = (st or {}).get("exp_avg"), (st or {}).get("exp_avg_sq")
        moments[n] = {"index": i,
                      "seen": (st or {}).get("step"),
                      "exp_avg_absmax": None if m is None else float(m.abs().max()),
                      "exp_avg_sq_absmax": None if v is None else float(v.abs().max()),
                      "count_nonzero": None if m is None else int((m != 0).sum())}
    deltas = {n: _maxabs(blob6[n], blob12[n]) for n in force12 if n in blob6}
    n_opt_force = len([n for n in (order or []) if FORCE_NAME_MARK in n])
    ev = {
        "optimizer_members_recorded": None if order is None else len(order),
        "optimizer_members_expected": cov.get("optimizer_members"),
        "named_tensors_in_model": cov.get("named_tensors"),
        "force_tensors_in_optimizer": n_opt_force,
        "force_tensors_in_adapter": {"step6": len(force6), "step12": len(force12)},
        "force_extra_trainable_from_coverage": len(expect_extra),
        "shape_verified_slots": shape_ok,
        "shape_mismatched": shape_bad[:5],
        "force_grad_nonzero_tensors": sum(1 for v in moments.values()
                                         if (v["exp_avg_absmax"] or 0) > 0),
        "force_grad_zero_tensors": sorted(n for n, v in moments.items()
                                          if not (v["exp_avg_absmax"] or 0) > 0),
        "force_step6_weight_absmax": {n: float(blob6[n].abs().max()) for n in force12[:3]},
        "force_param_sample": {k: moments[k] for k in sorted(moments)[:3]},
        "note": "warmup LR at step 6 is 3.5e-7, below one bf16 ulp, so adapter weight "
                "deltas over six steps say nothing; the stored Adam moments are the "
                "record of whether gradients were non-zero",
    }
    ok8 = (order is not None and len(order) == cov.get("optimizer_members")
           and n_opt_force == len(force12) == len(force6) == len(expect_extra) > 0
           and force6 == force12 == expect_extra
           and shape_ok == len(blob6) and not shape_bad
           and len(moments) == len(force12)
           and all((v["exp_avg_absmax"] or 0) > 0 for v in moments.values())
           and all((v["exp_avg_sq_absmax"] or 0) > 0 for v in moments.values()))
    check("C8 every force module is an optimizer member with proven non-zero gradients",
          ok8, dict(ev, weight_delta_step6_to_12=deltas))

    # ---- C9 / C10 -----------------------------------------------------------
    for name, key in (("C9 dataset force seam report is green", "dataset_force_verify.json"),
                      ("C10 deployment reload report is green", "publish_verify.json")):
        f = led / key
        doc = json.loads(f.read_text()) if f.exists() else None
        check(name, bool(doc) and not doc.get("failed"),
              {"report": str(f), "exists": f.exists(),
               "passed": None if doc is None else doc.get("passed"),
               "failed": None if doc is None else doc.get("failed"),
               "n_checks": None if doc is None else len(doc.get("checks", []))})

    # ---- C11 ----------------------------------------------------------------
    boundaries = [5000 * i for i in range(1, 9)]
    mono_after_warmup = all(ladder[i + 1] <= ladder[i] for i in range(WARMUP, TOTAL))
    peak = max(range(TOTAL + 1), key=lambda i: ladder[i])
    peak_ok = abs(ladder[peak] - BASE_LR) < 1e-9 and WARMUP - 2 <= peak <= WARMUP + 2
    check("C11 one cosine: peak at the warmup end, monotone after it, eta_min at 40000",
          peak_ok and abs(ladder[WARMUP] - BASE_LR) < 1e-9
          and mono_after_warmup
          and abs(ladder[TOTAL] - BASE_LR * 0.01) < 1e-12
          and all(ladder[b] < ladder[b - 1] for b in boundaries),
          {"lr_at_peak": ladder[peak], "peak_step": peak, "warmup_steps": WARMUP,
           "lr_at_40000": ladder[TOTAL], "eta_min_expected": BASE_LR * 0.01,
           "lr_at_5k_boundaries": {str(b): ladder[b] for b in boundaries},
           "monotone_after_warmup": mono_after_warmup})

    report = {"passed": sum(1 for c in checks if c["ok"]),
              "failed": [c["name"] for c in checks if not c["ok"]],
              "checks": checks,
              "reference_ladder_first13": ladder[: SEG2_STOP + 1],
              "lr_at_40000_reference": ladder[TOTAL],
              "warmup_steps": WARMUP,
              "train_records_parsed": {"seg1": len(s1), "seg2": len(s2)}}
    Path(a.report).write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print(json.dumps({"passed": report["passed"], "failed": report["failed"]}, indent=2))
    return 0 if not report["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
