"""One boundary, six questions, answered from files the run itself wrote.

The spec says that at the end of every 5,000-step segment these six things get checked:
global_step, scheduler, sampler, loss, the calibration fingerprint, and parameter
integrity. The driver already refuses to start a segment whose predecessor disagrees with
the plan, so this script is not the guard -- it is the *record*: it re-derives each answer
from a second source (the ledger, the node, the log, the uploaded calibration JSON, the
frozen rehearsal contract) and writes a report per boundary that says how the two sources
compared.

Read-only. Touches no network, loads nothing onto the GPU, and never writes inside a
checkpoint directory, so it is safe to run while the next segment is already training.

  python scripts/segment_check.py --step 5000
  python scripts/segment_check.py --step 12 --node-dir ... --run-dir ... --log ...
"""
import argparse
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import torch  # noqa: E402

import backup40k as bk  # noqa: E402
import drive40k as plan  # noqa: E402

REL_TOL = 1e-12
RECORD_START = re.compile(r"(?m)^\d{2}/\d{2} \[\d{2}:\d{2}:\d{2}\]")
FIELD = {
    "step": re.compile(r"step=(\d+)/(\d+)"),
    "loss": re.compile(r"(?<!_)loss=([0-9.eE+-]+)"),
    "loss_action": re.compile(r"loss_action=([0-9.eE+-]+)"),
    "loss_force": re.compile(r"loss_force=([0-9.eE+-]+)"),
    "loss_video": re.compile(r"loss_video=([0-9.eE+-]+)"),
    "lr": re.compile(r"(?<!_)lr=([0-9.eE+-]+)"),
    "force_valid_step_frac": re.compile(r"force_valid_step_frac=([0-9.eE+-]+)"),
    "force_weight_mean": re.compile(r"force_weight_mean=([0-9.eE+-]+)"),
    "speed": re.compile(r"speed=([0-9.eE+-]+) step/s"),
}
LOSSES = ("loss", "loss_action", "loss_force", "loss_video")


def fnum(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def read_json(path):
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(errors="replace"))
    except ValueError:
        return {}


def read_jsonl(path):
    if not Path(path).is_file():
        return []
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            rows.append({"_unparseable": line[:200]})
    return rows


def parse_progress(log_path):
    """Every trainer progress record in a segment log, in order.

    `rich` wraps one record over several physical lines and pads them, but it breaks at
    word boundaries, so `key=value` never splits. Records are found by their timestamp
    column and rejoined with single spaces before the fields are pulled out.
    """
    path = Path(log_path)
    if not path.is_file():
        return []
    text = path.read_text(errors="replace").replace("\r", "\n")
    starts = [m.start() for m in RECORD_START.finditer(text)]
    out = []
    for a, b in zip(starts, starts[1:] + [len(text)]):
        blob = " ".join(text[a:b].split())
        m = FIELD["step"].search(blob)
        if not m:
            continue
        rec = {"step": int(m.group(1)), "stop_step": int(m.group(2)), "wall": blob[:24]}
        for key, rx in FIELD.items():
            if key == "step":
                continue
            mm = rx.search(blob)
            rec[key] = fnum(mm.group(1)) if mm else None
        out.append(rec)
    return out


def _vals(key, rows):
    return [r[key] for r in rows if isinstance(r.get(key), float) and math.isfinite(r[key])]


def _block(key, rows):
    v = _vals(key, rows)
    return None if not v else {"n": len(v), "min": min(v), "max": max(v),
                               "mean": statistics.fmean(v)}


def summarize(recs, window=20):
    """Stats per loss field over the whole segment, plus head/tail windows."""
    tail = recs[-window:] if len(recs) >= window else recs
    head = recs[:window]
    out = {"records": len(recs),
           "steps_covered": [recs[0]["step"], recs[-1]["step"]] if recs else None,
           "non_finite_records": sum(1 for r in recs for k in LOSSES
                                     if r.get(k) is not None and not math.isfinite(r[k]))}
    for key in LOSSES + ("lr", "force_valid_step_frac", "force_weight_mean", "speed"):
        out[key] = {"whole": _block(key, recs), "head": _block(key, head),
                    "tail": _block(key, tail)}
    return out


def ratio_or_none(a, b):
    if not isinstance(a, float) or not isinstance(b, float) or b <= 0:
        return None
    return a / b


def curve(step):
    return bk.lr_at_local(int(step))


def close(a, b):
    return (isinstance(a, float) and isinstance(b, float)
            and abs(a - b) <= REL_TOL * max(abs(b), 1e-12))


def check_global_step(step, seg, node_dir, run_dir, log_override=None):
    led = [r for r in read_jsonl(run_dir / "seg_ledger.jsonl")
           if r.get("event") == "segment_end" and r.get("global_step") == step]
    drv_path = run_dir / "driver_ledger.jsonl"
    drv = [r for r in read_jsonl(drv_path) if r.get("stop_step") == step]
    st = read_json(node_dir / "trainer_state.json")
    log = Path(log_override) if log_override else run_dir / "logs" / ("seg%d.log" % seg)
    prog = parse_progress(log)
    last = prog[-1]["step"] if prog else None
    ev = {"boundary_step_under_check": step,
          "is_a_production_node": step % plan.NODE == 0 and step // plan.NODE == seg,
          "node_dir": str(node_dir), "log": str(log),
          "seg_ledger_segment_end": [
            {k: r.get(k) for k in ("global_step", "hit_stop", "lr_climbed", "epoch",
                                   "batch_in_epoch")} for r in led],
          "driver_ledger_record_present": (len(drv) >= 1) if drv_path.is_file() else None,
          "trainer_state_global_step": st.get("global_step"),
          "last_progress_line_step": last, "progress_records": len(prog)}
    ok = (step == st.get("global_step") == last and len(led) == 1
          and led[0].get("hit_stop") is True
          # the rehearsal ledger predates the lr_climbed field; S2 asserts the lr itself
          and led[0].get("lr_climbed") is not True
          and (not drv_path.is_file() or len(drv) >= 1))
    return ok, ev


def flatten_base_lrs(sd):
    """Every ``base_lrs`` the saved scheduler state carries, top level or inner.

    ``SequentialLR`` keeps its two stage schedulers inside the state dict, so the base
    learning rate -- 1e-4, the peak the warmup ramps toward -- is not necessarily at the
    top. Whichever level declares it has to agree with the frozen plan.
    """
    found = []
    if isinstance(sd, dict):
        if isinstance(sd.get("base_lrs"), (list, tuple)):
            found.append([float(x) for x in sd["base_lrs"]])
        for key in ("schedulers", "_schedulers"):
            for sub in (sd.get(key) or []):
                if isinstance(sub, dict):
                    found += flatten_base_lrs(sub)
    return found


def check_scheduler(step, node_dir, run_dir):
    sd = torch.load(node_dir / "scheduler.pt", map_location="cpu", weights_only=False)
    opt = torch.load(node_dir / "optimizer.pt", map_location="cpu", weights_only=False)
    st = read_json(node_dir / "trainer_state.json")
    want = curve(step)
    led = read_jsonl(run_dir / "seg_ledger.jsonl")
    here = [r for r in led if r.get("global_step") == step
            and r.get("event") in ("segment_end", "segment_start")]
    bases = flatten_base_lrs(sd)
    ev = {"scheduler_last_epoch": sd.get("last_epoch"),
          "scheduler_keys": sorted(sd.keys()) if isinstance(sd, dict) else None,
          "scheduler_base_lrs": bases,
          "scheduler_milestones": sd.get("_milestones"),
          "lr_saved_in_optimizer_pt": float(opt["param_groups"][0].get("lr", -1.0)),
          "lr_recorded_in_trainer_state": (st.get("extra") or {}).get("lr"),
          "ledger_lr_at_this_step": [r.get("lr") for r in here],
          "curve_at_step": want,
          "cross_check_vs_driver_table": bk.boundary_cross_check(step)}
    ok = (sd.get("last_epoch") == step
          and close(ev["lr_saved_in_optimizer_pt"], want)
          and close((st.get("extra") or {}).get("lr"), want)
          and all(close(r.get("lr"), want) for r in here)
          and all(b == [plan.LR] for b in bases)
          and sd.get("_milestones") == [int(plan.TOTAL * 0.05)]
          and ev["cross_check_vs_driver_table"].get("equal", True))
    return ok, ev


def measured_dataset_size(run_dir):
    """``len(train_dataset)`` as the live run itself printed it, not as frames.

    The corpus is 42,681 *frames*, but the model trains on 33-frame windows, and
    ``A3VideoDataset`` counts ``sum(ep_len - num_frames + 1)`` over episodes -- for the
    frozen 100-episode corpus that is 42,681 - 100*32 = 39,481. Dividing the frame count
    by the batch size would therefore predict an epoch 8 % longer than the one the
    sampler actually walks, and the sampler check would blame the checkpoint for the
    checker's own arithmetic. The trainer prints the number it built, so read that line;
    fall back to the frame arithmetic only if some future run loses the log.
    """
    log = Path(run_dir) / "logs" / "seg1.log"
    if log.is_file():
        text = log.read_text(errors="replace").replace("\r", "\n")
        # rich wraps "Train/val dataset size: 39481/39481" across two padded columns
        joined = " ".join(text.split())
        m = re.search(r"Train/val dataset size:\s*(\d+)/(\d+)", joined)
        if m and m.group(1) == m.group(2):
            return int(m.group(1)), "trainer log line (seg1.log)"
    cfg = Path(run_dir) / "config.yaml"
    if not cfg.is_file():
        return None, "no log line and no config"
    import math as _m
    import yaml
    try:
        doc = yaml.safe_load(cfg.read_text())
    except Exception:  # noqa: BLE001
        return None, "config unreadable"
    train = ((doc or {}).get("data") or {}).get("train") or {}
    num_frames = int(train.get("num_frames") or 0)
    rows = None
    for d in (train.get("dataset_dirs") or []):
        s = Path(d) / "summary.json"
        if s.is_file():
            try:
                rows = json.loads(s.read_text()).get("rows")
                eps = json.loads(s.read_text()).get("episodes")
            except (ValueError, AttributeError):
                return None, "summary.json unreadable"
            if isinstance(rows, int) and isinstance(eps, int) and num_frames:
                return rows - eps * (num_frames - 1), "derived from summary.json frames"
            break
    return None, "no measured size"


def dataset_batches(run_dir):
    """One epoch's cost in optimizer steps, from the dataset size the run measured.

    ``ResumableEpochSampler`` yields ``len(dataset)`` indices per epoch and the loader
    batches them with ``drop_last`` left at its default of False, so the last batch is
    short and the epoch is ``ceil(size / (batch_size * num_processes))`` steps -- one
    process here, gradient accumulation one, so micro steps and optimizer steps agree.
    """
    size, how = measured_dataset_size(run_dir)
    if not isinstance(size, int) or size <= 0:
        return None
    per_step = plan.BATCH
    return {"train_dataset_len": size, "size_source": how, "batch_size": per_step,
            "batches_per_epoch": -(-size // per_step),
            "last_batch_samples": size % per_step or per_step}


def check_sampler(step, node_dir, run_dir):
    """The resume-critical offsets, and one dataset length that explains every node.

    A boundary node records where the loader has to skip to when the next segment starts:
    ``resume_batch_offset`` is the batch index inside the epoch and must equal
    ``batch_in_epoch``. ``global_step == epoch * batches_per_epoch + batch_in_epoch``, and
    batches_per_epoch has to fall out of that as one positive integer shared by every node
    -- two different values would mean the epoch wrapped against two different dataset
    lengths, which cannot happen unless the data changed mid-run.
    """
    dirs = [Path(node_dir)]
    light = Path(run_dir) / "checkpoints" / "light"
    if light.is_dir():
        dirs += [d for d in sorted(light.iterdir())
                 if d.is_dir() and d != Path(node_dir)]
    nodes = []
    for d in dirs:
        st = read_json(d / "trainer_state.json")
        if not st or not isinstance(st.get("sampler"), dict):
            continue
        nodes.append({"dir": str(d), "global_step": st.get("global_step"),
                      "epoch": st.get("epoch"), "batch_in_epoch": st.get("batch_in_epoch"),
                      "sampler": st["sampler"]})
    nodes = [n for n in nodes if isinstance(n["global_step"], int)]
    if not nodes:
        return False, {"error": "no node with a sampler state under %s" % run_dir}
    bpe, bad = None, None
    for n in nodes:
        g, e, b = n["global_step"], n["epoch"], n["batch_in_epoch"]
        if not all(isinstance(x, int) for x in (g, e, b)):
            bad = {"node": n, "reason": "counters are not integers"}
            break
        if e == 0:
            # inside the first pass the batch index *is* the step count
            if g != b:
                bad = {"node": n, "reason": "epoch 0 but batch_in_epoch != global_step"}
                break
            if bpe is not None and g >= bpe:
                bad = {"node": n, "reason": "a wrap should have happened by this step",
                       "batches_per_epoch": bpe}
                break
        elif g <= b or (g - b) % e:
            bad = {"node": n, "reason": "no integer batches_per_epoch solves "
                                        "global_step = epoch*batches_per_epoch + "
                                        "batch_in_epoch"}
            break
        else:
            cand = (g - b) // e
            if bpe is None:
                bpe = cand
            elif cand != bpe:
                bad = {"node": n, "reason": "two different dataset lengths explain "
                                            "the nodes", "first": bpe, "second": cand}
                break
    seeds = {n["sampler"].get("seed") for n in nodes}
    sizes = {n["sampler"].get("batch_size") for n in nodes}
    procs = {n["sampler"].get("num_processes") for n in nodes}
    offsets_ok = all(n["sampler"].get("resume_batch_offset") == n["batch_in_epoch"]
                     for n in nodes)
    ds = dataset_batches(run_dir)
    predicted = (ds or {}).get("batches_per_epoch")
    # A node whose epoch is still 0 must not have reached the predicted epoch length, and
    # a derived epoch length must be the predicted one; either way the corpus is pinned.
    ds_ok = (predicted is None or bpe is None or bpe == predicted)
    ds_ok = ds_ok and all(n["batch_in_epoch"] < predicted for n in nodes
                         if n["epoch"] == 0 and predicted is not None)
    ev = {"nodes_checked": len(nodes), "derived_batches_per_epoch": bpe,
          "dataset": ds, "derived_matches_corpus": ds_ok,
          "offsets_match_counters": offsets_ok,
          "seeds": sorted(seeds, key=str), "batch_size": sorted(sizes, key=str),
          "num_processes": sorted(procs, key=str),
          "sampler_fields": sorted(nodes[0]["sampler"]),
          "first_and_last_node": [nodes[0]["dir"], nodes[-1]["dir"]],
          "per_node": [{k: n[k] for k in ("global_step", "epoch", "batch_in_epoch")}
                       for n in nodes]}
    ok = (bad is None and offsets_ok and len(seeds) == 1 and sizes == {plan.BATCH}
          and procs == {1} and ds_ok)
    if bad:
        ev["inconsistency"] = bad
    return ok, ev


def check_loss(step, seg, run_dir, log_override=None):
    log = Path(log_override) if log_override else run_dir / "logs" / ("seg%d.log" % seg)
    prog = parse_progress(log)
    if not prog:
        return False, {"error": "no progress records parsed from %s" % log}
    sm = summarize(prog)
    drift = [abs(r["lr"] - curve(r["step"])) / max(curve(r["step"]), 1e-12)
             for r in prog if isinstance(r.get("lr"), float)]
    ev = {"log": str(log), "max_relative_lr_drift_in_log_vs_curve":
          max(drift) if drift else None,
          # the progress line prints lr to three significant figures, so display rounding
          # alone can move it by 5e-4; a restarted warmup would be off by orders of
          # magnitude, which is what this threshold is there to catch
          "lr_drift_tolerance": 2e-3,
          "tail_over_whole_loss_ratio": ratio_or_none((sm["loss"]["tail"] or {}).get("mean"),
                                                      (sm["loss"]["whole"] or {}).get("mean")),
          "summary": sm}
    prev = run_dir / "logs" / ("seg%d.log" % (seg - 1))
    if seg > 1 and prev.is_file() and not log_override:
        psm = summarize(parse_progress(prev))
        pm = (psm["loss"]["tail"] or {}).get("mean")
        hm = (sm["loss"]["head"] or {}).get("mean")
        ev["previous_segment_tail_mean_loss"] = pm
        ev["this_segment_head_mean_loss"] = hm
        ev["boundary_loss_ratio"] = ratio_or_none(hm, pm)
    ok = (sm["non_finite_records"] == 0
          and sm["loss"]["whole"] is not None
          and all(r["stop_step"] == step for r in prog)
          and (_block("force_valid_step_frac", prog) or {}).get("min", 0.0) >= 1.0
          and (ev["max_relative_lr_drift_in_log_vs_curve"] or 0.0) <= 2e-3
          and (ev["tail_over_whole_loss_ratio"] or 1.0) <= 3.0
          and (ev.get("boundary_loss_ratio") is None
               or ev["boundary_loss_ratio"] <= 3.0))
    return ok, ev


def check_calibration(node_dir):
    st = read_json(node_dir / "trainer_state.json")
    fp = st.get("force_calibration")
    ref = bk.frozen_contract()
    closure, ev = bk.calibration_closure(bk.calibration_source(), fp)
    ev["matches_frozen_contract"] = fp == ref.get("force_calibration")
    ev["contract_fingerprint"] = ref.get("force_calibration")
    ev["node_global_step"] = st.get("global_step")
    return bool(ev["matches_frozen_contract"] and closure), ev


def check_params(node_dir, step):
    st = read_json(node_dir / "trainer_state.json")
    ref = bk.frozen_contract()
    files = {k: (node_dir / v).is_file() for k, v in (st.get("files") or {}).items()}
    sizes = {p.name: p.stat().st_size for p in sorted(Path(node_dir).iterdir())
             if p.is_file()}
    order = st.get("optimizer_member_order") or []
    ref_order = ref.get("optimizer_member_order") or []
    ok = (st.get("format") == "fasterwam-lora-light-state-v3"
          and st.get("global_step") == step
          and files and all(files.values()) and set(sizes) >= set(bk.STATE_FILES)
          and st.get("adapter_bytes") == ref.get("adapter_bytes")
          and st.get("coverage") == ref.get("coverage")
          and order == ref_order)
    return ok, {"format": st.get("format"), "files_present": files, "sizes": sizes,
                "adapter_bytes": st.get("adapter_bytes"),
                "contract_adapter_bytes": ref.get("adapter_bytes"),
                "coverage": st.get("coverage"), "contract_coverage": ref.get("coverage"),
                "optimizer_members": len(order), "contract_members": len(ref_order),
                "member_order_identical": order == ref_order}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, required=True, help="global_step at the boundary")
    ap.add_argument("--segment", type=int, help="default: step / NODE_INTERVAL")
    ap.add_argument("--run-dir", default=str(plan.RUN))
    ap.add_argument("--node-dir", help="default: the driver's own light_dir(step)")
    ap.add_argument("--log", help="override the segment log (rehearsal runs live elsewhere)")
    ap.add_argument("--report", help="default: _v1_stage/backup40k/segment<step>_check.json")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    seg = args.segment or (args.step // plan.NODE) or 1
    node_dir = Path(args.node_dir) if args.node_dir else plan.light_dir(args.step)
    report = Path(args.report) if args.report else \
        plan.STAGE / "backup40k" / ("segment%06d_check.json" % args.step)

    rows = [("S1 global_step agrees across ledger, node and log",
             lambda: check_global_step(args.step, seg, node_dir, run_dir, args.log)),
            ("S2 scheduler: one 40k curve, continued, not restarted",
             lambda: check_scheduler(args.step, node_dir, run_dir)),
            ("S3 sampler: the offsets a resume replays",
             lambda: check_sampler(args.step, node_dir, run_dir)),
            ("S4 loss: finite, force-valid, no break at the boundary",
             lambda: check_loss(args.step, seg, run_dir, args.log)),
            ("S5 calibration fingerprint closes on the uploaded JSON",
             lambda: check_calibration(node_dir)),
            ("S6 parameter integrity against the frozen contract",
             lambda: check_params(node_dir, args.step))]

    out = {"step": args.step, "segment": seg, "run_dir": str(run_dir),
           "node_dir": str(node_dir), "started": time.strftime("%FT%TZ")}
    checks, failed = [], []
    for name, fn in rows:
        try:
            ok, ev = fn()
        except Exception as exc:  # noqa: BLE001
            ok, ev = False, {"error": "%s: %s" % (type(exc).__name__, exc)}
        checks.append({"name": name, "ok": bool(ok), "evidence": ev})
        print(("PASS " if ok else "FAIL ") + name)
        print("     " + bk.mask(json.dumps(ev, sort_keys=True, default=str))[:1800])
        if not ok:
            failed.append(name)
    out["checks"] = checks
    out["passed"] = sum(1 for c in checks if c["ok"])
    out["failed"] = failed
    out["finished"] = time.strftime("%FT%TZ")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(bk.mask(json.dumps(out, indent=2, sort_keys=True, default=str)))
    print(json.dumps({"report": str(report), "passed": out["passed"],
                      "failed": failed}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
