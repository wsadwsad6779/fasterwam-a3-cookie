"""Collect the accepted low-force and correction demos into one machine-readable manifest.

Read-only with respect to the recordings. For each directory it reads `summary.json` plus
the `adjudication_*.json` that the read-only re-scorers (adjudicate_correct.py,
adjudicate_c3.py, adjudicate_wc.py) already wrote next to the recordings, and a recording
enters the manifest only when its own family's `strict_accept` is true. The families are
kept apart by the key the drivers themselves wrote into `retry_log[0].anomaly_kind`
(excessive_pad_force = C-1, wrong_closure = C-2, empty_grasp / stuck_no_force = C-3), so a
single episode can never be counted in two families.

Each entry carries the npz size and sha256 plus the few numbers the verdict was made on
(abnormal-half peak, recovery-half peak, hold force, lift, placement), so a later
image-re-recording or BC run can prove it is the same trajectory and re-check the decision
without loading the arrays.

Writes exactly one new file (default /workspace/a3_eval/corpus_manifest_v1.json) and
refuses to overwrite it.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

FAMILY_BY_KIND = {
    "excessive_pad_force": "C1_press_too_deep",
    "wrong_closure": "C2_wrong_closure",
    "empty_grasp": "C3_closed_but_empty",
    "stuck_no_force": "C3_closed_but_empty",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def num(ep, path, default=None):
    cur = ep
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def correct_dirs(dirs, out):
    """C-1 from adjudication_correct_v2.json, C-3 from adjudication_c3.json."""
    for d in dirs:
        dir_ = Path(d)
        summary = load_json(dir_ / "summary.json")
        if summary is None:
            print("SKIP no summary", d)
            continue
        kind = {}
        for row in summary:
            log = row.get("retry_log") or []
            kind[int(row["seed"])] = (log[0].get("anomaly_kind") if log else "-")
        adj1 = load_json(dir_ / "adjudication_correct_v2.json")
        if adj1:
            for e in adj1["episodes"]:
                if not e.get("strict_accept"):
                    continue
                seed = int(e["seed"])
                if FAMILY_BY_KIND.get(kind.get(seed)) != "C1_press_too_deep":
                    continue
                segs = {s["kind"]: s["segments"] for s in e["segments"]}
                out.append({"family": "C1_press_too_deep", "dir": str(dir_), "seed": seed,
                            "variant": e.get("variant"), "anomaly_kind": kind.get(seed),
                            "cut": (e["cuts"] or [None])[0],
                            "abnormal_peak_n": segs.get("abnormal", {}).get("peak"),
                            "abnormal_frames_over_1p6n": segs.get("abnormal", {}).get("over_1p6"),
                            "recovery_peak_n": segs.get("recovery", {}).get("peak"),
                            "recovery_frames_over_2n": segs.get("recovery", {}).get("over_2p0"),
                            "steps": e.get("steps"), "npz": "expert_correct_%d.npz" % seed})
        adj3 = load_json(dir_ / "adjudication_c3.json")
        if adj3:
            for e in adj3["episodes"]:
                if not e.get("strict_accept"):
                    continue
                seed = int(e["seed"])
                out.append({"family": FAMILY_BY_KIND.get(kind.get(seed), "C3_closed_but_empty"),
                            "dir": str(dir_), "seed": seed, "variant": e.get("variant"),
                            "anomaly_kind": kind.get(seed), "cut": e.get("cut"),
                            "abnormal_peak_n": num(e, ("abnormal", "peak")),
                            "closed_through_air": e.get("closed_through_air"),
                            "abnormal_closure_peak_n": num(e, ("abnormal_clamp", "closure_peak_pad_n")),
                            "abnormal_max_cookies_at_grip": num(e, ("abnormal_clamp", "max_cookies_at_grip")),
                            "abnormal_min_opening": num(e, ("abnormal_clamp", "min_opening")),
                            "recovery_peak_n": num(e, ("recovery", "peak")),
                            "hold_n": num(e, ("recovery_clamp", "closure_peak_pad_n")),
                            "lift_mm": num(e, ("recovery_clamp", "carried_lift_mm")),
                            "steps": e.get("steps"), "npz": "expert_correct_%d.npz" % seed})


def wc_dirs(dirs, out):
    for d in dirs:
        dir_ = Path(d)
        adj = load_json(dir_ / "adjudication_wc.json")
        if adj is None:
            print("SKIP no adjudication_wc", d)
            continue
        for e in adj["episodes"]:
            if not e.get("strict_accept"):
                continue
            out.append({"family": "C2_wrong_closure", "dir": str(dir_), "seed": int(e["seed"]),
                        "variant": e.get("recover_variant"), "anomaly_kind": "wrong_closure",
                        "cut": e.get("cut"),
                        "abnormal_peak_n": e.get("wrong_closure_peak_pad_n"),
                        "abnormal_max_cookies_at_grip": e.get("wrong_closure_max_at_grip"),
                        "recovery_peak_n": num(e, ("recovery_clamp", "carry_peak_pad_n")),
                        "hold_n": num(e, ("recovery_clamp", "closure_peak_pad_n")),
                        "lift_mm": num(e, ("recovery_clamp", "carried_lift_mm")),
                        "upright": num(e, ("recovery_clamp", "placed_upright")),
                        "steps": e.get("steps"),
                        "npz": "expert_wrongclosure_%d.npz" % int(e["seed"])})


def single_dirs(dirs, out):
    """Plain low-force single-cookie successes: the drivers own per-clamp accept is the rule."""
    # replay_forces rows have no row-level "accept" at all (only replay_correct writes that),
    # and their row-level "success" is the env's whole-task flag, false for a 1-clamp episode.
    for d in dirs:
        dir_ = Path(d)
        summary = load_json(dir_ / "summary.json")
        if summary is None:
            print("SKIP no summary", d)
            continue
        for row in summary:
            clamp = (row.get("clamps") or [{}])[-1]
            if not clamp.get("accept"):
                continue
            seed = int(row["seed"])
            out.append({"family": "S1_single_lowforce", "dir": str(dir_), "seed": seed,
                        "variant": row.get("variant"), "anomaly_kind": "-",
                        "hold_n": clamp.get("closure_peak_pad_n"),
                        "carry_peak_n": clamp.get("carry_peak_pad_n"),
                        "lift_mm": clamp.get("carried_lift_mm"),
                        "max_cookies_at_grip": clamp.get("max_cookies_at_grip"),
                        "upright": clamp.get("placed_upright"),
                        "steps": row.get("steps"), "npz": "expert_single_%d.npz" % seed})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--correct", nargs="*", default=[])
    ap.add_argument("--wc", nargs="*", default=[])
    ap.add_argument("--single", nargs="*", default=[])
    ap.add_argument("--out", default="/workspace/a3_eval/corpus_manifest_v1.json")
    args = ap.parse_args()

    entries = []
    correct_dirs(args.correct, entries)
    wc_dirs(args.wc, entries)
    single_dirs(args.single, entries)

    for e in entries:
        npz = Path(e["dir"]) / e["npz"]
        if npz.exists():
            e["npz_bytes"] = int(npz.stat().st_size)
            e["npz_sha256"] = sha256(npz)
        else:
            e["npz_missing"] = True

    tally = {}
    for e in entries:
        tally[e["family"]] = tally.get(e["family"], 0) + 1
    sources = []
    for d in args.correct:
        for name in ("adjudication_correct_v2.json", "adjudication_c3.json"):
            p = Path(d) / name
            if p.exists():
                sources.append({"path": str(p), "bytes": int(p.stat().st_size),
                                "sha256": sha256(p)})
    for d in args.wc:
        p = Path(d) / "adjudication_wc.json"
        if p.exists():
            sources.append({"path": str(p), "bytes": int(p.stat().st_size),
                            "sha256": sha256(p)})

    out = Path(args.out)
    if out.exists():
        print("REFUSING to overwrite", out)
        return
    blob = json.dumps({
        "note": ("Strict-accept only. Families are separated by the anomaly_kind the "
                 "recording drivers themselves wrote into retry_log[0], so no episode "
                 "appears in two families. C-1 requires the abnormal half to exceed 1.6 N "
                 "for at least 3 frames (adjudicate_correct.py after the or/and fix); "
                 "C-3 requires a closure that held nothing and a recovery grasp with hold "
                 "in [0.3, 1.6] N; C-2 requires a commanded release of the wrongly held "
                 "batch. Image frames are NOT in these recordings - see docs_task2 6.6."),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tally": tally,
        "sources": sources,
        "entries": sorted(entries, key=lambda e: (e["family"], e["dir"], e["seed"])),
    }, indent=2)
    out.write_text(blob, encoding="utf-8")
    print("WROTE", out, "bytes", len(blob), "sha", hashlib.sha256(blob.encode()).hexdigest()[:16])
    print("TALLY", json.dumps(tally))


main()
