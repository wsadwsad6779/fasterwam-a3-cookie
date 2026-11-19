"""Phase2: CPU-only regression for the corrected execution-chain recorder (real shapes).

No model, no simulator, no torch. Exercises the FIXED Capture against the REAL ForcePolicy
shape contract: raw=(n,7) (model native output), denorm=(1,n,7) (bridge return after unsqueeze),
rewritten=(n,7) (post-pm1 rows), adapter=(n,14). Confirms copy isolation, 32-predict/10-execute
row mapping, short chunks / early stop, and explicit rejection of illegal shapes / missing fields.
Also contrasts the archived ORIGINAL recorder, which wrongly rejects raw=(n,7).

Exit 0 iff every check passes. Writes recorder_cpu_regression.json.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = Path("/workspace/eval_restore_20261117")
FIXED_PATH = HERE / "execution_chain_capture_fixed.py"
ORIG_PATH = BASE / "deploy" / "capture" / "execution_chain_capture.py"
REPORT = HERE / "recorder_cpu_regression.json"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fixed = load_module("ecc_fixed", FIXED_PATH)
original = load_module("ecc_orig", ORIG_PATH)
Capture = fixed.Capture

# Small real files for hash_files (shape test only; not the 11 GB weights).
ART = {"weights": [str(FIXED_PATH)], "config": [str(FIXED_PATH)], "source": [str(FIXED_PATH)]}
RNG = np.random.default_rng(0)


def obs(state=None):
    s = np.zeros(16, dtype=np.float32) if state is None else np.asarray(state, dtype=np.float32)
    return {"observation.state": s}


def info(req, app):
    return {"requested_action": np.asarray(req, np.float32),
            "applied_action": np.asarray(app, np.float32)}


def chunk_arrays(n):
    return (RNG.standard_normal((n, 7)).astype(np.float32),
            RNG.standard_normal((1, n, 7)).astype(np.float32),
            RNG.standard_normal((n, 7)).astype(np.float32),
            RNG.standard_normal((n, 14)).astype(np.float32))


RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append({"name": name, "pass": True})
    except Exception as e:
        RESULTS.append({"name": name, "pass": False, "error": f"{type(e).__name__}: {e}"})


def expect_raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} but none raised")


# 1. real shapes raw(32,7)/denorm(1,32,7) accepted, ranks preserved
def t_real_32():
    cap = Capture(ART)
    cap.reset(obs())
    raw, den, rew, adp = chunk_arrays(32)
    cap.chunk(0, raw, den, rew, adp)
    assert cap.arrays["chunk_0_raw"].shape == (32, 7), cap.arrays["chunk_0_raw"].shape
    assert cap.arrays["chunk_0_raw"].ndim == 2
    assert cap.arrays["chunk_0_denorm"].shape == (1, 32, 7)
    assert cap.arrays["chunk_0_denorm"].ndim == 3
    assert cap.arrays["chunk_0_rewritten"].shape == (32, 7)
    assert cap.arrays["chunk_0_adapter"].shape == (32, 14)
    assert cap.chunks[0]["predicted_rows"] == 32


# 2. short chunk raw(3,7)/denorm(1,3,7)
def t_real_3():
    cap = Capture(ART)
    cap.reset(obs())
    raw, den, rew, adp = chunk_arrays(3)
    cap.chunk(0, raw, den, rew, adp)
    assert cap.arrays["chunk_0_raw"].shape == (3, 7)
    assert cap.arrays["chunk_0_raw"].ndim == 2
    assert cap.arrays["chunk_0_denorm"].shape == (1, 3, 7)
    assert cap.arrays["chunk_0_denorm"].ndim == 3
    assert cap.chunks[0]["predicted_rows"] == 3


# 3. in-place modification isolation (incl. pm1-style view rewrite of denorm)
def t_copy_isolation():
    cap = Capture(ART)
    cap.reset(obs())
    raw, den, rew, adp = chunk_arrays(32)
    snap = (raw.copy(), den.copy(), rew.copy(), adp.copy())
    cap.chunk(0, raw, den, rew, adp)
    raw[:] = 999.0
    den[:] = 999.0
    rew[:] = 999.0
    adp[:] = 999.0
    # pm1 rewrite is an in-place edit of a VIEW of the bridge result (den[0])
    view = den[0]
    view[..., -1] = np.where(view[..., -1] >= 0.0, 1.0, -1.0)
    for key, orig in zip(("raw", "denorm", "rewritten", "adapter"), snap):
        assert np.array_equal(cap.arrays[f"chunk_0_{key}"], orig), f"{key} polluted by in-place edit"


# 4. predict 32, execute only first 10 -> row/step/frame mapping
def t_map_32_10():
    cap = Capture(ART)
    cap.reset(obs())
    raw, den, rew, adp = chunk_arrays(32)
    cap.chunk(0, raw, den, rew, adp)
    for row in range(10):
        handed = adp[row].copy()
        sel = cap.selected(handed)
        assert np.array_equal(sel, adp[row])
        st = np.zeros(16, np.float32)
        st[7] = 0.5 - 0.01 * row
        cap.executed(sel, obs(st), info(handed, np.zeros(16, np.float32)))
    assert len(cap.mapping) == 10
    assert cap.chunks[0]["executed_rows"] == 10
    assert cap.frame == 10
    for i, m in enumerate(cap.mapping):
        assert m["chunk_row"] == i and m["step"] == i
        assert m["obs_frame_before"] == i and m["obs_frame_after"] == i + 1
        assert np.array_equal(cap.arrays[f"step_{i}_handed14"], adp[i])
    # only the 10 executed rows are in the trajectory, never the 32 predicted
    assert len(cap.mapping) != 32


# 5. early termination on a short chunk + over-execution rejected
def t_early_stop_short():
    cap = Capture(ART)
    cap.reset(obs())
    raw, den, rew, adp = chunk_arrays(3)
    cap.chunk(0, raw, den, rew, adp)
    for row in range(2):
        sel = cap.selected(adp[row].copy())
        cap.executed(sel, obs(), info(adp[row], np.zeros(16, np.float32)))
    assert cap.chunks[0]["executed_rows"] == 2 and cap.frame == 2
    # executing a 3rd row is allowed (predicted_rows=3) but a 4th must be rejected
    sel = cap.selected(adp[2].copy())
    cap.executed(sel, obs(), info(adp[2], np.zeros(16, np.float32)))
    expect_raises(lambda: cap.selected(adp[0].copy()))


# 6. illegal shapes / missing fields explicitly rejected
def t_reject_bad_raw_batch():
    cap = Capture(ART); cap.reset(obs())
    raw, den, rew, adp = chunk_arrays(32)
    expect_raises(lambda: cap.chunk(0, raw[None, ...], den, rew, adp))  # raw (1,32,7) now illegal


def t_reject_bad_denorm_rank():
    cap = Capture(ART); cap.reset(obs())
    raw, den, rew, adp = chunk_arrays(32)
    expect_raises(lambda: cap.chunk(0, raw, den[0], rew, adp))  # denorm (32,7) illegal


def t_reject_bad_dims():
    cap = Capture(ART); cap.reset(obs())
    raw, den, rew, adp = chunk_arrays(32)
    expect_raises(lambda: cap.chunk(0, np.zeros((32, 8), np.float32), den, rew, adp))
    cap2 = Capture(ART); cap2.reset(obs())
    expect_raises(lambda: cap2.chunk(0, raw, den, rew, np.zeros((32, 7), np.float32)))


def t_reject_nonfinite():
    cap = Capture(ART); cap.reset(obs())
    raw, den, rew, adp = chunk_arrays(32)
    raw[0, 0] = np.nan
    expect_raises(lambda: cap.chunk(0, raw, den, rew, adp))


def t_reject_missing_info():
    cap = Capture(ART); cap.reset(obs())
    raw, den, rew, adp = chunk_arrays(32)
    cap.chunk(0, raw, den, rew, adp)
    sel = cap.selected(adp[0].copy())
    expect_raises(lambda: cap.executed(sel, obs(), {"applied_action": np.zeros(16, np.float32)}))


def t_reject_missing_state():
    cap = Capture(ART)
    expect_raises(lambda: cap.reset({"observation.force": np.zeros(18, np.float32)}))


def t_reject_handed_mismatch():
    cap = Capture(ART); cap.reset(obs())
    raw, den, rew, adp = chunk_arrays(32)
    cap.chunk(0, raw, den, rew, adp)
    bad = adp[0].copy(); bad[0] += 1.0
    expect_raises(lambda: cap.selected(bad))


def t_reject_frame_mismatch():
    cap = Capture(ART); cap.reset(obs())
    raw, den, rew, adp = chunk_arrays(32)
    expect_raises(lambda: cap.chunk(1, raw, den, rew, adp))  # frame 1 != current 0


# 7. contrast: archived ORIGINAL rejects the REAL raw=(n,7); FIXED accepts it
def t_original_rejects_real_raw():
    OCap = original.Capture
    cap = OCap(ART); cap.reset(obs())
    raw, den, rew, adp = chunk_arrays(32)
    expect_raises(lambda: cap.chunk(0, raw, den, rew, adp))  # original bug: expects (1,n,7)


def t_fixed_accepts_real_raw():
    cap = Capture(ART); cap.reset(obs())
    raw, den, rew, adp = chunk_arrays(32)
    cap.chunk(0, raw, den, rew, adp)
    assert cap.arrays["chunk_0_raw"].ndim == 2


for nm, fn in [
    ("real_32_raw2d_denorm3d", t_real_32),
    ("real_3_short_chunk", t_real_3),
    ("copy_isolation_incl_pm1_view", t_copy_isolation),
    ("map_predict32_execute10", t_map_32_10),
    ("early_stop_short_chunk_overexec_rejected", t_early_stop_short),
    ("reject_raw_with_batch_dim", t_reject_bad_raw_batch),
    ("reject_denorm_wrong_rank", t_reject_bad_denorm_rank),
    ("reject_wrong_last_dims", t_reject_bad_dims),
    ("reject_nonfinite", t_reject_nonfinite),
    ("reject_missing_info_field", t_reject_missing_info),
    ("reject_missing_state", t_reject_missing_state),
    ("reject_handed_mismatch", t_reject_handed_mismatch),
    ("reject_chunk_frame_mismatch", t_reject_frame_mismatch),
    ("original_rejects_real_raw_BUG", t_original_rejects_real_raw),
    ("fixed_accepts_real_raw", t_fixed_accepts_real_raw),
]:
    check(nm, fn)

passed = sum(1 for r in RESULTS if r["pass"])
verdict = "PASS" if passed == len(RESULTS) else "FAIL"
report = {"NORMALIZER_IDENTITY": "RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL",
          "phase": "Phase2 recorder CPU regression (real shapes)",
          "fixed_path": str(FIXED_PATH), "original_path": str(ORIG_PATH),
          "total": len(RESULTS), "passed": passed, "verdict": verdict, "checks": RESULTS}
REPORT.write_text(json.dumps(report, indent=2, sort_keys=True))
print(json.dumps({"verdict": verdict, "passed": passed, "total": len(RESULTS),
                  "failures": [r for r in RESULTS if not r["pass"]]}, indent=2))
sys.exit(0 if verdict == "PASS" else 1)