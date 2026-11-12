"""CPU-only test for the 14-wide proprio interface fix. Runs with numpy alone -- no torch, no
simulator, no weights -- so it can be executed anywhere, including on a machine that has neither
the pod's model nor its GPU.

It covers the two halves of the fix, and nothing else:

* `eval40k.build_proprio14` -- the arithmetic. Columns 0-7 must be the deployment worker's own
  eight numbers unchanged, columns 8-13 must be the six frozen channels minus the calibration's
  `zero_load_ref_median`, computed in float64 and cast to float32 once, the way
  `build_v1_dataset.py` built the stored corpus. Each guard (worker width, force width, missing
  tare, non-finite tare) must raise, because silently returning a wrong-width proprio is how the
  8-vs-14 mismatch reached the normaliser in the first place.
* `policy40k.ForcePolicy` -- the call site. This is what makes the fix apply to the batch
  evaluation and not only to the smoke test: `_infer_kwargs` must consult `proprio_fn` when the
  driver supplied one, fall back to the worker's `build_proprio` when it did not, and normalise
  whatever it gets with `bridge.normalize_proprio`, the worker's own function.

`torch` is replaced by a recording stub: this test never calls a model, it only checks which
array would have been handed to it.

    py -3.13 proprio14_test.py        (or python3, or /opt/venv/bin/python on the pod)
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval40k as E                                        # noqa: E402
from force_stream import CHANNELS, ForceContractError      # noqa: E402

BASE8 = np.array([0.10, 0.20, 0.30, -0.40, 0.50, 0.60, 0.01, 0.02], dtype=np.float32)
ZERO6 = [-0.000118, -10.72115, -0.028227, 0.0, 0.0, -0.018633]
CALIB = {"version": "force-calib-v1", "channels": list(CHANNELS),
         "mean": [0.0] * 6, "std": [0.733885, 0.588731, 0.180894,
                                    0.681404, 0.685467, 1.396991],
         "scale": 1.0, "zero_load_ref_median": list(ZERO6)}

FAILS = []
RUN_CHECKS = [0]


def check(name, ok, detail=""):
    RUN_CHECKS[0] += 1
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else "   -> " + str(detail)))
    if not ok:
        FAILS.append(name)


def force18(values_by_channel):
    v = np.zeros(18, dtype=np.float32)
    for c, x in values_by_channel.items():
        v[c] = x
    return v


ALL18 = (np.arange(18, dtype=np.float32) + 1.0)     # channel 0 -> 1.0 ... channel 17 -> 18.0
RAW6 = ALL18[list(CHANNELS)].astype(np.float64)     # [1, 2, 3, 13, 14, 17]


class FakeWorker:
    """Stands in for `infer_a3_worker`; hands out exactly eight columns like the real one."""

    DEFAULT_PROMPT = "{task}"

    def __init__(self, base=BASE8):
        self._base = np.asarray(base)
        self.build_calls = 0

    def build_proprio(self, obs):
        self.build_calls += 1
        return self._base.copy()

    def prepare_fasterwam_images_official(self, obs):
        return obs["images"], np.zeros((1, 3, 4, 4), dtype=np.float32)

    def normalize_proprio(self, proprio, processor):
        self.normalised = np.asarray(proprio)
        return self.normalised

    def denormalize_action(self, action, processor):
        return np.asarray(action, dtype=np.float64)

    def quat_wxyz_to_matrix(self, quat):
        return np.eye(3)


def test_arithmetic():
    obs = {"observation.force": ALL18.copy(), "images": None}
    p14 = E.build_proprio14(obs, FakeWorker(), CALIB)
    want_force = RAW6 - np.asarray(ZERO6)
    check("width and dtype: one float32 vector of 14, not 8 and not float64",
          p14.shape == (E.PROPRIO_DIM,) and p14.dtype == np.float32, (p14.shape, p14.dtype))
    check("columns 0-7 are the worker's own eight numbers, bit-for-bit",
          np.array_equal(p14[:8].astype(np.float64), BASE8.astype(np.float64)))
    check("columns 8-13 are the six frozen channels minus the frozen zero-load tare",
          np.array_equal(p14[8:], want_force.astype(np.float32)),
          (p14[8:].tolist(), want_force.astype(np.float32).tolist()))
    check("the tare is subtracted, not divided out and not re-normalised "
          "(mean/std calibration belongs to the force conditioner, not here)",
          np.allclose(p14[8:], want_force, rtol=0, atol=1e-6))
    check("the six columns come from channels %s of the env's 18-wide vector, in that order, "
          "and no other channel is used" % (list(CHANNELS),),
          list(CHANNELS) == [0, 1, 2, 12, 13, 16]
          and np.allclose(p14[8:], ALL18[[0, 1, 2, 12, 13, 16]].astype(np.float64)
                          - np.asarray(ZERO6), atol=1e-6)
          and not np.any(np.isclose(p14[8:], ALL18[[3, 4, 5, 6, 7, 8]].astype(np.float64)
                                    - np.asarray(ZERO6))))
    facts = dict(E.PROPRIO14_FACTS)
    check("the last call is recorded for the report, with the tare it used",
          facts.get("dim") == 14 and facts.get("base_dim") == 8
          and facts.get("channels") == list(CHANNELS)
          and facts.get("zero_load_ref_median") == list(ZERO6)
          and np.allclose(facts.get("residual6", []), want_force, atol=1e-6), facts)

    for name, W, calib, obs_ in [
        ("worker returns 7 columns -> raise, do not pad",
         FakeWorker(BASE8[:7]), CALIB, obs),
        ("worker returns 14 columns (someone else already appended) -> raise",
         FakeWorker(np.arange(14, dtype=np.float32)), CALIB, obs),
        ("calibration without zero_load_ref_median -> raise",
         FakeWorker(), {k: v for k, v in CALIB.items() if k != "zero_load_ref_median"}, obs),
        ("non-finite tare -> raise",
         FakeWorker(), dict(CALIB, zero_load_ref_median=[0.0, float("nan")] + [0.0] * 4), obs),
    ]:
        try:
            E.build_proprio14(obs_, W, calib)
            check(name, False, "no exception")
        except ForceContractError as exc:
            check(name, True)
            print("      raised: %s" % str(exc)[:120])
    narrow = {"observation.force": np.zeros(16, dtype=np.float32)}
    try:
        E.build_proprio14(narrow, FakeWorker(), CALIB)
        check("env force vector narrower than channel 16 -> raise", False, "no exception")
    except ForceContractError as exc:
        check("env force vector narrower than channel 16 -> raise", True)
        print("      raised: %s" % str(exc)[:120])


# --- the call site, with `torch` stubbed out -----------------------------------------------

class _Rec:
    def __init__(self, arr):
        self.arr = np.asarray(arr)
        self.ops = []

    def to(self, **kw):
        self.ops.append(("to", kw))
        return self

    def unsqueeze(self, dim):
        self.ops.append(("unsqueeze", dim))
        return self


class FakeTorch(types.ModuleType):
    def from_numpy(self, a):
        return _Rec(a)

    def as_tensor(self, v, dtype=None):
        return _T(v)

    def allclose(self, a, b, rtol=1e-5, atol=1e-8):
        return bool(np.allclose(_raw(a), _raw(b), rtol=rtol, atol=atol))

    def __getattr__(self, name):
        if name == "no_grad":
            return _NoGrad
        if name in ("float32", "float64"):
            return name
        raise AttributeError(name)


class _T:
    """The slice/sub/div/tolist surface that `check_stats_width` uses off `torch`."""

    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float64)

    def reshape(self, *s):
        return _T(self.a.reshape(*s))

    @property
    def shape(self):
        return self.a.shape

    def double(self):
        return _T(self.a)

    def tolist(self):
        return self.a.tolist()

    def __getitem__(self, i):
        return _T(self.a[i])

    def __neg__(self):
        return _T(-self.a)

    def __sub__(self, o):
        return _T(self.a - _raw(o))

    def __truediv__(self, o):
        return _T(self.a / o)


def _raw(x):
    return x.a if isinstance(x, _T) else np.asarray(x, dtype=np.float64)


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _maps(**kw):
    class _Map(dict):
        __getattr__ = dict.__getitem__
    return _Map(**kw)


def test_callsite():
    sys.modules["torch"] = FakeTorch("torch")
    for mod in [m for m in list(sys.modules) if m.startswith("policy40k")]:
        del sys.modules[mod]
    import policy40k as P

    cfg = _maps(data=_maps(train=_maps(num_frames=33, action_video_freq_ratio=4)),
                EVALUATION=_maps(rand_device="cpu", tiled=False, binarize_gripper=True,
                                 num_inference_steps=10, negative_prompt="", text_cfg_scale=1.0),
                eval_num_inference_steps=10)

    class Model:
        torch_dtype = np.float32

        def infer_action(self, **kw):
            raise AssertionError("the model must not be called by _infer_kwargs")

    seen = []

    class W2(FakeWorker):
        def normalize_proprio(self, proprio, processor):
            seen.append(np.asarray(proprio).reshape(-1).copy())
            return np.asarray(proprio)

    obs = {"observation.force": ALL18.copy(), "images": None,
           "observation.eef_pose": np.zeros(7, dtype=np.float64)}
    hist = np.zeros((16, 6), dtype=np.float32)
    mask = np.zeros(16, dtype=bool)

    W = W2()
    calib = CALIB
    p_with = P.ForcePolicy(Model(), None, cfg, None, "cpu", bridge=W,
                           instruction="pick", proprio_fn=lambda o: E.build_proprio14(o, W, calib))
    kw = p_with._infer_kwargs(obs, hist, mask, 7)
    check("ForcePolicy takes the proprio from proprio_fn when the driver supplied one",
          np.asarray(kw["proprio"]).reshape(-1).shape == (14,)
          and W.build_calls == 1,   # the 8-wide builder is still what produces columns 0-7
          (np.asarray(kw["proprio"]).shape, W.build_calls))
    check("what proprio_fn produced is the 14 training columns, and it is the worker's own "
          "normalize_proprio that was asked to normalise it",
          len(seen) == 1 and seen[0].shape == (14,)
          and np.array_equal(seen[0][:8].astype(np.float64), BASE8.astype(np.float64)))
    p_with.calls = 0

    W3 = W2()
    p_without = P.ForcePolicy(Model(), None, cfg, None, "cpu", bridge=W3, instruction="pick")
    kw2 = p_without._infer_kwargs(obs, hist, mask, 7)
    check("proprio_fn defaults to None, which keeps the released worker's 8-wide path for the "
          "CPU fixtures instead of breaking them",
          p_without.proprio_fn is None and W3.build_calls == 1
          and np.asarray(kw2["proprio"]).reshape(-1).shape == (8,),
          (W3.build_calls, np.asarray(kw2["proprio"]).shape))
    check("the driver that builds a real policy passes the shared builder, so the batch "
          "evaluator and the smoke test cannot diverge",
          "proprio_fn=lambda obs: build_proprio14(obs, W, calib)"
          in (HERE / "eval40k.py").read_text(encoding="utf-8").replace("  ", " "))
    src = (HERE / "policy40k.py").read_text(encoding="utf-8")
    check("there is exactly one proprio source in the model call site and it prefers proprio_fn",
          src.count("self.proprio_fn or self.bridge.build_proprio") == 1
          and src.count("normalize_proprio(build(obs)") == 1)
    check("the frozen per-chunk noise table is untouched by this fix",
          "np.random.default_rng([int(seed), int(call)])" in src)


def test_stats_width():
    """`check_stats_width` -- the stats file has to be *proved* to be the one training used.

    The bands below are the ones measured out of
    `/workspace/a3_data/a3_fw_v1g/dataset_stats.json` on the pod (2026-11-12): a 14-wide state
    whose first eight columns carry the released base dataset's own band and whose last six are
    exactly +/- the residual std in `force_calib_v1.json`. That coincidence is the whole reason
    `min/max` mode computes `residual / std` for the force columns (scale 2/(2*std) = 1/std,
    offset -1-(1/std)(-std) = 0), so a file whose force band is merely *symmetric* is not enough --
    the half-width has to be the calibration's own number.
    """
    sys.modules.setdefault("torch", FakeTorch("torch"))
    base_min = [0.05463, 0.090315, 0.763831, -1.224807, -2.433783, -1.233328,
                0.012252, -0.040235]
    base_max = [0.133504, 0.302857, 0.919999, 2.480117, 1.23362, 2.443289,
                0.040235, -0.012252]
    std = list(CALIB["std"])

    def stats_with(force_min, force_max, width=None):
        lo = base_min + list(force_min)
        hi = base_max + list(force_max)
        if width is not None:
            lo, hi = lo[:width], hi[:width]
        return {"action": {"default": {"global_min": [-1.0] * 7, "global_max": [1.0] * 7}},
                "state": {"default": {"global_min": lo, "global_max": hi,
                                      "global_std": [0.1] * len(lo)}}}

    good = stats_with([-s for s in std], std)
    facts = E.check_stats_width(good, "synthetic:v1g", calib_std=std)
    check("a faithful stats file passes: 14-wide state, symmetric force band, half-width equal to "
          "the calibration std",
          facts["keys"] == ["default"] and set(facts["bands"]) == {"default"}
          and facts["bands"]["default"]["force_band_is_symmetric_std"]
          and set(facts["widths"]) == {"default/global_min", "default/global_max",
                                       "default/global_std"}
          and all(v == E.PROPRIO_DIM for v in facts["widths"].values())
          and np.allclose(facts["bands"]["default"]["half_width_force_columns"], std, atol=1e-9),
          facts["widths"])
    check("what it read is recorded in STATS_FACTS for the report, band and all",
          E.STATS_FACTS.get("bands", {}).get("default", {}).get("min")
          == base_min + [-s for s in std]
          and E.STATS_FACTS.get("bands", {}).get("default", {}).get("max") == base_max + std
          and E.STATS_FACTS["bands"]["default"].get("force_halfwidth_equals_calib_std") is True)

    for name, st, cstd in [
        ("8-wide state (the released LIBERO-shaped file) -> raise, name the file and the fix",
         stats_with([], []), std),
        ("force band symmetric but +/-1.0, not the residual std -> raise: min/max mode would stop "
         "being residual/std", stats_with([-1.0] * 6, [1.0] * 6), std),
        ("force band only +/-1e-3 off the calibration std -> raise (outside the float32 round-trip)",
         stats_with([-s - 1e-3 for s in std], [s + 1e-3 for s in std]), std),
        ("state block present but no entry carries global_min/global_max -> raise: the width was "
         "never actually checked",
         {"state": {"default": {"global_mean": [0.0] * 14}}}, std),
        ("calibration lists 5 stds against 6 banded force columns -> raise",
         good, std[:5]),
    ]:
        try:
            E.check_stats_width(st, "synthetic:%s" % name, calib_std=cstd)
            check(name, False, "no SystemExit")
        except SystemExit as exc:
            check(name, True)
            print("      raised: %s" % str(exc)[:150])

    check("a band only 1e-7 off the calibration std still passes, so the check tolerates the "
          "float32 the file stores but not a re-derived number (measured gap <= 2.5e-08)",
          E.check_stats_width(stats_with([-s - 1e-7 for s in std], [s + 1e-7 for s in std]),
                              "synthetic:tolerance", calib_std=std)["bands"]["default"][
              "force_halfwidth_equals_calib_std"] is True)
    src = (HERE / "eval40k.py").read_text(encoding="utf-8")
    check("the cross-check is not optional on the real path: build_policy hands it the calibration "
          "it just loaded, and it hands it before the merged artifact is opened",
          src.count("check_stats_width(stats, args.stats, calib_std=calib[\"std\"])") == 1
          and src.index("check_stats_width(stats, args.stats, calib_std=calib[\"std\"])")
          < src.index("model.load_checkpoint(str(merged))"))
    check("the smoke reads the band by what the file actually contains, not by a hardcoded key, "
          "and refuses a file with more than one state band",
          "bands = E.STATS_FACTS.get(\"bands\") or {}" in (HERE / "smoke40k.py").read_text(
              encoding="utf-8")
          and "if len(bands) != 1:" in (HERE / "smoke40k.py").read_text(encoding="utf-8"))


def test_channel_gather():
    """`select_channels` flattens, so the corpus-wide gather must not call it on the table.

    Measured on the pod 2026-11-12: `select_channels(force18)` where `force18` is the stored
    (42681, 18) array returns a (6,) vector -- six numbers out of row 0, because the function does
    `np.asarray(...).reshape(-1)` for the single-observation case the live path uses. A proof that
    called it that way would silently compare six values against 256,086 and fail, or worse, be
    rewritten to compare shapes. So the harness gathers with the vectorised index and separately
    proves the function applied row-by-row is that same gather.
    """
    from force_stream import select_channels

    rng = np.random.default_rng(11)
    table = rng.normal(scale=3.0, size=(40, 18)).astype(np.float32)
    check("select_channels on a whole (N, 18) table flattens: six numbers out of row 0, not a "
          "column slice -- which is why the corpus proof does not call it that way",
          select_channels(table).shape == (6,)
          and np.array_equal(select_channels(table), table[0, list(CHANNELS)]))
    check("called the way the live path calls it -- one 18-vector at a time -- it is exactly the "
          "frozen column slice, in order, for every row",
          np.array_equal(np.asarray([select_channels(v) for v in table]),
                         table[:, list(CHANNELS)]))
    check("the frozen channel set is the only one the gather can produce: channels %s in that "
          "order, and no neighbouring channel substitutes for one of them"
          % (list(CHANNELS),),
          list(CHANNELS) == [0, 1, 2, 12, 13, 16]
          and np.array_equal(select_channels(ALL18), ALL18[[0, 1, 2, 12, 13, 16]])
          and not np.array_equal(select_channels(ALL18), ALL18[[0, 1, 2, 3, 4, 5]])
          and not np.array_equal(select_channels(ALL18), ALL18[[16, 13, 12, 2, 1, 0]]))


if __name__ == "__main__":
    test_arithmetic()
    test_channel_gather()
    test_callsite()
    test_stats_width()
    print("checks run: %d, failures: %d" % (RUN_CHECKS[0], len(FAILS)))
    if FAILS:
        print("FAILED: " + "; ".join(FAILS))
        raise SystemExit(1)
    print("ALL PASS")
