"""CPU proof that the recorder's panel wiring is the wiring, without MuJoCo, a model or a GPU.

`record40k.PanelSink` is the piece that turns a live rollout into the composed video, and four of
its five jobs are invisible in the finished mp4: which column of the env's force vector became which
headline number, which replan's which row became which action-chain line, where the event ticks come
from, and whether the frame it is drawing is the frame whose force it shows. A fifth job -- not
drawing the model's *predicted* force next to a measurement -- is invisible by design, so it is
asserted against the sink's own code, docstrings and comments stripped.

So this drives `PanelSink` with hand-made observations whose force values encode their own frame
number, and a stub `ActionSpy` whose chunks each carry a different number. Every expectation is
recomputed here rather than read back from the code path under test.

    py -3.13 panelsink_selftest40k.py      # local: numpy + PIL + cv2
    python3 panelsink_selftest40k.py       # pod: the same, with imageio/libx264
"""
from __future__ import annotations

import ast
import inspect
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import panelvideo40k as PV  # noqa: E402
import record40k as R  # noqa: E402
from force_stream import FORCE_OBS_KEY  # noqa: E402

RESULTS = []
REPLAN = 10
HORIZON = 32
N_FRAMES = 40


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print("%-66s %s  %s" % (name, "PASS" if ok else "FAIL", detail), flush=True)


def _raises(fn):
    try:
        fn()
    except (Exception, SystemExit) as exc:  # the sink answers a bad observation with SystemExit,
        return "%s: %s" % (type(exc).__name__, str(exc)[:110])  # which is not an Exception
    return None


def _code_of(fn_or_class):
    """Source rendered back from its AST, with every docstring and comment gone.

    Two checks below forbid a token from appearing at all. Against raw `getsource` they would be
    answered by the prose that explains why the token is *not* used -- so they are run on code only,
    which is the thing that could actually read a prediction.
    """
    tree = ast.parse(inspect.getsource(fn_or_class))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def force_for(f):
    """An 18-wide vector in which every channel says which frame it came from.

    Each channel gets a different slope, so a column swapped for another one changes the value, not
    just the label -- which is the failure this file exists to catch.
    """
    return np.array([(col + 1) * (f + 1) for col in range(PV.FORCE18_W)], dtype=np.float64)


def obs_for(f, size=None):
    s = size or PV.CAM_SRC
    return {"observation.images.front": np.full((s, s, 3), 30 + f % 200, dtype=np.uint8),
            "observation.images.left_wrist": np.full((s, s, 3), 120 + f % 120, dtype=np.uint8),
            FORCE_OBS_KEY: force_for(f)}


def step_rec(f):
    """What `EnvProbe` puts in `steps` for video frame f, including the reset frame.

    Frame 0 is the reset observation, so it carries no action at all -- `record_reset` hands
    `handed_14d=None`, and the panel must then draw no action line. From frame 1 on the command belongs
    to control step f-1, and column 12 carries that step number so a one-frame shift between the
    images, the forces and the actions changes a value rather than only a label.
    """
    if f == 0:
        return {"frame": 0, "handed_14d": None, "requested_14d": None, "terminated": False,
                "cookies_in_target": None, "safety_reason": None}
    cmd = [0.0] * 14
    cmd[6] = cmd[13] = 1.0
    cmd[12] = float(f - 1)
    return {"frame": f, "handed_14d": list(cmd), "requested_14d": list(cmd), "terminated": False,
            "cookies_in_target": None, "safety_reason": None}


class FakeSpy:
    """`ActionSpy.rows` with a known value per chunk, so a row can be traced back to its replan."""

    def __init__(self, n_chunks=4, wild_chunk=2):
        self.rows = []
        for c in range(n_chunks):
            model = np.zeros((1, HORIZON, 7), dtype=np.float32)
            denorm = np.zeros((HORIZON, 7), dtype=np.float32)
            pm1 = []
            for r in range(HORIZON):
                model[0, r] = float(c) + 0.01 * r
                denorm[r] = 100.0 * c + r
                pm1.append(np.full(7, 100.0 * c + r, dtype=np.float32))
            # one dimension of one chunk pushed far past the +-5 band, like the 44.37 excursion
            if c == wild_chunk:
                model[0, 3, 4] = 44.37
                denorm[3, 4] = 44.37
            self.rows.append({"model_out": model, "denorm": denorm, "pm1": pm1})


def main():
    out = Path(tempfile.mkdtemp(prefix="panelsink_selftest_"))

    # --- the chain lookup: which replan's which row produced control step f
    spy = FakeSpy()
    for f, want_chunk, want_row in ((1, 0, 0), (10, 0, 9), (11, 1, 0), (25, 2, 4), (40, 3, 9)):
        got = R.chain_for_frame(spy, REPLAN, f)
        check("control step %d reads chunk %d row %d of the action chain" % (f, want_chunk,
                                                                            want_row),
              got.get("denorm_row") is not None
              and abs(got["denorm_row"][0] - (100.0 * want_chunk + want_row)) < 1e-6
              and abs(got["pm1_row"][0] - (100.0 * want_chunk + want_row)) < 1e-6,
              "denorm_row[0]=%s" % (got.get("denorm_row") or [None])[0])
    wild = R.chain_for_frame(spy, REPLAN, 2 * REPLAN + 4)      # chunk 2, row 3
    check("the 44.37 in chunk 2 dim d_ry shows as that chunk's absmax, not a later chunk's",
          abs(wild["model_absmax_dims"][4] - 44.37) < 1e-3
          and max(wild["model_absmax_dims"][:4] + [wild["model_absmax_dims"][5]]) < 4.0,
          "absmax per dim=%s" % [round(x, 2) for x in wild["model_absmax_dims"]])
    check("and as the fraction of that chunk's rows past the +-5 band (%d/32)" % 1,
          abs(wild["model_over5_frac"][4] - 1.0 / HORIZON) < 1e-9
          and all(v == 0.0 for i, v in enumerate(wild["model_over5_frac"]) if i != 4),
          "over5=%s" % [round(x, 4) for x in wild["model_over5_frac"]])
    check("a frame past every recorded chunk yields no chain rather than a stale one",
          R.chain_for_frame(spy, REPLAN, REPLAN * len(spy.rows) + 1) == {},
          str(R.chain_for_frame(spy, REPLAN, REPLAN * len(spy.rows) + 1)))

    # --- the sink: one composed frame per control step, force taken from the env
    sink = R.PanelSink(out / "seed11_step040000_composed_panel.mp4", fps=20, seed=11,
                       step="step_040000", total_frames=N_FRAMES, replan=REPLAN)
    sink.chain_fn = lambda f: R.chain_for_frame(spy, REPLAN, f)
    anns, imgs = [], []
    real_add = sink.rec.add

    def spy_add(front, left, ann):
        anns.append(ann)
        imgs.append((int(front[0, 0, 0]), int(left[0, 0, 0])))
        return real_add(front, left, ann)

    sink.rec.add = spy_add
    for f in range(N_FRAMES):
        sink.add_frame(obs_for(f), step_rec(f), None)
    stats = sink.close()
    check("composed mp4 written by %s, one frame per control step" % stats["backend"],
          stats["frames_written"] == N_FRAMES and stats["bytes"] > 20000,
          "%d frames, %.1f KB, %s" % (stats["frames_written"], stats["bytes"] / 1024.0,
                                      stats["wh"]))
    back = PV.read_frames(stats["path"])
    check("it decodes back with the same frame count and geometry",
          len(back) == N_FRAMES and back[0].shape == (PV.PANEL_H, 2 * PV.CAM_W + PV.PANEL_W, 3),
          "%d frames, %s" % (len(back), None if not back else list(back[0].shape)))

    want = {c: [(PV.PAD_COLS[c] + 1) * (f + 1) for f in range(N_FRAMES)] for c in PV.PAD_ORDER}
    got = {c: [round(v, 6) for v in sink.traces[c]] for c in PV.PAD_ORDER}
    check("every headline channel is the column it claims, per frame, with no offset",
          got == want, "l_inner first/last=%s/%s expected %s/%s"
          % (got["l_inner"][0], got["l_inner"][-1], want["l_inner"][0], want["l_inner"][-1]))
    check("the four traces are one sample per emitted frame and agree on length",
          all(len(sink.traces[c]) == stats["frames_written"] for c in PV.PAD_ORDER),
          str({c: len(sink.traces[c]) for c in PV.PAD_ORDER}))
    check("the stats name each headline channel by its env column, for anyone re-deriving the curve",
          stats["pad_channel_columns"] == {"l_inner": 12, "l_outer": 13,
                                           "r_inner": 14, "r_outer": 15},
          str(stats["pad_channel_columns"]))

    # --- the frame/step alignment the deliverable is judged on: one control step, four kinds of proof
    a0 = anns[0]
    check("frame 0 is the reset observation: force measured, no action line and no zero-filled one",
          a0.get("executed_row") is None and a0.get("ik_cmd_row") is None
          and a0["ctrl_step"] == 0 and a0["chunk"] == 0
          and abs(a0["pad_n"]["l_inner"] - force_for(0)[12]) < 1e-9,
          "executed=%s ik=%s ctrl=%s l_inner=%s" % (a0.get("executed_row"), a0.get("ik_cmd_row"),
                                                    a0["ctrl_step"], a0["pad_n"]["l_inner"]))
    drift = []
    for k in range(1, N_FRAMES):
        a = anns[k]
        if (a["ctrl_step"] != k - 1 or a["chunk"] != (k - 1) // REPLAN
                or a.get("executed_row") is None or a["executed_row"][12] != float(k - 1)
                or a.get("ik_cmd_row") != a["executed_row"]
                or abs(a["pad_n"]["l_inner"] - force_for(k)[12]) > 1e-9
                or imgs[k] != (30 + k % 200, 120 + k % 120)):
            drift.append(k)
    check("every later frame's command, IK input, force and both cameras come from step frame-1",
          not drift, "offending frames: %s" % drift[:5])
    check("the executed line can only come from handed_14d: the chain supplies no such key",
          "executed_row" not in R.chain_for_frame(spy, REPLAN, 25)
          and "executed_row" not in _code_of(R.chain_for_frame),
          str(sorted(R.chain_for_frame(spy, REPLAN, 25))))

    # --- event ticks, which are the answer to "when did each fingertip touch"
    sink2 = R.PanelSink(out / "_ticks.mp4", fps=20, seed=17, step="step_040000",
                        total_frames=30, replan=REPLAN)
    onset = {"l": 7, "r": 19}
    for f in range(30):
        rec = step_rec(f)
        obs = obs_for(f)
        fv = obs[FORCE_OBS_KEY]
        for col, side in ((12, "l"), (13, "l"), (14, "r"), (15, "r")):
            fv[col] = 0.0 if f < onset[side] else float(col - 11) * 100.0 * (1 if side == "l"
                                                                             else -1)
        sink2.add_frame(obs, rec, None)
        if f == 12:
            sink2.emergency.append({"frame": 12, "args": ["force limit"]})
    ticks_at_end = sink2._ticks(29)
    tf = sorted((t["text"].split()[0], t["frame"]) for t in ticks_at_end)
    check("each side's contact tick is drawn at the frame its own channel crossed the floor",
          tf == [("L_pad", onset["l"]), ("R_pad", onset["r"]), ("emergency", 12)], str(tf))
    check("a tick that has not happened yet is not drawn",
          sink2._ticks(onset["l"] + 1)
          and all(t["frame"] <= onset["l"] + 1 for t in sink2._ticks(onset["l"] + 1)),
          str([(t["text"], t["frame"]) for t in sink2._ticks(onset["l"] + 1)]))
    sink2.close()

    shared = []
    sink3 = R.PanelSink(out / "_shared.mp4", fps=20, seed=1, step="x", total_frames=5,
                        replan=REPLAN)
    sink3.emergency = shared
    shared.append({"frame": 3, "args": ["force limit exceeded"]})
    check("an emergency stop reaches the panel through one shared list, not a copy",
          any(t["text"] == "emergency stop" and t["frame"] == 3 for t in sink3._ticks(4))
          and "panel.emergency = probe.emergency" in inspect.getsource(R.run_episode_capture),
          str(sink3._ticks(4)))
    cap_src = inspect.getsource(R.run_episode_capture)
    check("the composed video is only asked for when the flag is on, and refuses a camera it lacks",
          "if ctx.panel_video:" in cap_src and "--rec-panel-video needs the front and left_wrist"
          in cap_src, "")

    # --- the refusals: each is a way a real recording could look fine and mean nothing
    def fresh(obs=None, rec=None, p14=None, name="_refuse"):
        s = R.PanelSink(out / (name + ".mp4"), fps=20, seed=1, step="x", total_frames=5,
                        replan=REPLAN)
        s.add_frame(obs or obs_for(0), rec or step_rec(0), p14)
        return s

    def repeated_index():
        s = R.PanelSink(out / "_repeat.mp4", fps=20, seed=1, step="x", total_frames=5,
                        replan=REPLAN)
        s.add_frame(obs_for(0), step_rec(0), None)
        s.add_frame(obs_for(1), step_rec(0), None)      # frame 0 twice

    cases = [
        ("an observation whose force is the 6-channel proprio subset, not the env's 18",
         lambda: fresh(obs=dict(obs_for(0), **{FORCE_OBS_KEY: np.zeros(6)}))),
        ("an observation with no force at all",
         lambda: fresh(obs={k: v for k, v in obs_for(0).items() if k != FORCE_OBS_KEY})),
        ("a camera that is not 256x256",
         lambda: fresh(obs=dict(obs_for(0), **{"observation.images.front":
                                               np.zeros((128, 128, 3), np.uint8)}))),
        ("the left_wrist camera missing",
         lambda: fresh(obs={k: v for k, v in obs_for(0).items()
                            if k != "observation.images.left_wrist"})),
        ("a NaN force from the simulator",
         lambda: fresh(obs=dict(obs_for(0), **{FORCE_OBS_KEY:
                                               np.array([np.nan] + [0.0] * 17)}))),
        ("a video frame index handed twice, which is how a force trace drifts a frame",
         repeated_index),
    ]

    for name, fn in cases:
        msg = _raises(fn)
        check("refuses to record: %s" % name, msg is not None, msg or "NO ERROR RAISED")

    src = _code_of(R.PanelSink)
    check("the panel's newtons can only come from the env: no prediction reaches the sink",
          "force_future" not in src and "pred" not in src and "FORCE_OBS_KEY" in src,
          "in code (docstrings and comments stripped): force_future=%s, 'pred'=%s, FORCE_OBS_KEY=%s"
          % ("force_future" in src, "pred" in src, "FORCE_OBS_KEY" in src))
    check("no pad channel is ever added into one number",
          "pad_l_n" not in src and not any(k in src for k in ("l_inner + ", "sum(self.traces")),
          "the only numbers the sink builds are the four per-channel readings: %s"
          % sorted(k for k in src.split() if "pad" in k)[:6])
    step_src = inspect.getsource(R.EnvProbe.install_on_env)
    reset_src = inspect.getsource(R.EnvProbe.record_reset)
    check("the panel is fed the same post-step observation object as that step's camera video",
          "self.sink.add(obs," in step_src and "self.panel.add_frame(obs," in step_src,
          "step=%s reset=%s" % ("self.panel.add_frame(obs," in step_src,
                                "self.panel.add_frame(obs," in reset_src))
    check("frame 0, before the policy moved anything, is in the composed video too",
          "self.panel.add_frame(obs," in reset_src, "record_reset feeds the panel")

    n_fail = RESULTS.count(False)
    print("\nartifacts: %s" % out)
    print("%s: %d/%d checks passed" % ("FAIL" if n_fail else "PASS",
                                       len(RESULTS) - n_fail, len(RESULTS)))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
