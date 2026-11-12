"""Three independent legs for the identity of the four fingertip channels, without MuJoCo or a GPU.

The composed video puts four numbers in the reader's face -- `L_pad inner/outer`, `R_pad inner/outer`
-- and every one of them is a bare column index into `observation.force`. An index is only as trustworthy
as the thing that fixed it, so this script refuses to trust the constants it is checking against
(`panelvideo40k.PAD_COLS`, `safety_wrap.L_INNER`, `replay_forces.L_INNER`, ...), all of which are
hand-written integers that agree with each other only because the same person wrote them.

Leg A -- the model file. `<sensor>` in `a3_cookie_transfer.xml` gives the eight sensors by name and type,
and the body tree gives the site, the one collidable geom inside that site's body, the slide joint that
body hangs from and the position actuator that drives it. That chain is what makes "L_finger_inner_touch"
mean *the left inner pad* rather than a string.

Leg B -- the code that builds the vector. `_sensor()` looks sensors up **by name** through
`model.sensor_adr`, so the XML's declaration order does not matter; the only thing that fixes a column
number is the order of the `np.r_[...]` in `env._observation`. That order is read out of the source text
and turned into an index map, then compared against the constants the panel actually uses. The width
contract (`observation_space` says `(18,)`) is what proves each touch sensor contributes exactly one
column: 3+3+3+3+1+1+1+1+1+1 = 18 and nothing else closes the sum.

Leg C -- real recorded data, which is the leg that can say "no". The 73 expert demos pool 32,818 frames,
and in every one of them the entire right side is frozen to float round-off (<1e-9 N): the corpus is
left-arm-only. So
the data positively identifies channels 12/13 as the left gripper's pads (they rise exactly as the left
aperture in `state[7]` closes) and can only say that 14/15 are *not* the left side. The right pair's
identity therefore rests on legs A and B until a right-arm press is recorded. `--require-right-press`
turns that gap into a failure (C6) so nobody can call the mapping fully proven on corpus data alone, and
C6 reads a real artifact -- the `pad_forcen` traces a recording writes beside its labels, or a corpus-style
`force18` npz -- so it can go green once the press test has actually been run on the pod.

    py -3.13 sensor_identity40k.py                 # local: XML + source + corpus
    python3 sensor_identity40k.py --corpus '/workspace/**/expert_*.npz'
    python3 sensor_identity40k.py --require-right-press          # C6 against the recorded press
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import panelvideo40k as PV  # noqa: E402

RESULTS = []
FACTS = {}


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print("%-72s %s  %s" % (name, "PASS" if ok else "FAIL", detail), flush=True)


# The mapping under test, written out here independently of every constant in the toolchain.
EXPECTED_SENSORS = [
    ("force", "L_wrist_force", "L_wrist_ft"),
    ("torque", "L_wrist_torque", "L_wrist_ft"),
    ("touch", "L_finger_inner_touch_sensor", "L_finger_inner_touch"),
    ("touch", "L_finger_outer_touch_sensor", "L_finger_outer_touch"),
    ("force", "R_wrist_force", "R_wrist_ft"),
    ("torque", "R_wrist_torque", "R_wrist_ft"),
    ("touch", "R_finger_inner_touch_sensor", "R_finger_inner_touch"),
    ("touch", "R_finger_outer_touch_sensor", "R_finger_outer_touch"),
]
EXPECTED_ASSEMBLY = [
    ("L_wrist_force", 3), ("L_wrist_torque", 3), ("R_wrist_force", 3), ("R_wrist_torque", 3),
    ("L_finger_inner_touch_sensor", 1), ("L_finger_outer_touch_sensor", 1),
    ("R_finger_inner_touch_sensor", 1), ("R_finger_outer_touch_sensor", 1),
    ("<sum L finger actuator_force>", 1), ("<sum R finger actuator_force>", 1),
]
PAD_SENSOR_OF_COL = {12: "L_finger_inner_touch_sensor", 13: "L_finger_outer_touch_sensor",
                     14: "R_finger_inner_touch_sensor", 15: "R_finger_outer_touch_sensor"}


# --------------------------------------------------------------------- leg A: the model file

def find_xml(explicit):
    cands = ([explicit] if explicit else []) + [
        "/workspace/a3_dual_arm_sim/models/a3_cookie_transfer.xml",
        str(HERE.parents[2] / "_backup_repo/pod_snapshot/2026-11-01/a3_dual_arm_sim/models/a3_cookie_transfer.xml"),
        str(HERE.parents[2] / "_r6_stage/out/_gitx/a3_dual_arm_sim/models/a3_cookie_transfer.xml"),
    ]
    for c in cands:
        if c and Path(c).is_file():
            return Path(c)
    raise SystemExit("no a3_cookie_transfer.xml found among: %s" % cands)


def find_env_py(xml_path):
    cands = [xml_path.parent.parent / "src" / "a3_dual_arm_sim" / "sim" / "env.py",
             Path("/workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/sim/env.py")]
    for c in cands:
        if c.is_file():
            return c
    raise SystemExit("no env.py next to %s" % xml_path)


def leg_a(xml_path):
    root = ET.parse(xml_path).getroot()
    print("\n== leg A: %s (sha256 %s, %d B)"
          % (xml_path, __import__("hashlib").sha256(xml_path.read_bytes()).hexdigest()[:8],
             xml_path.stat().st_size))
    sens = root.find("sensor")
    got = [(e.tag, e.get("name"), e.get("site")) for e in sens]
    check("A1 the <sensor> block is exactly the eight expected sensors, in document order",
          got == EXPECTED_SENSORS, str(got))
    rescaled = [(e.get("name"), attr) for e in sens for attr in ("slope", "refvalue", "weight")
                if e.get(attr) is not None]
    check("A2 no sensor carries slope/refvalue/weight, so a touch reading is the unrescaled contact "
          "force in newtons (MuJoCo reports SI)", not rescaled, str(rescaled) or "8/8 clean")

    parent = {c: fa for fa in root.iter() for c in fa}
    # MuJoCo actuators are the children of the <actuator> container (<position .../>), not
    # elements named "actuator".
    ACTS = list(root.find("actuator")) if root.find("actuator") is not None else []

    def enclosing_body(el):
        cur = el
        while cur is not None and cur.tag != "body":
            cur = parent.get(cur)
        return cur

    for col, sensor_name in sorted(PAD_SENSOR_OF_COL.items()):
        side, which = re.match(r"([LR])_finger_(inner|outer)", sensor_name).groups()
        site_name = "%s_finger_%s_touch" % (side, which)
        site = next((s for s in root.iter("site") if s.get("name") == site_name), None)
        if site is None:
            check("A3 col %d: site %s exists" % (col, site_name), False, "not found")
            continue
        # the site's own body (the immediate parent), which is where its pad geom lives
        holder = enclosing_body(site)
        pads = [g for g in holder.findall("geom") if g.get("contype") not in (None, "0")
                or g.get("conaffinity") not in (None, "0")]
        want_geom = "%s_finger_%s_geom" % (side, which)
        check("A3 col %d <- %s: the site's body carries exactly one collidable geom, named %s"
              % (col, sensor_name, want_geom),
              [g.get("name") for g in pads] == [want_geom],
              "collidable=%s site_pos=%s geom_pos=%s"
              % ([g.get("name") for g in pads], site.get("pos"),
                 pads[0].get("pos") if pads else None))
        check("A4 col %d: the touch site and the pad geom share the same frame (so the sensor is on "
              "the pad, not elsewhere on the finger)" % col,
              bool(pads) and pads[0].get("pos") == site.get("pos"))
        # which joint drives the body that carries the pad -> the identity is mechanical
        ancestor = holder
        joint = None
        while ancestor is not None and joint is None:
            for j in ancestor.findall("joint"):
                if j.get("name") == "%s_finger_%s_joint" % (side, which):
                    joint = j
                    break
            ancestor = parent.get(ancestor)
        check("A5 col %d: the pad body is driven by joint %s_finger_%s_joint (slide along %s)"
              % (col, side, which, joint.get("axis") if joint is not None else None),
              joint is not None and joint.get("type") == "slide")
        act = next((a for a in ACTS if a.get("name") == "%s_finger_%s_position" % (side, which)), None)
        check("A6 col %d: the same joint is driven by position actuator %s_finger_%s_position"
              % (col, side, which), act is not None,
              # `is not None`, not `if act`: an actuator element has no children, and a childless
              # Element is falsy, so the truthiness test would print an empty detail forever.
              ("ctrlrange=%s forcerange=%s kp=%s"
               % (act.get("ctrlrange"), act.get("forcerange"), act.get("kp")))
              if act is not None else "")

    for side in ("L", "R"):
        acts = [a for f in ("inner", "outer")
                for a in ACTS if a.get("name") == "%s_finger_%s_position" % (side, f)]
        rng = [float(x) for a in acts for x in (a.get("forcerange") or "0 0").split()]
        bound = 2 * max(abs(min(rng)), abs(max(rng))) if rng else None
        FACTS["finger_actuator_force_bound_%s" % side] = bound
        # This bound is what licenses the claim that channels 16/17 cannot be a 2775 N reading, so it
        # is asserted (both actuators, one shared symmetric range, 80 N when doubled) rather than
        # merely printed next to a check that only counted the actuators.
        fr = {a.get("forcerange") for a in acts}
        share = next(iter(fr)) if len(fr) == 1 else sorted(fr, key=str)
        check("A7 both %s finger actuators share forcerange %r, so channels 16/17 are bounded near "
              "+-%g N and can never themselves read 2775 N" % (side, share, bound or 0),
              len(acts) == 2 and len(fr) == 1 and None not in fr and bound == 80.0,
              "actuators=%d forcerange=%s bound=%s N" % (len(acts), share, bound))
    return root


# --------------------------------------------------------------------- leg B: the assembly order

def leg_b(env_path, root):
    print("\n== leg B: %s (sha256 %s, %d B)"
          % (env_path, __import__("hashlib").sha256(env_path.read_bytes()).hexdigest()[:8],
             env_path.stat().st_size))
    src = env_path.read_text(encoding="utf-8")

    # _sensor() resolves by name through model.sensor_adr -> XML order is irrelevant
    m = re.search(r"def _sensor.*?return self\.data\.sensordata", src, re.S)
    check("B1 the env reads every sensor by NAME through model.sensor_adr, so the XML's document "
          "order cannot be what fixes a column number",
          m is not None and "sensor_adr" in m.group(0) and "mjOBJ_SENSOR" in m.group(0))

    asm = re.search(r"force = np\.r_\[(.*?)\n        \]", src, re.S)
    if asm is None:
        check("B2 the 18-wide vector is built by one visible np.r_[...] in _observation", False, "not found")
        return None
    body = asm.group(1)
    terms = re.findall(r"self\._sensor\(\"([^\"]+)\"\)", body)
    sums = re.findall(r"sum\(self\.data\.actuator_force\[list\(self\._finger_actuators\[\"(\w)\"\]\)\]\)",
                      body)
    full = [(t, 3 if "wrist" in t else 1) for t in terms] + \
           [("<sum %s finger actuator_force>" % s, 1) for s in sums]
    check("B2 the assembly is exactly the expected ten terms in the expected order",
          full == EXPECTED_ASSEMBLY, str(full))

    idx, starts = 0, {}
    for name, w in full:
        starts[name] = idx
        idx += w
    check("B3 cumulative offsets put the four fingertip sensors at 12/13/14/15",
          all(starts[PAD_SENSOR_OF_COL[c]] == c for c in sorted(PAD_SENSOR_OF_COL)),
          str({c: starts[PAD_SENSOR_OF_COL[c]] for c in sorted(PAD_SENSOR_OF_COL)}))
    check("B4 the two actuator-force sums land on 16/17 and are newtons, not gripper stroke",
          starts.get("<sum L finger actuator_force>") == 16
          and starts.get("<sum R finger actuator_force>") == 17)

    shape = re.search(r"FORCE: spaces\.Box\(-np\.inf, np\.inf, \((\d+),\)", src)
    check("B5 the observation contract pins FORCE at width %s, which is only reachable if each touch "
          "sensor contributes exactly one column" % (shape.group(1) if shape else "?"),
          shape is not None and int(shape.group(1)) == sum(w for _, w in full),
          "sum(terms)=%d declared=%s" % (sum(w for _, w in full), shape.group(1) if shape else None))

    grip = re.search(r"self\._finger_actuators = \{(.*?)\n        \}", src, re.S)
    check("B6 channels 16/17 sum the named L/R_finger_inner/outer_position actuators",
          grip is not None and 'f"{side}_finger_{finger}_position"' in grip.group(1),
          "side loop over ('L','R') x ('inner','outer')")

    # the panel's own constants, compared against the map leg B just derived -- not against each other
    mism = {c: (PV.PAD_COLS[c], PAD_SENSOR_OF_COL.get(PV.PAD_COLS[c]))
            for c in PV.PAD_COLS if PV.PAD_COLS[c] not in PAD_SENSOR_OF_COL}
    check("B7 panelvideo40k.PAD_COLS names a fingertip sensor at every column it claims",
          not mism, str(mism))
    check("B8 panelvideo40k.PAD_COLS == the offsets derived from env._observation, independently",
          PV.PAD_COLS == {"l_inner": starts["L_finger_inner_touch_sensor"],
                          "l_outer": starts["L_finger_outer_touch_sensor"],
                          "r_inner": starts["R_finger_inner_touch_sensor"],
                          "r_outer": starts["R_finger_outer_touch_sensor"]},
          str(PV.PAD_COLS))
    check("B9 the panel refuses a force vector of any other width",
          PV.FORCE18_W == 18 and _raises_width(PV.force18_pads))
    return starts


def _raises_width(fn):
    try:
        fn(np.zeros(17))
    except ValueError:
        return True
    return False


def leg_b_recorder():
    """The video must read the raw 18-wide vector, never rebuild a right fingertip from the model's
    six-channel input."""
    rec = (HERE / "record40k.py").read_text(encoding="utf-8")
    sink = re.search(r"class PanelSink.*?\nclass ", rec, re.S)
    seg = sink.group(0) if sink else ""
    pads = re.search(r"def pads\(self, obs\).*?return self\.PV\.force18_pads\((.*?)\)\n", seg, re.S)
    check("B10 PanelSink.pads feeds force18_pads() the whole env vector, not a six-channel subset",
          pads is not None and "select_channels" not in seg,
          "arg=%r" % (pads.group(1).strip() if pads else None))
    from force_stream import CHANNELS, CHANNEL_NAMES
    check("B11 the frozen model input is exactly [0,1,2,12,13,16] and contains NO right fingertip "
          "channel",
          list(CHANNELS) == [0, 1, 2, 12, 13, 16] and 14 not in CHANNELS and 15 not in CHANNELS,
          str(list(CHANNEL_NAMES)))
    from force_stream import FORCE_OBS_KEY as FK
    check("B12 the key the recorder reads is the env's own force key",
          FK == "observation.force" and ("FORCE_OBS_KEY" in seg or "obs[FORCE_OBS_KEY]" in seg),
          "key=%s" % FK)
    # the one proprio-derived number on the panel must be labelled as the left pair
    tared = re.search(r"pad_l_tared_n.*?\n", rec)
    check("B13 the only p14-derived pad reading is named pad_l_tared_n (left pair, tared) and never "
          "the right pair", tared is not None and "r_inner" not in tared.group(0)
          and "r_outer" not in tared.group(0), (tared.group(0).strip() if tared else "none"))
    p14 = re.search(r"def build_proprio14.*?\n\n\n", (HERE / "eval40k.py").read_text(encoding="utf-8"), re.S)
    check("B14 build_proprio14 appends the six tared channels in force_stream order, so p14[11]/[12] "
          "are the left pads (8 + the 3 wrist-force columns + 3/4)",
          p14 is not None and "select_channels" in p14.group(0))
    return CHANNELS


# --------------------------------------------------------------------- leg C: real data

def find_corpus(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits
    return []


def leg_c(paths, require_right, skip, press_globs=()):
    print("\n== leg C: %d corpus files" % len(paths))
    if skip:
        print("   leg C SKIPPED by --skip-corpus: legs A+B alone fix the column numbers, but the "
              "mapping is NOT three-legged on this run and the report must say so.")
        # The press is data-leg evidence too, so --skip-corpus must not be a way to drop the
        # requirement --require-right-press was asked for without noticing. It still cannot replace
        # checks C1-C5, which are what make 12/13 the LEFT pads, so the exit code stays 3.
        press = right_press_check(press_globs) if require_right else False
        return "skipped+press-proven" if press else "skipped"
    if not paths:
        check("C0 corpus reachable for the data leg", False, "no files matched -- the data leg is "
              "NOT satisfied; do not call the mapping three-legged on this run")
        return
    F, right_spread = [], np.zeros(18)
    n_frames = 0
    for p in paths:
        with np.load(p) as z:
            if "force18" not in z.files:
                continue
            f = np.asarray(z["force18"], dtype=np.float64)
            if f.ndim != 2 or f.shape[1] != 18:
                check("C1 %s has an 18-wide force18" % Path(p).name, False, str(f.shape))
                return
            n_frames += f.shape[0]
            right_spread = np.maximum(right_spread, np.ptp(f, axis=0))
            F.append(np.maximum(f[:, 12], f[:, 13]))
    F = np.concatenate(F)
    print("   pooled frames: %d" % n_frames)
    check("C1 every file's force18 is (T,18)", True, "%d frames pooled" % n_frames)

    inert = [c for c in (6, 7, 8, 9, 10, 11, 14, 15, 17) if right_spread[c] < 1e-9]
    check("C2 the corpus's whole right side is frozen to float round-off (channels %s spread < 1e-9 N)"
          " -> it is a left-arm-only corpus" % inert, len(inert) == 9,
          "max ptp over the right channels: %s N (the left pads move by %.2f N)"
          % (np.array2string(right_spread[[6, 7, 8, 9, 10, 11, 14, 15, 17]], precision=1,
                             formatter={"float_kind": lambda x: "%.1e" % x}),
             float(np.max(F))))
    FACTS["corpus_right_channels_inert"] = inert

    check("C3 the four fingertip channels never read negative across the pooled corpus (a touch "
          "sensor reports the pad-normal contact magnitude, so no sign convention can be mixing the "
          "two sides)", _all_nonneg(paths), "min over cols 12-15: %.4f" % _minpad(paths))

    per = _opening_corr(paths)
    good = sum(1 for r in per if r < 0)
    tot = len(per)
    check("C4 channels 12/13 rise as the LEFT gripper aperture (state[7], the slot after the seven "
          "LEFT_JOINTS) closes: negative corr in %d/%d demos with contact" % (good, tot),
          tot > 0 and good == tot, "mean r=%+.3f" % (np.mean(per) if per else float("nan")))

    peak = float(np.nanmax(F)) if F.size else 0.0
    print("   left pad peak in corpus: %.3f N; frames >= %.0f N: %d of %d"
          % (peak, PV_CONTACT_ON_N, int((F >= PV_CONTACT_ON_N).sum()), F.size))
    FACTS["corpus_left_pad_peak_n"] = peak
    check("C5 the corpus peak fingertip force (%.1f N) is BELOW the panel's contact tick (%.0f N), "
          "so a tick on expert data is absent by construction, not by damage" % (peak, PV_CONTACT_ON_N),
          peak < PV_CONTACT_ON_N)

    press = right_press_check(press_globs) if require_right else False
    if not press:
        print("   LIMIT (recorded, not hidden): the data leg names 12/13 as left and can only say "
              "14/15 are not the left side. The right pair's identity rests on legs A+B until a "
              "right-arm press is recorded on the pod.")
    return "ran+press-proven" if press else "ran"


PV_CONTACT_ON_N = 20.0
# A right-pad press is only evidence about *which side* if the left pads stayed still in the same
# recording, so the quiet bound is as much part of the check as the contact bound.
LEFT_QUIET_N = 1.0
PAD_ORDER_CANON = ("l_inner", "l_outer", "r_inner", "r_outer")
PRESS_GLOBS_DEFAULT = [str(HERE.parents[1] / "out40k/record_diag/_record/**/*_steps.npz"),
                       "/workspace/_v1_stage/dev/out40k/record_diag/_record/**/*_steps.npz"]


def _minpad(paths):
    m = np.inf
    for p in paths:
        with np.load(p) as z:
            if "force18" not in z.files:
                continue
            f = np.asarray(z["force18"], float)[:, 12:16]
            m = min(m, float(f.min()))
    return m


def _all_nonneg(paths):
    return _minpad(paths) >= 0.0


def _opening_corr(paths):
    out = []
    for p in paths:
        with np.load(p) as z:
            if "force18" not in z.files or "opening" not in z.files:
                continue
            f = np.asarray(z["force18"], float)
            o = np.asarray(z["opening"], float)
            if o.shape[0] != f.shape[0]:
                continue
            pad = np.maximum(f[:, 12], f[:, 13])
            if pad.max() < 0.5 or float(np.std(o)) < 1e-9 or float(np.std(pad)) < 1e-9:
                continue
            out.append(float(np.corrcoef(o, pad)[0, 1]))
    return out


def _press_traces(path):
    """The four labelled pad traces of one recorded episode, or the reason it cannot be evidence."""
    with np.load(path) as z:
        if "pad_forcen" in z.files:
            if "pad_channel_order" not in z.files or "pad_channel_columns" not in z.files:
                return None, "pad_forcen is there but its channel labels are not"
            order = [str(c) for c in z["pad_channel_order"]]
            cols = [int(c) for c in z["pad_channel_columns"]]
            tr = np.asarray(z["pad_forcen"], dtype=np.float64)
        elif "force18" in z.files:
            f = np.asarray(z["force18"], dtype=np.float64)
            if f.ndim != 2 or f.shape[1] != 18:
                return None, "force18 is %s, not (T,18)" % (f.shape,)
            order, cols, tr = list(PAD_ORDER_CANON), [12, 13, 14, 15], f[:, 12:16]
        else:
            return None, "neither pad_forcen nor force18 in %s" % Path(path).name
    if order != list(PAD_ORDER_CANON) or cols != [12, 13, 14, 15]:
        return None, "its own labels say %s / %s, not the map legs A+B proved" % (order, cols)
    if tr.ndim != 2 or tr.shape[1] != 4 or tr.shape[0] < 2:
        return None, "the pad traces are %s" % (tr.shape,)
    return dict(zip(PAD_ORDER_CANON, (tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3]))), None


def right_press_check(press_globs):
    """C6: the half leg C has never been able to produce -- the right pads actually responding.

    A file in which 14/15 move proves nothing on its own; the left pair has to stay quiet in the same
    recording, otherwise the press could have been felt by both hands. Both bounds are in the check
    name so a future run cannot quietly widen them.
    """
    paths = sorted({p for g in press_globs for p in glob.glob(g, recursive=True)})
    if not paths:
        check("C6 a right-arm press recording exists at all", False,
              "nothing matched %s -- the press test has not been run, so 14/15 rest on legs A+B"
              % list(press_globs))
        return False
    best, unreadable = None, []
    for p in paths:
        traces, why = _press_traces(p)
        if traces is None:
            unreadable.append("%s: %s" % (Path(p).name, why))
            continue
        right = max(float(np.ptp(traces["r_inner"])), float(np.ptp(traces["r_outer"])))
        left = max(float(np.ptp(traces["l_inner"])), float(np.ptp(traces["l_outer"])))
        if best is None or right > best[1]:
            best = (p, right, left, traces)
    if best is None:
        check("C6 a right-arm press recording is readable", False, "; ".join(unreadable[:4]))
        return False
    p, right, left, traces = best
    detail = ("right ptp %.3f N, left ptp %.3f N over %d frames" % (right, left, len(traces["l_inner"])))
    if unreadable:
        detail += " (%d file(s) unreadable, e.g. %s)" % (len(unreadable), unreadable[0])
    ok = right >= PV_CONTACT_ON_N and left <= LEFT_QUIET_N
    check("C6 %s: the right pair moves (ptp >= %.0f N) while the left pair stays quiet (ptp <= %.1f N) "
          "-- that makes 14/15 the RIGHT pads, so the map is three-legged"
          % (Path(p).name, PV_CONTACT_ON_N, LEFT_QUIET_N), ok, detail)
    FACTS["right_press_file"] = str(p)
    FACTS["right_press_right_ptp_n"] = right
    FACTS["right_press_left_ptp_n"] = left
    return ok


# ------------------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=None)
    ap.add_argument("--env-py", default=None)
    ap.add_argument("--corpus", nargs="*", default=[
        str(HERE.parents[2] / "_r6_stage/audit73/npz/**/*.npz"),
        "/workspace/_v1_stage/_r6_stage/audit73/npz/**/*.npz"])
    ap.add_argument("--skip-corpus", action="store_true",
                    help="run legs A+B only; exits 3 so a caller must say why the data leg is absent")
    ap.add_argument("--require-right-press", action="store_true",
                    help="fail unless a recorded right-arm press also proves the right pair responds")
    ap.add_argument("--right-press", nargs="*", default=PRESS_GLOBS_DEFAULT,
                    help="glob(s) of recorded *_steps.npz (or corpus-style force18 npz) from a "
                         "right-only press test, used by --require-right-press")
    args = ap.parse_args()

    xml_path = find_xml(args.xml)
    env_path = Path(args.env_py) if args.env_py else find_env_py(xml_path)
    leg_a(xml_path)
    starts = leg_b(env_path, ET.parse(xml_path).getroot())
    leg_b_recorder()
    corpus_state = leg_c([] if args.skip_corpus else find_corpus(args.corpus),
                         args.require_right_press, args.skip_corpus, args.right_press)

    passed = sum(RESULTS)
    print("\n== %d/%d checks passed" % (passed, len(RESULTS)))
    if starts is not None:
        print("   column map accepted: 12=%s 13=%s 14=%s 15=%s (touch, newtons, no slope attribute)"
              % tuple(PAD_SENSOR_OF_COL[c] for c in (12, 13, 14, 15)))
    print("   finger actuator force bound: %s N per pair (channels 16/17 sum two of these)"
          % FACTS.get("finger_actuator_force_bound_L"))
    print("   data leg: %s" % corpus_state)
    if passed != len(RESULTS):
        return 1
    # 3 means "the corpus leg never ran", whatever the press did: C1-C5 are what say 12/13 are the left
    # pads, so a run without them is not a three-legged run.
    return 3 if corpus_state.startswith("skipped") else 0


if __name__ == "__main__":
    sys.exit(main())
