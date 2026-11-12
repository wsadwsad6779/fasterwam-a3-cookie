"""Does `sensor_identity40k.py` actually refuse a wrong channel map, or does it only print PASS lines?

`sensor_identity40k.py` claims three legs of evidence for columns 12-15 of the env's 18-wide force
vector. A claim like that is worth as much as the failure it would notice, so this file breaks the
model XML seven different ways -- each one a mistake a real recording could be built on -- and checks
that the gate exits non-zero and names the check that caught it. An XML edit that leaves the gate
green is a hole in the gate, and this script says which one.

The eighth thing it grades is C6, the check that refuses to call the map three-legged until a right-arm
press has been recorded. Since no press recording exists yet, it is graded against four synthetic ones
built the way `record40k` writes a steps npz: one honest press that C6 must accept, and three that it
must refuse (both hands pressed, the right pads never reaching the contact tick, and a file whose own
labels are traded). Without them `--require-right-press` would be a switch that can never turn green,
which is another word for a switch nobody runs.

Nothing here touches the corpus, the weights or the pod: the tampered XML and the synthetic press files
go to temp directories and the data leg is skipped, because legs A+B are what the model file can be made
to falsify.

    py -3.13 sensor_identity_tamper40k.py
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sensor_identity40k as SI  # noqa: E402  (same discovery, so this runs on the pod too)

XML = SI.find_xml(None)
ENV_PY = SI.find_env_py(XML)

INNER_ACT = ('<position name="L_finger_inner_position" joint="L_finger_inner_joint" kp="400.0" '
             'ctrlrange="0 0.0425" forcerange="-40 40" />')
OUTER_ACT = ('<position name="L_finger_outer_position" joint="L_finger_outer_joint" kp="400.0" '
             'ctrlrange="0 0.0425" forcerange="-40 40" />')

# Each entry: what a real recording would then be wrong about, the edit, and the check that must fail.
TAMPERS = [
    ("finger actuator force bound disappears, so +-80 N is no longer a fact about cols 16/17",
     lambda s: s.replace(INNER_ACT, INNER_ACT.replace(' forcerange="-40 40"', ""), 1), "A7"),
    ("one finger actuator can push 500 N, so cols 16/17 stop having a bound worth quoting",
     lambda s: s.replace(INNER_ACT, INNER_ACT.replace("-40 40", "-500 500"), 1), "A7"),
    ("the two L pads disagree about their limit, so 'the pair is bounded by 80 N' is a guess",
     lambda s: s.replace(OUTER_ACT, OUTER_ACT.replace("-40 40", "-40 9"), 1), "A7"),
    ("left inner and right inner trade names, and the panel draws the right pad as L_pad",
     lambda s: (s.replace('name="L_finger_inner_touch_sensor"', 'name="__SWAP__"')
                .replace('name="R_finger_inner_touch_sensor"', 'name="L_finger_inner_touch_sensor"')
                .replace('name="__SWAP__"', 'name="R_finger_inner_touch_sensor"')), "A1"),
    ("a touch sensor is silently scaled, so the number on the panel is not newtons",
     lambda s: s.replace('<touch name="L_finger_outer_touch_sensor"',
                         '<touch slope="0.01" name="L_finger_outer_touch_sensor"', 1), "A2"),
    ("a touch sensor is offset, so a zero reading no longer means no contact",
     lambda s: s.replace('<touch name="R_finger_outer_touch_sensor"',
                         '<touch refvalue="5" name="R_finger_outer_touch_sensor"', 1), "A2"),
    ("the pad geom stops colliding, so the sensor hangs on a body that can never feel anything",
     lambda s: re.sub(r'(<geom name="L_finger_inner_geom"[^>]*?)contype="2" conaffinity="1"',
                      r'\1contype="0" conaffinity="0"', s, count=1), "A3"),
]


def run(xml_path: Path):
    return subprocess.run([sys.executable, str(HERE / "sensor_identity40k.py"),
                           "--xml", str(xml_path), "--env-py", str(ENV_PY), "--skip-corpus"],
                          capture_output=True, text=True)


# C6 is the check that turns "the right pair is only proved by legs A+B" into a refusal, so the press
# recording it claims to grade has to be able to fail it. Each case writes one steps npz exactly the way
# record40k writes it -- pad_forcen plus its two label arrays -- and says what the verdict must be. A
# PASS verdict here is not proof the press happened; it only proves the check would notice if it did.
PRESS_CASES = [
    ("a right-only press reaches contact while the left pads stay flat",
     dict(right=40.0, left=0.0), "PASS"),
    ("both hands were pressed, so the file says nothing about which side is which",
     dict(right=40.0, left=12.0), "FAIL"),
    ("the right pads never reach the panel's 20 N contact tick",
     dict(right=0.2, left=0.0), "FAIL"),
    ("the recording labels its own traces the other way round",
     dict(right=40.0, left=0.0, order=("r_inner", "r_outer", "l_inner", "l_outer"),
          cols=(14, 15, 12, 13)), "FAIL"),
]


def _press_npz(tmp_dir: Path, right, left, order=("l_inner", "l_outer", "r_inner", "r_outer"),
               cols=(12, 13, 14, 15)):
    tr = np.zeros((60, 4))
    hit = np.arange(60) >= 20
    (tr[hit, 2], tr[hit, 3], tr[hit, 0]) = (right, right / 2.0, left)
    p = tmp_dir / "press_steps.npz"
    np.savez_compressed(p, pad_forcen=tr, pad_channel_order=np.asarray(order),
                        pad_channel_columns=np.asarray(list(cols)))
    return p


def press_cases():
    tmp_dir = Path(tempfile.mkdtemp(prefix="sensor_press_"))
    glob_ = str(tmp_dir / "*_steps.npz")
    holes = []
    print("\n== the right-arm press check (C6), graded against synthetic press recordings ==")
    for what, kw, want in PRESS_CASES:
        for f in tmp_dir.glob("*"):
            f.unlink()
        _press_npz(tmp_dir, **kw)
        r = subprocess.run([sys.executable, str(HERE / "sensor_identity40k.py"), "--skip-corpus",
                            "--require-right-press", "--right-press", glob_],
                           capture_output=True, text=True)
        line = next((l.strip() for l in r.stdout.splitlines() if l.strip().startswith("C6")), "")
        got = "PASS" if " PASS" in line else ("FAIL" if " FAIL" in line else "NO C6 LINE")
        # --skip-corpus keeps the exit at 3 (checks C1-C5, which name the LEFT pads, never ran), so a
        # green C6 must not be able to pull the run back to 0.
        want_rc = 3 if want == "PASS" else 1
        ok = got == want and r.returncode == want_rc
        print("%-5s C6 %-62s verdict=%s rc=%d" % ("PASS" if ok else "FAIL", what[:60], got, r.returncode))
        if not ok:
            print("      expected verdict %s and rc %d, line: %s" % (want, want_rc, line[:110]))
            holes.append(what)
    return holes


def main() -> int:
    src = XML.read_text(encoding="utf-8")
    tmp = Path(tempfile.mkdtemp(prefix="sensor_tamper_")) / "tampered.xml"
    xml_holes = []
    for i, (what, edit, want_check) in enumerate(TAMPERS):
        t = edit(src)
        if t == src:
            print("FAIL  tamper %d edited nothing: %s" % (i, what))
            xml_holes.append(what)
            continue
        tmp.write_text(t, encoding="utf-8")
        r = run(tmp)
        failed = [l.strip() for l in r.stdout.splitlines() if " FAIL" in l]
        named = [l for l in failed if l.startswith(want_check)]
        passed = next((l for l in r.stdout.splitlines() if "checks passed" in l), "?")
        ok = r.returncode != 0 and bool(named)
        print("%-5s %-62s rc=%d  %s" % ("PASS" if ok else "FAIL", want_check + " " + what[:58],
                                        r.returncode, passed))
        if ok:
            print("      caught by: %s" % named[0][:120])
        if not ok:
            print("      %s" % (r.stderr.strip().splitlines() or ["(no output)"])[-1][:160])
            xml_holes.append(what)
    print("\nbaseline (untampered XML, data leg skipped):")
    r = run(XML)
    print("  rc=%d  %s" % (r.returncode, next((l for l in r.stdout.splitlines()
                                               if "checks passed" in l), "?")))
    if r.returncode != 3:
        print("  FAIL  the untampered gate must exit 3 here (legs A+B only, corpus skipped)")
        xml_holes.append("baseline exit code")
    press_holes = press_cases()
    holes = xml_holes + press_holes
    print("%s: %d/%d XML tampers refused, %d/%d press cases graded"
          % ("FAIL" if holes else "PASS", len(TAMPERS) - len(xml_holes), len(TAMPERS),
             len(PRESS_CASES) - len(press_holes), len(PRESS_CASES)))
    for h in holes:
        print("  hole: %s" % h)
    return 1 if holes else 0


if __name__ == "__main__":
    sys.exit(main())
