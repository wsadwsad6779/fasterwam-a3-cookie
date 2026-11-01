"""Per-segment force check for correction demos.

The episode-level force numbers in replay_correct.py mix the abnormal pre-correction
segment (which is supposed to be there) with the recovery segment (which must be
clean). This splits them at the recorded failure step and reports both sides.
"""
import json
import sys
from pathlib import Path

import numpy as np

for d in sys.argv[1:]:
    rows = json.loads((Path(d) / "summary.json").read_text())
    print("==", d)
    for r in rows:
        f = Path(d) / ("expert_correct_%d.npz" % r["seed"])
        if not f.exists():
            print("   seed", r["seed"], "missing npz")
            continue
        z = np.load(f)
        pad = np.maximum(z["pad_force_inner"], z["pad_force_outer"])
        cuts = [q["at_step"] for q in r["retry_log"]] or [0]
        segs = []
        prev = 0
        for cut in cuts + [len(pad)]:
            segs.append((prev, cut))
            prev = cut
        parts = []
        for i, (a, b) in enumerate(segs):
            w = pad[a:b]
            if len(w) == 0:
                continue
            parts.append("seg%d[%d:%d] n=%d peak=%.3f over1.6=%d over2=%d"
                          % (i, a, b, len(w), float(w.max()),
                             int(np.sum(w > 1.6)), int(np.sum(w > 2.0))))
        print("   seed=%s accept=%s nclamps=%d recover_peak=%.3f recover_over2=%d"
              % (r["seed"], r["accept"], len(r["clamps"]),
                 float(pad[cuts[-1]:].max()) if len(pad[cuts[-1]:]) else 0.0,
                 int(np.sum(pad[cuts[-1]:] > 2.0))))
        for p in parts:
            print("        ", p)
