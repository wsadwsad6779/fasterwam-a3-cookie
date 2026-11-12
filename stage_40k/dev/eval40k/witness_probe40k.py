"""Why did the batch record no `raw14` per chunk? Decide it on the CPU, before re-running anything.

`install_chunk_recorder` writes `raw14`/`mapped_no_clamp14` only when the witness is not None, and
`make_proprio_witness` returns None when `W`, `calib` or `stats_band_vectors()` is None. All 11
recorded episodes have the keys present and set to None, so the witness was None for the whole node
-- which is why `proprio_band_map_pairs == 0` and why the +-5 clamp interlock was vacuous. This
script rebuilds the three inputs the way `eval40k.main` does, on the CPU, and prints which one is
None. It does not load a model and does not touch the results tree.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import eval40k as E
import smoke40k as S


def main() -> None:
    from fasterwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    calib = E.read_json(str(E.CALIB_PATH))
    stats = load_dataset_stats_from_json(str(E.V1G_STATS))
    facts = E.check_stats_width(stats, str(E.V1G_STATS), calib_std=calib["std"])
    print("STATS_FACTS keys: %s" % sorted(E.STATS_FACTS))
    print("bands keys: %s (need exactly 1)" % sorted(facts["bands"]))
    for k, b in sorted(facts["bands"].items()):
        print("  band %s: len(min)=%d len(max)=%d PROPRIO_DIM=%d"
              % (k, len(b["min"]), len(b["max"]), E.PROPRIO_DIM))
    band = S.stats_band_vectors()
    print("stats_band_vectors -> %s" % ("None" if band is None else
                                        "ok lo=%s hi=%s" % (np.round(band[0], 4).tolist(),
                                                             np.round(band[1], 4).tolist())))
    calib_facts = E.resolve_calib(str(E.CALIB_PATH), None)
    print("resolve_calib -> %s" % ("None" if calib_facts is None else sorted(calib_facts)))

    E.use_stage_fw()
    for p in (str(Path("/workspace/fasterwam_a3_bridge/bridge")),
              str(Path("/workspace/fasterwam_a3_bridge/bridge/_next")),
              str(Path("/workspace/a3_dual_arm_sim/src"))):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        import infer_a3_worker as W
        print("infer_a3_worker imported: True")
    except Exception as exc:  # noqa: BLE001
        W = None
        print("infer_a3_worker import failed: %r" % (exc,))
    witness = S.make_proprio_witness(W, calib_facts)
    print("make_proprio_witness -> %s" % ("None" if witness is None else "callable"))
    if witness is not None:
        print("verdict: with the stats file and calibration loaded here the witness builds, so the "
              "batch must have taken a different branch -- check the order of check_stats_width "
              "against install_chunk_recorder in run_node.")


if __name__ == "__main__":
    main()
