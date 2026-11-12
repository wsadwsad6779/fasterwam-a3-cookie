"""Read-only: what do the archived nodes actually record for the sampler and counters?

The boundary checker has to assert an arithmetic relation for the sampler offsets, and the
relation is only worth asserting once it has been observed. Prints both rehearsal nodes
(step 6 and step 12, six steps apart) side by side, plus the live run's batch/dataset
settings, so the invariant is read off numbers instead of assumed.
"""
import json
from pathlib import Path

ROOT = Path("/workspace/_v1_stage")

for step in ("000006", "000012"):
    p = ROOT / ("rehearsal65/run/checkpoints/light/step_%s/trainer_state.json" % step)
    if not p.is_file():
        print(step, "MISSING", p)
        continue
    d = json.loads(p.read_text())
    print("== node step %s ==" % step)
    for key in ("global_step", "epoch", "batch_in_epoch", "sampler", "extra",
                "adapter_bytes", "format"):
        if key in d:
            print("   %-16s %s" % (key, json.dumps(d[key])[:400]))
    if "force_calibration" in d:
        print("   force_calibration ", json.dumps(d["force_calibration"])[:300])

cfg = (ROOT / "run40k/config.yaml")
if cfg.is_file():
    print("\n== run40k config: batch / dataset / workers / num_processes ==")
    for i, line in enumerate(cfg.read_text().splitlines(), 1):
        if any(k in line.lower() for k in ("batch", "dataset", "worker", "process",
                                            "num_rep", "action_exec", "state_dim")):
            print("   %4d  %s" % (i, line.rstrip()))

print("\n== live sampler progress on the running segment ==")
log = ROOT / "run40k/logs/seg1.log"
if log.is_file():
    lines = [l for l in log.read_text(errors="replace").replace("\r", "\n").splitlines()
             if "step=" in l]
    print("   first:", lines[0].strip() if lines else None)
    print("   last :", lines[-1].strip() if lines else None)
    print("   count of progress lines:", len(lines))
