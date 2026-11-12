"""Reproduce the trainer's resume exactly, then predict the rehearsal's own LR log.

probe2 loaded scheduler.pt alone and got an lr near the warmup floor. That is not what the
trainer does: `load_light_state` restores `optimizer.pt` first, and
`Optimizer.load_state_dict` copies every non-`params` key of each saved param group --
including `lr` -- into the live groups. So the optimizer arrives already carrying the lr the
run had at that step, and `scheduler.pt` only has to keep the *continuation* on the curve.

This probe therefore does three things, cheapest first:

  1. read the lr actually stored inside each rehearsal node's optimizer.pt;
  2. replay the trainer's resume order (fresh build -> optimizer groups -> scheduler state)
     and step it the way the training loop steps it;
  3. take the step-6 node, simulate the resumed segment, and compare the predicted lr
     sequence against the lr the real GPU run printed in its own log.

Step 3 is the whole question: if a cold resume predicted from the archived bytes reproduces
the trainer's measured 7..12 line-for-line, the resume is on the frozen 40k curve.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/_v1_stage/fw/src")
sys.path.insert(0, "/workspace/_v1_stage/fw/scripts")

import torch  # noqa: E402
import drive40k as plan  # noqa: E402
from fasterwam.trainer import Wan22Trainer  # noqa: E402

LIGHT = Path("/workspace/_v1_stage/rehearsal65/run/checkpoints/light")
# The trainer's own log from the real second segment (steps 7..12 after a cold restart at 6).
LOGGED = [3.498500e-07, 4.00e-07, 4.50e-07, 5.00e-07, 5.50e-07, 6.00e-07, 6.50e-07]


def build():
    class Shim:
        def __init__(self, optimizer):
            self.optimizer = optimizer
            self.learning_rate = plan.LR

    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([p], lr=plan.LR, betas=(0.9, 0.95), weight_decay=1e-2)
    sch = Wan22Trainer._build_scheduler(Shim(opt), "cosine", plan.TOTAL,
                                         int(plan.TOTAL * 0.05))
    return opt, sch


def analytic(n):
    w = int(plan.TOTAL * 0.05)
    if n <= w:
        return plan.LR * (1.0 / w + (1.0 - 1.0 / w) * (n / float(w)))
    eta = plan.LR * 0.01
    return eta + (plan.LR - eta) * (1 + torch.cos(
        torch.tensor((n - w) / float(plan.TOTAL - w)) * torch.pi)).item() / 2


def resume(step):
    """The order load_light_state uses, minus the 200 MB of tensors."""
    node = LIGHT / ("step_%06d" % step)
    opt, sch = build()
    o_st = torch.load(node / "optimizer.pt", map_location="cpu", weights_only=False)
    for live, saved in zip(opt.param_groups, o_st["param_groups"]):
        for k, v in saved.items():
            if k != "params":
                live[k] = v
    s_st = torch.load(node / "scheduler.pt", map_location="cpu", weights_only=False)
    sch.load_state_dict(s_st)
    return node, opt, sch, o_st, s_st


print(json.dumps({"torch": torch.__version__, "nodes": sorted(p.name for p in LIGHT.iterdir())}))

print("\n== 1) lr stored inside each node's optimizer.pt (param_groups, not tensors)")
for step in (6, 12):
    node = LIGHT / ("step_%06d" % step)
    o_st = torch.load(node / "optimizer.pt", map_location="cpu", weights_only=False)
    st = json.loads((node / "trainer_state.json").read_text())
    s_st = torch.load(node / "scheduler.pt", map_location="cpu", weights_only=False)
    print("   step=%2d optimizer.pt lr=%r  trainer_state extra.lr=%r  "
          "scheduler last_epoch=%r sched0.last_epoch=%r"
          % (step, o_st["param_groups"][0].get("lr"), st.get("extra", {}).get("lr"),
             s_st.get("last_epoch"), s_st["_schedulers"][0].get("last_epoch")))

print("\n== 2) trainer-order resume at step 6, then the loop's own stepping")
node, opt, sch, o_st, s_st = resume(6)
used = []
# `while global_step < stop:` -> next(data_iter); backward; optimizer.step(); scheduler.step();
# global_step += 1.  The lr a step *uses* is whatever the optimizer carries at that moment.
for k in range(1, 8):
    used.append({"global_step_after": 6 + k, "lr_used_for_that_step":
                 float(opt.param_groups[0]["lr"])})
    opt.step()
    sch.step()
for row in used:
    n = row["global_step_after"]
    want_prev = analytic(n - 1)
    print("   step %2d used lr=%.12g   analytic curve(%d)=%.12g   %s"
          % (n, row["lr_used_for_that_step"], n - 1, want_prev,
             "ok" if abs(row["lr_used_for_that_step"] - want_prev) <= 1e-15 else "OFF CURVE"))

print("\n== 3) predicted vs the real second-segment log (rounded the way the log rounds)")
pred = [r["lr_used_for_that_step"] for r in used][:len(LOGGED)]
rows = [{"logged": "%.6e" % a, "predicted": "%.6e" % b, "agree": abs(a - b) <= 5e-13}
        for a, b in zip(LOGGED, pred)]
for r in rows:
    print("   logged=%s predicted=%s %s" % (r["logged"], r["predicted"],
                                            "ok" if r["agree"] else "MISMATCH"))
print("   all agree:", all(r["agree"] for r in rows))

print("\n== 4) same resume at step 12, first three continued steps")
node, opt, sch, o_st, s_st = resume(12)
cont = []
for k in range(1, 4):
    cont.append({"step": 12 + k, "lr_used": float(opt.param_groups[0]["lr"]),
                 "want_curve_at": 12 + k - 1, "want": analytic(12 + k - 1)})
    opt.step()
    sch.step()
for c in cont:
    print("   step %d lr_used=%.12g want=%.12g %s"
          % (c["step"], c["lr_used"], c["want"],
             "ok" if abs(c["lr_used"] - c["want"]) <= 1e-15 else "OFF CURVE"))

print("\n== 5) for contrast: scheduler state loaded WITHOUT the optimizer lr")
opt2, sch2 = build()
sch2.load_state_dict(torch.load(LIGHT / "step_000012/scheduler.pt", map_location="cpu",
                               weights_only=False))
print("   lr after load alone = %.12g  (this is what probe2 measured)"
      % float(opt2.param_groups[0]["lr"]))
opt2.step()
sch2.step()
print("   lr after one more step = %.12g   analytic curve(13)=%.12g"
      % (float(opt2.param_groups[0]["lr"]), analytic(13)))
