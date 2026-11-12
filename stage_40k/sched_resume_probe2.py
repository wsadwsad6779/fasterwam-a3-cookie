"""What does the trainer's resume actually do to the LR, measured rather than reasoned?

`backup40k.resume_test` replayed `scheduler.step()` dozens of times on top of an
already-advanced SequentialLR and landed on an lr that matched neither curve(step) nor
curve(step+1). That number says nothing about the run: the trainer never replays, it
builds a scheduler, calls `load_state_dict(saved)`, and then steps once per optimizer
step. So the question this probe answers is what *that* sequence produces.

Three candidate assertions, all cheap, all on the rehearsal node's own bytes:

  A  the saved state_dict is what a fresh scheduler has after `step` calls
     (identical state => identical continuation, no arithmetic needed);
  B  after `load_state_dict`, the optimizer's lr is already curve(step), i.e. the next
     optimizer step uses exactly the lr an uninterrupted run would have used;
  C  the first `step()` after the load gives curve(step+1), the lr after that step.

Also prints the torch version and the source of SequentialLR.load_state_dict/step, because
the answer depends on which of them re-applies lr to the optimizer.
"""
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/_v1_stage/fw/src")
sys.path.insert(0, "/workspace/_v1_stage/fw/scripts")

import torch  # noqa: E402
from torch.optim.lr_scheduler import SequentialLR  # noqa: E402

import drive40k as plan  # noqa: E402
from fasterwam.trainer import Wan22Trainer  # noqa: E402

NODE = Path("/workspace/_v1_stage/rehearsal65/run/checkpoints/light/step_000012")
STEP = 12
WANT_FIELDS = ("last_epoch", "base_lrs", "_milestones", "_step_count", "_gammas",
               "_last_gamma")


def build():
    class Shim:
        def __init__(self, optimizer):
            self.optimizer = optimizer
            self.learning_rate = plan.LR

    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=plan.LR,
                            betas=(0.9, 0.95), weight_decay=1e-2)
    return opt, Wan22Trainer._build_scheduler(Shim(opt), "cosine", plan.TOTAL,
                                              int(plan.TOTAL * 0.05))


def fields(sd):
    """Scalar fields of a scheduler state_dict, inner schedulers included."""
    out = {}
    for k in WANT_FIELDS:
        if k in sd:
            out[k] = sd[k]
    inner = sd.get("_schedulers") or sd.get("schedulers") or []
    for i, sub in enumerate(inner):
        if isinstance(sub, dict):
            for k in ("last_epoch", "base_lrs", "start_factor", "end_factor",
                      "total_iters", "T_max", "eta_min", "_step_count"):
                if k in sub:
                    out["sched%d.%s" % (i, k)] = sub[k]
    return out


def analytic(n):
    """curve(n): the lr sitting in the optimizer once global_step == n."""
    w = int(plan.TOTAL * 0.05)
    if n <= w:
        return plan.LR * (1.0 / w + (1.0 - 1.0 / w) * (n / float(w)))
    t = plan.LR * 0.01
    return t + (plan.LR - t) * (1 + torch.cos(
        torch.tensor((n - w) / float(plan.TOTAL - w)) * torch.pi)).item() / 2


def diff(a, b, label):
    keys = sorted(set(a) | set(b))
    return [{"field": k, label: a.get(k, "<absent>"), "other": b.get(k, "<absent>")}
            for k in keys if a.get(k, "<absent>") != b.get(k, "<absent>")]


print(json.dumps({"torch": torch.__version__, "plan": {"TOTAL": plan.TOTAL,
                                                       "LR": plan.LR,
                                                       "warmup": int(plan.TOTAL * 0.05)}}))
print("\n== SequentialLR.load_state_dict")
print(inspect.getsource(SequentialLR.load_state_dict))
print("== SequentialLR.step")
print(inspect.getsource(SequentialLR.step))

saved = torch.load(NODE / "scheduler.pt", map_location="cpu", weights_only=False)
state = json.loads((NODE / "trainer_state.json").read_text())
print("\n== node: global_step=%s extra.lr=%r" % (state.get("global_step"),
                                                 state.get("extra", {}).get("lr")))
print("   saved fields:", json.dumps(fields(saved), sort_keys=True))

ref_opt, ref = build()
ref_fields_after_build = fields(ref.state_dict())
print("\n== A) fresh stepped %d times vs the saved state" % STEP)
for _ in range(STEP):
    ref.step()
ref_after = fields(ref.state_dict())
print("   lr after %d steps: %.12g   analytic curve(%d): %.12g"
      % (STEP, float(ref_opt.param_groups[0]["lr"]), STEP, analytic(STEP)))
print("   field differences (saved vs fresh-stepped):", json.dumps(
    diff(fields(saved), ref_after, "saved"), sort_keys=True))
A_ok = diff(fields(saved), ref_after, "saved") == []

res_opt, res = build()
print("\n== B) load_state_dict into a scheduler that was only just built")
print("   before load: lr=%.12g" % float(res_opt.param_groups[0]["lr"]))
res.load_state_dict(saved)
lr_on_load = float(res_opt.param_groups[0]["lr"])
print("   after  load: lr=%.12g   analytic curve(%d)=%.12g   match=%s"
      % (lr_on_load, STEP, analytic(STEP), abs(lr_on_load - analytic(STEP)) <= 1e-15))
B_ok = abs(lr_on_load - analytic(STEP)) <= 1e-15
print("   loaded fields equal the fresh-stepped fields:",
      json.dumps(diff(fields(res.state_dict()), ref_after, "loaded")))

print("\n== C) the first steps after that load, against the uninterrupted curve")
C_ok = True
for i in range(1, 5):
    res.step()
    got = float(res_opt.param_groups[0]["lr"])
    want = analytic(STEP + i)
    C_ok &= abs(got - want) <= 1e-15
    print("   call %d -> lr=%.12g  analytic curve(%d)=%.12g  %s"
          % (i, got, STEP + i, want, "ok" if abs(got - want) <= 1e-15 else "OFF CURVE"))

print("\n== the replay the old R2 did (for the record)")
old_opt, old = build()
old.load_state_dict(saved)
for _ in range(max(STEP - 1, 0)):
    old.step()
print("   old R2 would report lr=%.12g at replayed step %d, analytic=%.12g"
      % (float(old_opt.param_groups[0]["lr"]), STEP - 1, analytic(STEP - 1)))

print("\n== VERDICT  A(saved state == fresh@step)=%s  B(lr after load == curve)=%s  "
      "C(next steps on curve)=%s" % (A_ok, B_ok, C_ok))
print("   => R2 must assert A, B and C; replaying step() on a loaded scheduler is wrong.")
