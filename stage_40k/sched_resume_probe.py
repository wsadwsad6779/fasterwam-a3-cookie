"""What does the saved scheduler state actually contain, and what does a resume produce?

The backup tool's resume test replays `sch.step()` after `load_state_dict` and landed on an
lr that matches neither the curve at the node's step nor at any other obvious index, so one
of two things is true and they have opposite meanings:

  * the trainer's resume really leaves the 40k curve -- a §四 blocker, the run must stop; or
  * the replay harness is wrong (stepping an already-advanced SequentialLR double-counts) --
    a bug in my test, and the honest replacement is to compare the saved state against the
    state a fresh scheduler has at that step, because identical state means identical
    continuation by construction.

Print both, plus the raw keys of the checkpoint, so the answer is read off a table instead of
argued from PyTorch internals. Uses the rehearsal node on disk; touches no network.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/_v1_stage/fw/src")
sys.path.insert(0, "/workspace/_v1_stage/fw/scripts")

import torch  # noqa: E402
import drive40k as plan  # noqa: E402
from fasterwam.trainer import Wan22Trainer  # noqa: E402

NODE = Path("/workspace/_v1_stage/rehearsal65/run/checkpoints/light/step_000012")
STEP = 12


def build():
    class Shim:
        def __init__(self, optimizer):
            self.optimizer = optimizer
            self.learning_rate = plan.LR

    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=plan.LR,
                            betas=(0.9, 0.95), weight_decay=1e-2)
    return opt, Wan22Trainer._build_scheduler(Shim(opt), "cosine", plan.TOTAL,
                                              int(plan.TOTAL * 0.05))


def brief(sd):
    out = {}
    for k, v in sd.items():
        if k in ("schedulers", "_schedulers"):
            out[k] = [brief(s) if isinstance(s, dict) else type(s).__name__ for s in v]
        elif k == "lr_lambdas":
            out[k] = ["<lambda>" if callable(x) else x for x in v]
        elif isinstance(v, (int, float, str, bool, list, type(None))):
            out[k] = v
        else:
            out[k] = type(v).__name__
    return out


print(json.dumps({"plan": {"TOTAL": plan.TOTAL, "NODE": plan.NODE, "LR": plan.LR}}, indent=2))

fresh_opt, fresh = build()
print("\n== fresh scheduler immediately after construction:", json.dumps(brief(fresh.state_dict()), indent=1)[:1200])

print("\n== reference: fresh scheduler stepped from 0, lr after each step")
ref = []
for i in range(1, STEP + 6):
    fresh.step()
    ref.append({"step": i, "lr": float(fresh_opt.param_groups[0]["lr"])})
for row in ref:
    print("   step=%2d lr=%.12g" % (row["step"], row["lr"]))

ck = torch.load(NODE / "scheduler.pt", map_location="cpu", weights_only=False)
print("\n== saved scheduler.pt state_dict:", json.dumps(brief(ck), indent=1)[:1600])

res_opt, res = build()
res.load_state_dict(ck)
print("\n== after load_state_dict, before any step: opt lr=%.12g last_epoch=%r" % (
    float(res_opt.param_groups[0]["lr"]), ck.get("last_epoch")))
seq = []
for i in range(1, 5):
    res.step()
    seq.append({"calls_after_resume": i, "lr": float(res_opt.param_groups[0]["lr"])})
for row in seq:
    print("   resume step-call %d -> lr=%.12g" % (row["calls_after_resume"], row["lr"]))

want_at = {r["step"]: r["lr"] for r in ref}
print("\n== verdict table")
print("   first lr produced after resuming      = %.12g" % seq[0]["lr"])
print("   fresh-from-zero curve at step %d      = %.12g" % (STEP + 1, want_at.get(STEP + 1)))
print("   fresh-from-zero curve at step %d      = %.12g" % (STEP, want_at.get(STEP)))
print("   recorded extra.lr in trainer_state    = %r" % json.loads(
    (NODE / "trainer_state.json").read_text())["extra"].get("lr"))
same = abs(seq[0]["lr"] - want_at.get(STEP + 1, -1)) <= 1e-15
print("   RESUME IS ON THE SAME CURVE:", same)
