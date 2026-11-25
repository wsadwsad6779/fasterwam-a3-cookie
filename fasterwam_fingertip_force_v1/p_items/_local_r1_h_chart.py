import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "R1_H_SWEEP_RAW.json")))
Hs = [32, 48, 64]
pos = {32: 1, 48: 2, 64: 3}
rng = np.random.default_rng(20261125)

mean_strict = [d["per_H"][str(h)]["mean_strict"] for h in Hs]
mean_relaxed = [d["per_H"][str(h)]["mean_relaxed"] for h in Hs]
grasps = [d["per_H"][str(h)]["grasps_mean"] for h in Hs]

fig, ax = plt.subplots(figsize=(8.5, 5.2))
for h in Hs:
    rows = d["per_H"][str(h)]["per_seed"]
    j = (rng.random(len(rows)) - 0.5) * 0.12
    ax.scatter(pos[h] + j, [r["strict"] for r in rows], s=16, color="C0", alpha=0.4)
    ax.scatter(pos[h] + j, [r["relaxed"] for r in rows], s=16, color="C1", alpha=0.30, marker="x")
ax.plot([pos[h] for h in Hs], mean_strict, "-o", color="C0", lw=2, label="strict in-box (mean)")
ax.plot([pos[h] for h in Hs], mean_relaxed, "-s", color="C1", lw=2, label="user-relaxed in-box (mean)")
for h in Hs:
    ax.annotate("grasps %.1f" % grasps[Hs.index(h)], (pos[h], max(mean_strict[Hs.index(h)], mean_relaxed[Hs.index(h)]) + 0.15),
                ha="center", fontsize=8, color="gray")
ax.set_xticks([pos[h] for h in Hs]); ax.set_xticklabels(["H=%d" % h for h in Hs])
ax.set_xlabel("action_horizon H (r=H execute full chunk; H multiple of 8)")
ax.set_ylabel("stable cookies in target (per episode /10)")
ax.set_title("R1 H-horizon sweep, F2 15k (zero-training) - H>32 is UNTRAINED (OOD)", fontsize=10)
ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
fig.text(0.01, 0.01, "F2 step_015000 sha 83aacf23 | seeds 7500-7509 | 600 steps | 5 denoise | exact_fill 0/10 all H\nH32 relaxed0.7/strict0.3 -> H48/H64 0.0/0.0: STOP escalation, no fine-scan (best<2). Observational in-box only, not a capability.",
         fontsize=6.5, color="gray")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "R1_H_SWEEP.png"), dpi=130)
print("wrote R1_H_SWEEP.png; means strict", mean_strict, "relaxed", mean_relaxed, "grasps", grasps)
