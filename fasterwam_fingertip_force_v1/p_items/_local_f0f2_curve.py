import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
steps = [3000, 6000, 9000, 12000, 15000]
F0_strict = [0, 0, 0, 0, 0]
F2_strict_r10 = [0, 0, 0, 0, 0]
F0_exact = [0, 0, 0, 0, 0]
F2_exact = [0, 0, 0, 0, 0]
# F2 15k diagnostic at alternate replan (NOT official r10), single-config
diag = {"r16": {"x": 15000, "strict": 0.2, "relaxed": 0.5}, "r32": {"x": 15000, "strict": 0.3, "relaxed": 0.7}}

fig, ax = plt.subplots(figsize=(9, 5.3))
xs = [s/1000 for s in steps]
ax.plot(xs, F0_strict, "-o", color="C0", lw=2, label="F0 (no force) strict in-box, official r10")
ax.plot(xs, F2_strict_r10, "-s", color="C3", lw=2, label="F2 (force) strict in-box, official r10")
ax.plot(xs, F0_exact, ":", color="C0", lw=1, alpha=0.6)
ax.plot(xs, F2_exact, ":", color="C3", lw=1, alpha=0.6)
ax.text(15.1, 0.03, "F0/F2 exact_fill = 0/10 (dotted)", color="gray", fontsize=8)
# annotations of the zero line
for s in steps:
    ax.annotate("0", (s/1000, 0), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7, color="gray")
# F2 15k diagnostics (not official)
ax.scatter([15], [0.3], marker="^", color="C3", s=70, zorder=5)
ax.scatter([15], [0.2], marker="v", color="C2", s=70, zorder=5)
ax.annotate("F2 r32 diag: strict0.3 / relaxed0.7", (15, 0.3), xytext=(-150, 24), textcoords="offset points", fontsize=8, color="C3", arrowprops=dict(arrowstyle="->", color="C3", lw=0.8))
ax.annotate("F2 r16 diag: strict0.2 / relaxed0.5", (15, 0.2), xytext=(-150, -30), textcoords="offset points", fontsize=8, color="C2", arrowprops=dict(arrowstyle="->", color="C2", lw=0.8))

ax.set_xticks(xs); ax.set_xticklabels(["%dk" % (s//1000) for s in steps])
ax.set_xlabel("training step (checkpoint)")
ax.set_ylabel("stable cookies in target (mean /10, official r10)")
ax.set_title("F0 vs F2 in-target success vs training step (official r10) — both 0 at every milestone")
ax.set_ylim(-0.05, 0.8); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")
fig.text(0.01, 0.01,
         "strict = sim cookies_in_target (tilt<=15). Source: cmp_f0_{3,6,9,12,15}k and step_0{03000..15000} SUMMARY_step_* (n=10 seeds each).\n"
         "F0: 0/10 at 3k/6k/9k/12k/15k. F2: 0/10 at 3k/6k/9k/12k/15k (r10). Only F2 15k r16/r32 DIAGNOSTIC (single-config, OOD-ish replan) is nonzero; not an official milestone score.",
         fontsize=6.5, color="gray")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "F0_F2_SUCCESS_VS_STEP.png"), dpi=130)
print("wrote F0_F2_SUCCESS_VS_STEP.png")
