# 2026-11-23 ~ 11-25 · 基线 / r-sweep / H-sweep / B 诊断

本周主体：官方 base(step_021700) 起训 F0(无加力) 到 15k 并冻结；F2(力 FiLM+C-02) 冻结 15800；r 扫描(10/16/32)、H 扫描(32/48/64)、B2-B5 teacher/消融/梯度诊断。

## 关键结论
- F0 3k–15k、F2 3k–15k(官方 r10)：strict in-box 0/10、exact_fill 0/10（F0 15k relaxed 也 0）。
- r 扫描（F2 15k）：relaxed 0/0.5/0.7，strict 0/0.2/0.3（r=10/16/32）；exact 全 0；r32 非零集中在 7503/7504。
- H 扫描（F2 15k 零训练）：H32 relaxed0.7/strict0.3 → **H48/64=0.0**；不 +16、不细扫；H>32=OOD。
- B2（六阶段 teacher）：4/6 阶段有样本（RELEASE/RESET=0），闭合命令在 chunk 内衰减。
- B3（力消融）：normal/zero/skip 差 ~0.003–0.012 → 力输入几乎不影响闭合。
- B5（真 L_force）：cos(grip,force)=+0.10 近正交；夹爪维梯度主导。
- B4（state clone）：INCONCLUSIVE（够不到）。
- 冻结身份：F0 9k=7ca235ba、F0 12k=d3f2a3a8、F0 15k=8ef07241、F2 15k=83aacf23、F2 15800=fa9d87e3（详见 P0_FROZEN_ID.json）。

## 文件
- `R_SWEEP_MEAN_IN_TARGET.png`
- `R_SWEEP_PLACEMENT_RATE.png`
- `R_SWEEP_EXACT_FILL_RATE.png`
- `R_SWEEP_GRASP_RATE.png`
- `R_SWEEP_GRIPPER_CLOSURE.png`
- `R_SWEEP_COST_PARETO.png`
- `R_SWEEP_SEED_HEATMAP.png`
- `R_SWEEP_RAW_RESULTS.csv`
- `R_SWEEP_SUMMARY.json`
- `R1_H_SWEEP_RAW.json`
- `R1_H_SWEEP.png`
- `R1_H_ESCALATION_REPORT.md`
- `R1_SHA256.json`
- `H0_9K_PROBE.json`
- `H0_9K_PROBE_FULL.json`
- `B2_SIXPHASE_TEACHER.json`
- `B2_PARTIAL_COVERAGE.md`
- `B3_FORCE_ABLATION_EXPANDED.json`
- `B5_LGRIP_LFORCE_GRADCOS.json`
- `B4_STATE_CLONE_043_028.json`
- `B5_FAIL.json`
- `F0_FROZEN_15K_FULL.json`
- `F0_12K_EVAL.json`
- `F0_12K_CKPT_VERIFY.json`
- `F0_15K_RELAXED.json`
- `P0_FROZEN_ID.json`
- `P1_RELAXED_GEOMETRY.json`
- `EVAL_SHA_STATUS.json`
- `F0_F2_SUCCESS_VS_STEP.png`
- `P_R1_H32_MEASURED.json`

HF 镜像: https://huggingface.co/datasets/wsadwsad6779/fasterwam-a3-cookie-task7/tree/main/history/2026-11-23_to_25_baseline_rH_B
