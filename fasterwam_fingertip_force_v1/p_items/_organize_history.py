import os, shutil, json

ROOT = os.path.dirname(os.path.abspath(__file__))  # p_items
REPO = os.path.dirname(ROOT)
HIST = os.path.join(REPO, 'history')
HF_BASE = 'https://huggingface.co/datasets/wsadwsad6779/fasterwam-a3-cookie-task7/tree/main'

F25 = '2026-11-23_to_25_baseline_rH_B'
F26 = '2026-11-26_denoise_gripper_semantics'
os.makedirs(os.path.join(HIST, F25), exist_ok=True)
os.makedirs(os.path.join(HIST, F26), exist_ok=True)

list25 = ['R_SWEEP_MEAN_IN_TARGET.png','R_SWEEP_PLACEMENT_RATE.png','R_SWEEP_EXACT_FILL_RATE.png',
 'R_SWEEP_GRASP_RATE.png','R_SWEEP_GRIPPER_CLOSURE.png','R_SWEEP_COST_PARETO.png','R_SWEEP_SEED_HEATMAP.png',
 'R_SWEEP_RAW_RESULTS.csv','R_SWEEP_SUMMARY.json','R1_H_SWEEP_RAW.json','R1_H_SWEEP.png','R1_H_ESCALATION_REPORT.md',
 'R1_SHA256.json','H0_9K_PROBE.json','H0_9K_PROBE_FULL.json','B2_SIXPHASE_TEACHER.json','B2_PARTIAL_COVERAGE.md',
 'B3_FORCE_ABLATION_EXPANDED.json','B5_LGRIP_LFORCE_GRADCOS.json','B4_STATE_CLONE_043_028.json','B5_FAIL.json',
 'F0_FROZEN_15K_FULL.json','F0_12K_EVAL.json','F0_12K_CKPT_VERIFY.json','F0_15K_RELAXED.json','P0_FROZEN_ID.json',
 'P1_RELAXED_GEOMETRY.json','EVAL_SHA_STATUS.json','F0_F2_SUCCESS_VS_STEP.png','P_R1_H32_MEASURED.json']
list26 = ['DENOISE_RUNTIME_CONFIG_AUDIT.json','DENOISE_FROZEN_OBS_RESULTS.json','GRIPPER_ACTION_SEMANTICS_AUDIT.json',
 'OFFICIAL_FASTERWAM_VS_A3_ADAPTER.md','OLD_12K_PROVENANCE_LEDGER.md','OLD_12K_VS_CURRENT.md','T7_NEXT_DECISION.md',
 'T7_GAP_ANALYSIS_20261126.md','POD_CONSOLE_CHECK.json','POD_RECOVERED.json',
 'F0_R_SWEEP_RESULTS.json','F0_R_SWEEP_RAW.csv']
script26 = ['scripts/_t7_eval_diag_v2.py','scripts/_p0_denoise_probe.py','scripts/_p1_denoise_frozen.py',
 'scripts/_f0_rsweep_collect.py','scripts/_t7_f0_rsweep_diag.sh']

def cp(src, dst):
    if os.path.exists(src):
        shutil.copy2(src, dst); return True
    return False

c1 = [f for f in list25 if cp(os.path.join(ROOT, f), os.path.join(HIST, F25, f))]
c2 = [f for f in list26 if cp(os.path.join(ROOT, f), os.path.join(HIST, F26, f))]
c2s = [f for f in script26 if cp(os.path.join(ROOT, f), os.path.join(HIST, F26, os.path.basename(f)))]

r25 = """# 2026-11-23 ~ 11-25 · 基线 / r-sweep / H-sweep / B 诊断

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
""" + '\n'.join('- `%s`' % f for f in c1) + """

HF 镜像: %s/history/%s
""" % (HF_BASE, F25)

r26 = """# 2026-11-26 · 去噪步数 / 夹爪动作语义 / 旧实验溯源 / F0 r-sweep

## 关键结论
- P0 运行时：scheduler=WanContinuousFlowMatchScheduler(shift5, train1000)；**scheduler_steps==K(5/10/20)** 真实生效；输出 [32,7]。
- P1 冻结观测 K 扫描：**K 越大闭合召回越低**（CARRY 0.708→0.438→0.438）→ **H1(5步太少)被否**；多步收敛到 mid-band ~0.44–0.46。
- 夹爪动作语义（源码级）：env 目标开度=`(g+1)/2`、eval `g=2a−1` ⇒ **目标开度=a 恒等，无解码 bug**；官方评估是 sign 二值；mid-band 0.43 ⇒ 连续抓不住 / sign 放不开（两者 exact 都 0）。0.28 非通用判据（用力+结果）。
- 旧"12k 入盒 3/6" = **NOT_VERIFIED**；最接近是 A3 LoRA/pm1 `in_target_final 8/2 success:false`（倒着进框）；旧无加力12k ckpt 缺失。
- F0 r=10/16/32 ×7500–7509 诊断（同 diag v2，含 trajectory/strict/relaxed/曾入盒）——用于回答"F0 是否像 F2 一样受益于 r32"。

## 文件
""" + '\n'.join('- `%s`' % f for f in (c2 + c2s)) + """

HF 镜像: %s/history/%s
""" % (HF_BASE, F26)

open(os.path.join(HIST, F25, 'README.md'), 'w', encoding='utf-8').write(r25)
open(os.path.join(HIST, F26, 'README.md'), 'w', encoding='utf-8').write(r26)

root = """# Task7 · FasterWAM + 指尖力 · 打包总述（按时间归档）

本仓 `fasterwam-a3-cookie` 的正式评估/训练证据按时间归档在 `history/` 下；HF 同步镜像：
https://huggingface.co/datasets/wsadwsad6779/fasterwam-a3-cookie-task7

## 目录结构（本分支）
- `history/2026-11-23_to_25_baseline_rH_B/` — F0/F2 里程碑、r/H 扫描、B2–B5 诊断（含 README.md）
- `history/2026-11-26_denoise_gripper_semantics/` — 去噪步数审计、夹爪动作语义审计、旧实验溯源、F0 r-sweep（含 README.md）
- 工作区（未归档的完整证据/脚本/中间件）仍在 `p_items/`、以及仓库根的历史文件。

## 分支索引（origin 实际状态，git ls-remote 核实）
- `fingertip-force-v1`：**唯一有内容的分支**（293 文件：p_items 全套证据、训练/评估脚本、本归档）。
- `main` / `a3-cookie-r27` / `archive-2026-11-01` / `backup-2026-11-01-r6-audit` / `rescue-40k-restore-20261117`：**origin 上为空树（0 文件）**。
- 本分支（`history-2026-11-24_to_11-30`）由 fingertip-force-v1 派生，仅**新增** history/ 与本文档，未改历史文件；设为默认分支。

> 注：若你认为"还有很多分支"，那可能在另一台机器/另一个仓库或官方 hustvl/FasterWAM；本仓 origin 经 ls-remote 只有以上 6 个。

## 冻结状态（勿改）
- F0 15k 冻结：`step_015000.pt` sha256 `8ef07241…`（strict/relaxed 0/10）。
- F2 暂停 15800：`fa9d87e3`（只读）。F0 9k `7ca235ba`、F0 12k `d3f2a3a8`、F2 15k `83aacf23`。
- 训练全停（F0 15k 冻结、F2 15800 冻结、F1 未启）。

## checkpoint 说明
大文件 checkpoint（各 ~3.35GB）在 pod `/workspace`（**销毁后 /workspace 持久保留**），未上传（体积+带宽）；sha 已写入各证据 json。

## 安全
HF token 仅在命令环境临时使用，**未写入任何文件/提交**；因已在对话中出现，建议尽快在 HuggingFace 后台 **rotate**。
"""

open(os.path.join(REPO, 'TASK7_HISTORY_README.md'), 'w', encoding='utf-8').write(root)
print('copied25', len(c1), 'copied26', len(c2), 'scripts', len(c2s))
