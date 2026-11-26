# 2026-11-26 · 去噪步数 / 夹爪动作语义 / 旧实验溯源 / F0 r-sweep

## 关键结论
- P0 运行时：scheduler=WanContinuousFlowMatchScheduler(shift5, train1000)；**scheduler_steps==K(5/10/20)** 真实生效；输出 [32,7]。
- P1 冻结观测 K 扫描：**K 越大闭合召回越低**（CARRY 0.708→0.438→0.438）→ **H1(5步太少)被否**；多步收敛到 mid-band ~0.44–0.46。
- 夹爪动作语义（源码级）：env 目标开度=`(g+1)/2`、eval `g=2a−1` ⇒ **目标开度=a 恒等，无解码 bug**；官方评估是 sign 二值；mid-band 0.43 ⇒ 连续抓不住 / sign 放不开（两者 exact 都 0）。0.28 非通用判据（用力+结果）。
- 旧"12k 入盒 3/6" = **NOT_VERIFIED**；最接近是 A3 LoRA/pm1 `in_target_final 8/2 success:false`（倒着进框）；旧无加力12k ckpt 缺失。
- F0 r=10/16/32 ×7500–7509 诊断（同 diag v2，含 trajectory/strict/relaxed/曾入盒）——用于回答"F0 是否像 F2 一样受益于 r32"。

## 文件
- `DENOISE_RUNTIME_CONFIG_AUDIT.json`
- `DENOISE_FROZEN_OBS_RESULTS.json`
- `GRIPPER_ACTION_SEMANTICS_AUDIT.json`
- `OFFICIAL_FASTERWAM_VS_A3_ADAPTER.md`
- `OLD_12K_PROVENANCE_LEDGER.md`
- `OLD_12K_VS_CURRENT.md`
- `T7_NEXT_DECISION.md`
- `T7_GAP_ANALYSIS_20261126.md`
- `POD_CONSOLE_CHECK.json`
- `POD_RECOVERED.json`
- `F0_R_SWEEP_RESULTS.json`
- `F0_R_SWEEP_RAW.csv`
- `scripts/_t7_eval_diag_v2.py`
- `scripts/_p0_denoise_probe.py`
- `scripts/_p1_denoise_frozen.py`
- `scripts/_f0_rsweep_collect.py`
- `scripts/_t7_f0_rsweep_diag.sh`

HF 镜像: https://huggingface.co/datasets/wsadwsad6779/fasterwam-a3-cookie-task7/tree/main/history/2026-11-26_denoise_gripper_semantics
