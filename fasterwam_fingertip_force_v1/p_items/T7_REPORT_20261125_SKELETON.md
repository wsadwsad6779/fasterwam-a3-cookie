# T7 报告骨架（2026-11-25 / V5 双轨）

状态：骨架；r16/r32 完成后由 `_t7_final_report.py` 生成正式 `T7_REPORT_20261125.md/json`。
所有数字以 `p_items/` 下证据文件为准；本骨架只做结构与已有结论的占位。

## 0. 结论摘要
- F0 stage1 3000 完成；3k 官方评估 exact 0/10、avg_in_target 0、close 0.976、open 1.88%；执行 median 0.4305、<0.30 5.9%。
- F2 3k 修正版（sha 342c4c4a）exact 0/10、in_target 0、grasps 11.6、open 1.75%；执行 median 0.3897、<0.30 1.05%。
- Base anchor（官方 step_021700）：exact 0、in_target 0、grasps 0.1、open 41.3%、carry 98.8% —— 基座“张开+接触多但不抓”。
- r10 官方口径 10 seeds：in_target user 0 / strict 0、exact 0/10、执行 <0.30 5.2%、median 0.4172；strict 锚点 10/10 与 15k milestone 一致。
- 结论方向：三向 3k 都 exact 0；F0/F2 都从基座的张开/接触行为偏移到中带/闭合侧；共享连续夹爪监督/均值回归仍是主因；力通路改变接触但不改变成功率。

## 1. 三向 3k（base / F0 / F2，同 seeds 7100-7109、V3.1、eval 37376d2c）
| model | in_target | exact | grasps | open_cmd | exec median | exec <0.30 | carry>0.5 |
|---|---|---|---|---|---|---|---|
| base | 0 | 0/10 | 0.1 | 0.413 | 0.755 | 0.0003 | 0.988 |
| F0 3k | 0 | 0/10 | 1.1 | 0.019 | 0.4305 | 0.059 | 0.059 |
| F2 3k | 0 | 0/10 | 11.6 | 0.0175 | 0.3897 | 0.0105 | 0.592 |

证据：`F0_EVAL_3K_DONE.json`、`F2_EVAL_3K_DONE.json`、`P17_F2_3K_CORRECTED.json`、`F0_F2_COMPARISON_3K.json`、`BASE_ANCHOR_V31.json`。

## 2. F2 3k 协议纠正（必写）
- 首次“F2 3k”评估因 ckpt 被归档到 /root，`build_model` 静默回退基座（summary checkpoint_sha=null）；该次数据作废并保留为 base anchor。
- 修正重跑用 `/root/fwv1_checkpoint_archive/step_003000.pt` sha `342c4c4a…`，SUMMARY `28b47a0b…`；`_t7_cmp_3k.py` 已改为 fail-closed。
- 证据：`P15_F2_3K_PROTOCOL_CORRECTION.json`、`P17_F2_3K_CORRECTED.json`。

## 3. 第 7 维静态审计（P18）
- normalizer dim6 0.28/1.0 → −1/+1，0.49→−0.4167，无裁剪/重映射。
- 7 维等权 flow MSE、lambda_action=1.0；无 pm1、无平滑。
- 标签 49.0% <0.30；执行 F0 5.9% / F2 1.05%。
- 证据：`P18_DIM7_STATIC_AUDIT.json`。

## 4. ForceZero / SkipFiLM（P19）
- forcezero/skipfilm 四项 exact 0、in_target 0；forcezero 7504 7 grasps、carry 0.258（normal 0/0.093）；skipfilm 降低 contact/carry。
- 力通路影响接触行为，不影响成功率。
- 证据：`P19_FORCE_ABLATIONS.json`。

## 5. Chunk 机制（R2/P16）
- 7504 r10 执行 6 个含闭合 chunk 仍 0 入盒；r32 3 个含闭合 chunk → 3 strict/5 user；7508 r32 4 个 → 0。
- r32 是个别 chunk/seed 运气，不是 r 因果。
- 证据：`P16_CHUNK_MECHANISM_7504_7508.json`。

## 6. r 曲线（待 r16/r32 完成后填）
- r10（官方口径）10 seeds：user 0 / strict 0、exact 0、grasps 1.9、exec<0.30 5.2%、median 0.4172；strict 锚点 10/10 与 15k milestone 一致。
- r16/r32：diagnostic-only，待 `R_COMPARE_PARTIAL_r16/32.json`；最终图 `R_SWEEP_MEAN_IN_TARGET.png` 等，横轴 r、纵轴 in_target，user 与 strict 两条线，显式标注 official/diagnostic。
- 若 r16/r32 也全 0：写“三档 r 均无入盒，truncation 非主因，与 R2 一致”。

## 7. GPU 并发 incident
- 06:27:51–06:39:36 teacher-forced + ForceZero/SkipFiLM 与 F2 3k 修正重跑重叠约 11 分钟，单卡单模型锁被违反。
- 评估按 seed 确定性；指标不受影响。防再发：base 链在 teacher3k/dim7 前有 `_t7_eval` 守卫，r sweep 未释放不启动。
- 证据：`GPU_RESOURCE_LOCK.json`、`R_SWEEP_PROGRESS.json#postr_contention_note`。

## 8. Teacher-forced 局限
- `TEACHER_FORCED_15K.json`：n=8，仅 CARRY_CLOSED / CONTACT_CLOSE 两个相位；APPROACH/RESET/RELEASE/DESCEND 未覆盖。
- 3k 修正版（`TEACHER_FORCED_3K.json`）在 r sweep 后同 val 批次跑；报告 6B 对比 15k vs 3k。
- 证据：`P20_TEACHER3K_DIM7_SPEC.json`。

## 9. 夹爪专用设计 A/B/C（只写规格，BLOCKED_USER）
- A 状态头 + 连续开度：1024-d 特征 → 3 态 + 连续开度，1–2M 参数；15800 起 2–4k 步；验收：10-seed r10、exact_fill、carry、<0.30/open 双峰。
- B 分布式损失：dim6 two-hot/HL-Gauss 或 mixture-density，其余 6 维 MSE；15800 起 3–6k 步；验收同 A + 不退化其他维。
- C Flow Matching 采样修复：先零训练（denoise 5→10/20、采样器/guidance、仅 dim6 约束）；有效再做 1–2k 步微调；验收同 A + 推理耗时。
- 均需用户规格 + 预算批准；当前不实现、不训练。

## 10. 边界
- 标签 V3 sha c4090dd2、exact_fill src==70、场景冻结 cef07926、eval 37376d2c 不动。
- F2 15800 暂停、STOP_TRAINING 不动；F2 续训 5200 步、F0 15k→21k、A/B/C 实现均 BLOCKED_USER。
- 不扩预算、不删文件（move-only）、token 不入库。