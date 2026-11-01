# FasterWAM → A3 饼干搬运：交接说明（2026-11-01 最终版）

> 这份文件是默认分支上的最新交接说明，取代 `docs/README_HANDOFF_r27_2026-10-27.md`（旧版保留在归档分支）。
> 它对应的现场是云端实例 `rgapi-one-click-24316-57b6b870`（anruicloud，Radeon 48 GB，容器内 `/workspace` 98 GB 已用 97 %）。
> 实例一旦关闭，`/workspace` 整体销毁，所以本仓库（代码与文档）加 Hugging Face 仓库（权重、数据、评估）就是恢复现场的全部内容。
> 2026-11-01 的完整归档材料——台账、恢复脚本、pod 快照、最终报告、worktree diff——在 `archive-2026-11-01` 分支；
> 默认分支只保留运行代码。下文提到 `scripts/`、`pod_snapshot/`、`archive/2026-11-01/` 的路径都在那个分支里。
> 本文件不代表研究目标已经达成；第 13 节把没做完的事逐条列清。

一句话结论：微调链 r1→r5 都可复现、loss 正常收敛，但**裸模型从来没有真正完成过一次「夹起单块 → 搬运 → 在盒内松手且留在盒内」**；r4 与 r5 的 12 条录制里 `single`、`rel_in_bin`、`rest_in_bin` 全是 0。当前方向已经改成「补低力示范语料 + 给力通道输入」，不是继续加步数。

必须先知道的一条硬损失：**r4/r5 已丢失的 `proprio_encoder` 末态参数不能由本次备份恢复**（第 8 节）。其余 8 类资产都在下面标了位置。

---

## 1. 研究目标

- 用视频-动作模型（FasterWAM，双相机图像 + 本体感知 → 末端位姿与夹爪开度）在 A3 双臂仿真里完成「把源区饼干搬进目标盒」，并让夹取接触力降到专家示范的量级（约 1.3 N）。
- 拆成三条能独立验收的子目标：
  1. 训练链可复现：同一份语料、同一份归一化统计、同一随机种子能重跑出同一权重；
  2. 接触力量化：指垫力与腕部力都要能逐帧读出来，并和确定性专家回放做对照；
  3. 任务完成度：以「松手后留在盒内的块数」为准，不以「末帧中心落在盒内 xy」为准。
- 本轮（2026-10-29 至 2026-11-01）的实际产出是第 2、3 条的**度量与证据**，以及第 1 条里 r5 续训的否定性结论。第 9、10 节是给下一轮的输入。

## 2. A3 任务说明

- 环境：`A3CookieTransferEnv`（`a3_dual_arm_sim`，MuJoCo 3.3.2 + robosuite 1.4.0），80 块源饼干，目标盒 `required = 10`。
- 观测：两个相机（正面 + 侧面）256×256 RGB，本体感知 8 维（末端位置/姿态与夹爪开度），控制频率 20 Hz，一次评估最长 600 步。
- 动作：末端位移 +  yaw + 夹爪开度；夹爪走 `--gripper-mode pm1`（归一化开度刻度，0 全闭、1 全开）。
- 评估器：`_r4_stage/run_matrix.py`（sha `4ddeb5ea…`），`--seeds 11 21 22 --replan 10 --max-steps 600 --position-noise 0 --yaw-noise 0 --batch-offset 0 --video`。
- 两档对照是本轮定下来的硬规矩：
  - **B0** = 裸模型，不套任何 Wrapper/Safety，模型输出原样进仿真（逐字 verbatim 100 %）；
  - **C0** = 接口包装 + Safety 阈值，允许改写动作，但必须同时报「官方值」和「被改写了多少步」。
  - 任何结论都要 B0/C0 成对出现，只报 C0 会被 Safety 的接管掩盖模型本身的行为。
- 上游 pinned commit（这两个必须一字不差，`a3_dual_arm_sim` 的 `ff552c4` 是错版）：

```
hustvl/FasterWAM        @ 83667817df0d4f823f39d90700e61ea2f432ac45   (Apache-2.0)
Eter0109/a3_dual_arm_sim @ 08ebe8cb370856051d0b32b07e691cc9b056716b
```

## 3. 原始专家示范的策略：五块一批、两批十块

- 原始 100 条专家示范来自 HF dataset `Eter0109/a3-front-close-left-100`，revision `8fdd8a7caf8275eba8be8a83b53a99176a084062`，42,681 帧、20 Hz。
- 它的任务定义不是「夹一块放一块」，而是**一次夹起一批、一批 5 块、两批凑满 10 块**（`docs_task1` 的 `cookie_2x10.md`）。规则：

```
每批 5 块，两批共 10 块；
第一次抓取抓 n 块，则序列是 [n, 10-n, n, 10-n]。
```

- 这条定性很重要，否则会把「一次夹 2 块」当成错误行为。第 9 节的纠正示范里，C-2 类就是按「一次夹 ≥2 块」这个真实任务口径采集的。
- 同一条专家轨迹在确定性回放下的接触力：指垫力中位 **1.39 N**、峰值 **1.54 N**（第 7 节拿它当尺度锚点）。

## 4. r1 → r5 训练历史

底座一律是 HF 上的 `libero/step_021700.pt`（11,117,679,929 B，sha256 `01be4efca565d1890fd8a7dc1507b01c00c8cc63c16ae252287492ba9ac7550d`）。LoRA 挂在 392 层 q/k/v/o 上，可训练参数 30.7 M。

| 轮次 | 起点 | 步数 | lr | batch | 语料 | 关键产物 |
|---|---|---|---|---|---|---|
| r1 | libero | 2000 | 2e-4 | 6 | 原始 100 条（`a3_data/a3_fw`） | `a3_lora_r1/step_002000_merged.pt`，11,117,977,659 B，sha256 `4091b02a23b3e35fb63b8bac38ea8e9c6d44439c052ed5aa5db151435a175601` |
| r2 | r1 merged | 3000 | 1e-4 | 6 | 同上 | `a3_lora_r2/step_003000_merged.pt`（在 HF `weights/r2/`） |
| r3 | r2 merged | 3000 | 1e-4 | 6 | 同上 | `a3_lora_r3/step_003000_merged.pt`（在 HF `weights/r3/`，r4/r5 的评估底座） |
| r4 | r3 merged | 4000 | 5e-5 | 6 | 原始 + dagger 近程（`a3_fw_dagger_near`） | `checkpoints/light/step_000500…step_004000` 各 5 个张量文件 |
| r5 | r4 light `step_004000` | global step 8000（本段 4000 步，lr 重新升温） | 5e-5（`restart_peak_lr`） | 6 | 同 r4 | `checkpoints/light/step_008000/` |

- r5 的 `lr_restart.json` 落在 `runs/a3_lora_r5/`，逐字字段：`resumed_global_step=4000`、`segment_steps=4000`、`restart_peak_lr=5e-05`、`warmup=200`、`cosine t_max=3800`、`dataset_len=39481`、`epoch_at_resume=1`、`batch_in_epoch=419`、`sample_offset=18000`。
- **r5 是 global step 8000，不是累计 16000 步**。旧文档里写 16000 的地方已作废。
- r4 还留了一次崩溃现场 `runs/a3_lora_r4_crashed_step500`，用来解释为什么 `step_000500` 那批权重不能当 r4 用。
- 启动命令模板（r2/r3 用的是 `accelerate launch scripts/train.py data=a3_2cam task=a3_fasterwam batch_size=6 max_steps=… learning_rate=… resume=…/libero/step_021700.pt`）在 `pod_snapshot/2026-11-01/_r4_stage/run_r4.sh`、`run_r5.sh` 里，逐字未改。

## 5. r4 与 r5 的 B0 / C0 评估表

同一权重、同一 seed、同一动作流；底座同为 `a3_lora_r3/step_003000_merged.pt`；唯一变量是 adapter（r4 `step_004000/adapter.pt` vs r5 `step_008000/adapter.pt`）。lr 重加热是残留的第二变量，所以结论要写成「续训 + 重加热」的联合效应。

### 5.1 B0 裸模型（官方字段，取各自 `summary_lora.json`）

| seed | run | `success` | 官方 `cookies_in_target` | `max_pad_force` N | `max_wrist_force` N | verbatim % |
|---|---|---|---|---|---|---|
| 11 | r4 | false | 0 | 25.526 | 13.63 | 100 |
| 11 | r5 | false | 0 | **30.972** | 18.04 | 100 |
| 21 | r4 | false | 0 | 25.955 | 93.99 | 100 |
| 21 | r5 | false | 0 | **37.178** | 95.02 | 100 |
| 22 | r4 | false | **6** | **865.429** | **333.05** | 100 |
| 22 | r5 | false | **2** | **27.726** | 97.86 | 100 |

### 5.2 C0 接口 + Safety（官方字段）

| seed | run | `success` | 官方 `cookies_in_target` | `max_pad_force` N | `max_wrist_force` N | verbatim % | `estop_steps` | `binfloor_steps` |
|---|---|---|---|---|---|---|---|---|
| 11 | r4 | false | 0 | 25.526 | 13.63 | 100.0 | 0 | 0 |
| 11 | r5 | false | 0 | 30.972 | 18.04 | 99.83 | 0 | 0 |
| 21 | r4 | false | 0 | 25.955 | 87.16 | 99.0 | 0 | 0 |
| 21 | r5 | false | 0 | **49.167** | 89.76 | 98.17 | **1** | 0 |
| 22 | r4 | false | **3** | 49.793 | 97.64 | 81.33 | **7** | **79** |
| 22 | r5 | false | **1** | 26.423 | 93.55 | 91.83 | 0 | **37** |

### 5.3 九项指标（同一打分器 `autonomy_metrics.py` 两侧重跑，逐条 verbatim）

`single` = 通过验收的单块抓—搬—放；`rel_in_bin` = 在盒内释放；`rest_in_bin` = 松手后仍留在盒内。

```
r5 (dir=/workspace/a3_eval/r5_b0_c0_paired, files=6)
lora_B0_11  events=0 single=0 single_lift=0 bundle=0 held_no_release=0 rel_in_bin=0 rest_in_bin=0 dur_med=0   car_med=0mm
            pad p50=22.46 p95=23.27 max=30.97 | >25N 14帧 >40N 0 | wrist p50=11.10 max=18.04 | flips obs=256 cmd=48 | 改写=0  | cit_end=0
lora_B0_21  events=1 single=0 single_lift=0 bundle=1 held_no_release=0 rel_in_bin=0 rest_in_bin=0 dur_med=1   car_med=0mm
            pad p50=22.11 p95=25.35 max=37.18 | >25N 88帧 >40N 0 | wrist p50=16.39 max=95.02 | flips obs=181 cmd=68 | 改写=0  | cit_end=0
lora_B0_22  events=3 single=0 single_lift=1 bundle=3 held_no_release=0 rel_in_bin=0 rest_in_bin=0 dur_med=8   car_med=2mm
            pad p50=18.00 p95=26.02 max=27.73 | >25N 116帧 >40N 0 | wrist p50=16.02 max=97.86 | flips obs=180 cmd=48 | 改写=0  | cit_end=4 (官方 2)
lora_C0_11  events=0 single=0 single_lift=0 bundle=0 held_no_release=0 rel_in_bin=0 rest_in_bin=0 dur_med=0   car_med=0mm
            pad p50=22.46 p95=23.07 max=30.97 | >25N 6帧  >40N 0 | wrist p50=11.12 max=18.04 | flips obs=193 cmd=56 | 改写=1  | cit_end=0
lora_C0_21  events=4 single=0 single_lift=0 bundle=4 held_no_release=1 rel_in_bin=0 rest_in_bin=0 dur_med=8   car_med=1mm
            pad p50=0.00  p95=25.11 max=49.17 | >25N 36帧 >40N 1 | wrist p50=10.96 max=89.76 | flips obs=185 cmd=104 | 改写=11 | cit_end=0
lora_C0_22  events=2 single=0 single_lift=1 bundle=2 held_no_release=1 rel_in_bin=0 rest_in_bin=0 dur_med=176 car_med=76mm
            pad p50=7.99  p95=25.30 max=26.42 | >25N 36帧 >40N 0 | wrist p50=11.01 max=93.55 | flips obs=220 cmd=76 | 改写=49 | cit_end=3 (官方 1)

r4 (dir=/workspace/a3_eval/r4_b0_c0_paired, files=6，同一打分器重跑)
lora_B0_11  events=0 single=0 single_lift=0 bundle=0 held_no_release=0 rel_in_bin=0 rest_in_bin=0 dur_med=0   car_med=0mm
            pad p50=22.48 p95=23.07 max=25.53 | >25N 1帧  >40N 0 | wrist p50=10.88 max=13.63 | flips obs=226 cmd=14 | 改写=0  | cit_end=0
lora_B0_21  events=4 single=0 single_lift=0 bundle=4 held_no_release=1 rel_in_bin=0 rest_in_bin=0 dur_med=2   car_med=0mm
            pad p50=4.93  p95=25.64 max=25.96 | >25N 67帧 >40N 0 | wrist p50=11.18 max=93.99 | flips obs=195 cmd=107 | 改写=0  | cit_end=0
lora_B0_22  events=2 single=0 single_lift=1 bundle=2 held_no_release=2 rel_in_bin=0 rest_in_bin=0 dur_med=254 car_med=72mm
            pad p50=38.15 p95=87.12 max=865.43 | >25N 452帧 >40N 250 | wrist p50=11.21 max=333.05 | flips obs=244 cmd=66 | 改写=0 | cit_end=6 (官方 6)
lora_C0_11  events=0 single=0 single_lift=0 bundle=0 held_no_release=0 rel_in_bin=0 rest_in_bin=0 dur_med=0   car_med=0mm
            pad p50=22.48 p95=23.07 max=25.53 | >25N 1帧  >40N 0 | wrist p50=10.88 max=13.63 | flips obs=226 cmd=14 | 改写=0  | cit_end=0
lora_C0_21  events=3 single=0 single_lift=0 bundle=3 held_no_release=0 rel_in_bin=0 rest_in_bin=0 dur_med=1   car_med=0mm
            pad p50=4.02  p95=24.87 max=25.96 | >25N 19帧 >40N 0 | wrist p50=11.26 max=87.16 | flips obs=207 cmd=107 | 改写=6  | cit_end=0
lora_C0_22  events=1 single=0 single_lift=1 bundle=1 held_no_release=1 rel_in_bin=0 rest_in_bin=0 dur_med=340 car_med=130mm
            pad p50=18.92 p95=26.92 max=49.79 | >25N 80帧 >40N 7 | wrist p50=11.10 max=97.64 | flips obs=257 cmd=68 | 改写=112 | cit_end=6 (官方 3)
```

r4 B0/22 同一条动作流下 Safety 的影子计数（不执行、只统计）是「会改写 75.83 % 的步、会硬停 249 步」；r5 三条 B0 的 `estop_steps` 与 `binfloor_steps` 都是 0。

怎么读这三张表：

- 拿到手的实质进步只有一项：seed 22 的灾难尖峰消失了（指垫 865.429 → 27.726 N，腕部 333.05 → 97.86 N），三条 seed 落进 25–37 N 同一档位。
- 常态峰值反而略升（11：25.5 → 31.0，21：26.0 → 37.2 N），C0 侧 seed 21 从 25.96 N 涨到 49.17 N。所以「续训不恶化接触力」不是无条件成立。
- `single/rel_in_bin/rest_in_bin` 12 条全 0，`success` 12 条全 false。r4 B0/22 的「官方 6 块」是碾压/挂爪（见第 6 节），不能记成入盒。
- 由此定的决策：不启动 20000 步续训，改走「低力 + 纠正示范语料（B）」加「力通道输入（C）」。

## 6. 「正确释放」与「官方入盒计数」是两个口径

- 官方 `cookies_in_target` 由 harness 当场写进 `summary_lora.json`，判据含朝向/静止等条件，但**末帧中心落在目标盒 xy 投影内**这一步会被「挂在闭合夹爪上、z 还停在源高度 0.768」的饼干满足。r4 B0/22 的 6 块就是这种状态：终帧 `opening≈0.37`，正确释放 0 次，稳定入盒 0 块。
- 自算口径有三个字段，来自离线 npz + `autonomy_metrics.py`：`xy_projection(末)`（最松）、`rel_in_bin`（在盒内松手）、`rest_in_bin`（松手后仍留在盒内，最严）。
- 三处不一致必须记住：r5 B0/22 自算 4 / 官方 2，r5 C0/22 自算 3 / 官方 1，r4 C0/22 自算 6 / 官方 3。差的都是挂爪那些块。
- 规矩：报告「成功入盒几块」一律用 `rest_in_bin`；要引官方值就写官方字段名并注明来源 json；两栏不得混用。
- 离线复现不了官方条件：评估 npz 里没有 `cookies_in_target`、`cookies_in_target_mask`、饼干四元数、`qvel`，只有 `cookie_xyz (600,80,3)`、`eef_xyz`、`opening` 和力通道。要逐帧对齐官方，得在 recorder 里多存一个 `cookies_in_target_mask`（只是多存一个 info 字段，不动模型/Wrapper/Safety/仿真执行逻辑）。
- 还有一个刻度坑：判定「夹着」用的 `opening < CLOSED_OPENING` 默认 0.42 是策略的 pm1 刻度；单块专家夹到 0.011、只开到 0.192，用 0.42 会把「夹着」读成「张开」。2026-10-31 起该阈值改读环境变量 `CLOSED_OPENING`（打策略用默认，打单块专家用 0.12）。同一份 npz、同一个打分器，只是刻度不同。

## 7. 接触力量级：模型 25–865 N，专家约 1.3 N

- 确定性专家回放（同一条轨迹、打分器重算）：指垫力中位 **1.39 N**、峰值 **1.54 N**。
- 模型侧：常态 p50 就在 18–22 N，seed 22 的 r4 峰值 865.43 N、>40 N 有 250 帧；`pad_force_inner` 单通道峰值 859.170 N、>2 N 的帧占 542/600。也就是说力不是偶发尖峰而是持续挤压。
- 量级差 0.45–0.59 N 与 25–49 N 这个跨度，是判断「缺示范、缺力输入」而不是「缺步数」的主要依据。
- C0 能把峰值压下来（865.43 → 49.79 N），代价是 112/600 步被改写；两档最后都没有释放入盒，所以它只是保护，不是技能。

## 8. r4/r5 的轻量 checkpoint 漏存了 `proprio_encoder`

事实链，已在 pod 上逐张量核对：

- 轻量 ckpt 只落 **784** 个张量、**30,670,848** 个元素；而优化器里实际有 **786** 个张量、**30,707,712** 个元素，差 **36,864** 个元素，正好是 `proprio_encoder` 的两层权重与偏置。
- 判据要用「是否在优化器成员里」，不能只看 `requires_grad`（本轮就差点用错）。
- 后果：r4 `step_004000` 与 r5 `step_008000` 的 `proprio_encoder` **末态参数只存在于内存里那一次评估，pod 上没有任何副本，本次 HF/GitHub 备份也无法恢复它**。能恢复的是 adapter.pt / optimizer.pt / rng.pt / scheduler.pt / trainer_state.json 与训练脚本、配置、语料。
- 修复代码（light_state v3，把 786 张量全量落盘）本体与测试在仓库 `pod_snapshot/2026-11-01/_r6_stage/`（`light_state_v3.py`、`test_light_state_v3.py`，v2 两件同目录）；HF 的 `tests/light_state_test*` 是它产出的 checkpoint 与自测目录，不是源码。它只保证「下一次训练不再漏存」，不追认 r4/r5。
- 因此 r4→r5 这条比较线**不能从权重层面重跑复现**，只能引用已归档的 npz/视频/json 读数。这一条在任何后续报告里都得写在前面。

## 9. 73 条新增示范（低力 + 纠正）

- 台账：`corpus_manifest_v1.json`（pod 路径 `/workspace/a3_eval/corpus_manifest_v1.json`，41,118 B，sha256 `b679fa6bd07e12c3be25e02f1d5aa3d14e3d785fe726e04f98b1f3a391dd0e00`），HF 上在 `eval/a3_eval/corpus_manifest_v1.json`。
- 组成（共 73 条）：

| 类别 | 条数 | 内容 |
|---|---|---|
| 单块低力 | 30 | 一次一块、指垫力压在专家量级 |
| C-1 | 17 | 到位后延迟闭合 / 半开状态纠正 |
| C-2 | 8 | 一次夹 ≥2 块的错误闭合纠正（按第 3 节的真实任务口径定性） |
| C-3 | 18 | 搬运途中与释放点纠正 |

- 转出来的训练数组在 `a3_data/a3_fw_dagger`（4 文件，652,802,564 B）与 `a3_data/a3_fw_dagger_near`（4 文件，917,773,169 B），两批都随 `frames.dat` + `arrays.npz` + `dataset_stats.json` + `summary.json` 一起归档。
- 语料到此收口：r5 延长训练已被测得无效（第 5 节），本轮决定不再新采集、不再加步数。
- 待办：第 24 项审计（逐条轨迹核查、按 301,056 B/帧重算磁盘、做 2–3 条带图 smoke）还没做完，见第 13 节。

## 10. 力感知输入：8D → 14D 方案（尚未训练）

- 现状：模型本体感知只吃 `state[:, :8]`，**没有任何力输入通道**。
- 方案 V1：把专家集自带的 18 维力里挑 6 个通道拼进去，`FORCE_V6 = [0, 1, 2, 12, 13, 16]`，8D → 14D；生成脚本 `build_force_arrays.py`，输出目录 `/workspace/a3_data/a3_fw_force`。
- 已归档产物：`data/a3_fw_force/arrays.npz`（5,243,625 B）与 `data/a3_fw_force/force_stats.json`（16,529 B，归一化统计）。
- 明确状态：**V1 只是数据与接线方案，从未训练、从未评估**。不能把它当成已有能力。等预算的 V0/V1 对照定义在 `docs_task3` / 任务 #25。

## 11. 归档位置与下载链接

模型、数据、评估、研究归档全部在 HF 仓库（公开，无需 token）：

```
https://huggingface.co/wsadwsad6779/fasterwam-a3-cookie-lora
```

当前 `main` 指针：`f2aaf55c6ce1b164de97c5b6ec2cd75e942d0de0`（2026-11-01T08:48:11Z，最后一次提交 "Archive M batch 1/1"）。
远端共 1,264 个文件；`MANIFEST/` 下 4 份台账可直接下载。

目录约定（本次归档新建，未覆盖任何既有文件）：

| HF 路径 | 内容 | 档 |
|---|---|---|
| `step_002000_merged.pt`、`trainer/**`、`data/frames.dat` | r1 权重与训练状态（2026-10 就已上传） | 既有 |
| `weights/r2/`、`weights/r3/` | **台账里的目标路径，远端并没有这两个目录**：三个 >2 GiB 整文件上传会冻结，实际只存了分片（见下一行与 11.2）；恢复必须装配，不能直接下载 | — |
| `weights/r4/`、`weights/r5/` | r4、r5 轻量 ckpt（step_000500…step_008000）：adapter / optimizer / rng / scheduler / trainer_state | P0/P1 |
| `parts60gz/<档>/<原路径>/part_NNN.gz` | 三个 >2 GiB 权重的 60 MiB 分片（r3 `full_state` 178 片、r3 与 r2 的 `step_003000_merged.pt` 各 177 片，共 532 片），gzip 压缩（原因与恢复方式见 11.2 与 `docs/RESTORE_NEW_INSTANCE.md` 第 4 节） | P0/P1 |
| `data/` | `a3_fw`、`a3_fw_dagger`、`a3_fw_dagger_near`、`a3_fw_force` 的数组、`frames.dat`、归一化统计 | P0 |
| `eval/a3_eval/`、`eval/bridge/` | B0/C0 的 `summary_lora.json`、`manifest_lora.json`、npz、打分日志、dagger 恢复件 | P0 |
| `videos/a3_eval/`、`videos/fasterwam_a3_bridge/` | 评估与桥接录像 | P0 |
| `runs/a3_lora_r{2,3,4,5}`、`runs/a3_lora_r4_crashed_step500` | `trainer_state.json`、`lr_restart.json` 等训练侧记录 | P0/P1 |
| `replay/bridge/` | 确定性回放的 `frames.dat` | P1 |
| `tests/light_state_test`、`tests/light_state_test2` | v3 产出的轻量 checkpoint（`adapter/optimizer/rng/scheduler`）与自测目录，**不是源码**；源码在仓库 `pod_snapshot/2026-11-01/_r6_stage/` | P1 |
| `research/**` | `_r4_stage` 脚本、`expert_seed{11,21,22}.npz`、`_selftest` | P1 |
| `MANIFEST/p0p1_manifest.tsv` 等 | 归档台账（每行：档、`path_in_repo`、pod 相对路径、字节数、sha256） | — |

原始专家数据集（第三方，只引用不重传）：

```
https://huggingface.co/datasets/Eter0109/a3-front-close-left-100/tree/8fdd8a7caf8275eba8be8a83b53a99176a084062
```

### 11.1 校验口径与 A–F 分类

每个文件上传前先算 sha256，与台账记录不符就拒绝上传；因此「上传成功」的定义是**远端可读回且字节与 sha256 相同**，仅文件大小相同不算验证通过。全量服务端对账与抽样真实回读见 `archive/2026-11-01/` 与 `PRE_SHUTDOWN_FINAL_REPORT.md`。

3,888 个原始 pod 文件的最终分类按**内容哈希**判定，不看当初的上传记录（完整表见归档分支 `PRE_SHUTDOWN_FINAL_REPORT.md` 开头的 A–F 表与 `docs/BACKUP_STATUS_ABC_FINAL.md`）：

| 档 | 含义 |
|---|---|
| A | HF 已备份且 SHA256 匹配（三个 >2 GiB 权重按 532 个分片装配后再核整文件 sha256） |
| B | GitHub 已备份，远端 clone 逐文件验证 |
| C | Windows 本机已备份，SHA256 匹配 |
| D | 固定上游公开仓库 revision，可按 sha256 重新取得 |
| E | 经明确判断不需要保留的可再生缓存 / 自测产物 / 空文件 |
| F | 无可靠副本，退租后会丢失 |

> 旧版文档里另有一套 A/B/C（A = pod 外有逐字节副本、B = 只核过大小、C = pod 唯一副本），已废弃，不要与新表混用。

### 11.2 >2 GiB 权重的真实状态（必须如实写）

- 这个 pod 到 `huggingface.co`、`github.com`、`cdn-lfs*`、`s3.amazonaws.com` 都不通，只有 `hf-mirror.com` 在允许清单上，且它代理写接口；经典 LFS 预签名会打到 `s3.amazonaws.com`（403），所以只能走 Xet。
- 现象：大文件上传中途冻结——进度条不再动、`/proc/<pid>/io` 的 rchar 不变、CPU 接近 0、`ss -tnp` 里已经没有 socket、xet 日志停在最后一行。进程不退出，`timeout` 到点才杀掉。
- **根因（2026-11-01 07:30Z 前后定位，推翻此前「64 MiB xorb 边界」的解释）**：冻结发生在服务端对该文件的某个 chunk 返回 `"result":"found"`（即远端已存在该 chunk）之后；xet 客户端收到 `found` 就静默挂住。作为对照，同目录、同样大小的分片在 dedup 查询返回 `not_found` 时正常传完：`part_000` 62,914,560 B / 5.7 s / 11.07 MB/s，gz 版 `part_100` 50,473,476 B / 12.2 s / 5.37 MB/s。该冻结已观测到 3 次（`part_001` 两次、`part_100` 一次），每次 xet 日志末条都是 `dedup_task … "result":"found"`。
- 这也解释了为什么 256 MiB / 900 MB / 11 GB 的文件「必挂」：文件越大，越可能含有远端已存在的 chunk（既有上传、权重里的重复块），dedup 命中的概率趋近 1；而小的、内容全新的文件常常整份都是 `not_found`，于是看起来像「小文件能传、大文件不能传」，实际判据是 dedup 命中而不是文件大小或 xorb 边界。
- 换真实路径 / 软链、单文件 / 两文件 / 八文件一个 commit、`HF_XET_FIXED_UPLOAD_CONCURRENCY=1`、`HF_XET_MIN_SPACING_BETWEEN_GLOBAL_DEDUP_QUERIES=86400`、关进度条，都无法绕开这个命中即挂死的行为；`HF_ENDPOINT`/`HF_HOME`/`HF_HUB_CACHE`/`HF_HUB_ENABLE_HF_TRANSFER=0` 与能正常传的那批完全一致。
- 绕行办法：把每个 60 MiB（62,914,560 B）分片用 `gzip -n -1` 重新压缩。压缩后字节全变，任何 chunk 都不可能已存在于远端，dedup 查询一律 `not_found`，上传恢复稳定——实测 8 片一批（402 MB）一个 `create_commit` 40 s 完成（约 10 MB/s），远端回读 sha256 与台账一致。`-n` 保证没有时间戳，压缩结果可复现；GNU gzip 没有 `-0`（会产出 0 字节空文件）。
- 三个整文件与其分片：`r3/full_state/step_003000.pt`（11,179,613,340 B，178 片）、`r3/step_003000_merged.pt`（11,117,977,659 B，177 片）、`r2/step_003000_merged.pt`（11,117,977,659 B，177 片），共 532 片；每片的字节数与 sha256 都写进台账。
- 恢复：`cat part_*.gz | gunzip` 即整文件（gzip 多 member 可串联），再比对整文件 sha256。步骤写在 `docs/RESTORE_NEW_INSTANCE.md` 第 4 节，台账是 `archive/2026-11-01/parts60gz_manifest.tsv`（列：档、分片 `path_in_repo`、pod 相对路径、片字节、片 sha256）。
- 只要有一个分片未读回校验通过、或整文件重装配 sha256 不符，这三个权重就仍是 C 档（pod 唯一副本），不得当作已备份。

### 11.3 上传手册（下次直接照这个走）

这一轮在上传上踩的坑、验证过有效的做法，按执行顺序写在这里。换机器换实例都适用。

**1) 端点与进程内环境**

这个 pod 只有 `hf-mirror.com` 在允许清单上（`huggingface.co`、`github.com`、`cdn-lfs*`、`s3.amazonaws.com` 全不通），
且它代理写接口；经典 LFS 预签名会打到 `s3.amazonaws.com`（403），所以只能走 Xet。上传器必须在**进程内**强制环境，
只写在 shell 里不够稳：

```python
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/hfstage'          # 缓存放 /root，别占 /workspace
os.environ['HF_HUB_CACHE'] = '/root/hfstage/hub'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'    # hf_transfer 不走这里
os.environ.pop('HF_HUB_DISABLE_XET', None)       # 必须允许 Xet
```

**2) 先算 sha256，再决定传不传**

每个文件上传前先算 sha256 与台账比对，不符就拒绝上传；`archive_state.jsonl` 记下 OK 的路径，重跑自动跳过，
所以「断了就跑同一个命令」是安全的。**不要**先传后算。

**3) 一批 8 个文件，别一次几百个**

实测一批 8 个文件（约 400 MB）一个 `create_commit` 40 s 左右（10–21 MB/s）。批次太大时失败要整批重来，
排查也难。外层用 `timeout -k 10 600` 包住，`rc != 0` 就杀掉重试；state 去重保证已完成的不会重传。

**4) dedup 命中会让 Xet 静默挂死——这是唯一真坑**

判据不是文件大小，而是服务端对某个 chunk 返回 `"result":"found"` 之后客户端挂住：
没有 socket、`/proc/<pid>/io` 的 rchar 不动、CPU≈0、xet 日志冻结、进程不退出。
同一批里 dedup 返回 `not_found` 的文件都正常传完。遇到这个现象不要加超时硬扛，直接换内容。

**5) 绕行办法：`gzip -n -1` 重压缩**

压缩后字节全变，任何 chunk 都不可能已存在于远端，dedup 一律 miss，上传立刻恢复。
`-n` 去掉时间戳保证可复现；GNU gzip 没有 `-0`（会产出 0 字节空文件）。
还原时 `cat part_*.gz | gunzip`（gzip 多 member 可串联），整文件 sha256 不变。
这一轮 532 个分片全部按这个方式传完，8 片一批，无一次挂死。

**6) 大于 2 GiB 的文件先切片**

切成 60 MiB（62,914,560 B）的字节区间，每片单独 gzip、单独校验、单独记录 `part_bytes`/`part_sha256`。
不要把整文件直接扔给 Xet；也不要在 `/workspace` 里拼一个巨型 tar.gz（这一轮 `/workspace` 只剩 2.8 GB）。

**7) 传完必须做两件独立的事**

- **服务端对账**：拉远端 tree，逐行比 `size` 与 LFS 的 `sha256`，缺一个、差一字节都算失败。
- **真实回读**：抽样把文件从 `resolve/main/<path>` 流式读回本地算 sha256（不落盘）。只在 pod 侧回读还不够，
  这一轮又在本地（另一条网络、另一次 DNS）抽了一批，两边都对上才算过。

**仅文件大小相同不算验证通过。**

**8) 小文件/代码往 pod 上传的坑（Jupyter contents API）**

- contents API 返回的 `format` 可能是 `'text'` 也可能是 `'base64'`，**先判断再解码**，否则 `atob` 直接抛
  `InvalidCharacterError`。
- 大段 base64 手写进脚本会静默损坏：实测一个 3,856 B 的文件大小一致、内容里 `gz[idx]` 被改成 `gs[idx]`。
  所以 PUT 之后**必须读回算 sha256**；不一致时按 256 B 分块定位到唯一差异块，用 pod 端 `sed -i` 定点修，不要整份重传。
- 更稳的做法：在页面里放一个 `<input type=file>`，用浏览器自动化把本地文件塞进去，再用 `FileReader` → base64 →
  contents API 写入。字节不经过「模型抄写」，本轮 4 个文件（含 12.5 KB 脚本）一次通过、sha 全一致。
- 终端 websocket 的命令：先发 `Ctrl-C` 再发命令；输出重定向到文件再用 contents API 读回（单次脚本执行 15 s 上限）。

**9) 缓存与磁盘**

这一轮 `HF_HOME` 缓存 + gz 分片在 `/root` 下占了约 88 GB（`/root` 有 1.6 TB），`/workspace` 全程没被撑爆。
收尾时这些中间产物只列清单、等确认后再清，不自动删。

### 11.4 代码上传到 GitHub 的经验（本轮实测）

仓库是私有的，pod 又只放行 `hf-mirror.com`（`github.com` 不通），所以代码侧的推送、clone、比对全部在本机做，
pod 上只跑「从 HF 取数据」的恢复演练。下面几条都是这一轮真踩到的。

**1) `.gitattributes` 写 `* -text`，否则克隆出来字节就变了**

本机 `core.autocrlf=true`。不加这一行，clone 时会把仓库里的 LF 转成 CRLF，脚本的 sha256 随之改变，
跟 pod 上对过账的文件不再是同一份字节。仓库根的 `.gitattributes` 只有一行：

```
* -text
```

**注意 add 顺序**：`.gitattributes` 必须与其它文件在同一个 commit 里**先**生效。这一轮它比 `README.md` 晚加，
结果 `README.md` 的 blob 是按旧规则归一化的；文件内容没丢，但「工作区字节 = 仓库 blob」这句话当时不成立。
下次先写 `.gitattributes`，再 `git add -A`。

**2) 推完必须另开目录全新 clone 再逐文件比**

不能拿 `git status` / 本地 `git diff` 代替——它们看的是同一个工作区。做法是 `git clone` 到新目录，
对「仓库里每个受管文件」比 sha256，并单独统计只差行尾的数量：

```
clone 1207 个受管文件：953 逐字节相同 / 254 仅 CRLF↔LF / 0 其他差异 / 0 缺失
archive/2026-11-01/p0p1_manifest.tsv      sha256 54f4e3669a46b3af58331abee2b55c350c1292516b692c19cb4bb0520d596b08
archive/2026-11-01/parts60gz_manifest.tsv sha256 307172a4f82d98f58391068c2ac4078ac4f8ff64efafa50470361456db6d0f04
```

台账和恢复脚本的 sha256 是钉死的常量，clone 回来对不上就说明传输/归一化出了问题，先别庆祝推送成功。

**3) 大文件和大目录不进 git**

- 11 GB 级的权重、分片、评估视频、replay 帧一律走 HF；`.gitignore` 显式排除，并在 README 里写清它们在哪。
- `pod_snapshot/` 里的 `FasterWAM/`、`a3_dual_arm_sim/` 上游 checkout 也不进 git：它们与 pinned commit 逐字节相同，
  且整棵上游树另有 `archive/2026-11-01/git_snapshot_20261101.tar.gz`。

**4) 推送前扫密钥，token 不落任何持久位置**

提交前全量扫一遍 token / 私钥 / 口令；这一轮扫描后只剩 `HF_TOKEN=...` 这类占位符和一条第三方报错文本。
HF token 只存在于旧 pod 的 `/root/.cache/huggingface/token`（0600），不写进脚本、Git 历史、日志或聊天正文；
实例销毁后自然失效，但仍建议在 HF 后台吊销。

**5) 分支与验收顺序**

新开 `archive-2026-11-01` 分支推送，不动既有分支；推送成功只算「发出去了」，真正通过的标准是第 2 条的
全新 clone 比对 + 台账 sha 对上 + 之后的真实恢复演练。

**6) 恢复脚本必须真跑一遍，静态审查看不出台账列错位**

`restore_new_instance.sh` 第一版是把「分片台账第 1 列当路径」写的，而这一列其实是档位（P0/P1）。
脚本照常启动、照常往下跑，只是 532 个分片一个都没装配，`--verify` 还把 532 个不存在的路径算成
`giants_missing`、最后仍然 `rc=0`——只看代码和本地 `bash -n` 完全发现不了。是空目录演练把它逼出来的。
所以：改完恢复脚本，先在新目录里跑一遍 `--parts` + `--verify`，确认末行是
`verify rows=715 ok=715 missing=0 sha_bad=0 giants_missing=0 giants_bad=0 named_missing=0`、退出码为 0，
再谈恢复成功。只跑正例还不够——反例（空目录全量 `--verify`、点名一个不存在的文件、`--weights` 给一个台账里没有的路径）
必须各自返回非零；这一轮正是靠「缺文件仍然 rc=0」才发现判定漏洞的。
（三轮回演练的过程与数字见 `PRE_SHUTDOWN_FINAL_REPORT.md` 第 10 节。）

## 12. 新实例恢复与重启步骤

一键脚本（先读它，别手敲）：

```
scripts/restore_new_instance.sh
docs/RESTORE_NEW_INSTANCE.md
```

最小路径，逐条对应新 GPU 实例要能恢复的 8 项：

1. **代码 + A3 环境** —— `git clone -b archive-2026-11-01 https://github.com/wsadwsad6779/fasterwam-a3-cookie.git`，跑 `scripts/restore_new_instance.sh --code --deps`；Python 3.12.3 + ROCm，mujoco 3.3.2、robosuite 1.4.0、libero/bddl/robomimic、transformers 4.57.6、accelerate 1.13.0；pip 走阿里云镜像，一次性把可能缺的全装上。上游两个仓库按第 2 节的 pinned commit 取，然后必须补这条软链（worker 路径硬编码依赖它）：

```
ln -sfn /workspace/FasterWAM /workspace/fasterwam_a3_bridge/FasterWAM
```

2. **r1–r5 权重 / adapter / 优化器 / 训练状态** —— `parts60gz/**`（三个 >2 GiB 权重要装配）、`weights/r4`、`weights/r5`，r1 在仓库根与 `trainer/**`；台账里 `weights/r2`、`weights/r3` 的整文件路径远端没有，别直接下。注意第 8 节的漏存事实。
3. **原始 100 条专家与转换数据** —— 数据集按 revision 重下（第三方，需你确认允许再分发才复制），或直接取 `data/**` 的 `frames.dat` + `arrays.npz`。
4. **73 条新增示范** —— `data/a3_fw_dagger*` + `eval/a3_eval/corpus_manifest_v1.json`。
5. **B0/C0 评估数据与视频** —— `eval/**`、`videos/**`（`--rest` 一起取）；本地还有一份 `FasterWAM_r3_r4_r5_实验视频合集.zip`（18,732,532 B）与 `SHA256SUMS.txt`。
6. **训练配置 / 归一化统计 / 采样器恢复逻辑** —— `runs/**`、`data/**/dataset_stats.json`、`pod_snapshot/2026-11-01/_r4_stage/run_r5.sh`（含 `lr_restart.json` 的续训接线）。
7. **light_state v3 修复代码与测试** —— `pod_snapshot/2026-11-01/_r6_stage/`（源码与测试）、`tests/**`（`--rest`，v3 产出的 checkpoint 与自测目录）。
8. **各阶段研究报告与后续计划** —— `docs/**`（含本 README 与旧版交接）与本仓库根的 `docs_task1…docs_task5`、`PRE_SHUTDOWN_FINAL_REPORT.md`。

大文件取法实测速率（新实例照这个顺序试，别一上来直连）：

| 取法 | 实测速率 |
|---|---|
| ModelScope | 36 MB/s |
| hf-mirror 6 路 Range 分段 | 26 MB/s |
| hf-mirror 单流 | 5.6 MB/s |
| huggingface.co 直连 | 1.7 MB/s |
| `hf download` 默认 | 2 MB/s |

11.1 GB 级别的权重用六路分段脚本，脚本在 `scripts/restore_new_instance.sh` 里（`fetch_range6`）。取完必须比对 sha256，本 README 里出现的每个字节数与 sha256 都以 `archive/2026-11-01/p0p1_manifest.tsv` 为准。

`--verify` 先逐行核对 `p0p1_manifest.tsv` 的 715 个文件，再对由 532 个分片重装配出的 3 个整文件核对整文件 sha256
（预期输出 `verify rows=715 ok=715 missing=0 sha_bad=0 giants_missing=0 giants_bad=0 named_missing=0`）。全量恢复后才允许 `missing=0`；
如果只试取了几个文件，`--verify` 要把这些路径一起带上（如 `--verify weights/r4/light/step_004000/adapter.pt`），
否则缺文件会被算成失败。任何下载或装配失败都会以非零退出码结束，并在最后打印 `RESTORE FAILED: N file(s) ...`。

## 13. 未完成工作与下阶段计划

明确未完成、且不得被当成已完成的：

- **严格意义的抓放成功：0。** 12 条录制里没有任何一次 `rest_in_bin > 0`，`success` 全 false。
- **light_state v3 只做到代码与自测，没接到真实训练做一致性 smoke**（任务 #21/#27）。下一次训练前必须先补这一步，否则又会漏存张量。
- **r4/r5 的 `proprio_encoder` 末态参数永久丢失，本次备份无法恢复**（第 8 节）。r4→r5 这条对比线只能引用归档读数，不能重跑。
- **V1 力感知输入未训练、未评估**（第 10 节）。
- **73 条语料的逐条审计与带图 smoke 未完成**（任务 #24）。
- **v3 真实模型验证、V1 训练、严格抓放成功都还没做过**，任何后续文档把它们写成已完成都是错的。
- 数据侧的外部依赖：`data/a3_front_close_left_100` 是第三方数据集，本次只在 HF 引用原 revision，没有把它复制进我们的仓库；是否允许再分发需要你自己确认。FasterWAM / Wan-AI 权重的再分发许可同样待确认。
- 本会话用过的 HF token 需要你去 HF 后台吊销；它没有写进任何脚本、日志、commit 或聊天正文，只存在于旧 pod 的 `/root/.cache/huggingface/token`（0600），实例销毁即失效。

下一轮建议的最小闭环（按依赖排序）：

1. 新实例上跑 `scripts/restore_new_instance.sh --code --deps`，再 `--all`。旧 pod 上已经用空目录做过全量演练（`weights`/`rest`/`data`/`parts`/`verify` 全过，`rows=715 ok=715 missing=0`，详见 `PRE_SHUTDOWN_FINAL_REPORT.md` 第 10 节），**没演练过的只剩代码与环境安装这两步**。

2. 用 v3 修复后的落盘逻辑跑 1-step smoke，核对张量数 = 786、元素数 = 30,707,712。
3. 把 73 条语料 + `a3_fw_force` 拼成 14D 输入训练集，先做等预算的 V0/V1 对照（同 step、同 lr 形状、同语料），评估一律 B0/C0 成对，报 `rest_in_bin` 与官方值两栏。
4. 只有在 `single > 0` 之后才讨论加长训练。

> 恢复验证的边界（不要外推）：演练是**文件级**的，跑在同一台旧 pod 的 `/root/drill*` 空目录里，只是强制走远端下载；
> `--code`、`--deps` 与**模型推理 smoke 都没有在另一台全新 GPU 实例上跑过**。所以不能把本归档描述成
> 「已在新机器上端到端复现」；新机器上首次恢复仍需按上面两步重跑并自己核对 `--verify` 的输出。

## 附：仓库内容与许可

- `pod_snapshot/2026-11-01/` —— 949 个本项目代码/文档/脚本文件（5,466,151 B）从旧 pod 逐个 sha256 对拍后复制。旧 pod 上那份快照共 2,003 个文件（39,077,618 B），其中 1,054 个是 `FasterWAM/` 与 `a3_dual_arm_sim/` 的上游 checkout；它们与 pinned commit 逐字节相同（见第 2 节），整棵上游树又已在 `archive/2026-11-01/git_snapshot_20261101.tar.gz` 里，所以没在 git 里再放一份，`.gitignore` 里显式排除。
- `archive/2026-11-01/` —— 归档台账（`p0p1_manifest.tsv`、`parts60_manifest.tsv`、`parts60gz_manifest.tsv`、`batch_plan_summary.txt`、`upload_batches.json`、`git_snapshot_20261101.tar.gz` + `git_snapshot_verify.json`、`parts_reassembly.tsv`）与上传器源码 `uploader/`（`hf_archive.py`、`run_slice.sh`、`go3.sh`、`verify_git_snapshot.py`）。台账同时放在 HF 的 `MANIFEST/`，恢复脚本在本地缺台账时按固定 sha256 从那里自举。
- `scripts/`、`docs/`、`bridge/`、`finetune/`、`tools/`、`results/` —— 原创代码与文档。
- 没有写入任何密钥：本分支不含 HF/GitHub token、私钥或账号口令，提交前做过全量扫描（`pod_snapshot` 内只剩 `HF_TOKEN=...` 这类占位符与一条第三方报错文本）。
- 旧 pod 的 `/workspace` 数据没有整体打包成巨型 tar.gz 上传（磁盘只剩 3.2 GB，且你要求避免这一步）；分片中间件一律放 `/root`，不占 `/workspace`。
- 实例终止由你本人确认执行，我不会代为关闭。
