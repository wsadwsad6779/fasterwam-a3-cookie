# 任务一：r5 从 r4 续训 4000 步（累计 16000 步）

运行目录：`/workspace/a3_train_runs/a3_lora_r5/`（pod）。
产物：`train.log`、`windows.log`、`preflight.txt`、`lr_restart.json`、`config.yaml`、
`checkpoints/light/step_00{4500,5000,5500,6000,...}`。
起点：`/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_004000`（只读）。

## 0 约束逐条核对

| 任务要求 | 落地方式 | 证据 |
| --- | --- | --- |
| 正确恢复 optimizer / scheduler / RNG / global step / sampler | 全部从 r4 轻量态读入，非重新初始化 | `preflight.txt`、`lr_restart.json` |
| 原始 100 条专家数据不变 | 训练数组目录 `a3_fw` 只读；本轮没有任何写它的脚本 | `dataset_len 39481`（=100 条 42681 帧切出的样本数）不变 |
| 不改动作表示与输入结构 | 仍 `action 7` / `proprio 8` / 两路横向拼接图像 | `a3_2cam.yaml:24,35,36` 未改 |
| 不加 Wrapper、不加新安全规则 | 训练侧无策略代码；评估侧沿用现有 B0/C0，未新增规则 | 本轮没有写入 `safety_wrap`/`a3_wrap` 的补丁 |
| 轻量 checkpoint | `save_every=500`，每个 ~185 MB（adapter 61 MB + optimizer 123 MB + scheduler/rng/state） | `ls -la …/step_004000` |
| 明确记录数据覆盖、loss、lr | 见 §4、§5 | `windows.log`、`train.log` |
| 保留 r4 checkpoint | r4 目录只读；续训前逐个 sha256 记档，续训后再校验一致 | §1 |
| 提交 r5 与 r4 的 C0 对照 | 训练结束后由 watcher 自动跑 `--controls B0,C0`，再用 `autonomy_metrics.py` 打分 | §6（待训练结束） |
| 暂不启动 r6 到 20000 | 没有为 r6 建任何目录或脚本 | — |

## 1 起点保真：r4 轻量态的字节级记录

`preflight.txt`（2026-10-31T17:47:05Z）在启动前逐文件记了 sha256：

```
adapter.pt        61595489  e8372c333eb2e454dd91c79ff597bd4ce6ae0aa3a2594c57b016e02ff40f916d
optimizer.pt     123478413  e8e9d350c412019daca8feec6390f3a02f7c9a949ed39c83d489ce3b24d29978
scheduler.pt          1675  f470a84f04d9cf0ef00698be1086f4342a5be09287ce8a703ebfb1fefe1e0abd
rng.pt                14557  3f6ed0124e60a58fc61863768582c794a6bb35a7d23be37c9abeacddccf93cc0
trainer_state.json       639  c5fd77224fff3cd53e5dbccddc15f6c785b950da5460883518d8a8ff6de9d5e3
```

底座（冻结权重）`/workspace/a3_train_runs/a3_lora_r3/step_003000_merged.pt`，
11117977659 字节，`base_sha256_head=1835f884d430f3f76c2cc120d8753cb66300ea3e9f79c4ecf2002636c62fee9f`。
代码身份也一并记档：`r4_patch.py 4095a536…`、`r5_patch.py a6709bbb…`、
`light_state.py a18b207c…`、`trainer.py c021d8f0…`、`train_light_r5.py 6bff9c21…`。

r4 目录写入前后都只读；`checkpoints/light/step_004000/` 的文件时间仍是 13:52，
说明续训没有回写起点。

## 2 恢复出来的训练状态

`trainer_state.json` 里 r4 结束时的状态（被 r5 读入而非重建）：

```
format fasterwam-lora-light-state-v1   global_step 4000   epoch 1   batch_in_epoch 419
sampler {seed 42, epoch 1, epoch_offset 0, resume_batch_offset 419,
         sample_offset 18000, batch_size 6, num_processes 1}
extra   {base_checkpoint …r3/step_003000_merged.pt,
         instruction "transfer 10 cookies into target box",
         lr 5.000000000000001e-07, lora_init_seed 42}
```

日志对应行（`train.log`）：

```
Applied LoRA to 392 layers (trainable params = 30670848)
LoRA mode: only lora_ params trainable.
Loading frozen base first: /workspace/a3_train_runs/a3_lora_r3/step_003000_merged.pt
LR restart: step=4000->8000 warmup=200 peak=5.000e-05 eta_min=5.000e-07
            (restored lr=5.000e-07, first restarted lr=2.500e-07)
sampler pass=1 windows=36967 head=27048 tail=3548
```

`lr_restart.json` 里还留下调度器恢复的细节：
`resumed_global_step 4000`、`segment_steps 4000`、`lr_restored 5.000000000000001e-07`、
`restart_peak_lr 5e-05`、`restart_warmup_steps 200`、`restart_cosine_t_max 3800`、
`restart_eta_min 5.000000000000001e-07`，以及内层 `LinearLR(start_factor 0.005, total_iters 200)`
与 `CosineAnnealingLR(T_max 3800)` 的 `last_epoch` 都从旧态读回。

"16000 步"是分段累计：r1 2000 + r2 3000 + r3 3000 + r4 4000 = 12000，r5 +4000 = 16000。
单段命令行是 `max_steps=8000`，因为 global step 是累计计数、不是本段计数。

## 3 数据与表示没动

- 数据：`dataset_len 39481`（由 100 条示范的 42681 帧按窗口切出），本轮未重建。
- 采样延续 r4 的 offset：`sample_offset 18000`、`resume_batch_offset 419`、`seed 42`，
  所以 r5 不会重复 r4 已经吃过的窗口顺序。
- 表示：`action_output_dim 7`、`proprio_output_dim 8`、`video_size [224, 448]`
  （两路横向拼接）、`action_video_freq_ratio 4`，全部保持原值；
  `concat_multi_camera: horizontal`、`num_output_cameras: 2`。
- 结构：没有加 Wrapper、没有加安全规则；`FW_SKIP_STATE=1` 是既有设置（跳过优化器内
  冻结态的冗余校验），不是新机制。

## 4 学习率与本段 loss

| 段 | 步 | lr 走向 |
| --- | --- | --- |
| r5 前 200 步 | 4000→4200 | warmup 线性爬回 5.0e-05（`log_every=20` 下第一个记录点是 4020 步的 5.23e-06） |
| r5 余下 3800 步 | 4200→8000 | cosine 衰减到 5.0e-07 |

`r5_report.py` 打出来的 lr 端点：`lr_unique_endpoints [4.98e-05, 4.99e-05, 5e-05]`，
也就是 `FW_LR_RESTART=1` 之后调度器确实把 lr 重新爬回了峰值再衰减，
末点 `step=8000 lr=5e-07`、首点 `step=4020 lr=5.23e-06` 都对得上。

**这条必须写进 r5 vs r4 的结论里**：r5 相对 r4 同时改了两个变量——训练步数（+4000）
和 lr 形状（r4 末段 lr 已经衰减到底，r5 用 warmup 重新拉回 5e-05 再退火）。
所以 r5 与 r4 的差值不能解释成"纯粹的更多训练量"，它是一个"续训 + 重加热"的组合实验；
`lr_restart.json` 里同时留了 restored 快照与 restarted 后的调度器状态，这一点是可核验的。

本段 loss（200 个记录点，按步区间分位取均值）：

| 分位 | 步区间 | mean loss | mean loss_action | mean loss_video |
| --- | --- | --- | --- | --- |
| 1 | 4020–4800 | 0.14054 | 0.02759 | 0.11294 |
| 2 | 4820–5600 | 0.14093 | 0.02017 | 0.12078 |
| 3 | 5620–6400 | 0.13798 | 0.02136 | 0.11662 |
| 4 | 6420–7200 | 0.15043 | 0.03449 | 0.11593 |
| 5 | 7220–8000 | 0.13370 | 0.02882 | 0.10488 |

首点 `loss=0.1505 action=0.0453 video=0.1052`，末点 `loss=0.181 action=0.0463 video=0.1347`；
最低单步 action loss `0.0011 @ step=5860`（同点 total 0.0904、video 0.0893、lr 3.01e-05）。

诚实的读数：**这 4000 步没有把 action 分支压低**。分位 1 到分位 5 的 `loss_action`
是 0.02759 → 0.02882（还略升），中间分位 4 反而最差 0.03449；真正下降的是
`loss_video`（0.11294 → 0.10488）。那个 0.0011 的单步最低值是单个 batch 的抽样波动
（batch=6、每 20 步才记一次），不能当趋势。这条读数直接进任务五的判据。

## 5 覆盖与 loss 记录

`windows.log` 的窗口覆盖对账（同一个数据集共 39480 个窗口）：

| run | records | distinct | max_index | 重复次数 min–max |
| --- | --- | --- | --- | --- |
| r5（本次 4000 步） | 24054 | 24054 | 39479 | 1–1（**零重复**） |
| r4（历史 4000 步） | 24049 | 22609 | 39480 | 1–2 |

- 24054 = 4009 个 step × batch 6，与"从 4000 续到 8000"的消耗量吻合，
  说明恢复出来的 sampler 计数器没有把已训窗口重放一遍。
- r5 这一整段每个窗口只用了一次，覆盖到数据集的 61%；
  对照 r4 有 1440 个窗口被用了两次（上一轮就发现过的 sampler 尾部重复），
  这一轮的 pass-1 窗口是连续的（CONTIGUOUS），没有踩到那个坑。
- 原始 100 条示范的数据本身未改动，表示与输入结构未改动（§3）。

检查点：`step_004500 / 5000 / 5500 / 6000 / 6500 / 7000 / 7500 / 8000` 全部落盘，
`a3_lora_r5` 合计 1.4 GB，r4 的 `step_004000` 原样保留未动。
结束时 `/workspace` 余量 3.42 GB（97% 已用）。

## 6 r5 vs r4 的 B0/C0 对照

评估已经跑完：`/workspace/a3_eval/r5_b0_c0_paired/` 六条录制（B0/C0 × seed 11/21/22，
`steps=600`、`replans=60` 全部一样），九项指标由同一个打分器分别打到
`r5_autonomy_metrics.txt`（2219 B）与 `r4_autonomy_metrics_rerun.txt`（2227 B），
r4 的原目录与原始 `summary_lora.json` 未写入。

两侧用的是**同一个文件、同一个 sha**：`/workspace/_r4_stage/autonomy_metrics.py`
（`a6979e8dff97f6270fed3b31495bf8679f5d4c3b7065e90d17d6c665111cdc45`），
`r5metrics.sh` 第 11–12 行两次调用只差输入目录；
`r4_b0_c0_paired/` 里所有文件的 mtime 仍是 10-31 16:32–17:04（原始评估那一次），
重跑只往 `/workspace/a3_eval/` 根下写了那个 txt。

三句话结论，逐条数字与三栏口径（官方 / 自算松 / 任务口径）在
`docs_task5_decision.md` §3.1–§3.3，这里不重复：

1. **12/12 条录制的 `success` 都是 false**，`single`（合格单块抓—搬—放）、
   `rel_in_bin`、`rest_in_bin` 在 r4 与 r5 两侧**全为 0**——多训这 4000 步
   没有产生过一次合格的"抓住—搬走—在盒里松手"。
2. **865 N 那种碾压机质消失了**（r5 B0/22 指垫峰 27.726 N、腕部 97.86 N），
   但常态峰值小幅上移（B0/11 25.53→30.97，B0/21 25.96→37.18 N），
   C0 侧 seed 21 峰值 25.96→49.17 N。所以是"病态减少 + 常态略恶化"，两半都要报。
3. **Safety 的接管没有变成可有可无**：r4 C0/22 是 7 步硬停 + 79 步盒底托举 + 112 步改写，
   r5 换成 C0/21 一步硬停 + C0/22 37 步盒底托举 + 49 步改写。接管点随权重迁移，量级仍在。

对照公平性（同 `run_matrix.py` sha、同底座、同 seeds、同 flags，唯一变量是 adapter）
与"这一步同时改了 lr 形状、不是纯训练量实验"这两条写在 `docs_task5_decision.md` §3 开头。

## 7 一个必须记进结论的发现：轻量态不覆盖全部可训练参数

`light_state.py:35-41` 的 `lora_named_parameters()` 只挑参数名里含 `lora_` 的参数，
`adapter_state_dict` / `save_light_state` 因此只保存 LoRA。对账结果（r4 `step_004000`）：

```
adapter.pt   : 784 个张量，全部 lora_ 命名，共 30670848 个元素
               keys = {format, lora, meta(global_step, base_checkpoint)}
               其中 proprio 相关张量：[]（空）
optimizer.pt : param_groups 1 个，参数 786 个，动量元素合计 30707712
差值         : 2 个参数、36864 个元素 = proprio_encoder.weight (4096,8) + bias (4096)
               占可训练总量 30670848 的 0.120 %
r3 merged ckpt keys : [mot, step, torch_dtype, proprio_encoder]
```

也就是说 `proprio_encoder` 是**可训练但不出货**：每次续训都从上一段的合并底座
（这里是 r3 merged）重新读入它，而 Adam 一阶/二阶动量却是上一段训练自己积累的旧值。
对 r4→r5 这条曲线的影响是：这 0.12 % 的参数每段都被"重置权重 + 保留旧动量"，
它的梯度历史与实际权重不匹配。r4/r5 的 8 维 proprio 语义没变，所以结论仍可解释为
"同一结构上的连续训练"，但**这个偏差在 V1（proprio 8→14）之前必须修**，
否则学到的力投影会在下一次续训时被静默丢掉。

修法（只动训练状态格式，不动模型/Wrapper/Safety/仿真）：轻量态增加一个
`extra_trainable` 字典，保存所有"可训练但名字不含 `lora_`"的参数；
装载时若该字典缺失则显式报错而不是静默跳过。r6 之前落地即可，本轮 r5 不改，
以保持 r5 与 r4 的可比性。

**这条对评估读数的直接后果（必须写进 r5 vs r4 的结论）**：两次对照的加载路径都是
"r3 合并底座 + 本轮 LoRA adapter"（`run_matrix.py` 打印
`trainable=30670848`，只有 LoRA 被挂上），所以评估里跑的其实是
**r3 的 `proprio_encoder` + r4 或 r5 的 LoRA**，不是训练末期那套完整权重。
三点推论：

1. r4 与 r5 犯的是同一个错、错法一致（同一个底座、同一个缺失分支），
   所以两者的**差值仍然可信**，它隔离出来的就是 LoRA 那 30.67 M 元素因多训 4000 步
   （外加 lr 重加热）产生的变化；
2. 但 r5 的**绝对表现**不能代表 r5 训练出来的模型：训练后期 LoRA 是跟着
   它自己那份一直在变的 proprio 输入一起适应的，评估时把那份 proprio 换成 r3 的，
   等于给策略喂了一个轻微离散的 8 维状态通路；
3. 因此任何"r5 学会了/没学会"的判定都只在这个混合体的意义上成立，
   而 `rest_in_bin` 之类的九项绝对值在 `extra_trainable` 修好之前**不可跨轮次当基线复用**。

## 8 运行监控

- 速度稳定 0.41 step/s（2.46 samples/s），4000 步 ≈ 2.7 h。
- 保存策略 `save_every=500` → 本段共 8 个检查点，每个约 185 MB，累计约 1.5 GB；
  `/workspace` 当前可用 4.1–4.5 GB（已用 96 %），够放下但不宽裕，
  所以评估脚本 `r5_eval.sh` 仍带"输出目录必须为空 + 剩余空间 ≥2 GB"的硬检查。
- 磁盘、进程、日志三处心跳由 agent 侧轮询，不依赖人工守。
