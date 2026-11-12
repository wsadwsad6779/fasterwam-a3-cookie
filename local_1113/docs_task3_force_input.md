# 任务三：把力观测喂给 Faster-WAM（V0 / V1 设计）

结论先说：力这一路不需要重新采数据。100 条专家示范的原始 LeRobot 集里每一帧都存了
`observation.force`（18 维，float32），只是训练用的派生数组 `a3_fw/arrays.npz` 没把它带上。
所以任务三的第一版改动是"重建数组 + 加宽一个线性层"，不是重采集，也不是重构模型。

## 1. 现在模型看见什么、吐出什么

配置与代码事实（全部按行号可查）：

| 项 | 值 | 位置 |
| --- | --- | --- |
| 图像输入 | 两路 224×224，横向拼接成 `[3, 224, 448]` | `configs/data/a3_2cam.yaml:13-24`（`concat_multi_camera: horizontal`，`num_output_cameras: 2`） |
| 本体感知维度 | 8 | `configs/data/a3_2cam.yaml:36` `proprio_output_dim: 8`，`shape_meta.state` 也是 8（`:19-21`） |
| 8 维的含义 | 左臂 q1..q7 + 夹爪，是关节量不是笛卡尔位姿 | 语料 `collection_summary.json` 的 `training_joint_order` |
| 动作输出 | 7 维，前 6 维是 delta、第 7 维夹爪绝对值 | `a3_2cam.yaml:35` + `delta_action_dim_mask`（`:37-38`） |
| 频率 | 20 Hz 控制，`action_video_freq_ratio: 4` | `a3_2cam.yaml:23`，语料 `summary.json` 的 `fps: 20` |

本体感知进模型只经过一个线性层，变成一个 token 拼进条件序列：

- `models/wan22/fastwam.py:76`：`self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim)`，
  其中 `text_dim = 4096`（`configs/model/fasterwam.yaml:22`），`proprio_dim` 直接引用
  `data.train.processor.proprio_output_dim`（`fasterwam.yaml:6`）。
- `fastwam.py:282-296` `_append_proprio_to_context()`：8 维→1 个 4096 维 token，追加到文本/图像条件后面。
- 训练侧 `fastwam.py:415-425` 要求 `sample["proprio"]` 必须存在且最后一维等于 `proprio_dim`；
  推理侧 `fastwam.py:868-876` 同样校验。形状校验点只有这几处，改宽度是配置级改动。

对设计最重要的两点：

1. `trainer.py:86-88` 把 `proprio_encoder` 的参数显式加进 AdamW；
   `trainer.py:296-299` 在"只训 DiT / LoRA 模式"里冻结一切之后再把它单独解冻
   （`proprio_encoder.train(); requires_grad_(True)`）。也就是说这一层从一开始就是可训练的，
   加宽它不需要打开任何新模块。
2. `fastwam.py:1194-1195` 加载它用的是 `load_state_dict(..., strict=True)`，
   所以 8→14 的宽度变化必须配一个"按列拷贝"的权重装载补丁，不能指望旧 checkpoint 直接对上。

## 2. 力信号从哪来，能不能在真机测到

仿真侧的 18 维定义在 `sim/env.py:375-388`，按拼接顺序：

| 索引 | 内容 | 单位 | 真机对应物 |
| --- | --- | --- | --- |
| 0:3 | 左臂腕部六维力传感器的三轴力 | N | 商品化 F/T 传感器（ATI Nano17 一类） |
| 3:6 | 左腕三轴力矩 | N·m | 同上 |
| 6:9 | 右腕力 | N | 同上 |
| 9:12 | 右腕力矩 | N·m | 同上 |
| 12 | 左指垫内侧接触 | N | 指尖触觉阵列或指尖电机电流标定 |
| 13 | 左指垫外侧接触 | N | 同上 |
| 14, 15 | 右指垫内/外 | N | 同上 |
| 16, 17 | 左/右指尖执行器合力 | N | 电机电流 × 力常数 |

采样与对齐：这些量都在 `sim/env.py` 里由 `mj_forward` 之后的传感器读数拼出，每个控制帧刷新一次，
20 Hz，与图像、`observation.state` 同一帧；不是异步高频力采样。语料侧 `meta/info.json` 明确写了
`observation.force float32 [18]`、`observation.velocity float32 [16]`、`observation.eef_pose float32 [14]`、
`timestamp float32 [1]`，50 个 parquet 文件合计 42681 行，与 `arrays.npz` 的 42681 帧一一对应，
所以时间戳同步是现成的，不需要插值。

明确不用的输入：`privileged_cookie_positions`、`privileged_cookie_in_target`、
`privileged_left_finger_contacts(i)`（`tasks/cookie_transfer.py:153-215`）这类只有仿真可见的
物体真值和接触身份。它们只允许出现在打分和标签生成里，不允许进模型输入。

## 3. 实测到的力长什么样（决定了该给哪几维）

同一批场景下的确定性回放（`/workspace/_r5_stage/replay_forces.py`，sha256 `a71cbd26a46cf34f...`）：

- 指垫通道是信号源：单块低力夹取时，首次接触 0.11-0.32 N、闭合峰值 0.47-0.58 N、
  搬运期间均值 0.578 N、释放后 0.005-0.011 N。2×10 专家代码里也写着同一件事：
  一块 6.3 mm 的饼干夹到底约 0.58 N。
- 腕部通道基本没有负载信息：语料静止帧的左腕力是 `[1.5e-5, -10.69, 1.2e-4]`，
  右腕 `+10.83 N`，这条 ±11 N 是夹爪自重压在传感器上的常数偏置；
  r4 裸策略跑动时腕力 p50 也只有 11.1-11.2 N，而指垫 p50 是 38 N。

所以 V1 取 6 维，只取正在干活的那条臂（语料 `training_arm_mode = left`）：

```
force6 = [ L_wrist_force_x, L_wrist_force_y, L_wrist_force_z,   # 0:3
           L_finger_inner_touch,                                  # 12
           L_finger_outer_touch,                                  # 13
           sum(L_finger_actuator_force) ]                         # 16
proprio_v1 = concat(proprio_v0(8), force6)                        # 14 维
```

腕部只给三轴力、不给力矩，理由是当前任务几乎不做侧向刮擦，力矩主要是自重力臂的镜像；
先让模型有一个"能察觉挤压"的最小通道，而不是一次性加 12 维。

## 4. 最小改动方案

改动清单（按依赖顺序）：

1. 重建训练数组：把 18 维力按上表切成 6 维，与现有 `proprio (42681, 8)` 行对齐后存成
   `proprio14`。这一步不产生新数据，只读 parquet。
2. 归一化（不能沿用 state 的 min/max）：
   - 每维先减"该回合前 20 帧的中位数"作为零载参考（腕部 y 会减掉约 -10.7 N 的重力基线）；
   - 再除以语料上该维 p95 增量的绝对值，夹在 `[-5, 5]`；
   - 统计量只在训练集上算，写进 `dataset_stats.json` 的独立键 `force`，与 `state`/`action` 分开。
   这个"回合起点中位数"在真机上同样可测（抓起之前悬空采 1 秒），不是仿真特权。
3. 配置：`a3_2cam.yaml:36` `proprio_output_dim: 8 → 14`，`shape_meta.state.raw_shape` 同步。
   `fasterwam.yaml` 不用改，它是引用。
4. 权重装载：`nn.Linear` 从 `weight (4096, 8)` 变 `(4096, 14)`。做法是新层的**前 8 列等于旧层、
   后 6 列全零**。零初始化保证第 0 步 V1 与 V0 输出逐元素相等，之后这 6 列才从数据里长出来。
   这个补丁写在 `--adapter` 装载路径旁边（bridge 侧的一个 patch 文件），不改仓库源码，
   避免影响已经在跑的 r5。

代价：新增 `6 × 4096 = 24576` 个权重 + 4096 偏置里被复用的那一份，即每个臂约 2.5 万参数
（相对现有 3067 万可训练参数的 0.08 %），轻量 checkpoint 多约 0.1 MB/臂，优化器状态多约 0.2 MB。
不增加 token 数（仍然是一个 proprio token），不增加显存量级。

## 5. V0 / V1 公平对照

| 维度 | V0（不加力） | V1（加 6 维力） |
| --- | --- | --- |
| 起点权重 | r5 `step_008000` | r5 `step_008000`（同一份） |
| 力观测 | 无 | 6 维，零初始化列 |
| 段长 | +2000 步 | +2000 步 |
| 优化器 | 新建 AdamW（两臂一致，避免继承偏置） | 同 |
| 学习率 | warmup 100 步（`trainer.py:99` 的 5 % 规则在 2000 步下就是 100）→ cosine 到 5e-7 | 同 |
| batch / seed | 6 / sampler seed 42，窗口顺序两臂对齐 | 同 |
| 数据 | 同一 100 条语料（+ 任务二新增示范时两臂同时加） | 同 |
| 评估 | B0 与 C0 配对，seed 11/21/22，`--replan 10 --max-steps 600 --gripper-mode pm1`，九项指标 | 同 |

看三个读数就够判断有没有收益：

1. B0 下搬运/闭合阶段的指垫力分布（p50、p95、max、>25 N 帧数）是否下降；
2. B0 的单块抓取次数、正确释放次数、稳定入盒数是否上升；
3. B0 的 `verbatim_pct` 是否上升——这一条最关键：如果裸模型自己就能保持低力，
   Safety 需要改写的步数自然下降，才是用户要的"不靠辅助规则也能抓放"。

力列对第 0 步的严格等价性由零初始化保证，因此 V1 与 V0 的差异只能来自训练过程本身。

## 6. 需要多少条带力观测的新示范

不需要为了"有力"而新增示范：42681 帧全部已经带力。需要新增的是为了"让力有用"的示范，
也就是力的分布里要有可区分的对照，这来自任务二：

- 30 条单块低力抓取（1 次夹取/回合，带场景抖动）：给出 0.4-0.6 N 量级的正确夹持区；
- 20 条纠正示范（在下压力越接近阈值时回退重新对准，或在错误闭合前张开重夹）：
  给出"高力→回退→低力"的轨迹，否则模型永远看不到"察觉力大之后该怎么办"的样本。
  只有第一类而不给第二类，力观测大概率只会被当成背景噪声。
- 首轮 30+20 条用于读出方向；要把 V1 变成正式重训，需要 100 条量级。

## 7. 可选支线：用力标签做接触状态预测

辅助任务可以做，但要说清标签从哪来、损失怎么进模型：

- 标签来源：指垫力 `max(pad_inner, pad_outer) > 0.05 N` 且连续 ≥3 帧算"接触"，
  间隙 ≤8 帧内合并成同一个事件；再叠一个"夹持中"状态（接触且张开度低于闭合阈值）。
  这是可测信号的量化，不是仿真接触真值（`privileged_left_finger_contacts` 那种），
  因此真机上同样能生成。
- 训练方式：现有损失只有两项，`fastwam.py:642`
  `loss_total = λ_video * loss_video + λ_action * loss_action`。
  辅助头要接成"从条件特征分叉的三分类头 + 自己的 λ_aux"，
  并且它监督的是图像/本体表征，不进入动作反传路径。
- 明确不能做的事：把仿真 `mj_forward` 之后的接触力读数当成一个可反传的损失项塞进 BC——
  它不在计算图里，是环境输出的常数，反传不动。它只能作为标签或观测。
- 定位：V1b 支线，等 V0/V1 的对照出结果再决定要不要投。

## 8. 风险与已知坑

1. 仿真与真机的触觉量程、噪声底、延迟不同。单块饼干 0.58 N 的信号量级偏低，
   真机指尖传感器噪声底若接近 0.2 N，信噪比会很差；需要选更高灵敏度的指尖传感器，
   或在仿真里把力放大到与真机一致的刻度后统一归一化。
2. 20 Hz 采样只有 50 ms 粒度，接触瞬间的尖峰会被吃掉——r4 的 865 N 是持续挤压而不是单次冲击
   （>40 N 共 250 帧），所以这个粒度够用；但如果以后依赖"撞击检测"就不够了。
3. 腕部三轴力可能被判成"与任务无关的常数+重力"而学成噪声；对策是保留它但监控权重范数，
   若 6 列里力列的范数增长明显小于 proprio 列，说明没用上。
4. 轻量 checkpoint 目前只存 LoRA（`light_state.py:35-41` 只挑名字含 `lora_` 的参数）。
   实测对账（r4 `step_004000` 轻量目录）：`adapter.pt` 里是 784 个张量、30670848 个元素，
   与日志 `trainable params = 30670848` 的 LoRA 部分一致；`optimizer.pt` 的单一参数组有
   786 个参数、动量总量 30707712 个元素，**多出的 2 个参数 = 36864 个元素，正好是
   `proprio_encoder` 的 `weight (4096, 8)` + `bias (4096)`**（占可训练量的 0.120 %）。
   r3 的合并权重里确实有 `proprio_encoder` 这个键，所以每次续训都是"从上一段合并权重
   重新初始化这一层 + 恢复属于上一段自己的 Adam 动量"。V1 加宽这一层之后，
   如果不先修这个格式，跨段续训会把学到的力投影丢掉、只留下陈旧的动量。
   修法是轻量态多存一个 `extra_trainable` 字典，这是训练状态格式，
   不改模型、不改 Wrapper、不改 Safety、不改仿真。

## 9. 已落地的数据准备（V1 的前置件，不含模型改动）

`/workspace/_r5_stage/build_force_arrays.py`（sha256 `4c4cbad32fee214aec3dd6738ad959dc59a140948636275909aef7c3827f6a73`）
已经把 §7 第 1、2 步做完，产物 `/workspace/a3_data/a3_fw_force/arrays.npz`（5.2 MB，新建目录，
`a3_fw` 原文件未动）加 `force_stats.json`。跑出来的对齐与统计：

- `episode_index` 与 `a3_fw/arrays.npz` 完全相等，`state[7]` 与 `opening` 最大绝对差 `0.0`
  → 18 维力与训练数组**逐帧同一行**，不需要插值、不需要重采。
- 100 个 episode，长度 417 / 426（中位）/ 448 帧。
- 一个必须记住的事实：`proprio` **不是**原始 state——两者最大绝对差 3.762，
  说明训练数组里的 proprio 已按 `dataset_stats.json` 归一化过。
  所以力通道也必须归一化到同一尺度，不能直接把牛顿值接在后面。
- 6 条力通道按"减去该回合前 20 帧中位数（零载参考）"后的分布：

```
通道            L_wrist_fx  L_wrist_fy  L_wrist_fz  L_pad_inner  L_pad_outer  L_gripper_act
零载参考中位      -0.0001    -10.7212      -0.0282       0.0000       0.0000       -0.0186
残差中位          -0.0001      -0.1216       0.0137       0.9382       0.9378        1.8955
残差 p95           0.0764       0.6108       0.2121       1.4450       1.4535        2.9156
残差 |max|         3.5879       3.5483       1.3152       1.5398       1.5180        2.9741
残差均值          -0.0900      -0.1837      -0.0005       0.7043       0.7085        1.4202
残差标准差         0.7339       0.5891       0.1823       0.6814       0.6855        1.3970
```

腕部三轴去掉重力后中位接近 0（|残差| p95 ≤0.61 N），指垫两条 0 ~ 1.54 N，
夹爪驱动力 0 ~ 2.97 N——这正是 V1 需要的"有信号但不淹没"的量程。
按 §7 的做法，除以各自的标准差（0.68 ~ 1.40）即可落到约 ±2 的尺度，
与已归一化的 8 维 proprio 同阶。数组里同时存了 `force6`（原始牛）和
`force6_zeroed`（去零载后的残差），训练时取后者，`force_stats.json` 里的
`residual_std` 就是除数。

## 10 接线点逐条核到行号（2026-10-31 21:20 只读复核，未改任何代码）

训练仓库是 `/workspace/FasterWAM`，hydra 入口 `scripts/train_light_r5.py`
（`config_name="train"`、`data=a3_2cam`）。下表每格都是当场 `sed -n` 读出来的，不是推断。

| # | 位置 | 现状（verbatim 要点） | V1 要做的事 |
| --- | --- | --- | --- |
| 1 | `prepare_a3_arrays.py:20-23` | 只读 `observation.state / eef_pose / action` 三列，**力根本没读** | 增读 `observation.force`（18 维，已在 parquet 里） |
| 2 | `prepare_a3_arrays.py:104-108` | `prop[:, :3]=位置`、`3:6=四元数换算`、`6=0.04*opening`、`7=-0.04*opening` → proprio 8 维 | 8 维后面接 6 维去零载力残差 |
| 3 | `configs/data/a3_2cam.yaml:17-20,36` | `state: raw_shape 8 / shape 8`、`proprio_output_dim: 8` | 新建 `a3_2cam_force.yaml` 把它扩到 14；`model/fasterwam.yaml:6` 是引用，不用改 |
| 4 | `a3_video_dataset.py:94-96,153` | `arr["action"] (N,7)`、`arr["proprio"] (N,8)`，`"state"` 直接取 `_proprio[s]` | 读同一 npz 的 `force6_zeroed` 并拼接 |
| 5 | `a3_video_dataset.py:87-93` + `:55` | 要求**同目录**有 `frames.dat`，并要求 `dataset_stats.json` | `a3_fw_force/` 目前只有 `arrays.npz`(5 243 625 B) 与 `force_stats.json`，这两样都缺：帧文件要与 `a3_fw` 对齐（同一批视频帧，只能软链/复制，不能重编码），stats 要为 14 维 state 块重算 |
| 6 | `fastwam_processor.py:272-276` | `sample["proprio"] = data["state"]` 后 `assert` 维度等于 `proprio_output_dim` | 扩维后这条断言是唯一"静默不匹配"的守卫 |
| 7 | `models/wan22/fastwam.py:76` | `self.proprio_encoder = nn.Linear(proprio_dim, text_dim)` = `4096×8 + 4096` = **36864 参数**（正好是 `docs_task1` §7 里轻量态漏掉的那 2 个张量） | 换成 `4096×14`，按 §4 第 4 步"前 8 列继承、后 6 列零初始化" |
| 8 | `fastwam.py:282-303 / 415-425` | proprio 只作为**一个** cross-attn token 进 context（`_append_proprio_to_context`） | 不增 token 数，显存不变（§4 末尾的代价核算成立） |
| 9 | `trainer.py:86-88,296-299` | AdamW 参数集与 `requires_grad_(True)` 的地方 | 若另起 `force_encoder` 分支要在这里登记为可训练 |
| 10 | `light_state.py:35`（`lora_named_parameters`）+ `:39`（`adapter_state_dict`）+ `:44`（`save_adapter`） | 三处都由 `"lora_" in name` 这一个过滤驱动 | **必须先修**，见下 |
| 11 | `light_state.py:55-60`（`load_adapter(model, path, strict=True)`，`expected = {name for name,_ in lora_named_parameters(model)}`） | 装载侧用的是**同一个过滤** | 见下 |
| 12 | `bridge/_next/common.py:217`（`adapter_meta = light_state.load_adapter(model, str(adapter))`，默认 `strict=True`，全仓没有 `strict=False`） | 评估装载走的还是那个过滤 | 同上 |
| 13 | `bridge/infer_a3_worker.py:95-105`（`build_proprio(obs)`：`extract_left_eef_state` 6 维 + `opening=clip(obs["observation.state"][7],0,1)` → `concat`，再 `normalize_proprio`） | 推理侧 proprio 也是 8 维 | 在这里接同样 6 维，顺序与训练侧逐字一致 |
| 14 | `_next/run_matrix.py:227-230` | 评估循环**已经在读** `obs[safety_wrap.FORCE_KEY]` 的 `L_INNER(12)/L_OUTER(13)/wrist(0:3)` | 力在评估侧本来就可得，缺的只是"进模型输入"这一步 |
| 15 | `sim/env.py:375-386` | 18 维顺序 = `L_wrist_force, L_wrist_torque, R_wrist_force, R_wrist_torque, L_finger_inner, L_finger_outer, R_finger_inner, R_finger_outer, L_act_sum, R_act_sum`；`safety_wrap.py:26-28` `L_INNER=12, L_OUTER=13` | §7 的 6 维切片 `[0,1,2,12,13,16]` 与之一致（已核对） |

**第 10–12 条合起来是一个比报错更危险的坑**：保存侧与装载侧共用
`"lora_" in name` 这一个过滤，所以新加的力编码器（名字不含 `lora_`）既不会被存进
`adapter.pt`，也不会在评估装载时被要求存在——**两侧一致地忽略它，`strict=True` 也不会响**。
结果是"V1 训了 2000 步、评估时力列还是零初始化"，而且没有任何异常。
判据不是"名字含 `lora_`"，而是"优化器里有没有它"。所以 V1 的实施顺序是：
先接 `docs_light_state_param_sets_audit.md` §5 的 v3（按参数对象身份取
`optimizer.param_groups` 成员，LoRA 与非 LoRA 分两个子字典存，写前算
`projected_bytes` 并受预算守卫；`requires_grad` **不能**当保存条件——训练期它等于
成员集合是因为 `trainer.py:286-299` 先做了冻结，任何不重放这段冻结的代码都会把它
放大成 1637 张量 / 87.1 亿元素，即 17.4 GB），再动第 3/4/7/13 条；否则 V1 的读数不可信。
好消息是命名不用操心：`force_encoder` 只要进了 `trainer.py:86-88` 的参数集，
v3 就会按成员身份自动带上它，不需要维护白名单。
（`scripts/export_lora_merged.py:31-41` 这条合并路径**已经**带着 `proprio_encoder`，
说明缺的只是轻量态这一条。）

## 11 需要批准的四个动作（当前都未执行）

> **落点路径先说清**（`docs_task5_decision.md` §5.4 实测）：这些文件在 pod 上各有两份，
> `PYTHONPATH` 决定活的那份是 `/workspace/FasterWAM/src/fasterwam/…`，
> 推理侧是 `/workspace/fasterwam_a3_bridge/bridge/_next/run_matrix.py`（md5 `89d93c9a…`，
> 已和 `fasterwam-a3-cookie` 那份 `fcb12a96…` 不一致）。
> 下面每一条都指**活副本**；改错副本是静默无效，不是报错。

1. 改 `FasterWAM/src/fasterwam/light_state.py:35` 的保存集（属于训练侧仓库源码，
   不在"模型动作/Wrapper/Safety/仿真执行逻辑"禁改清单里，但会改变 checkpoint 内容）；
2. 新建 `configs/data/a3_2cam_force.yaml` + `prepare_a3_arrays`/`a3_video_dataset`/
   `build_proprio` 三处扩列（训练与推理同尺度，缺一处就废）；
3. 给 `a3_fw_force/` 补 `frames.dat`（与 `a3_fw` 同一批帧）与 14 维 state 的
   `dataset_stats.json`；
4. V0/V1 双臂各 2000 步的新训练跑（r5 已于 20:31 到 `max_steps=8000` 正常结束，
   这是"第二个训练"，按监督要求必须批）
   ——磁盘 22:05 实测 **3.407 GB**，两臂各 2000 步的轻量态约 0.77 GB/次保存
   （`a3_lora_r5` 4000 步 8 个检查点 1.4 GB 折算），这一项本身不紧；
   **紧的是带图语料**：73 条严格合格示范重采带图要 **9.88 GB**（`docs_task2` §6.6 实测帧数
   × 301,056 B/帧），缺口 6.47 GB，必须先定归档/删除策略；
   但 2–3 条 smoke（0.41–0.61 GB）现在就能放下。逐文件清单与依赖见 `docs_task5_decision.md` §5。


## 12 归一化两条规格已用实测钉死（2026-11-01，只读，未改任何代码）

§4 第 2 步原来写的是"除以语料上该维 p95 增量的绝对值"，§9 落地时用的是
`residual_std`。两者必须选一个，因为 V1 只有一个数能进 `dataset_stats.json`。
用 `a3_fw_force/arrays.npz` 现成的 `force18`（42681×18，100 回合）算了一遍：

| 通道 | 残差 p95 N | 残差 std N | 残差 \|max\| N | 除 std 后 p95 | 除 std 后 \|max\| | 除 p95 后 \|max\| |
| --- | --- | --- | --- | --- | --- | --- |
| L_wrist_fx | 0.0764 | 0.7339 | 3.5879 | 0.10 | 4.89 | **46.9** |
| L_wrist_fy | 0.6108 | 0.5891 | 3.5483 | 1.04 | 6.02 | 5.81 |
| L_wrist_fz | 0.2121 | 0.1823 | 1.3152 | 1.16 | 7.21 | 6.20 |
| L_pad_inner | 1.4450 | 0.6814 | 1.5398 | 2.12 | 2.26 | 1.07 |
| L_pad_outer | 1.4535 | 0.6855 | 1.5180 | 2.12 | 2.22 | 1.04 |
| L_gripper_act | 2.9156 | 1.3970 | 2.9741 | 2.09 | 2.13 | 1.02 |

**除数定为 `residual_std`（下表最后一列就是选它的理由）。** `L_wrist_fx` 这条通道
几乎全程是 0（p95 只有 0.0764 N），拿 p95 当除数会把每一个偶发的横向碰触放大到
10–47 倍、然后被 `[-5, 5]` 截断——信号被截断成常数，等于没有输入。
除 std 之后 pad 两条与夹爪驱动力都落在 ±2.3、腕部三轴常态 ±1 附近，
和已经归一化好的 8 维 proprio 同阶；`[-5, 5]` 只会削到腕部那几条 6–7 倍标准差的稀有尖峰。

**零载参考定为全局常数，不再逐回合算。** §4 写的是"该回合前 20 帧的中位数"，
训练时能算，推理时不行——评估回合刚开始还没有 20 帧可看中位数，
要么退化用"已有帧的中位数"（和训练不一致），要么等 20 帧（模型前 20 步没有力输入）。
实测逐回合前 20 帧中位数的离散度，结论是这条根本不用逐回合算：

```
通道            100 个回合的参考值    std       min          max
L_wrist_fx      median -0.0001      0.0083   -0.0288      0.0059
L_wrist_fy      median -10.7212     0.0211   -10.7833    -10.7129
L_wrist_fz      median -0.0282      0.0225   -0.0675      -0.0020
L_pad_inner     median  0.0000      0.0000    0.0000       0.0000
L_pad_outer     median  0.0000      0.0000    0.0000       0.0000
L_gripper_act   median -0.0186      0.0062   -0.0317      -0.0068
```

指垫两条在**全部 100 个回合**里前 20 帧中位数都精确等于 0.0000（未加载就是 0），
腕部 fy 是 -10.72 N 的重力基线、100 个回合最大相差 0.07 N。
把这个最大离散 0.07 N 换算到归一化尺度是 `0.07 / 0.5891 = 0.12`，
是 pad 通道常态幅度（2.1）的 6 %。**所以用一个全局向量减，训练和推理可以逐位一致**，
代价是每条通道最多 0.12 个归一化单位的偏置，而换来的是推理侧不需要等待、不需要缓冲。
`force_stats.json` 里 `zero_load_ref_median` 就是这个向量，`zero_load_ref_per_episode`
保留作这张表的出处。

落到实装上，第 11 节的第 3 项要写进 `dataset_stats.json` 的是
（**注意**：§13 已把运行时归一化定成"14 项 `state` normalizer 一次过"，
所以下面这块的用途是**推导那 6 项 normalizer 参数的出处与口径**，
不是第二个运行时归一化器——两边参数必须由同一份测量算出来，不能各填各的）：

```
force: {
  channels: [0, 1, 2, 12, 13, 16],          # 18 维里的切片，与 a3_dual_arm_sim/sim/env.py:375-386 对过
  names:    [L_wrist_fx, L_wrist_fy, L_wrist_fz, L_pad_inner, L_pad_outer, L_gripper_act],
  unit: N, hz: 20,
  zero_load_ref: [-0.0001, -10.7212, -0.0282, 0.0, 0.0, -0.0186],
  divisor:       [0.7339, 0.5891, 0.1823, 0.6814, 0.6855, 1.3970],   # residual_std
  clip: [-5.0, 5.0]
}
```

顺带一条要改的数据产物：`arrays.npz` 里现在的 `force6_zeroed` 是按**逐回合**参考减出来的，
换成全局参考需要重算一版。这一步是只读重算 + 新文件名（`arrays_globalref.npz`），
不覆盖现有 5 243 625 字节那份；算完要重新对一次 `episode_index` 与 `a3_fw/arrays.npz` 相等、
`state[7]` 与 `opening` 最大绝对差仍为 `0.0`。这条也归进 §11 第 3 项的批准范围。

## 13 三处扩列的精确落点（2026-11-01 只读实测，未改任何代码）

为了让批准之后的改动是一次到位的，先把行号与现状全部抄实。全部取自活副本
（§11 开头那条 `PYTHONPATH` 说明）。

**① `FasterWAM/src/fasterwam/light_state.py:34-35`** —— 保存集过滤器，全文只有一行：

```
def lora_named_parameters(model):
    return [(name, param) for name, param in model.named_parameters() if "lora_" in name]
```

`adapter_state_dict`（:38-39）和读取侧都走这一个函数，所以 `proprio_encoder.weight/bias`
（14×4096 = 36,864 参数）在存和读两侧**对称地**被丢掉——不报错，只是新训出来的输入权重
从来没进过 checkpoint。修法就是让这个过滤器额外接受 `proprio_encoder.` 前缀。

**② `FasterWAM/src/fasterwam/datasets/lerobot/a3_video_dataset.py`** —— `A3VideoDataset`（:37）：

```
:87   frames_path = data_dir / "frames.dat"
:96   self._proprio = np.ascontiguousarray(arr["proprio"], dtype=np.float32)   # [N, 8]
:99   # frames.dat is headerless raw bytes (not a .npy), so build the memmap explicitly.
:106  frames.dat size mismatch: {actual_bytes} != {expected_bytes}      ← 有硬校验
:122  self._state_key = shape_meta["state"][0]["key"]
:131  processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(pretrained_norm_stats)))
:153  "state": {self._state_key: torch.from_numpy(self._proprio[s].copy())}
```

宽度是从 `arrays.npz` 的 `proprio` 列数**自动带出来**的（:96 直接把整列搬进内存，
:153 原样塞进 batch），所以数据集这侧不需要改逻辑，需要改的是**数组本身**：
`prepare_a3_arrays.py`（唯一一份，在 `fasterwam-a3-cookie/finetune/`）要产出 `proprio [N, 14]`。
:106 那个 `frames.dat` 尺寸校验说明 `a3_fw_force/` 必须真的有一份帧流，
不能只靠 symlink 蒙过去——这正是 §11 第 3 项的内容。

**③ 推理侧 `fasterwam_a3_bridge/bridge/infer_a3_worker.py:95-100`** —— 现在是这样：

```
def build_proprio(obs):
    eef6 = extract_left_eef_state(obs)
    opening = float(np.clip(obs["observation.state"][7], 0.0, 1.0))
    fingers = a3_opening_to_panda_fingers(opening)
    return np.concatenate([eef6, fingers]).astype(np.float32)
```

（6 维 eef + 2 维指关节 = 8，和 `:96` 那条 `[N, 8]` 对得上。）
调用点在 `predict_action_chunk`：`proprio = normalize_proprio(build_proprio(obs), processor)`。

**这里有一个必须现在就定下来的设计点，否则会做出一个静默不一致的 V1。**
现有 `dataset_stats.json` 的键只有 `action / state / num_episodes / num_transition`，
归一化是**整条 proprio 一次过** `state` normalizer。V1 有两条可选写法：

- (a) `build_proprio` 返回 14 维**原始值**（8 维老 proprio + 6 维牛顿力），
  `state` normalizer 扩成 14 项；训练与推理共用同一个 `normalize_proprio`。
- (b) `build_proprio` 内部先用 §12 那个独立 `force` 块把 6 维归一好再拼，
  `state` normalizer 保持 8 项、另加 `force` 块。

(b) 的问题不是不能做，而是它把"力的归一化"放到了 normalizer 之外，
于是 §12 里 `zero_load_ref / divisor / clip` 这套数在训练侧（数组准备时算好）
和推理侧（`build_proprio` 里现算）**是两个独立的实现**，
任何一边改动都不会让另一边报错，只会让模型推理时看到的力分布和训练时不一样——
就是 §12 前半段花力气排除掉的那类训练/推理错配。
**所以定为 (a)**：`build_proprio` 只负责把 6 维原始力拼上（切片 `[0,1,2,12,13,16]`
减全局 `zero_load_ref`、除 `residual_std`、截断 `[-5, 5]` 这三步也归 normalizer 之外不做，
即连减法和除法都进 `state` 的 14 项统计里，靠 mean/std 或 min/max 表达），
§12 那张 `force` 块降级成**推导这 6 项 normalizer 参数的出处**，
运行时不再单独存在。这样力通道的尺度只有一处定义，改一处两边同时生效。

**验证顺序（批准之后，按监督要求）**：逐文件 contents API 文本 PUT →
逐文件 sha256 与本地对拍 → 每个改动文件单独 `py_compile` →
`--max-steps 2` 的 dry run 打印 `batch["state"]` 的最后一维宽度、
与 `dataset_stats.json` 的 `state` 项数对拍 → 才允许正式跑。
