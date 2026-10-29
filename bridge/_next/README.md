# fasterwam_next — 阶段 B/C 实验脚本

这套脚本只做一件事：把"Faster-WAM 到底学到了什么"变成可复现的数字。它不改
`bridge/` 里已有的任何文件，`run_wrapped.py` 的 npz 格式也保持原样。

## 0. 先说结论性的前提

- **2×2 的全称**：`{raw, lora}` 权重 × `{safety, full}` 控制层。四格用同一批
  seed、同一次 reset，因而是配对的。
- **归一化必须跟着权重走**：LoRA 训练时用的是
  `/workspace/a3_data/a3_fw/dataset_stats.json`（`a3_2cam.yaml` 里的
  `pretrained_norm_stats`），发布版基座用的是
  `checkpoints/fasterwam_release/libero/dataset_stats.json`。`infer_a3_worker.py`
  默认只读后者；把 A3 stats 灌给 raw、或把 LIBERO stats 灌给 lora，都会让动作尺度
  静默错掉。`common.py` 里每个模型自带 stats，跑之前先看 manifest 打印。
   **这也是目前最值得先排查的一个混淆项**：r26/r27 究竟用的哪个 stats，交接日志
   里没有记录。
- **固定场景 vs 随机场景是两套配置**，不覆盖同一份 baseline。`--position-noise 0`
  保持训练场景；扰动只在 Stage C 单独开。
- **`same_column` 专家是脚本专家**，不是人类遥操作。LoRA 学的是"在这个场景里模仿
  这个脚本"，所以 teacher forcing 问的是"拟合得像不像"，不是"会不会新技能"。

## 1. 文件

| 文件 | 作用 | 需要 mujoco/torch |
|---|---|---|
| `common.py` | 权重/stats 注册、manifest 指纹、`Policy.chunk()` | 是 |
| `env_factory.py` | 建环境；把 `position_noise_m` 真正接进 `task_config`；`reset()` 显式给 `randomize_cookies` | 是 |
| `safety_wrap.py` | 只做接口 + 安全，不读 `env`，力从 `obs["observation.force"][12:14]` 取 | 否 |
| `oracle_probe.py` | 统计控制层读了多久仿真真值（`cookie_positions` / `data`） | 否 |
| `run_matrix.py` | 阶段 B2 的 2×2 闭环 | 是 |
| `run_teacher_forcing.py` | 阶段 B1：专家驱动，模型只在专家轨迹上被查询；含不需要权重的动作链自检 | 否（`--check-only`）/ 是 |
| `analyze_runs.py` | 汇总 2×2：可控率、改写幅度、成功率、Oracle 读取 | 否（只要 numpy） |
| `calibrate.py` | 阶段 C0：验证噪声真的生效、核对力布局、量出专家自身峰值力 | 是 |
| `gripper_label_audit.py` | 用 `joint_cmd` 校验训练数据里的夹爪标签 | 否（只要 numpy） |
| `selftest.py` | 本地自检（安全层 no-op、计数器、分析通道） | 否 |

## 2. 放到 pod 上（直接从 GitHub 取）

```bash
# 1) 本仓库（只跑阶段 B/C 的话，只需要 bridge/_next/）
git clone -b a3-cookie-r27 https://github.com/wsadwsad6779/fasterwam-a3-cookie /workspace/fasterwam-a3-cookie
mkdir -p /workspace/fasterwam_a3_bridge
cp -r /workspace/fasterwam-a3-cookie/bridge /workspace/fasterwam_a3_bridge/bridge

# 2) 仿真代码 —— 版本必须对，见下面的警告
git clone https://github.com/Eter0109/a3_dual_arm_sim /workspace/a3_dual_arm_sim
git -C /workspace/a3_dual_arm_sim checkout 08ebe8c   # 本地验证过的那一版

# 3) 自检
cd /workspace/fasterwam_a3_bridge/bridge && source /opt/venv/bin/activate
export PYTHONPATH=/workspace/fasterwam_a3_bridge/bridge/_next:$PYTHONPATH
python3 -m py_compile _next/*.py && python3 _next/selftest.py
```

`_next/` 是新目录，不动 `bridge/` 里任何已有脚本；`common.py` 自己会把 `bridge/`、
`a3_dual_arm_sim/src`、`FasterWAM` 加进 `sys.path`，所以 `PYTHONPATH` 只是为了让
`import common` 能找到。

### ⚠️ `a3_dual_arm_sim` 的版本：`ff552c4` 是错的

仓库根 README 和 `docs/README_REBUILD_FASTERWAM_A3.md` 里写的
`Eter0109/a3_dual_arm_sim @ ff552c4`（2026-09-28 14:47）是**重构前的扁平布局**
（`src/a3_dual_arm_sim/cookie_transfer.py`、`batch_expert.py`），而 `bridge/` 里的
import 是

```python
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv
from a3_dual_arm_sim.controllers.same_column_batch_expert import A3SameColumnBatchExpert
```

`tasks/`、`controllers/` 两个包是从 `8088b31`（`refactor: drop legacy shims and prune
historical experiments`，2026-09-28 21:44）才开始存在的。按 `ff552c4` 克隆再跑，
第一行 import 就 `ModuleNotFoundError`。

| 版本 | 布局 | 能不能用 `bridge/_next` |
|---|---|---|
| `ff552c4`（2026-09-28 14:47） | 扁平 | ❌ ImportError |
| `8088b31`（2026-09-28 21:44）起 | `tasks/` + `controllers/` | ✅ 下限 |
| **`08ebe8c`（2026-10-21）**，在 `origin/main` 上 | 同上 | ✅ **本地验证用的就是它** |

选 `08ebe8c` 而不是 `origin/main` 的 HEAD，因为本文 §6 那些本地数字是在它上面跑出来的；
`main` 之后又有十几次改动（`batch_expert`、采集器、基准），没在本工作里验证过。要用新版本
就先把 `selftest.py` + `calibrate.py` 重跑一遍再信数字。

**pod 上那份现在是什么版本，仍然没有记录。** 到 pod 上第一件事：

```bash
git -C /workspace/a3_dual_arm_sim rev-parse --short HEAD
```

不是 `08ebe8c` 就在 pod 上把 §6 的本地数字当成"待复验"，不要直接引用。

## 3. 执行顺序

### A. 自检（1 分钟）

```bash
python3 _next/selftest.py
```

### B0. 权重的 manifest（每个模型 1–2 分钟，含 11 GB 权重加载）

```bash
python3 _next/run_matrix.py --model raw  --controls safety --seeds 999 --max-steps 2 \
    --out outputs/_smoke_raw
python3 _next/run_matrix.py --model lora --controls safety --seeds 999 --max-steps 2 \
    --out outputs/_smoke_lora
```

看 `outputs/_smoke_*/manifest_*.json` 里的 `normalizer.action.min/max`：

- raw 的 x/y/z 范围应当接近 ±1（LIBERO 归一化）；
- lora 的应当是 A3 实测范围（x≈[-0.064, 0.087]、y≈[-0.115, 0.299]）。

如果两个文件一样，说明 stats 被写死成同一份，先停下来查 `FASTERWAM_A3_STATS`。

注意 `--seeds 999 --max-steps 2` 只是加载检查，不是实验。

### B1. Teacher forcing：拟合能力（每个模型约 10–20 分钟）

先跑不需要权重的动作链自检（几秒，且决定了后面所有数字对不对）：

```bash
python3 _next/run_teacher_forcing.py --check-only --seeds 7001
```

它把专家发出的 14 维 A3 动作按 adapter 的**逆变换**换回 LIBERO 单位，再和仿真真正
走出来的位移、转角、夹爪开合比较。语义来自源码而不是猜测：

| 通道 | 约定 | 来源 |
|---|---|---|
| 平移 | 世界系，1 单位 = 0.025 m | `sim/actions.py` 直接加在 `site_xpos` 上 |
| 旋转 | **局部 EEF 系**，1 单位 = 0.10 rad | `mju_quatIntegrate` 右乘，`V3` 用 `R.T @` 逆回去 |
| 夹爪 | `2*opening-1` 的**设定值**，+1 = 全开，每步最多走 0.08 | `env.py` 的 `max_gripper_step` 夹取 |

`pos_err_mm_p95` 里含着 IK 跟踪误差和关节限速，所以不要求 0；但它必须是**毫米级**。
如果 p95 到了几厘米，说明单位和坐标系里有一个是错的，后面的 MAE 全部无意义。

然后跑模型：

```bash
python3 _next/run_teacher_forcing.py --model raw  --seeds 7001 7002 7003 --query-stride 20 --out outputs/tf_raw
python3 _next/run_teacher_forcing.py --model lora --seeds 7001 7002 7003 --query-stride 20 --out outputs/tf_lora
```

目标不是专家的动作，而是**用训练公式从轨迹重建的标签**（`0.05 m`、`0.5 rad`、夹爪
±1 + 1e-4 死区）。`chunk7` 本来就在这个空间里，所以比较不需要经过 adapter，也就
不会把 adapter 的误差算进模型头上。

`--seeds 7001+` 是训练（`seed_start: 6001`，6001–6100）之外的种子。**但**固定场景下
专家轨迹与 seed 无关，seed 只改变外观随机化，所以这一组衡量的是"换外观还拟合不拟合
得上"，不是轨迹泛化。真正的轨迹泛化在 Stage C，用 `--position-noise 0.003` 再跑一遍。

输出里看四件事：

- `horizons.h0.pos_mae_mm` 与 `expert_pos_scale_mm` 的比值 —— 误差相对运动量的大小；
- `horizons.h31` 是否明显劣化 —— 32 步内漂移；
- `by_phase.APPROACH / DESCEND / CLOSE` 的分段误差 —— 抓空通常出在 descend/close；
- `grip.*` —— 见下。

**夹爪符号是这一轮必须钉死的一件事。** `predict_action_chunk` 在返回前会把夹爪通道
改写一遍，规则取决于环境变量 `FASTERWAM_A3_GRIPPER_MODE`：

- `pm1`：`where(a>=0, +1, -1)`，保留"标签 ±1、+1=夹紧"；
- `libero01`（默认）：先 `a*2-1` 再取反，隐含假设模型反归一化后是"0/1、1=夹紧"。

A3 训练标签是 ±1。发布版基座的反归一化输出是 0/1 还是 ±1，代码里没有记录、也无法从
本地文件推出来 —— 如果是 ±1，默认分支就会把方向反过来，而 adapter 又做了 `-sign`，
一共两次取反。所以脚本默认把两种模式在**同一批查询**上各跑一遍，让数据来定：

| 字段 | 含义 |
|---|---|
| `grip.sign_agree_moving` | 专家真正在开/合时，模型符号与标签符号一致的比例 |
| `grip.hold_bands.*.model_pos_rate` | 专家夹住不动时，按测量开合量分三段统计 `chunk7[6] > 0` 的比例 |
| `a3_h0.grip_sign_agree` | 经过 adapter 之后，与专家 14 维动作第 6 维同号的比例 |

判读：`sign_agree_moving` 高的那个模式是对的；两个都不高，说明模型没学会夹爪时序，
而不是符号问题；两个都接近 100%，说明 `-sign` 恰好抵消了转换，属于隐患而不是故障。
`model_pos_rate` 故意不写"open"，因为哪个符号代表张开随模式而变。

### B2. 2×2 闭环（每格 2–5 分钟/seed，两个模型分进程跑）

```bash
python3 _next/run_matrix.py --model raw  --controls safety,full --seeds 11 21 22 23 24 \
    --out outputs/stageB
python3 _next/run_matrix.py --model lora --controls safety,full --seeds 11 21 22 23 24 \
    --out outputs/stageB
python3 _next/analyze_runs.py outputs/stageB
```

必须**一个进程一个模型**：两个 5.56B 权重同时驻留会把 48 GB 吃穿。

`analyze_runs.py` 会先跑一条自检：`safety` 格的 verbatim 应当 ≈100%（只有安全规则
触发时才下降）。如果 safety 格显著低于 100%，说明安全层在改动作，那一格就不是
"低干预"了，结果不能当成裸策略能力。

### C0. 噪声与安全阈值标定（约 10 分钟）

```bash
python3 _next/calibrate.py --config /workspace/a3_dual_arm_sim/configs/cookie_batch.yaml
```

本地已经跑过一轮（数字见 §6），云端这一步主要是在 pod 的仿真版本上**复核**，并且回答
三个必须先回答的问题：

1. `--position-noise 0.003` 到底有没有换掉场景（打印每个饼干的 xy 位移，z 单列 ——
   z 是落位堆叠的结果，不是采样出来的）；
2. `obs["observation.force"][12:14]` 是否等于 `env._sensor("L_finger_inner/outer_touch_sensor")`；
3. 脚本专家自身的峰值夹持力是多少 —— 本地 1.46 N，而这个专家自己把 8 N 判成卡死，
   所以 `soft` 可以压到 3–6 N 而不碰合法抓取，`hard` 保持执行器上限（~40 N）。
   **不要**再用几何探针的 13.9 N 当依据。

同一支 `calibrate.py` 还会打印左腕六维力里 `|f|` 的分位数。夹爪触觉只能看到"抓得紧不紧"，
看不到"有没有顶进桌面"——顶桌只会在腕力上留痕。本地实测左腕 |f| 在只是举着胳膊时就有
10.7 N（主要在自重和重力补偿），所以腕力判据必须先减基线再定阈值。`run_matrix.py` 已经把
每步的 `wrist_force` 记进 npz，但**还没有**据此设第二条安全规则：等这批数据出来再定。

拿到数字后：

```bash
python3 _next/run_matrix.py --model lora --controls safety --seeds 11 21 \
    --soft-force <X> --hard-force <Y> --out outputs/stageB_calib
```

### C1. 扰动场景（2.5 mm 缝，必须逐级加）

```bash
for n in 0.001 0.002 0.003; do
  python3 _next/run_matrix.py --model lora --controls safety --seeds 11 21 22 \
      --position-noise $n --out outputs/stageC_n$n
done
```

`--position-noise 0.003` **已经超过**排缝宽度：本地实测每个饼干 xy 位移 p95 是
4.1–5.2 mm（采样界是 ±3 mm，超出部分来自落位时的相互挤压），已经比 2.5 mm 的排缝宽。
再往上加只会看到全失败，分不出"模型不行"和"场景不可能"。要"刚好卡在缝边"，起点是
`0.001`（p95≈1.4 mm）。

## 4. 判读规则（先写下来，避免事后解释）

- **LoRA 有用**：在 `safety` 格里 lora 的 `cookies_mean` / 到达率高于 raw，且
  `analyze_runs.py` 的配对差值方向一致（5 个 seed 里至少 4 个同向）。
- **LoRA 只在包装器下有用**：`full` 格有差异、`safety` 格没有 —— 说明收益来自
  包装器补位，不是策略本身。
- **包装器主导**：`full` 格的 `verbatim_%` 低（open/preclose/descend ≈0），
  `rewrite_mm` 与 `model_mm` 同量级或更大，同时 `oracle_reads` 每步都在涨。
  这时 `full` 的成功数不能用来说明模型能力。
- **拟合好但闭环差**：teacher forcing 的 h0 误差小（< 1 mm），但 `safety` 格失败。
  这指向分布偏移 / 累积误差，而不是数据没学到 —— 下一步是数据增广，不是加 rank。
- **拟合差**：teacher forcing 的 h0 误差与 `expert_pos_scale_mm` 同量级（比如
  >30%），先查 stats 配对、动作签名、gripper 约定，再谈训练容量。

## 5. 已经确认的数据问题（本地可复现）

`gripper_label_audit.py` 对 `/workspace/a3_data/a3_fw/arrays.npz`（也就是训练用的那份）
的结论：

| 指标 | 数值 |
|---|---|
| 总帧数 | 42,581 |
| 夹爪指令在动的帧 | 24.56%（其中标签与指令方向一致 94.97%） |
| 夹爪指令静止且处于夹持状态（`joint_cmd < 0.5`） | 30,535 帧 |
| 其中被标成 **open** | **33.86%** |
| 夹爪完全张开且静止（`joint_cmd > 0.9`） | 1,586 帧 |
| 其中被标成 **close** | 45.27% |
| 与存储指令矛盾的标签总计 | **27.2%** |

机制：`prepare_a3_arrays.py` 用**测量到的**夹爪开合量的帧间差来定标签，静止时沿用上一个
标签。指垫一接触饼干，测量值会回弹（0.507 → 0.441），这个回弹被当成"张开"，于是整个
夹持保持段都被标成 open。`joint_cmd`（存下来的夹爪指令）在同一段里是恒定的 0.4525。

后果：模型被训练成"夹住的同时张开"，而 adapter 又把夹爪二值化成 `-sign`，于是执行层
只能输出 ±1。这解释了为什么夹爪必须由 wrapper 接管，也解释了为什么离线 L1 看起来"还行"——
模型忠实复现的是一个错的标签。

这不一定是 LoRA 失败的唯一原因，但在它修好之前，任何"模型能不能自己抓"的结论都不干净。
修法（先不改，等 B1 确认）：把 `act[:,6]` 改成 `clip(-Δjoint_cmd / 0.08, -1, 1)`
（LIBERO 符号，+1 夹紧），连续值覆盖 75.44% 的静止帧（≈0），饱和只占 4.23%；同时把
adapter 的 `-sign(action[6])` 改成 `-action[6]`，并保持 `binarize_gripper=false`。
**注意**：现有 checkpoint 是在旧标签上训的，所以阶段 B 的 2×2 必须继续用旧 adapter，
否则两列不在同一基准上。

重训前还有一个更便宜的检查：B1 的 `grip.sign_agree_moving`。如果正确模式下这个数已经很高，
说明模型学到的是**命令**的方向、只是标签在静止段被写错；如果很低，说明模型学到的就是
标签本身（含错误），那么重训收益最大。两种情况对应不同的下一步，先量再决定。

## 6. 本地已验证 / 未验证

本地用的是 `E:/Innovative_practice/WAM/a3_dual_arm_sim`，commit
**`08ebe8c`（2026-10-21，在 `origin/main` 上）**，纯 CPU，装在
`.venv_mujoco/` 里，与 `bridge/` 无关。**下面这些数字只在 `08ebe8c` 上成立**；pod 上那份
的 commit 到 pod 后用 `git -C /workspace/a3_dual_arm_sim rev-parse --short HEAD` 打出来，
不同就要重跑 §C0 复核（见 §2 的警告）。

```bash
cd C:/Users/ASUS/Documents/Qoder/2026-10-27/675ee83d
A3_CONFIG_PATH="E:/Innovative_practice/WAM/a3_dual_arm_sim/configs/cookie_batch.yaml" \
PYTHONPATH="E:/Innovative_practice/WAM/a3_dual_arm_sim/src;fasterwam_next" \
  .venv_mujoco/Scripts/python.exe fasterwam_next/run_teacher_forcing.py \
  --check-only --seeds 7001 --out outputs/local_check
```

实测结果（seed 7001，固定场景）：

| 项目 | 数值 | 意义 |
|---|---|---|
| 专家重放 | 473 步，`cookies_in_target=10`，`failed=False` | `--max-steps 600` 够用；teacher forcing 的观测在专家流形上 |
| 平移链自检 | p50 0.1 mm，p95 1.1 mm | `0.025 m`/世界系约定正确 |
| 旋转链自检 | p50 0.3 mrad，p95 3.1 mrad | 局部 EEF 系约定正确 |
| 夹爪方向自检 | 20 帧中 100% 一致 | `2*opening-1` 的设定值语义正确 |
| 专家指垫力 | p95 1.45 N，max 1.46 N | 远低于自身判定卡死的 8 N |
| 左腕 \|f\| | p50 10.7 N，max 14.3 N | 主要是自重，做触地判据必须先减基线 |
| 噪声 3 mm | xy p95 4.1–5.2 mm，z 受落位影响可达 16 mm | 旋钮真的生效 |

「框架自检 + 源码核对」：

- 安全层对 `[-1,1]` 内的正常动作是逐位 no-op；40 N 立即停；连续 3 步 30 N 后退；
  空夹爪低于箱沿时抬升。
- `OracleProbe` 只统计外部读取，不统计 `env.step` 内部访问。
- `analyze_runs.py` 的 verbatim / 改写幅度 / 分阶段拆分在合成 npz 上正确。
- `obs["observation.force"]`（`contracts.FORCE`）的 12/13 维在本地实测等于
  `env._sensor("L_finger_inner/outer_touch_sensor")`。
- 训练数据 `seed_start: 6001`、`policy: same_column`、`stored_action_mode:
  joint_position`，`position_noise_m` 默认 0 且 collector 用
  `randomize_cookies: False` —— 所以训练集是固定场景。
- 动作链自检的数学：`build_targets()` 能从轨迹精确重建位置/旋转标签（往返误差
  <1e-12），`chain_audit()` 能把专家的 14 维动作按 adapter 的逆变换还原回轨迹
  （合成数据上 p50 误差为 0、夹爪方向 100% 一致），`align()` 在轨迹末尾正确丢弃
  越界 horizon。见 `selftest.py` 的 `teacher forcing math` 一节。
- 专家返回的就是 14 维 `cartesian_delta`（`validate_action` 会硬性拒绝其它形状），
  所以老脚本里的 `action[:14]` 是无害的，不需要按 16 维关节动作去改写 —— 这一条
  曾经被记成 bug，核对源码后撤回。

本轮因为"在本地真的跑了一遍"而修掉的两处错误（都会让云端实验白跑）：

| 问题 | 原写法 | 后果 |
|---|---|---|
| 力的键名 | `obs["force"]` | 真实键是 `observation.force`，`run_matrix.py` 第一步就 KeyError |
| 箱子内壁半宽 | `(0.061, 0.033)` | 真实内壁是 `(0.057, 0.029)`（0.063/0.035 减 6 mm 壁厚）；原值把 4 mm 外的点也当成"箱内"，抬升规则会在箱口外侧误触发 |

未验证（必须云端跑出来才算数）：

- 两个 checkpoint 的 stats 到底是不是各自配对（交接日志里没有记录）。
- pod 上的 `a3_dual_arm_sim` 与本地 `08ebe8c` 是否同版本（决定上面那些数字能不能直接引用）。
- 动作链自检在**带相机渲染**的路径上是否仍然成立（本地只跑了 `render_cameras=False`；
  渲染只影响图像通道，不影响动力学）。
- `grip.sign_agree_moving` 在两个 `FASTERWAM_A3_GRIPPER_MODE` 下分别是多少 ——
  这既验证夹爪符号，也是"模型学会了在夹持时张开"的直接证据（用现有 checkpoint
  就能验，不需要重训）。
- 2×2 各格的实际耗时（按 r26 的 76–517 s 估，每格 2–5 分钟）。
- 模型侧的 `soft_force_n` 该定多少（专家峰值 1.46 N 只说明下限可以压到 3–6 N，
  真正的上限要看模型撞墙时的力分布）。
- 噪声场景下专家本身还能不能完成（Stage C 的前置条件）。
