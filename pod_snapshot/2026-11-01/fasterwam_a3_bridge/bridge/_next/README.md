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
  **r26/r27 用的是哪一份，已经从存下来的 npz 反推出来了**：r27 的
  `actions_7d` 里 y 的范围是 `[-0.105, +0.310]`，与 A3 stats 的 `[-0.115, 0.299]`
  吻合，而 LIBERO stats 的 x/y/z 是 ±0.9375 的对称范围。所以 r26/r27 是
  **A3 权重配 A3 stats**，配对没错，这一项可以从混淆清单里划掉。
- **夹爪的模式必须跟着权重走**（同一个 env var 不可能对两个模型都对），见 B0 的表。
  r26/r27 没有设 `FASTERWAM_A3_GRIPPER_MODE`，走的是默认 `libero01`；对一个用 ±1
  标签训出来的 checkpoint，这个分支会把号翻掉。那两次的夹爪由 wrapper 全程接管，
  所以没暴露出来 —— 但 `safety` 格会暴露。
- **固定场景 vs 随机场景是两套配置**，不覆盖同一份 baseline。`--position-noise 0`
  保持训练场景；扰动只在 Stage C 单独开。
- **`same_column` 专家是脚本专家**，不是人类遥操作。LoRA 学的是"在这个场景里模仿
  这个脚本"，所以 teacher forcing 问的是"拟合得像不像"，不是"会不会新技能"。

## 1. 文件

| 文件 | 作用 | 需要 mujoco/torch |
|---|---|---|
| `common.py` | 权重/stats 注册、manifest 指纹、`Policy.chunk()` | 是 |
| `env_factory.py` | 建环境；`position_noise_m` 真正接进 `task_config`；`batch_offset_m` 整批刚性平移；`scene_sanity()` 最低有效性门 | 是 |
| `safety_wrap.py` | 只做接口 + 安全，不读 `env`，力从 `obs["observation.force"][12:14]` 取 | 否 |
| `oracle_probe.py` | 统计控制层读了多久仿真真值（`cookie_positions` / `data`） | 否 |
| `run_matrix.py` | 闭环矩阵；`--controls C0,C1..C5` 按能力放权（C0=接口+安全，C5 必须等于 C0），`full/guard/force` 保留为历史对照 |
| `run_teacher_forcing.py` | 阶段 B1：专家驱动，模型只在专家轨迹上被查询；含不需要权重的动作链自检 | 否（`--check-only`）/ 是 |
| `analyze_runs.py` | 汇总 2×2：可控率、改写幅度、成功率、Oracle 读取；按「状态机阶段 × 通道」给放权比例，并做 `C5 vs C0` 配对与动作流 trace check | 否（只要 numpy） |
| `calibrate.py` | 阶段 C0：验证噪声真的生效、核对力布局、量出专家自身峰值力 | 是 |
| `scene_audit.py` | 阶段 C0b：加噪后饼干有没有重叠 / 倾斜 / 离开源盘；可选多种子重放专家 | 是 |
| `b0_probe.py` | 阶段 B0：夹爪伺服方向、adapter 符号、checkpoint 实际输出（A/B 不需要权重） | 是（C 需要） |
| `gripper_label_audit.py` | 用 `joint_cmd` 校验训练数据里的夹爪标签 | 否（只要 numpy） |
| `selftest.py` | 本地自检（安全层 no-op、计数器、分析通道） | 否 |

## 2. 放到 pod 上（直接从 GitHub 取）

**一条命令**（推荐，代码 + 依赖 + 权重一起恢复；细节见仓库 `deploy/README.md`）：

```bash
HF_TOKEN=... bash deploy/bootstrap_pod.sh --smoke
source /workspace/activate_fasterwam_a3.sh
```

下面是同一件事的手工版本，用来核对脚本做了什么：

```bash
# 1) 本仓库（只跑阶段 B/C 的话，只需要 bridge/_next/）
git clone -b a3-cookie-r27 https://github.com/wsadwsad6779/fasterwam-a3-cookie /workspace/fasterwam-a3-cookie
mkdir -p /workspace/fasterwam_a3_bridge
cp -r /workspace/fasterwam-a3-cookie/bridge /workspace/fasterwam_a3_bridge/bridge

# 2) 仿真代码 —— 版本必须对，见下面的警告
git clone https://github.com/Eter0109/a3_dual_arm_sim /workspace/a3_dual_arm_sim
git -C /workspace/a3_dual_arm_sim checkout 08ebe8cb370856051d0b32b07e691cc9b056716b

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

**2026-10-29 定的路线（以此为准）**：目标是"模型逐步完全接管"，所以不追求脚本专家或
wrapper 的成功率。

1. **B0** 权重 manifest + **夹爪方向实测**：确认每个 checkpoint 配的是自己的 stats 和夹爪
   模式（各 2 分钟），再跑 `_next/b0_probe.py` 证明"模型说闭合 → 指垫真的合上"，
   而不是只靠编码推断；
2. **E0 基线（C0）**：无扰动固定场景，`--controls C0`，看 LoRA 自己能完成到哪一步。
   **每个种子都保留完整轨迹**（npz 的 `pre_14d` / `post_14d` / `changed` / `phase` /
   `applied_canonical` / 指垫力 / 腕力），至少一个种子加 `--video`；0/10 也要留下，
   因为这一步的产物是"失败发生在哪个阶段"，不是分数。`analyze_runs.py` 现在同时按
   **状态机阶段 × 通道**给 verbatim：全程 0% 可能只是"该通道只在 hold 阶段被交还，
   而这一局没走到 hold"，不能当成实现错误；反过来 C1 的 xyz 全程 0% 也可能完全正常；
3. **按能力放权**：一级只交还一个能力，每级保留上一级作对照。**每一级单独跑、单独分析，
   不要 C1→C5 连着跑完**：

   | 阶段 | 交还给模型 | 仍由 wrapper 负责 |
   |---|---|---|
   | `C1` | 搬运 xyz | 接近、抓取目标、下降/闭合、释放、姿态 |
   | `C2` | + 接近 xyz | 抓取目标、下降/闭合、释放、姿态 |
   | `C3` | + 抓取目标选择 | 下降/闭合、释放、姿态 |
   | `C4` | + 下降与夹爪闭合 | 释放/回收、姿态 |
   | `C5` | + 释放/回收与姿态 | 无（只剩安全层） |

   `C0` = 只做接口转换 + 安全层，是整条阶梯的终点参照；`C5` 与 `C0` 的执行逻辑相同，
   所以**必须与 `C0` 逐步动作完全相同**（本地 stub 已验证 trace diff = 0，但那是 stub，
   真实 Faster-WAM 推理下要再确认一次，`analyze_runs.py` 会打印 trace check）。
   哪一级掉下来就停在这一级：先看该阶段的阶段 × 通道 verbatim 和失败帧，判断是
   "模型没有这个能力"还是"wrapper 还在挡"，需要时用这一级的轨迹做 LoRA 微调，
   **只重跑这一级**，再继续下一级。不要加新的任务级规则去补；
4. **训练与放权必须耦合**：没有权重更新时 `C5` 与 `C0` 是同一串动作，走完阶梯只等于
   把辅助一路撤掉，不会自动变好。所以每一级失败后的 LoRA 微调是阶梯的一部分。
   真正的成功判据是：**`C5` 比初始 `C0` 明显更好**（进箱饼干数 / 成功率），
   而不是"我们执行到了 C5"；
5. **整批平移**（`--batch-offset`）作为唯一的位置扰动，测"模型能不能用视觉找到饼干"，放在
   放权稳定之后。逐块 `--position-noise` 只作诊断，不混进同一项实验。

四档 `full / guard / force / safety` 保留为历史对照（`--controls full,guard,force`），
但不再作为研究主线：它们一次删掉多个保护机制，失败无法归因。

**不做**：完整 B1 teacher forcing 扫描、B2 的 2×2、B3，以及扰动场景下的专家成功率复测
（已有的一次性诊断数据留在下面 C0b/C1）。

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

**夹爪方向必须实测，不能只看编码。** `b0_probe.py` 分三层，前两层不需要权重：

```bash
python3 _next/b0_probe.py --out outputs/b0                 # A 伺服方向 + B adapter 符号
python3 _next/b0_probe.py --model raw  --out outputs/b0_raw # 再加 C：checkpoint 实际输出
python3 _next/b0_probe.py --model lora --out outputs/b0_lora
```

- A：不带模型，`a3[6]=+1` 必须把夹爪张开、`-1` 必须闭合（本地实测 1.000 → 0.000）；
- B：LIBERO `+1`（close）经 `action_adapter_v3` 必须变成 `a3[6]=-1`（本地实测成立）；
- C：跑一个 chunk，打印原始/适配后的夹爪值，并真的走 40 步看指垫怎么动。起始张开时
  模型第一帧就要闭合 → `sign_suspect=true`，先查 `gripper_mode` 再谈闭环。

A/B 任一项失败时脚本 `exit 1`，此时不要开闭环。

**夹爪那一列已经查清，结论会改变 B1 的跑法。** 两份 stats 的第 7 维范围不同：

| 模型 | stats 文件 | action[6] 范围 | 夹爪标签记法 | 该走哪个分支 |
|---|---|---|---|---|
| raw | LIBERO `libero/dataset_stats.json`（sha256 `c06dc67f…`，40,952 B） | **`[0.0, 1.0]`** | 0/1 | `libero01`（`a*2-1` 后 `*-1`） |
| lora | A3 `a3_fw/dataset_stats.json` | **`[-1.0, 1.0]`** | ±1（+1=夹紧，`prepare_a3_arrays.py` 写的） | `pm1`（`where(a>=0,+1,-1)`） |

所以**一个全局的 `FASTERWAM_A3_GRIPPER_MODE` 不可能同时对两个模型正确**：把 `pm1` 用在
raw 上，0/1 会被读成"全部 >=0 → 全部 +1"；把 `libero01` 用在 lora 上，±1 会先被
`a*2-1` 变成 `[-3,1]` 再被 `*-1` 翻号。模式改成跟 checkpoint 走，写在
`common.MODELS[*]["gripper_mode"]` 里，`Policy.chunk()` 每次调用前推进环境变量。
B2 每格的 summary 和 manifest 都有 `gripper_mode` 字段，事后能确认那一格用的是哪种；
要覆盖就用 `--gripper-mode pm1|libero01`。

lora 的 `pm1` 是**算术确定**的，不是猜：A3 标签 +1=夹紧是我们自己在
`prepare_a3_arrays.py` 里定的，adapter 做 `out6 = -sign(a6)`（A3 的 -1 = 闭合），
所以 LIBERO 空间的 +1 必须是"夹紧"；`pm1` 保留这个号，`libero01` 的 `*-1` 把它翻掉。
raw 那一格则多一个未知：LIBERO 数据集的 0/1 里 1 是"夹紧"还是"张开"没有本地证据，
上表的 `libero01` 只在"1=夹紧"时成立。B1 的 `grip.sign_agree_moving` 就是对 raw 这一格
的判决 —— 两种模式都接近 0%，说明两个分支都在翻号，那需要给 worker 加一个不取反的分支，
而不是在这张表里二选一。

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

A3 训练标签是 ±1，发布版基座是 0/1 —— 这不是猜的，两份 stats 的第 7 维范围直接写着
（见 B0 的表）。所以两个模型各自需要一个分支，`--gripper-modes` 默认仍把两种都在**同一批
查询**上各跑一遍，但目的从"哪一个是瞎的"变成了"验证算术推出来的默认值对不对"：

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
2. `obs["observation.force"][12:14]` 是否等于 `env._sensor("L_finger_inner/outer_touch_sensor")`。
   **只在 reset 时比对没有意义** —— 那一刻所有力通道都是 0。`calibrate.py` 现在在专家
   重放的整个过程里逐步比对，并在结尾打印"接触步数 / 最大差"。本地是 223 个接触步、
   最大差 0.00 N，所以安全层读 `obs` 而不是 `_sensor` 是等价的，不是近似；
3. 脚本专家自身的峰值夹持力是多少 —— 本地 1.46 N，而这个专家自己把 8 N 判成卡死，
   所以 `soft` 可以压到 3–6 N 而不碰合法抓取，`hard` 保持执行器上限（~40 N）。
   **不要**再用几何探针的 13.9 N 当依据。注意这个 1.46 N 只在**标称场景**成立：
   同一个专家在对缝失败时会读到 36–48 N（见 C1 的表），所以 40 N 的硬阈值会在
   "专家式失败"时就开火，这是设计意图。

同一支 `calibrate.py` 还会打印左腕六维力里 `|f|` 的分位数。夹爪触觉只能看到"抓得紧不紧"，
看不到"有没有顶进桌面"——顶桌只会在腕力上留痕。本地实测左腕 |f| 在只是举着胳膊时就有
10.7 N（主要在自重和重力补偿），所以腕力判据必须先减基线再定阈值。`run_matrix.py` 已经把
每步的 `wrist_force` 记进 npz，但**还没有**据此设第二条安全规则：等这批数据出来再定。

拿到数字后：

```bash
python3 _next/run_matrix.py --model lora --controls safety --seeds 11 21 \
    --soft-force <X> --hard-force <Y> --out outputs/stageB_calib
```

### C0b. 场景有效性（一次性诊断，已完成，不再扫）

`scene_audit.py` 读的是 reset 之后、50 步落位之后**真实**的饼干位置（`env.cookie_positions`），
不是采样值。5 个种子 × 5 档噪声：

| `position_noise_m` | 最小 AABB 间隙 | 穿透对数 | 最大穿透 | 最大倾角 | 倾斜 >5° | z 最大偏离 | 最大位移 | 有效性门 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.40 mm | 0 | 0.00 mm | 0.00° | 0 | 0.00 mm | 0.00 mm | 5/5 通过 |
| 0.0005 | -0.00 mm | 0 | 0.00 mm | 0.02° | 0 | 0.01 mm | 0.68 mm | 5/5 通过 |
| 0.001 | -0.89 mm | 1–5 | 0.89 mm | 1.40° | 0 | 0.60 mm | 1.36 mm | **5/5 不通过** |
| 0.002 | -2.78 mm | 6–13 | 2.78 mm | 7.63° | 2 | 6.07 mm | 5.04 mm | **5/5 不通过** |
| 0.003 | -4.33 mm | 19–24 | 4.33 mm | 12.52° | 15 | 15.35 mm | 9.34 mm | **5/5 不通过** |

机制是几何决定的：饼干 50 × 6.33 × 25 mm，列间距 50.4 mm、行间距 8.833 mm，所以名义
列间隙只有 **0.4 mm**、行间隙 **2.5 mm**。逐块独立 `U(-n, n)` 采样在 1 mm 时就必然把相邻
列推进彼此（实测残余穿透 0.9 mm），2 mm 起出现倾斜，3 mm 有一半以上饼干歪斜、z 相差 1.5 cm。
**所以逐块噪声不是"同一个任务换个位置"，而是换了一个不可能完成的任务。**

现在 `env_factory.scene_sanity()` 是唯一保留的检查（只看：穿透 > 0.5 mm、饼干离开源盘、
位姿非有限），`run_matrix.py` 在 reset 后调用它，不通过直接中止并报出种子与穿透量，
避免无效场景混进结果；`--allow-invalid-scene` 只用于明确想看退化场景的场合。

### C1. 位置扰动：改成整批平移

同一批种子上的**专家诊断记录**（一次性，不再复测；seed = 7001–7005，`max_steps=900`）：

| `position_noise_m` | 10/10 的种子 | `cookies_in_target` | 失败点 |
|---|---|---|---|
| 0 | 5/5 | 10 10 10 10 10 | — |
| 0.0005 | 2/3 | 10, 6, 10 | `DESCEND_TO_PLACE`：搬运中饼干滑出批次 |
| 0.001 | 2/5 | 7, 6, 10, 10, 5 | `VERIFY_RELEASE` / `DESCEND_TO_PLACE` / `LIFT` 各一次 |
| 0.002 | 0/3 | 0, 6, 5 | 48.27 N 撞停（41 步）/ 滑出 / 滑出 |
| 0.003 | 0/1 | 5 | 36.28 N 撞停 |

两件事同时成立，而且必须分开说：

- **0.5 mm 时场景仍然有效（零穿透），专家已经 1/3 失败**——失败原因是五块一起抓的批次
  余量本来就小，不是场景坏了；
- 1 mm 起场景本身开始穿透，2/3 mm 明确无效。

所以"扰动下的成功率"既不能当模型能力，也不能当任务难度：它主要测的是批次抓取的随机余量。

**采用的做法**：位置扰动一律用 `--batch-offset`，整批饼干**刚性平移**同一个向量
（`env_factory.apply_batch_offset`，从 `env.np_random` 抽样，可复现）。实测 3 mm / 5 mm
下相对排列、倾角、z 全部不变（与名义网格的偏差 0.000 mm），"饼干离开源盘"为 0；源盘在
x 方向还有 ±12 mm、y 方向 ±8 mm 的余量，5 mm 以内是安全的。这样扰动只改变"批次在桌上
的哪里"，才是"Faster-WAM 能否用视觉找到它"的问题。

```bash
# E0 基线（无扰动，只留接口 + 安全层；每个种子留全轨迹，至少一个录像）
python3 _next/run_matrix.py --model lora --controls C0 --seeds 11 21 22 --video \
    --out outputs/E0_C0
python3 _next/analyze_runs.py outputs/E0_C0

# 按能力放权：一次只跑一级，分析完再决定下一级跑不跑
python3 _next/run_matrix.py --model lora --controls C1 --seeds 11 21 22 --out outputs/E0_C1
python3 _next/analyze_runs.py outputs/E0_C1     # 阶段 × 通道 verbatim + 失败帧

# 该级失败 -> 用这一级的轨迹做 LoRA 微调 -> 只重跑这一级，再继续
# python3 _next/run_matrix.py --model lora --controls C2 ... --out outputs/E0_C2

# 最终验证：C0 和 C5 放进同一个目录，analyze 才会做配对比较。
# C5 必须与 C0 动作逐帧相同（trace check），且真实任务指标要明显高于初始 C0。
python3 _next/run_matrix.py --model lora --controls C0,C5 --seeds 11 21 22 \
    --out outputs/E0_final
python3 _next/analyze_runs.py outputs/E0_final

# 整批平移（先 3 mm，再 5 mm），等放权稳定之后再做
python3 _next/run_matrix.py --model lora --controls C0 --seeds 11 21 22 \
    --batch-offset 0.003 --out outputs/E1_offset3
python3 _next/analyze_runs.py outputs/E1_offset3
```

`run_matrix.py` 的 npz 每步都存了模型原始动作（`pre_14d`）、wrapper 输出（`post_14d`）、
仿真真正执行的 16 维 canonical 动作（`applied_canonical`）、是否被改动（`changed`）、
包装器阶段（`phase`）、指垫力和腕力，`analyze_runs.py` 再给出 xyz / 旋转 / 夹爪三个通道
各自的放权比例——这三个数就是"模型真正自己做了多少"。E0 即使 0/10 也要保留，因为要回答的
是"卡在哪一段"，不是"成功了几个"。


## 4. 判读规则（先写下来，避免事后解释）

- **LoRA 有用**：在 `safety` 格里 lora 的 `cookies_mean` / 到达率高于 raw，且
  `analyze_runs.py` 的配对差值方向一致（5 个 seed 里至少 4 个同向）。
- **LoRA 只在包装器下有用**：`full` 格有差异、`safety` 格没有 —— 说明收益来自
  包装器补位，不是策略本身。
- **包装器主导**：`full` 格的 `verbatim_%` 低（open/preclose/descend ≈0），
  `rewrite_mm` 与 `model_mm` 同量级或更大，同时 `oracle_reads` 每步都在涨。
  这时 `full` 的成功数不能用来说明模型能力。
- **全程 verbatim 不能单独判读**：先看 `analyze_runs.py` 的"阶段 × 通道"表。C1 的 xyz
  全程 0% 很正常（xyz 只在 hold 阶段交还，而这一局几乎都在接近段）；如果某个通道在它
  应该被交还的阶段里仍然是 0%，那才是实现问题。
- **C5 的判据是"比初始 C0 明显更好"**：`analyze_runs.py` 的 `C5 vs C0` 段落会打印
  `d_cookies` / `d_success` / 各通道 verbatim 差值和 trace check。没有权重更新时
  `C5≡C0`（trace diff 应为 0），这时"走完阶梯"不构成任何结论；只有真实指标提升才算
  模型接管了任务。
- **拟合好但闭环差**：teacher forcing 的 h0 误差小（< 1 mm），但 `safety` 格失败。
  这指向分布偏移 / 累积误差，而不是数据没学到 —— 下一步是数据增广，不是加 rank。
- **拟合差**：teacher forcing 的 h0 误差与 `expert_pos_scale_mm` 同量级（比如
  >30%），先查 stats 配对、动作签名、gripper 约定，再谈训练容量。
- **扰动下的差异不算能力**：任何 Stage C 的比较必须先贴出**同一噪声档下专家自己的
  数字**。专家在 1 mm 就失败（见表），所以那几档的两列差异只反映失败方式的不同；
  没有专家基线时，这一栏写"不可判读"，不写"模型更鲁棒"。

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
| 专家 + 1 mm 噪声 | 5 种子：2 个 10/10；7 / 6 / 5，失败在三个不同阶段 | 1 mm 是"余量"，不是硬阈值；且与 seed 7001 单跑完全复现（536 步） |
| 专家 + 2 mm 噪声 | 3 种子：0 个 10/10（0 / 6 / 5），一个 41 步撞停 48.3 N | 场景本身已穿透 2.8 mm，不能当评测条件 |
| 整批平移 3/5 mm | 相对排列偏差 0.000 mm，倾角 0°，z 不变，出盘 0 | `apply_batch_offset` 是刚性平移，机制可用 |
| 有效性门 | E0 0/5 不通过；1 mm 5/5 不通过；2 mm 5/5 不通过；平移 3/5 mm 0/5 不通过 | `scene_sanity()` 恰好拦住被破坏的场景，不误伤 E0 与平移 |
| 力布局（接触中） | 223 个接触步，`obs[12:14]` 与 `_sensor` 最大差 0.00 N | 安全层读 obs 与读传感器等价 |
| 分通道放权 | stub 冒烟：`safety` 三通道 100%，`full` xyz/rot 0%、grip 60% | `analyze_runs.py` 能分开报告三个通道 |
| 夹爪方向（B0-A） | `a3[6]=+1` → opening 1.000；`-1` → 0.000 | 伺服语义与 A3 记法一致，`+1` 开、`-1` 合 |
| adapter 符号（B0-B） | LIBERO `+1` → `a3[6]=-1.0`；`-1` → `+1.0` | `out6=-sign(a6)` 与伺服方向能串起来用 |
| 能力放权阶梯（C0–C5，stub） | C1 xyz 0%/grip 0%；C2、C3 xyz 100%/grip 0%；C4 grip 100%；C5 全 100% | 每级只动一个通道，grip 辅助在 C4 才交还 |
| C5 泄漏检查 | `C5` 走 wrapper（全部释放）再进安全层，与 `C0` 执行动作 60 步 max diff = 0.000e+00，oracle 读取 0，phase=`raw` | 全部释放后 wrapper 不再写任何通道；与安全层串联后仍逐帧等于 C0 |
| 阶段 × 通道统计 | 合成 npz：C1 的 `hold` 段 xyz 100%、`open/descend` 段 0%；stub 矩阵 C1 全程 xyz 0%（只走到 open/preclose/descend） | `analyze_runs.py` 能区分"这一级没交还"和"这一局没走到该阶段" |
| C5 vs C0 判据 | 合成数据：`d_cookies=+0.0`、trace diff 0 | 没训练时 C5≡C0 是预期，输出会明确写"这不等于模型提升" |

「框架自检 + 源码核对」：

- 安全层对 `[-1,1]` 内的正常动作是逐位 no-op；40 N 立即停；连续 3 步 30 N 后退；
  空夹爪低于箱沿时抬升。
- `OracleProbe` 只统计外部读取，不统计 `env.step` 内部访问。
- `analyze_runs.py` 的 verbatim / 改写幅度 / 分阶段拆分在合成 npz 上正确。
- `obs["observation.force"]`（`contracts.FORCE`，18 维）的 12/13 维在本地实测等于
  `env._sensor("L_finger_inner/outer_touch_sensor")`，且在 223 个**接触**步上误差
  仍为 0.00 N（reset 时两边都是 0，那次比对不构成证据）。
- `run_matrix.py` 的闭环在本地用 stub policy 跑通了整条链（`safety` 格 verbatim 100%、
  oracle 读取 0；`full` 格 verbatim 0%、`cookie_positions`/`data`/`_sensor` 计数随步数增长）。
  stub 常驻不动，所以这只证明管线通，不证明能力。相机在这台机器上关掉了（没有 EGL），
  pod 上 r26/r27 已经证明能渲。
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

本轮因为"在本地真的跑了一遍"而修掉的三处错误（都会让云端实验白跑或白读数据）：

| 问题 | 原写法 | 后果 |
|---|---|---|
| 力的键名 | `obs["force"]` | 真实键是 `observation.force`，`run_matrix.py` 第一步就 KeyError |
| 箱子内壁半宽 | `(0.061, 0.033)` | 真实内壁是 `(0.057, 0.029)`（0.063/0.035 减 6 mm 壁厚）；原值把 4 mm 外的点也当成"箱内"，抬升规则会在箱口外侧误触发 |
| 每格的 `replans` | `policy.calls` 跨格累加 | 一条命令跑 `--controls safety,full` 时，`full` 格的 `replans` 会把 `safety` 格的次数算进来（3+3=6）。`run_one` 现在每个 episode 开头清零，`chunk_idx` 也随之从 0 开始 |

未验证（必须云端跑出来才算数）：

- pod 上的 `a3_dual_arm_sim` 与本地 `08ebe8c` 是否同版本（决定上面那些数字能不能直接引用）。
- 动作链自检在**带相机渲染**的路径上是否仍然成立（本地只跑了 `render_cameras=False`；
  渲染只影响图像通道，不影响动力学）。
- `grip.sign_agree_moving`：对 lora 是验证 `pm1`（应当很高），对 raw 是判决分支
  （`libero01` 高 = 1 是"夹紧"；两个都低 = 两个分支都在翻号）。用现有 checkpoint
  就能验，不需要重训。
- 2×2 各格的实际耗时（按 r26 的 76–517 s 估，每格 2–5 分钟）。
- 模型侧的 `soft_force_n` 该定多少（专家峰值 1.46 N 只说明下限可以压到 3–6 N，
  真正的上限要看模型撞墙时的力分布）。


---

## DAgger 纠正示范：从失败状态接管（近失败轮）

r3 的 C0 失败都发生在模型自己的抓取—抬升链条上。这一轮不再用固定早期步数，而是按每条 C0 trace 的失败事件选接管点；只做两件事：把接管状态接回来、把接回来的完整动作链转成训练格式。专家只在离线数据生成里出现，在线仍是 Faster-WAM + Adapter + Safety。

### 失败机理（C0 trace 的结论）

- seed11：t≈90 起指垫第一次压到饼干（t=95 实测 6.3 N，t=100 起 21–26 N），饼干被挤出列。t=150 时源列已压实，专家在同一状态 DESCEND 直接被 insertion blocked 挡回（30.7 / 25.3 N）。t≥500 的“开爪前瞬间”全部不可恢复（506/530 以 7 块续持，闭合立刻 32.1 / 27.9 N）。
- seed21：t≈172 起同样压堆，t=190 时只有单侧指垫接触（5.75 N / 0 N），闭合链永远不成立（CLOSE 421 步超时）。t=226 时模型已夹住 6 块并抬升，随后提前开爪；用 resume-held 从“已持握”状态续走，没有重新接近，避免二次压堆。
- seed22：不是压堆，而是闭合位姿偏移；在失败前的 t=56/68/72 接管都能救回。

### 接管结果（本轮 10 个点）

| seed | take | 方式 | 结果 |
| --- | --- | --- | --- |
| 11 | 70 / 80 / 85 | 标准专家 | 三条都 10/10 入盒，峰值指垫力 1.4 / 1.4 / 1.6 N（t≤85 是压堆前最后一个干净窗口） |
| 11 | 150 / 160 / 170 | 标准专家 | 失败：DESCEND 插入受阻（30.7 / 25.3 N / 余同） |
| 11 | 506 / 530 | resume-held 7 块 | 失败：闭合即 32.1 / 27.9 N，结构性卡死 |
| 11 | 570 / 579 | 标准专家 | 失败：同上，源列已压实 |
| 21 | 226 | resume-held 6 块，从 LIFT 起 | 成功：两批共 11 块入盒，recovered=true |
| 21 | 190 | resume-held 5 块，从 CLOSE 起 | 失败：单侧接触，CLOSE 421 步超时 |
| 21 | 214 | 标准专家 | 失败：指垫 24–28 N |
| 22 | 56 / 68 / 72 | 标准专家 | 三条都 10/10 入盒 |

不可恢复点的共同特征：指垫已经在饼干堆里饱和（20–32 N），换专家也只能把堆继续压紧。可恢复点的共同特征：指垫还没接触饼干堆，或夹持链仍然完整（226 的 6 块全部在 9–18 mm 内）。

限制要说清楚：“提前开爪”类只有 seed21 226 一个可恢复状态。trace 里 274 / 406 / 484 / 514 的开爪事件中，饼干与指垫已经拉开 65–107 mm，属于已经松开而不是“夹着开爪”，救回来也不是同一类纠正。

### 纠正数据集

7 条成功 take 转成数据集（multi-root：outputs/dagger_near + outputs/dagger_resume）：

- 7 episodes、3,048 transitions、frames.dat 917,618,688 B（=3048×224×448×3）；占 base 42,681 的 7.1%。
- 每条结束时目标盒内 10 块（21:226 为 11 块）。

check_a3_dagger_dataset.py 用训练配置实例化通过：越出 base min/max：action 0.15%、state 0.88%；越出 q01/q99：action 1.45%、state 10.56%；夹爪 +1（闭合）占比 0.588，每条 episode 开合 17–40 次。

动作一致性：位置 label/专家逐步余弦 0.78–0.85、量级比 ≈0.49（0.025/0.05，整个数据集共有）；旋转 ≈0.1 mrad/step，方向噪声主导；夹爪方向一致率 0.93–0.95。

### r4 方案（尚未启动）

- 起点：r3 merged，LoRA 继续微调；超参与 r3 一致（batch 6、lr 1e-4、max_steps 3000、resume 自 r3 merged）。
- 数据：base 42,681 + 纠正 3,048。采样权重 base:new = 1:4，每 epoch 有效样本 42,681 + 12,192 = 54,873，纠正占 22.2%。7 条 episode 不重采样会被 100 条 base 淹没；权重再高则容易背下这 7 个场景。
- 显存：r3 稳态 28.5 GB / 48 GiB（batch 6），r4 同配置不变，余量约 19 GB。
- 磁盘：r4 需要 merged 11.1 G + checkpoints/weights ≈ 11 G ≈ 22 G；当前只剩 8.5 G，至少腾 14 G。候选是 r1 merged（11 G）与 r2 merged（11 G），两者都已被 r3 取代，C0 trace 与指标保留。清理需要确认后再做。
- 评价：r4 与 r3 用同 seed（11/21/22）、同 C0 配置、同 wrapper 版本各跑一遍；只看饼干入盒数与饼干自身运动，另报 Wrapper / Safety 干预比例。
