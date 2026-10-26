# Faster-WAM × LIBERO × A3 饼干夹取：阶段总结与 AI 复现交接 README

**更新日期：2026-10-23｜语言：中文｜当前优先任务：修复 A3 夹爪越界并完成可复核的闭环测试**

> **阅读顺序与证据边界：** 本文件是对 `docs/legacy/README_FasterWAM_A3_NATIVE_PATHS_2026-10-21.md` 的**增量核对、纠错与整理**，不是把旧实验抹掉。本文优先于旧文档；旧文档仅保留历史详情，遇到冲突以本文件新证据为准。当前 AI 能访问的是用户上传的旧 README、A3 交接 ZIP、桥接 Python 源文件、`outputs.zip`、新一轮 3 个 CSV、组会材料及公开 Faster-WAM **固定 commit** 的源码；**不能直接访问用户 E 盘的真实 Git 仓库，也不能访问已退出的 Runpod 工作区**。因此未核验的 E 盘文件、脚本、仓库修改、原始 JSON/日志及 checkpoint 实际存在性一律标为待核实。所有附送脚本不在新 Pod 上端到端执行过。
>
> **立刻需要记住：有两轮不同的 LIBERO 实验，绝不能混同。** ① 较早的“**40 个不同任务 × 每任务 1 次**”smoke test，40/40（旧文档 §9）；② **2026-10-22 新完成的“10 个固定任务 × 4 个评测 seed”**，也是 40/40（本文 §3）。最新的 seed 实验不是 40 个不同任务，也不是完整 LIBERO 的正式基准。A3 **尚无成功饼干抓取**。

## 0. 新 AI 五分钟交接：先读这里

| 事项 | 截至 2026-10-23 的准确状态 |
|---|---|
| 项目目的 | 使用 Faster-WAM 官方发布的 LIBERO 模型作闭环评测，并尝试**不重新训练**、零样本迁移到自建 A3 双臂 MuJoCo 饼干夹取场景；后续根据效果判断是否需要数据采集/微调 |
| 已完成的最新实验 | **10 任务 × seed 11/23/37/53 × 每组合 1 episode = 40 条**，CSV 40/40 成功；四组 seed 对应的 40 份保存 YAML 配置曾在 Runpod 逐一核验通过 |
| 多 seed 观察 | 起始画面基本一致，部分轨迹、手腕相机画面、动作幅度略有差异；**没有直接比较动作张量，不能把差异全部归因于模型 seed** |
| seed 因果解释 | 官方固定版本 `eval_libero_single.py` 会将 `cfg.seed` 传给 `infer_action`，也会传给 LIBERO `get_libero_env`，后者执行 `env.seed(seed)`；固定 `initial_states[0]` 仍不完全控制所有环境随机性 |
| A3 适配进展 | 图像/本体状态适配、7D→14D 动作转换、历史单次真实模型推理、短闭环**前三个动作**有执行；接下来左夹爪转换后 `+1.015625` 超出 `[-1,1]`，触发异常，**没成功夹取** |
| A3 最合理的下一步 | **先在本地检查文件与离线动作后处理**；从固定 commit 官方 `binarize_gripper` 逻辑核对并补齐手写 worker 缺失的 `np.sign`，通过离线检查后才租 GPU 做短闭环验证；尚未证明修复有效 |
| 当前本地项目位置（历史实测） | Windows `E:\Innovative_practice\WAM\` / WSL `/mnt/e/Innovative_practice/WAM/` 下 `FasterWAM/`、`a3_dual_arm_sim/`、`fasterwam_a3_bridge/`；本轮**未重新读 E 盘** |
| 云端情况 | 相关 Runpod 已退出/历史实例存在数据丢失风险；**不要假定 `/workspace/FasterWAM` 仍存在**，也不要复用旧 SSH 地址 |
| **最省钱的下一步** | 本地 WSL `git`、`find`、Python 版本和 A3 桥接代码只读核验（§9.1），**不重跑已完成 40 次 LIBERO** |

**交互偏好：** 用户是计算机本科生；命令必须注明“【本地 Windows CMD】”“【本地 WSL Ubuntu】”或“【Runpod】”，尽量一次只给 **1～2 条**，等用户贴运行结果后再继续；优先解释为什么执行、什么输出算成功，避免术语堆砌。GPU 是付费资源，必须先完成本地核验与备份，再启动云端。

**证据标签：** `[上传文件实检]` 此会话确实读取/运行检查；`[历史终端核验]` 用户曾贴终端结果但现已无法连回原机；`[固定 commit 源码核验]` 以固定 URL 检查源码；`[旧交接档案]` 只能证明旧交接记录的内容；`[新编脚本静态测试]` 通过离线语法/模拟文件检查但未跑 GPU；`[待核实]` 无法直接确认。

## 1. 项目目标和边界：究竟在做什么

Faster-WAM（World Action Model）是利用视觉、语言、机器人状态及未来信息生成机器人操作动作的模型。本项目先复用**官方已训练权重**，在 LIBERO/robosuite/MuJoCo 验证模型与推理链路；再通过观测与动作接口转换，将同一权重尝试用于与官方机器人/摄像机/动作语义不一致的 **A3 双臂饼干夹取**。

**已做：** 官方预训练权重部署；两轮不同的 LIBERO 单 episode 评测；A3 相机、状态、动作适配和部分闭环。**未做：** 从零训练 Faster-WAM、微调 checkpoint、采集可用于训练的 A3 示教集、实现动态 SparseMoT/新模型、A3 完整成功抓取、标准 50-trial 全 LIBERO 统计、LIBERO-Plus/OOD 实验。不要将“部署推理”称为“训练成功”。

参考： [Faster-WAM 官方仓库](https://github.com/hustvl/FasterWAM) · [官方权重](https://huggingface.co/hustvl/FasterWAM) · [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) · [A3 仿真仓库](https://github.com/Eter0109/a3_dual_arm_sim)。

## 2. 核心环境、版本、模型与校验码

### 2.1 最新多 seed 实验的真实终端输出 `[历史终端核验]`

| 项目 | 成功实验使用的版本/内容 |
|---|---|
| Faster-WAM commit | `83667817df0d4f823f39d90700e61ea2f432ac45` |
| LIBERO commit | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| checkpoint | `checkpoints/fasterwam_release/libero/step_021700.pt` |
| checkpoint SHA-256 | `01be4efca565d1890fd8a7dc1507b01c00c8cc63c16ae252287492ba9ac7550d` |
| 归一化统计 | `checkpoints/fasterwam_release/libero/dataset_stats.json` |
| stats SHA-256 | `c06dc67fcf819b04e2720c90e3d02e0bc70283af9bca3cc77a7a162432fd0910` |
| **最新 LIBERO venv** | **Python 3.10.22**、torch **2.7.1+cu128**、MuJoCo **3.3.2** |
| GPU / driver | **NVIDIA RTX 6000 Ada Generation**（48GB）/ **595.91.07** |
| 云端工程根 | `/workspace/FasterWAM`；LIBERO Python `.venvs/libero/bin/python` |
| 必要系统参数 | `MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`、`HF_HUB_ENABLE_HF_TRANSFER=0` |
| 必要 Hydra 覆盖 | `model.redirect_common_files=false`（避免历史 DiffSynth 401） |

**重要：** 旧 README 中的 **RTX A6000**、**Python 3.10.18** 属于更早的 LIBERO/A3 运行实例，不是上表**2026-10-22 多 seed**实验的 GPU/解释器。旧环境曾成功，但新实验审计应使用本表。`reproduction_info.txt` 是用户在 Runpod 亲自运行并贴出内容的元数据；实际文件目前是否已同步 E 盘**待核实**。SHA 用于确认模型字节一致，不是在线下载凭据。

### 2.2 需要准备的大文件

- Faster-WAM 官方仓库 `libero/step_021700.pt` 和 `libero/dataset_stats.json`：放到 `checkpoints/fasterwam_release/libero/`，**注意 HF 仓库路径从 `libero/` 开始**。
- Wan 2.2：`Wan2.2_VAE.pth`（历史约 2.82GB）、`models_t5_umt5-xxl-enc-bf16.pth`（历史约 11.4GB），放到 `checkpoints/Wan-AI/Wan2.2-TI2V-5B/`。
- Wan 2.1 的 `google/umt5-xxl/*` tokenizer 文件，放到 `checkpoints/Wan-AI/Wan2.1-T2V-1.3B/`。
- `checkpoint` ≠ Wan 模型全套；有 `step_021700.pt` 不说明推理所需 VAE/T5 已齐。
- A3 项目要求 Python **≥3.12**，而 Faster-WAM/LIBERO 使用独立的 **3.10**；不要在一个 venv 强行混装。历史用 NPZ/标志文件在同一容器的两个独立进程间通信。

## 3. **最新阶段：10 个固定 LIBERO 任务 × 4 seed（2026-10-22）**

### 3.1 设计、任务清单及 seed 含义

使用 checkpoint、10 个任务列表、各任务的 `trial_idx=0`（第一次预设初始状态）、单卡、运行参数不变；只改变 `cfg.seed` 为 **11、23、37、53**。这四个整数**没有官方特殊含义**，只是易记录、可重复的不同固定值。每个 seed × 10 任务只跑一次，合计 **40 episodes**，不是每个任务 40 次，也不是 40 个不同任务。

固定任务文件 `.runtime/eval_tests/selected_10_tasks.txt` 的准确内容：

```text
libero_spatial,0
libero_spatial,1
libero_object,0
libero_object,1
libero_goal,0
libero_goal,1
libero_10,0
libero_10,1
libero_10,2
libero_10,3
```

- `EVALUATION.num_trials=1`，因此每个任务只用 LIBERO 保存的**第 0 条初始状态**；不引入新的初始状态扰动。
- `cfg.seed` 是本次评测随机种子；配置里的 `MULTIRUN.task_sample_seed=42` 仅用于随机任务抽样，与显式指定上述任务文件是不同概念，**不需更改为 11/23/37/53**。
- **固定初始状态不等于固定全部环境随机性。** 固定版本源码先 `env.reset(); env.set_init_state(initial_state)`，同时将 `cfg.seed` 传入 `get_libero_env` 并调用 `env.seed(seed)`；动作生成也接收同一个 `cfg.seed`。所以当前实验考察的是**评测 seed 联合影响下的闭环表现**，不是独立隔离模型 seed 的实验。参见 §5。

### 3.2 六个历史结果批次（SSH 中断后手工续跑）

| 结果目录 `.runtime/eval_tests/results/` 下 | seed | 原始 JSON 文件数 | 状态 |
|---|---:|---:|---|
| `seed11_pilot_retry2/` | 11 | 1 | 成功 pilot |
| `seed11_remaining9/` | 11 | 6 | SSH 断线后留有 6 个 |
| `seed11_missing3/` | 11 | 3 | 手工仅补缺少 3 个 |
| `seed23/` | 23 | 10 | 全部 |
| `seed37/` | 37 | 10 | 全部 |
| `seed53/` | 53 | 10 | 全部 |
| **合计** | | **40** | 六份 `failed_tasks.txt` 均为空；`failed_tasks` 表示**程序级失败**而非任务不成功 |

历史运行脚本位置 `.runtime/eval_tests/run_batch.sh`（接口为 `bash .runtime/eval_tests/run_batch.sh SEED TASK_FILE BATCH_NAME`），但**该脚本的确切原文没有包含在当前上传文件中**。另附 `scripts/run_selected_seed_batch.sh` 是根据已验证官方命令**新重构**的复现辅助脚本，**不是冒充用户原始 `run_batch.sh`**；需恢复到目标固定 commit 后在 GPU 上验证。

### 3.3 原始 CSV 核对结果 `[上传文件实检]`

`docs/evidence/all_40_episode_results.csv` 含 **40 行**有效条目；以 `(seed,task_suite,task_id)` 为键 **40 个唯一组合**，每行 `success=1` 且 `total_episodes=1`。`seed_success_rates.csv` 与 `suite_success_rates.csv` 与之匹配。

| seed | 成功 / episode | 成功率 | 平均 `duration_seconds`（任务记录耗时） |
|---|---:|---:|---:|
| 11 | 10/10 | 100% | 46.78 s |
| 23 | 10/10 | 100% | 46.67 s |
| 37 | 10/10 | 100% | 44.78 s |
| 53 | 10/10 | 100% | 44.79 s |
| **总计** | **40/40** | **100%** | **45.76 s/episode** |

| 任务集合 | 成功 / episode | 成功率 |
|---|---:|---:|
| `libero_spatial` | 8/8 | 100% |
| `libero_object` | 8/8 | 100% |
| `libero_goal` | 8/8 | 100% |
| `libero_10` | 16/16 | 100% |

上述 `duration_seconds` 是 JSON 的整个任务记录耗时，不是纯动作推理时延，也不能据此推断 seed 加快了动作。相同任务在四个 seed 下视频**定性**观察到动作幅度、手腕视角、执行过程存在轻微区别，但没有定量动作差异测量。这里 100% 只表示**本次测试选定的 10 个任务 / 固定 trial0**；不得声称完整 LIBERO benchmark 100% 或 OOD 泛化 100%。

### 3.4 seed 修改与证据链

- `[历史终端核验]` 用户运行专门校验脚本逐项读取每个 JSON 对应 `eval_config_gpu0_task*.yaml`；输出 `seed11...` 三批分别 1、6、3 份，另外三个 seed 各 10 份；最后：`ALL 40 SEED CONFIGURATIONS VERIFIED`。
- `[上传文件实检]` 三份统计 CSV 的全部 40 条记录与上述 seed、任务集合完全一致。
- **不能从 CSV 独立重验 YAML**，因为 JSON/YAML 原始归档没有在本会话完整提供；只可记录此前用户亲自执行且提供的校验输出。在归档回到本地后使用 `scripts/verify_seed_configs.py` 再查一次。

### 3.5 输出、备份与视频

```
.runtime/eval_tests/
├── selected_10_tasks.txt          # 固定任务文件；历史云端存在，当前本地需核验
├── remaining_9_tasks.txt         # seed11 恢复辅助文件
├── seed11_missing3.txt           # seed11 恢复辅助文件
├── run_batch.sh                  # 历史运行脚本：本会话没有字节副本
├── export_results.py            # 历史导出脚本：当前上传仅 CSV
├── reproduction_info.txt        # 含版本、两个 SHA-256；是否已下载 E 盘待查
├── all_40_episode_results.csv   # 已重新上传并检查；交接包 docs/evidence 有副本
├── seed_success_rates.csv
├── suite_success_rates.csv
└── results/
    ├── seed11_pilot_retry2/
    ├── seed11_remaining9/
    ├── seed11_missing3/
    ├── seed23/
    ├── seed37/
    └── seed53/
```

每批目录下有 `libero_spatial/`、`libero_object/`、`libero_goal/`、`libero_10/` 中对应任务的 `gpu0_taskN_results.json`、`eval_config_gpu0_taskN.yaml`、`videos/*.mp4`、日志与 `failed_tasks.txt`；一个 seed11 任务横跨三个批次，**不要简单以目录数来算 seed 数**。

历史输出压缩包名：`/workspace/fasterwam_libero_40_episodes.tar.gz` 或后来建议生成的 `..._final.tar.gz`；**当前会话没有两者的原始压缩包，且未直接检验它们现在是否已落地 E 盘**。近期组会 ZIP 包含三个 CSV 和用于展示的三个合成视频，不等同于 40 个原始 episode 的可复现原始档案。下载前核对实际存在的文件，别推断旧 Pod 仍在线。

## 4. **更早阶段：40 个不同 LIBERO 任务 × 1 次 smoke test**

旧交接文档确有另一轮已完成评测：`libero_spatial`、`libero_object`、`libero_goal`、`libero_10` 四套件**各 10 个任务**，共 **40 个不同任务，每任务 1 次，40/40 成功**；历史累计任务时间 `19m21s`，云端早期机型 **RTX A6000**，原归档为 `experiment_backups/fasterwam_libero_40tasks_smoke.tar.gz`（约 8.5MB）。该数字既不表示最新多 seed 实验，也不表示标准 50-trial 基准。历史归档下载 / `tar -tzf` 曾成功，但目前具体本地目录仍待核查。

旧 README §8–§11 还保留单任务 smoke test、40 个不同任务的完整旧命令与故障诊断；可参阅 `docs/legacy/README_FasterWAM_A3_NATIVE_PATHS_2026-10-21.md`。研究统计时务必将两轮实验分为**不同的表格**，不合并成所谓“80 个独立初始状态”或“80 个不同任务”。

## 5. seed 机制：为什么是相同初始画面却有轻微轨迹变化

在 **Faster-WAM 固定 commit** `83667817…` 的公开源码中已核对以下事实：

1. [`configs/sim_libero.yaml`](https://github.com/hustvl/FasterWAM/blob/83667817df0d4f823f39d90700e61ea2f432ac45/configs/sim_libero.yaml) 定义 `EVALUATION.num_trials: 50`、`replan_steps: 10`、`binarize_gripper: true`、`MULTIRUN.task_sample_seed: 42`；实验以 Hydra override 把 trial 数改为 1。
2. [`eval_libero_single.py`](https://github.com/hustvl/FasterWAM/blob/83667817df0d4f823f39d90700e61ea2f432ac45/experiments/libero/eval_libero_single.py) 使用 `task_suite.get_task_init_states(task_id)`；若 `num_trials=1`，每次都访问下标 0；执行 `env.reset()` 与 `env.set_init_state(initial_state)`；并将 `cfg.seed` 提供给 `infer_action`。
3. 同一文件 `run_single_task()` 将 `cfg.seed` 传入 `get_libero_env(...)`；[`libero_utils.py`](https://github.com/hustvl/FasterWAM/blob/83667817df0d4f823f39d90700e61ea2f432ac45/experiments/libero/libero_utils.py) 执行 `env.seed(seed)`，注释明确提示**固定初始状态时 seed 似乎仍能影响物体位置**。
4. 推理动作后处理还可能开启 `binarize_gripper`，相关逻辑对 A3 修复有重要意义。

**当前能断言：** 四个评测 seed 被配置、传给模型与环境；**任务结果对本次变化稳定**，视频有细微差异。**不能断言：** 不同 seed 导致的差异完全来自模型，或者在相同环境、相同输入下输出一定不同。下一步因果对照应统一 `env.seed(0)` 与初始 state，**仅**令 `infer_action(seed=11/23/37/53)` 变化；保存**同一首帧 + 本体状态 + prompt**的动作张量，并比较 `max_abs_diff`、L2 距离、后续轨迹和成功率。先提交这一隔离改动的 Git diff，避免混入原基线。

## 6. A3 饼干夹取对接：已完成、尚未完成与代码实证

### 6.1 三个不同工程和环境（绝不能混用）

| 工程 | Windows | 本地 WSL | 用途 |
|---|---|---|---|
| Faster-WAM | `E:\Innovative_practice\WAM\FasterWAM` | `/mnt/e/Innovative_practice/WAM/FasterWAM` | 预训练模型/官方 LIBERO 推理，GPU 环境 Python **3.10** |
| A3 仿真 | `E:\Innovative_practice\WAM\a3_dual_arm_sim` | `/mnt/e/Innovative_practice/WAM/a3_dual_arm_sim` | 双臂机械臂/夹爪/饼干环境，Python **≥3.12** |
| A3 桥接 | `E:\Innovative_practice\WAM\fasterwam_a3_bridge` | `/mnt/e/Innovative_practice/WAM/fasterwam_a3_bridge` | 观测适配、动作适配、测试脚本、输出视频 |

这些路径在 **2026-10-21 曾被用户本地 WSL `ls` 验证**，但**新一轮未直接检查**。历史 `fasterwam_a3_bridge.zip`（约 197MB）是用户自己的旧归档，与额外的 `FasterWAM_A3_Handoff_NATIVE_2026-10-21.zip`（约 1.6MB，包含辅助脚本）是不同文件。解压额外交接包**不是运行现有 A3 文件的必需步骤**，更不能用归档覆盖 E 盘当前源码。

### 6.2 数据接口和实际 A3 闭环路径 `[旧交接档案 + 上传 Python 源码实检]`

| 数据流 | 当前适配办法 | 事实边界 |
|---|---|---|
| A3 图像 | 读取前方相机 + 左手腕 RGB，两路尺寸处理成 `[224,224]`，横向拼接，`uint8[224,448,3]` / 模型侧 `float32[1,3,224,448]`，归一化到 `[-1,1]` | 有离线 NPZ 和图；`vision_adapter.py` 与 `vision_adapter_official.py` **实现不同**，需核实实际运行用的是哪一个 |
| A3 机器人状态 | `observation.state` 为 16D（左夹爪索引 7、右夹爪索引 15），`eef_pose` 为双臂 14D（每臂 xyz+quat/wxyz） | 左臂 xyz+rotvec+两指近似长度构成模型 8D proprio；`0.04*opening, -0.04*opening` 是**近似**，未标定成真 Panda 状态 |
| Faster-WAM 输出 | 官方 LIBERO 预训练 checkpoint 输出 **32×7** 动作 chunk（历史一次实例）；模型动作需反归一化、夹爪变换 | 上传的原 worker 中缺少官方可选二值化（如下） |
| A3 执行 | `action_adapter_v3.py` 将 **LIBERO 单臂 7D** 转为 **A3 双臂 cartesian_delta 14D**，右臂不移动，右夹爪保持状态 | A3 需要所有通道在 `[-1,1]`；坐标、旋转、速度/时间语义尚未全面独立验证 |
| 模型与仿真通信 | 历史 A3 Python3.12 ↔ 模型 Python3.10：同一容器两个进程，NPZ 文件 + `request.flag`/`done.flag`/`stop.flag` | **本地 WSL 与 Runpod 是两台不同机器**；跨机器无法直接共享本地绝对路径，需另设文件传输/网络通道 |

`action_adapter_v3.py` 的**当时实际逻辑**：线性 3D 动作 `x_A3 = 2*x_LIBERO`（假设 LIBERO 一单位 0.05m，A3 一单位 0.025m）；旋转 `world_rotvec = 0.5 * model_rotvec`、`local_rotvec = R_eef.T @ world_rotvec`、`A3_rotation=local_rotvec/0.1`（假设 A3 一单位 0.1rad）；左夹爪 `g_A3 = -g_LIBERO`；最后检查 14 个通道均不超过 `1+1e-6`。**这些转换是工程假设和历史源码事实，尚不是经全部姿态、接触情形严格标定成功的物理等价映射。**

旧交接 ZIP 的可靠内容清单：`bridge_snapshot/` 28 个非重复原始 `.py`、`recovered_historical/infer_a3_worker.py`、`recovered_historical/run_a3_closed_loop.py`、`experimental/infer_a3_worker_official_gripper.py`、`experimental/action_adapter_v3_gripper_guard.py`、`scripts/preflight.py` 等，以及 `assets/outputs_original.zip`。本次在 `docs/legacy/A3_Handoff_NATIVE_2026-10-21.zip` 原样留一份。**其中历史 worker 由旧聊天恢复，不是用户此刻 E 盘工作区的真实快照**，须先做 hash/diff 再替换。

### 6.3 已经真实达到哪一步

- `[旧交接档案]` A3 的 FK/IK、关节跟踪、精度、夹爪、相机预处理等有多份源脚本和输出视频 `outputs.zip`（已读取 ZIP 文件条目）；多数是**人为给定动作控制测试**，不是 Faster-WAM 已经执行的策略动作。
- `[旧交接档案]` 模型曾生成一次 `(32,7)` 的 action chunk；云端常驻 worker 曾报告 `Faster-WAM worker ready` 和约 `2.48s` 的一次请求（**不是官方论文纯时延指标**）。
- `[旧交接档案]` 真正的 A3 短闭环 `Replan 1` 成功执行前 **3 个动作**，之后 V3 动作适配报错；**没有成功的完整饼干抓取**，**没有可信的多 seed A3 抓取成功率**。

### 6.4 夹爪越界：必须保留的原始诊断

```text
=== Replan 1 ===
Predicted chunk: 32 actions, executing first 5
step=001 chunk_step=1/5 reward=0.000 left_gripper_cmd=-0.992
step=002 chunk_step=2/5 reward=0.000 left_gripper_cmd=-0.992
step=003 chunk_step=3/5 reward=0.000 left_gripper_cmd=-1.000
ValueError: Converted action exceeds A3 limits:
[-0.025635  0.087891 -0.021973  0.205539 -0.023557  0.060376
  1.015625   0.        0.        0.        0.        0.        0.        1.      ]
```

**错误位置是索引 6（第 7 个元素，左夹爪），不是第 4 个旋转通道 `0.205539`**。模型经过历史后处理后的夹爪约为 `-1.015625`，V3 翻转变成 A3 `+1.015625`，超出 `[-1,1]`。旋转空间分量均在范围内，不要盲目缩放所有动作，也不要改变 `action_horizon` 当作修复夹爪越界。

在**固定 commit 官方源**中：`configs/sim_libero.yaml` 默认 `EVALUATION.binarize_gripper: true`，`eval_libero_single.py` 在动作反归一化与符号处理后执行：

```python
if bool(cfg.EVALUATION.get("binarize_gripper", False)):
    action[..., -1] = np.sign(action[..., -1])
```

而上传交接包 `recovered_historical/infer_a3_worker.py` 确认只做 `_denormalize_action`、`action[..., -1] * 2 - 1` 和 `invert_gripper_action`，**缺少上述可选官方二值化**。因此合理的**候选修复**是在实际 worker 中确认目标 cfg 后补齐与官方一致的 `np.sign`，让夹爪离散到 `-1/0/+1`，再交 V3；同时记录修复前/后的动作差异。`experimental/infer_a3_worker_official_gripper.py` 可供比对，但**其余代码可能不匹配当前 checkout**；修复需离线测试、代码审查、短闭环实测。**此修复尚未被真实 GPU A3 测试证明成功，不得写成“已修复”。**

备选防御手段：仅夹爪通道允许小幅越界时警告并限制到 `[-1,1]`，**不是优先方案**；禁止无记录地对全部 14D 输出做统一 clip。先保留失败日志和源代码，再修复。

## 7. 已遇到的主要问题：症状 → 原因 → 处理 → 结果

| 症状 | 已知原因或最强线索 | 已实际采用 / 候选处理 | 当前状态 |
|---|---|---|---|
| Hydra 报 `configs/data` 不在 | rsync 排除 `data/` 时误排除了嵌套 `configs/data/` | rsync 改为只排除根目录 `/data/`，或直接固定 commit git clone | `[历史修复]` |
| `third_party/LIBERO` Git checkout 异常 | 上传目录冒充真正 Git 克隆 | 使用正式 clone/pinned commit，而非拷贝 `.git` 缺失目录 | `[历史修复]` |
| MuJoCo `libEGL.so.1` 缺失 | headless GPU 镜像系统 EGL 库缺失 | 安装合适的 `libegl1`，设置 `MUJOCO_GL=egl`/`PYOPENGL_PLATFORM=egl` | `[历史修复]` |
| `DiffSynth-Studio` HTTP 401 | `model.redirect_common_files=true` 转向受限资源 | **`model.redirect_common_files=false`** 使用官方 Wan `.pth` | `[历史实测成功]` |
| `hf_transfer` ModuleNotFound | 环境变量误启用 HF transfer | `HF_HUB_ENABLE_HF_TRANSFER=0` | `[历史实测成功]` |
| 手动导入 `fasterwam` 失败 | 不在 Python `sys.path` | `PYTHONPATH="$PWD/src:$PWD" ...`；官方脚本原本设路径 | `[历史修复]` |
| `scp` 卡在 0% | 旧环境默认协议不兼容/传输停滞 | 使用 `scp -O`；新 Pod 要实际核对是否仍需要 | `[旧交接档案]` |
| Seed11 中间断线 | SSH 中断且已完成 6/9 | 不相信“运行命令退出”即结果全丢；先数 JSON，再只补缺 3，后来 40 文件完整 | `[历史终端核验]` |
| Runpod Stop 后文件丢失 | 没有持久化 Volume，远程临时文件系统 | **Stop 前打包并拉回本地**；新 Pod 重建环境 | `[历史经历]` |
| `MULTIRUN.task_sample_seed=42` 未变化 | 这是任务抽样 seed 而非评测 seed | 固定显式任务文件，逐个核对每份实际 YAML 中 `seed` | `[历史核验通过]` |
| A3 左夹爪 `+1.015625` | worker 可能漏掉官方二值化；最终通道越界已确证 | 首选离线核对后补 `np.sign`；再小范围闭环 | `[未修复；候选方案]` |
| A3 测试误认高精度为控制器成功 | FK 数值一致不等于末端物理跟踪误差 | 另测真实 tracking/IK/执行误差和 seed 分布 | `[待验证]` |

## 8. 新 Runpod **从零复现 LIBERO**：按阶段执行（不要随意直接开跑）

**适用范围：** 以下基于旧成功命令、固定 commit 的官方入口和当前 CSV 重建，单独的用户 `run_batch.sh` 原文当前缺失。**不能保证云端镜像、网络/模型下载、依赖在将来完全无变化，必须先分阶段检查再付费运行。** 如果只想继续 A3 工作，**不建议为了验证 README 而重跑 40 episodes**。

### 8.1 先准备本地 / 检查可用资源（免费）

【本地 Windows CMD】`wsl -d Ubuntu`。随后【本地 WSL Ubuntu】：

```bash
cd /mnt/e/Innovative_practice/WAM
for p in FasterWAM a3_dual_arm_sim fasterwam_a3_bridge; do test -d "$p" && echo "OK $p" || echo "MISSING $p"; done
for p in FasterWAM a3_dual_arm_sim; do echo "=== $p ==="; git -C "$p" rev-parse HEAD 2>&1; git -C "$p" status --short 2>&1 | head -40; done
```

```bash
cd /mnt/e/Innovative_practice/WAM
find fasterwam_a3_bridge -maxdepth 1 -type f -name '*.py' -printf '%f\n' | sort
find fasterwam_a3_bridge/.venv -maxdepth 3 \( -name python -o -name python.exe \) -print 2>/dev/null
python3 --version
```

**成功标准：** 真实三目录存在、查看 Git SHA/工作区改动、A3 有可执行 Python 3.12；如不一致先修复路径及依赖，**禁止直接覆盖用户改动**。本会话无法代替用户操作 E 盘。

### 8.2 【Runpod】开新 Pod、固定版本、安装环境

- 选择历史验证过的 48GB 显存等级 GPU，确认磁盘足够放 **11GB checkpoint + 约14GB Wan + 两套 venv + 仿真与缓存**，旧记录 120GB 容器盘曾够用；这不是最低硬件保证。
- 如 Pod 无持久化 Volume，**Stop 可能清空 `/workspace`**。下载大文件前先确定备份策略；SSH 退出不停止 GPU 计费。
- 先核查：`nvidia-smi`、`df -h /workspace`、`git --version`，再进行下列命令。

```bash
# 【Runpod】仅当 /workspace/FasterWAM 尚不存在时执行，不覆盖旧工作区
cd /workspace
git clone https://github.com/hustvl/FasterWAM.git
cd FasterWAM
git checkout 83667817df0d4f823f39d90700e61ea2f432ac45
git rev-parse HEAD
```

```bash
# 【Runpod】依当前仓库安装说明先保证 uv 已安装，再执行官方脚本
cd /workspace/FasterWAM
bash scripts/setup/install_core.sh
bash scripts/setup/install_libero.sh
.venvs/libero/bin/python -c 'import sys,torch,mujoco;print(sys.version);print(torch.__version__,mujoco.__version__)'
git -C third_party/LIBERO rev-parse HEAD
```

若 `uv: command not found`，按 [uv 官方安装指引](https://docs.astral.sh/uv/) 安装；若 EGL 无法导入，安装提供 `libEGL.so.1` 的系统包并设 `MUJOCO_GL=egl` 和 `PYOPENGL_PLATFORM=egl`。务必与上面固定 LIBERO commit 对照，有偏差先查安装脚本的 pin。**不要擅自改 `uv.lock` / `pyproject.toml` 或假设系统 Python 3.12 就是模型的 Python。**

### 8.3 【Runpod】只下载必要权重和 Wan 组件

如果本地已保存 checkpoint，可上传原样文件；若从 Hugging Face 重新获取，遵循官方路径（会产生下载时延/计费）：

```bash
# 【Runpod】脚本需要网络与 huggingface_hub，未再在新 Pod 完整实测
cd /workspace/FasterWAM
export HF_HUB_ENABLE_HF_TRANSFER=0
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
.venvs/libero/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('hustvl/FasterWAM', allow_patterns=['libero/step_021700.pt','libero/dataset_stats.json'], local_dir='checkpoints/fasterwam_release',max_workers=1)
snapshot_download('Wan-AI/Wan2.2-TI2V-5B', allow_patterns=['Wan2.2_VAE.pth','models_t5_umt5-xxl-enc-bf16.pth'], local_dir='checkpoints/Wan-AI/Wan2.2-TI2V-5B',max_workers=1)
snapshot_download('Wan-AI/Wan2.1-T2V-1.3B', allow_patterns=['google/umt5-xxl/*'], local_dir='checkpoints/Wan-AI/Wan2.1-T2V-1.3B',max_workers=1)
PY
sha256sum checkpoints/fasterwam_release/libero/step_021700.pt checkpoints/fasterwam_release/libero/dataset_stats.json
```

**验收：** 上述 SHA-256 与 §2.1 的两个值完全一致；路径确实存在；VAE/T5/tokenizer 齐备。不要用 `allow_patterns=['fasterwam_release/libero/*']`，这是 HF 仓库中的错误前缀，会导致 `Fetching 0 files`。此组合下载命令由历史片段整理，**新云端尚未端到端验证**。

### 8.4 【Runpod】确认单任务可以闭环（最小付费测试）

```bash
cd /workspace/FasterWAM
mkdir -p .runtime/eval_tests
printf 'libero_spatial,0\n' > .runtime/eval_tests/one_task.txt
HF_HUB_ENABLE_HF_TRANSFER=0 \
CKPT_PATH="$PWD/checkpoints/fasterwam_release/libero/step_021700.pt" \
DATASET_STATS_PATH="$PWD/checkpoints/fasterwam_release/libero/dataset_stats.json" \
NUM_GPUS=1 MAX_TASKS_PER_GPU=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
bash scripts/eval_fasterwam_libero.sh \
  MULTIRUN.task_file="$PWD/.runtime/eval_tests/one_task.txt" \
  MULTIRUN.chunk_size=1 MULTIRUN.create_only=false MULTIRUN.dry_run=false \
  model.redirect_common_files=false EVALUATION.num_trials=1 \
  EVALUATION.visualize_future_video=false seed=11 \
  EVALUATION.output_dir="$PWD/.runtime/eval_tests/results/preflight_seed11"
```

**这是可复现最小测试的重构命令，仍会消耗一次 GPU 推理**；与早期原单任务 pilot 的部分输出路径、seed 可能不同。验收 `gpu0_task0_results.json` 中 `successes=1,total_episodes=1`、视频存在、`eval_config_gpu0_task0.yaml` 有 `seed: 11`。不成功时先查看日志，不要反复盲跑。

### 8.5 【Runpod】四组 seed 任务 / 脚本运行

复制交接包 `scripts/run_selected_seed_batch.sh` 到云端 `FasterWAM` 同名目录后，默认只输出命令、不启动 GPU；必须显式确认 `CONFIRM_PAID_RUN=YES` 才会执行。**这是新编脚本，不能取代用户真实旧 `run_batch.sh` 而不经检查。**

```bash
# 【Runpod】任务文件创建，不能覆盖已有结果
cd /workspace/FasterWAM
mkdir -p .runtime/eval_tests
cat > .runtime/eval_tests/selected_10_tasks.txt <<'TASKS'
libero_spatial,0
libero_spatial,1
libero_object,0
libero_object,1
libero_goal,0
libero_goal,1
libero_10,0
libero_10,1
libero_10,2
libero_10,3
TASKS
```

```bash
# 【Runpod】免费预览实际命令；下一条确认后才会跑
bash scripts/run_selected_seed_batch.sh --dry-run
# 需要 GPU 且确认费用后，使用 tmux 执行：
# tmux new -s wam_seed
# CONFIRM_PAID_RUN=YES bash scripts/run_selected_seed_batch.sh --execute
```

**提示：** 已成功 40 episode 不必重复跑。若意外 SSH 中断，先核验 `*_results.json` 与 `failed_tasks.txt`，**只创建缺失任务文件重新执行**，不要将文件数当作完整性证明；结果目录必须使用新名称，避免覆盖旧数据。旧脚本 `run_batch.sh` 若本地仍保存，应优先备份并逐项核对参数，然后再决定是否使用新脚本。`tmux` 断开不是停止运行，`Ctrl-B D` 可分离，`tmux attach -t wam_seed` 可重新连接。

### 8.6 【本地 WSL 或 Runpod】验证 40 行 CSV；有原始 JSON/YAML 时额外核验

```bash
# 【本地 WSL】本交接包展开后的示例
cd /mnt/e/Innovative_practice/WAM/FasterWAM
python3 scripts/verify_multiseed_csv.py docs/evidence/all_40_episode_results.csv \
  --seed-summary docs/evidence/seed_success_rates.csv \
  --suite-summary docs/evidence/suite_success_rates.csv
```

**注意：** 上面路径只有在将整个本交接包内容**谨慎合并**到 FasterWAM 工程根后才成立；若还没有合并，请在已解压交接包根目录运行脚本。验收打印 `VALIDATION PASSED: exactly 40 unique episode records`。这只证明**CSV 记录**完整。

如果已有旧原始六批 `/workspace/FasterWAM/.runtime/eval_tests/results/`，可执行：

```bash
# 【Runpod】需 PyYAML，LIBERO venv 已安装 hydra 一般可用；离线新脚本
.venvs/libero/bin/python scripts/verify_seed_configs.py \
  --csv .runtime/eval_tests/all_40_episode_results.csv \
  --base .runtime/eval_tests
```

这一步逐条核对 CSV 关联的 JSON、YAML 中 seed 字段；仅当原始六批文件齐全时才会成功。新跑的批次名称与历史不同，需先根据目录生成其 CSV，并保证 `batch`、`result_json` 指向实际路径。

### 8.7 【Runpod】重新导出三份 CSV（若六个历史原始批次存在）

原始的 `export_results.py` 当前没有字节副本；本交接提供**新写的替代导出脚本** `scripts/export_multiseed_results.py`，从原始 JSON 读取并按批次名推定 seed，完整检查 40 个组合/重复键/缺失键与程序失败清单，写出 3 个 CSV，**不覆盖原始 JSON**：

```bash
cd /workspace/FasterWAM
.venvs/libero/bin/python scripts/export_multiseed_results.py \
  --results-root .runtime/eval_tests/results \
  --output-dir .runtime/eval_tests/reexported_2026
```

默认批次映射为历史六个目录；新批次请看脚本 `--help` 指定映射。执行前确定结果树存在。**本会话只对已上传 CSV 做离线验证，没有从用户的原始 40 个 JSON 重新导出过**。

### 8.8 【Runpod → 本地 WSL】打包、拷贝与校验

```bash
# 【Runpod】存下真实输出目录、配置、日志和视频；按现有文件判断
cd /workspace/FasterWAM
export ARCHIVE=/workspace/fasterwam_libero_40_episodes_backup.tar.gz
tar -czf "$ARCHIVE" -C .runtime/eval_tests \
  results all_40_episode_results.csv seed_success_rates.csv \
  suite_success_rates.csv selected_10_tasks.txt
ls -lh "$ARCHIVE"
sha256sum "$ARCHIVE"
```

如果当前的三个 CSV 不在 `.runtime/eval_tests/`，请先从导出目录复制进去，**不能原样运行并假设目录齐全**。`reproduction_info.txt`、`run_batch.sh` 与其他任务文件如实际存在，应追加到打包清单；`tar -czf` 创建新档案而不是修改旧文件。生成后用 `tar -tzf` 验证清单。

【本地 WSL Ubuntu】在新 Pod 控制面板确认的**当前** IP、端口、密钥路径下下载；用占位符替换，不复用 2026 年旧实例：

```bash
mkdir -p /mnt/e/Innovative_practice/WAM/seed_eval_results
scp -O -P <NEW_SSH_PORT> -i ~/.ssh/id_ed25519_runpod \
  root@<NEW_POD_IP>:/workspace/fasterwam_libero_40_episodes_backup.tar.gz \
  /mnt/e/Innovative_practice/WAM/seed_eval_results/
cd /mnt/e/Innovative_practice/WAM/seed_eval_results
tar -tzf fasterwam_libero_40_episodes_backup.tar.gz | head
```

**这里 `<NEW_SSH_PORT>` / `<NEW_POD_IP>` 是占位符，必须替换；不能一键原样执行。** 将 Runpod `sha256sum` 和本地 `sha256sum` 对照，确认备份可读、大小和内容正确，**之后才 Stop Pod**。

## 9. A3：从当前状态继续，而不是重跑全部 LIBERO

### 9.1 第一步：本地只读核验（零 GPU）

先在【本地 Windows CMD】`wsl -d Ubuntu`，再在【本地 WSL Ubuntu】：

```bash
cd /mnt/e/Innovative_practice/WAM
ls -lah
find fasterwam_a3_bridge -maxdepth 1 -type f -name '*.py' -printf '%f\n' | sort
for p in FasterWAM a3_dual_arm_sim; do git -C "$p" rev-parse HEAD; git -C "$p" status --short; done
```

```bash
python3 --version
find fasterwam_a3_bridge/.venv -maxdepth 3 \( -name python -o -name python.exe \) -print 2>/dev/null
ls -lh FasterWAM/checkpoints/fasterwam_release/libero/{step_021700.pt,dataset_stats.json} 2>&1
```

**验收：** 记录 A3 实际 Git commit、A3 3.12 venv 是 Linux `.venv/bin/python` 还是 Windows `.venv/Scripts/python.exe`、桥接版本、checkpoint 是否在本地；任何差异先报告而不覆盖。若用户无 `.venv/bin/python`，不能自动使用 Windows venv 在 WSL 执行。

### 9.2 下一步技术验证顺序（不要一次付费运行所有步骤）

1. **保留真实工作区快照**：备份 A3 工程和桥接文件，记录 `git status`，对比 `docs/legacy/A3_Handoff_NATIVE_2026-10-21.zip` 的 `bridge_snapshot/` 和 E 盘同名文件，不覆盖本地已修改版本。
2. **廉价离线动作验证**：在 A3 Python 3.12 venv 验证 `FasterWAMA3ActionAdapterV3.convert` 对边界值的处理；在对应的 Faster-WAM Python3.10 环境检查实际 worker 是否缺官方 `binarize_gripper` 逻辑。用模拟动作检查 `g=-1.015625` 经 `np.sign` 变 `-1` 后 A3 `+1`，对正常空间分量维持不变。
3. **补全后处理（需代码审查）**：只在真正控制路径对应 worker 中补齐官方可选二值化；**先 diff，再保存有版本号的补丁**，不能擅自全通道 clip 或更改控制器硬参数。
4. **本地 A3 控制器验证**：先 seed0 小范围人造动作、短序列；必要时 seed1～5，区分相机/物体随机化与固定几何。历史 A3 默认 `position_noise_m=0` / `yaw_noise_rad=0`，即改变 seed 不一定改变初始物体位置。
5. **最后才开启新 GPU**：按 §8 恢复 Faster-WAM Python3.10、单任务预热，再同一云端独立运行 A3 Python3.12；文件 IPC 目录必须真的在两个进程共享的文件系统内；一次只做有限步数、异常即停、保存动作/异常和视频。
6. **验收条件**：5 个以上连续动作不再因夹爪超界中断；完整短闭环具备观测→动作→执行→新观测；随后才开始完整饼干抓取（成功判定应区分“夹住”“抬起”“转移”“放下”）。若性能不足，再考虑 A3 示教/微调；不提前声称零样本成功率。

### 9.3 A3 最小复现的具体文件与操作前提

- 用户历史真实短闭环入口存放于 `docs/legacy/A3_Handoff_NATIVE_2026-10-21.zip` 内：`FasterWAM_A3_Handoff_2026-10-21/recovered_historical/run_a3_closed_loop.py`，模型侧为同目录 `infer_a3_worker.py`。它们是**历史对话恢复版本**，不是已证实的当前工作区版本；先比较代码 SHA/差异再用。
- 原 runner 用绝对路径 `/workspace/FasterWAM/ipc` 交换 `observation.npz`、`actions.npz`、`request.flag`、`done.flag`；`run_a3_closed_loop.py` 固定 `env.reset(seed=0)`、`REPLAN_STEPS=5`、`MAX_STEPS=400`、`WAIT_TIMEOUT=300`、`INSTRUCTION="Pick up the cookie with the left robot arm."`。**这不是一个安全的小步数测试入口**，正式使用前必须改为可配置短步数、超时和日志路径，并先清除或区分上一轮过时 IPC 标志。更不能在个人 WSL 与远程 Pod 上直接共享这些路径。
- **候选修复对照方式：** 在**本地 WSL**可用 `unzip -p docs/legacy/A3_Handoff_NATIVE_2026-10-21.zip 'FasterWAM_A3_Handoff_2026-10-21/experimental/infer_a3_worker_official_gripper.py'` 只读查看候选代码；与真实 `FasterWAM/infer_a3_worker.py` 逐行核对。不要原样覆盖真实文件。
- **无 GPU 单点回归例子**（在 NumPy 可用的本地 Python 中运行）：

```bash
python3 - <<'PY'
import numpy as np
libero_after_original_worker = -1.015625
libero_after_official_binarize = float(np.sign(libero_after_original_worker))
a3_left_gripper = -libero_after_official_binarize
assert a3_left_gripper == 1.0
print('OFFLINE GRIPPER CHECK PASSED:', a3_left_gripper)
PY
```

这只证明**一个已知边界数值的数学修复合理**，不证明 A3 完整仿真或机器人控制成功。若 Python 缺 NumPy，先检查现有 A3 venv，而不要为了该小测试租 GPU。

### 9.4 A3 控制准确率 / 多 seed 要有可重复协议

- **目的 A**：对已做的 FK/IK、真实末端跟踪、夹爪范围等在 seed=0 对照后换 seed1～5，记录真实误差分布；仅运行数值正运动学断言不等于机械臂实际控制精度达标。
- **目的 B**：零样本 Faster-WAM 饼干单臂抓取；明确固定机器人、任务语言、相机、场景布局、动作 chunk/replan 步数与每 episode 最大步数。真实抓取之前**不要宣布任何成功率**。
- 对 A3 `CookieTransferTaskConfig` 的历史查证显示 `position_noise_m=0`、`yaw_noise_rad=0` 且 `randomize_boxes=False` 常为默认；**seed 改变不等于一定改变饼干几何位置**。如要测试抗扰动，必须显式开启合理非零随机扰动并记录它。

## 10. 目录结构和需要备份的文件（实际路径与理想布局区分）

下面是**预期/历史结构，不保证用户当前磁盘逐项存在**：

```text
E:\Innovative_practice\WAM\
├── FasterWAM/                          # 官方 Git checkout；本交接包 README 可放此处
│   ├── README.md                       # 本文（推荐先备份原项目的 README）
│   ├── configs/sim_libero.yaml
│   ├── configs/model/fasterwam.yaml
│   ├── configs/task/libero_fasterwam_2cam224_1e-4.yaml
│   ├── scripts/eval_fasterwam_libero.sh
│   ├── experiments/libero/{run_libero_manager.py,eval_libero_single.py,eval_libero_chunk.py}
│   ├── src/fasterwam/
│   ├── third_party/LIBERO/
│   ├── checkpoints/fasterwam_release/libero/{step_021700.pt,dataset_stats.json}
│   ├── .runtime/eval_tests/             # 仅当本地已恢复多 seed 原始归档才存在
│   │   ├── selected_10_tasks.txt
│   │   ├── results/seed*/
│   │   ├── all_40_episode_results.csv
│   │   ├── seed_success_rates.csv
│   │   └── suite_success_rates.csv
│   ├── experiment_backups/            # 旧 40 个不同任务 smoke test 归档；历史存在
│   └── docs/legacy/README_FasterWAM_A3_NATIVE_PATHS_2026-10-21.md
├── a3_dual_arm_sim/                    # 独立 Git 仓库，SHA 待本地查
├── fasterwam_a3_bridge/               # 用户自有 bridge 与 outputs；不是本 ZIP 自动同步
├── fasterwam_a3_bridge.zip            # 用户 197MB 旧归档；不在本小交接包里
└── seed_eval_results/                 # 多 seed 归档/CSV/视频的可能本地位置；待核查
```

**本交接 ZIP 的准确内容**：顶层 `README.md`、四个 `scripts/` 脚本、`docs/evidence/` 下三份 CSV、10 任务文件和用户终端环境信息转录、`docs/VERIFICATION_REPORT.md`、`docs/legacy/` 下完整旧 README 与 A3 交接 ZIP。上传 CSV 可独立检查；但原始 40 JSON、每 episode 视频、模型权重和三套真实 Git 工作区**不在交接 ZIP 中**。这些大资源必须仍从用户磁盘备份或官方源获取。

| 重要资料 | 来源 | 是否必须保留 / 作用 |
|---|---|---|
| `step_021700.pt` + `dataset_stats.json` | 官方模型 | **必须**（或能重新准确下载 + hash 验证），推理和归一化 |
| `eval_config_gpu0_task*.yaml` + 原始 JSON | 运行结果 | **必须**，逐任务真实 seed 与成败证据 |
| 视频、任务日志 | 运行结果 | **建议全部保留**，复查轨迹/定位失败原因 |
| `run_batch.sh`、任务列表和执行命令 | 用户运行脚本 | **必须**，真实重跑路径；原脚本目前未上传，待找回 |
| `reproduction_info.txt` | 用户实际运行输出 | **必须保留**，commit / checkpoint hash / GPU / Python 版本 |
| A3 bridge 原始 `.py` / 控制器参数 / 测试视频 | E 盘 A3 工作区 | **必须**，后续修复真实来源 |
| `docs/legacy/...` | 旧 README 与 A3 交接归档 | **保留且不要覆盖当前代码**，包含旧历史审计 |

## 11. 当前最重要的待核实清单（不能自行编造）

1. **E 盘真实状态**：`FasterWAM`、`a3_dual_arm_sim` 当前 Git SHA、分支、dirty 文件、是否等于 §2.1 的 Faster-WAM commit；特别是 A3 SHA。
2. **完整 seed 原始档案**：`fasterwam_libero_40_episodes_final.tar.gz`、其中 40 份 JSON/YAML、视频、日志是否已传回 E 盘并通过 SHA-256/`tar -tzf`；历史 `run_batch.sh` 完整正文在哪里。
3. **`reproduction_info.txt` 本地副本**是否保存；若没有，用本文用户贴过的实际输出重建说明，但明确是从历史终端转录，**不伪装成当前设备重新运行**。
4. **A3 最新工作区**：现有 `.venv` 解释器是 Python 3.12 还是 Windows venv？现有 `infer_a3_worker.py`/`run_a3_closed_loop.py` 是否与交接档案相同？是否已经另外尝试修复夹爪（本轮无最新代码证据）。
5. **官方配置参数确切落地值**：最新多 seed 实验 `manager_config.yaml` / 各任务 YAML 中 `action_infer_mode`、`num_inference_steps`、`replan_steps`、图像处理、seed 如何设置；本 README 根据固定 commit 官方默认值和部分历史配置描述，**未对 40 份 YAML 完整做参数审计**。
6. **环境因果问题**：四组不同 seed 的初始观测是否逐像素一致，环境 seed 与模型 seed 单独干预后的动作张量有无差异；目前仅视频定性观察。
7. **A3 抓取评价**：真实任务完成判定、是否接触与抬升、摩擦/碰撞和运动控制安全性；目前不存在可信完整闭环成功结果。
8. **GPU 可恢复性**：新 Pod 计费、磁盘容量/持久化卷、当前 SSH IP 均需未来现场确认；以前旧 Pod IP/端口不能直接使用。
9. **新增脚本执行情况**：本交接包新写的导出、验证、批处理脚本虽进行了离线检查，但未使用全部远程原始文件和 GPU 做端到端验证。

## 12. 下一个 AI 的可执行计划与验收标准

### 优先级 P0：零费用保护现场（下一个对话立刻执行）

- 【本地 WSL】先运行 §9.1 的**两条**只读命令；收集 Git SHA/dirty 状态、桥接实际脚本和 Python 版本。
- 确认 E 盘**原始 seed 归档 + 历史旧实验归档**是否均存在并通过 `tar -tzf`；先备份，不要覆盖。
- 记录具体差异：本地 `FasterWAM` 是否已更新，与 2026-10-22 commit 是否一致；对 A3 worker 做**只读** diff。**验收：** 有清晰的目录清单、版本和原始备份。

### 优先级 P1：修复 A3 夹爪越界并真正完成小闭环

- 在离线 Python 中模拟 `-1.015625` 的夹爪输出，确认官方 `np.sign` 后正确映射 `+1`，其余空间通道不被修改；审查 worker 实际调用路径与 `EVALUATION.binarize_gripper`。
- 形成最小差异补丁，先测动作范围与失败用例，明确左右夹爪语义；再恢复 GPU 两套隔离 venv，单次推理与最多数步闭环试验。
- **验收：** 无夹爪超界异常且能够连续重规划/执行动作，保存真实日志、动作、画面；**不要求此时一次就能抓取成功**。

### 优先级 P2：固定协议评估 A3 夹取

- 明确饼干数量/场景 seed/几何扰动/摄像头/任务描述/夹爪力度/成功定义；逐段验证接近→接触→夹起→转移→放置。
- 先 3～5 个可重复种子作 pilot，足够后再按 20～50 次统一 protocol 统计成功率、失败类型和系统时延；若成功率低再考虑专家轨迹与微调。
- **验收：** 每次输出 `seed、init_state、success、steps、failure_reason、video、action_log`，能独立复查。

### 优先级 P3：研究性 seed 分析、实验归档

- 为纯模型随机性实验**环境 seed 固定、模型 seed 变化**，同一首帧直接对比动作张量；在有预算时扩大 initial states。不要把固定初始场景 + 10 个任务的 40/40 当成论文级统计显著性。
- 整理 CSV、原始 JSON、配置、视频、模型 sha、代码 diff，形成报告图表；可以使用已经制作的组会 PPT，但视频合成文件不替代原始数据。

## 13. 旧历史、其他文档与接手提醒

- **最重要的旧参考**：`docs/legacy/README_FasterWAM_A3_NATIVE_PATHS_2026-10-21.md`，保留完整 1354 行旧交接记录，内容包括早期 40 不同任务评测、历史双 Python A3 工程、A3 FK/IK 测试脚本解释及失败复现。
- **旧 A3 辅助文件 ZIP**：`docs/legacy/A3_Handoff_NATIVE_2026-10-21.zip`，只用于离线对照/找回源文件；不要直接覆盖用户 E 盘现有桥接目录。
- **目前无访问 E 盘权限**：本文件是**可下载后安装的 README 更新候选**，并没有直接修改用户实际 `E:\Innovative_practice\WAM\FasterWAM\README.md`。用户应先备份原本 README，再根据实际仓库状况审查合并。
- **不要泄露 SSH 密钥、访问令牌、密码或私人凭据**；旧 SSH 地址与价格只在历史档案作为过去记录，不代表现仍有效。

### 下一位 AI 第一句话该怎么做

> 我已经阅读 `README.md` 的最新版本和 `docs/legacy` 历史资料。先不租 Runpod，也不重跑已经完成的 40 次 seed 评测。请在**本地 WSL Ubuntu**依次运行 README §9.1 的两组只读命令，将输出给我。我会以真实文件确认 A3 工作区、Python3.12、当前 worker 的 `binarize_gripper` 处理和模型 checkpoint/原始备份；确认后仅修改 A3 必要的最小代码差异，先离线验证夹爪 `+1.015625` 越界问题，再考虑开启付费 GPU 进行短闭环。

---

**交接质量检查：** 新阶段的 40 CSV 已离线验证；对应旧 Runpod 的 YAML 40/40 逐项检查有用户终端输出作为历史证据；固定 commit 的官方源码中 seed 传递及 `binarize_gripper` 可在上述精确链接核对；A3 候选 worker 的缺失处理已通过归档 Python 源文件核对。**未取得 E 盘文件系统或原始 40 JSON/YAML，因此不能声称已独立端到端重跑、恢复了真实本地 README，或已修复 A3 抓取。**
