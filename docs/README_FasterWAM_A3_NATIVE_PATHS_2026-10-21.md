# Faster-WAM × LIBERO × A3 饼干抓取：完整项目 README / 交接复现手册

> **最新状态时间：2026-10-21。** 本 README 由已上传的 2026-10-16 原交接文档、2026-10-19～20 的完整聊天记录、33 份 Python 上传（去重后 28 份）、`outputs.zip` 与公开项目源码共同核对整理。主要研究任务已从“LIBERO 官方复现”推进到“Faster-WAM 零样本 A3 饼干夹取与成功率”，但尚未完成可成功的 A3 闭环。
>
> **最重要的诚实限制：此交接包提供尽可能完整的源文件、可执行检查工具和云端恢复脚本，但不能保证在一台新 Pod 上无条件零错误、一键跑成。** 理由：用户 E 盘真实状态与 A3/Faster-WAM 当前 Git SHA 尚未读取；原始 A3 仓库、Faster-WAM 11GB 权重与额外 Wan 模型不包含在这份小压缩包里；Runpod 是付费且无持久化的远程计算环境；恢复脚本未在全新 GPU 上端到端运行验证。下文明确划分 `[历史已运行]`、`[源码静态核验]`、`[当前官方源码]`、`[新编写未端到端运行]` 和 `[待实测]`。**禁止下一位 AI 把这些不同等级混为一谈。**
>
> **下一位 AI 的首要原则：先做本地零费用核验；一轮只给 1～2 条命令；不要重新做 LIBERO 40×1；绝对不要在远程 GPU 持续计费期间编写无关调试脚本。**

## 2026-10-21 再次纠正：**现有 E 盘三个项目目录才是权威工作区**（最高优先级）

用户现在已经实测：在 Windows CMD 运行 `wsl -d Ubuntu` **成功**；在 WSL 中 `cd /mnt/e/Innovative_practice/WAM` 后确实有 `FasterWAM/`、`a3_dual_arm_sim/`、`fasterwam_a3_bridge/`、`fasterwam_a3_bridge.zip`（约 197 MB）。资源管理器截图只展示 `fasterwam_a3_bridge/` 的**部分**内容：`.venv/`、`__pycache__/`、`outputs/`、`action_adapter*.py`、若干 `diagnose_*.py`；**不能据此断言所有其他 `.py` 都缺失**。截图里的 `__pycache__` 不能当作源码或环境备份。

**更正旧建议：不要为了运行现有 A3 测试，必须先安装交接 ZIP 或创建第四个项目目录。** `FasterWAM_A3_Handoff_2026-10-21/` 只是一份**可选、额外的交接资料**，在用户 `ls -la` 时并不存在。此前调用其 `scripts/preflight.py` 出现 `No such file or directory`，**只是调用了尚未解压的路径**；不是 WSL、A3 或 Python 出了问题。

**真实本地路径，以此为准：**

| 用途 | Windows 路径 | WSL 路径 |
|---|---|---|
| 官方 Faster-WAM 主项目 | `E:\Innovative_practice\WAM\FasterWAM` | `/mnt/e/Innovative_practice/WAM/FasterWAM` |
| A3 仿真仓库 | `E:\Innovative_practice\WAM\a3_dual_arm_sim` | `/mnt/e/Innovative_practice/WAM/a3_dual_arm_sim` |
| **现有主要实验桥接文件** | `E:\Innovative_practice\WAM\fasterwam_a3_bridge` | `/mnt/e/Innovative_practice/WAM/fasterwam_a3_bridge` |
| **用户原始 197 MB ZIP** | `E:\Innovative_practice\WAM\fasterwam_a3_bridge.zip` | `/mnt/e/Innovative_practice/WAM/fasterwam_a3_bridge.zip` |
| **新版小交接 ZIP（仅在用户自行下载后）** | `E:\Innovative_practice\WAM\FasterWAM_A3_Handoff_NATIVE_2026-10-21.zip` | `/mnt/e/Innovative_practice/WAM/FasterWAM_A3_Handoff_NATIVE_2026-10-21.zip` |

**交接 ZIP 到底多了什么？** 用户先前逐批上传的 28 份非重复原始 `.py` 都在 `bridge_snapshot/`，目的是留存历史快照；以 E 盘原始同名 `.py` 优先，**不要覆盖**。交接包另外新增、不能从该局部截图确认原目录是否含有的内容有：`scripts/preflight.py`、`scripts/seed_batch.py`、`scripts/seed_scene_check.py`、`scripts/offline_action_check.py`、`scripts/verify_bundle.py`、`scripts/test_gripper_guard.py`、`scripts/cloud_prepare_env.sh`、`scripts/cloud_archive_results.sh`；`recovered_historical/infer_a3_worker.py`、`recovered_historical/run_a3_closed_loop.py`；以及 `experimental/` 中两个候选夹爪处理补丁。还包含此前上传的 `outputs_original.zip` 和原版交接 README。**是否与你实际 197MB ZIP 重复未知，因为没收到该 ZIP 内容清单。** 小交接 ZIP **不包含 11GB checkpoint、Python venv、A3 全仓库或者可直接运行的大模型环境**。

### 现在直接从原始工作区继续：两条只读检查（本地 WSL）

```bash
cd /mnt/e/Innovative_practice/WAM
find fasterwam_a3_bridge/.venv -maxdepth 3 \( -name python -o -name python.exe \) -print 2>/dev/null; python3 --version; command -v python3.12 || true
```

```bash
find fasterwam_a3_bridge -maxdepth 1 -type f -name '*.py' -printf '%f\n' | sort; git -C a3_dual_arm_sim rev-parse --short HEAD
```

收到这两段**实机输出**再确定 `.venv` 是 Windows 还是 Linux、Python3.12 是否可用、有哪些测试代码。本地 seed0 首测应优先直接使用用户原始 `fasterwam_a3_bridge/test_fk_consistency.py`（如该文件实际存在），**不依赖交接包**；运行前确认 `a3_dual_arm_sim` 的真实安装/import 路径和 `MUJOCO_GL`。批量 seed 工具是便利工具，可把交接 ZIP 中的**单一脚本**单独复制到 `fasterwam_a3_bridge/seed_batch.py`，不必为它解压整个包，更不可假定已复制。

**如果用户愿意单独保存小交接包**：把它看作归档，放在 E 盘 WAM 根目录或任意备份位置，和原始代码隔离。历史文档里所有要求先解压交接包才能做本地核验的命令均属于**可选工作流**，不应再作为唯一入口。未来 Runpod `/workspace/...` 仍为**另一台电脑的临时路径**，与 E 盘原始项目不可混淆。

---

## 2026-10-21 用户本地终端实测更正（最高优先级）

**证据**：用户在 Windows CMD 执行 `wsl -d Ubuntu` 成功，进入 `asus@LAPTOP-I6M87GO6` 的 **本地 WSL Ubuntu**；执行 `cd /mnt/e/Innovative_practice/WAM; ls -la` 得到以下**真实存在**的一级条目：

```text
E:\Innovative_practice\WAM\
├── .cache/                     # 已见目录
├── Faster-WAM.pdf              # 已见文件
├── FasterWAM/                  # 已见目录，内容尚未在本轮逐项核对
├── a3_dual_arm_sim/            # 已见目录，Git SHA 未知
├── fasterwam_a3_bridge/       # 已见目录，用户截图确认 .venv/、__pycache__/、outputs/、多个 .py
└── fasterwam_a3_bridge.zip    # 已见文件，197446732 字节（约 188 MiB）；不是本交接 ZIP
```

**不存在的目录**：当时 E 盘根目录下**没有** `FasterWAM_A3_Handoff_2026-10-21/`。因此此前命令
`python3 FasterWAM_A3_Handoff_2026-10-21/scripts/preflight.py --wam-root "$PWD"`
报 `No such file or directory` 的原因是**交接包尚未拷贝并解压到指定位置**，**不是** Ubuntu 启动失败、脚本语法问题或必须重装 Python。

**已通过实际 ZIP 内部目录表验证**：本交接 ZIP 自带顶层 `FasterWAM_A3_Handoff_2026-10-21/`，包含 `README.md`、`scripts/preflight.py`、`scripts/seed_batch.py`、`scripts/seed_scene_check.py` 等。先把本 ZIP 保存到 `E:\Innovative_practice\WAM\` 再解压，不要把现有的 `fasterwam_a3_bridge.zip` 错认成交接包，也不要覆盖 `fasterwam_a3_bridge/`。本 README 所称 `FasterWAM_A3_Handoff_2026-10-21/` 是**解压后将出现的目录，不是用户上传截图时已存在的目录**。

### 可选：要使用交接包附带辅助脚本时才解压（本地 WSL Ubuntu）

1. 在 Windows 浏览器下载**本次更新的 `FasterWAM_A3_Handoff_NATIVE_2026-10-21.zip`**，用资源管理器复制到 `E:\Innovative_practice\WAM\`；不要移动、重命名或解压现有的 `fasterwam_a3_bridge.zip`。
2. 在**已打开的 Ubuntu 终端**中逐条运行：

```bash
cd /mnt/e/Innovative_practice/WAM
python3 -m zipfile -t FasterWAM_A3_Handoff_NATIVE_2026-10-21.zip
```

```bash
python3 -m zipfile -e FasterWAM_A3_Handoff_NATIVE_2026-10-21.zip .
python3 FasterWAM_A3_Handoff_2026-10-21/scripts/preflight.py --wam-root "$PWD"
```

`zipfile -e` 解压目标 `.` 为当前 WAM 目录；压缩包内部**已经有** `FasterWAM_A3_Handoff_2026-10-21/` 顶层目录，不要再手工套一层同名文件夹。`preflight.py` 只读清点：**不会**导入 MuJoCo、不会启动 Runpod、不会安装依赖。

### 关于当前 `.venv` 和运行 Python 的更正

截图只证明 `fasterwam_a3_bridge/.venv` **目录存在**，不能证明它是 Linux venv、Python 3.12、可直接被 WSL 运行，甚至不能证明环境内包完整。下个操作应当执行：

```bash
ls -l /mnt/e/Innovative_practice/WAM/fasterwam_a3_bridge/.venv/bin/python /mnt/e/Innovative_practice/WAM/fasterwam_a3_bridge/.venv/Scripts/python.exe 2>/dev/null || true
python3 --version
```

若仅有 `Scripts/python.exe`，把它视作 Windows venv，需要另行核实在 Windows Python 3.12 下的工作状态；不能在 WSL 直接调用相同的 POSIX `.venv/bin/python` 路径。**在确认真实解释器和 A3 依赖前不要运行 seed_batch**。之后运行脚本时使用当前项目中实际存在的 `test_fk_consistency.py` 等名称；用户截图显示了部分文件但未展示整个目录清单，不能假定所有 ZIP 中的 28 个脚本文件都在 E 盘现存目录。

### 证据边界

README 中第 §0～§17 节的 `/workspace/...` 是历史 **Runpod 云端路径**；`/mnt/e/...` 才是当前 **本地 WSL 路径**。`E:\...` 是 Windows 路径。交接包内 `bridge_snapshot/` 是已上传代码的归档，**不是**现存 E 盘 bridge 的完整镜像；用户还有 `fasterwam_a3_bridge.zip`，其内部文件与 `.venv` 未上传、无法在本对话证实。本轮未见 E 盘下 `FasterWAM` 的 checkpoint、`a3_dual_arm_sim` 的真实版本内容以及 `.venv` 解释器，因此此前各条只能保留为历史记录或待核验，不得升格为当前实测。

---

## 最新总览：下一位 AI 先读（优先级高于下面历史 §0～§17）

| 核心事实 | 最新结论 |
|---|---|
| 用户主要目标 1 | A3 机械臂、夹爪、FK/IK、观察等已做的测试换 **不同 seed** 做重复性检查；先证明 seed 是否改变场景，再统计误差；主要在本地 Python 3.12 进行，不需 GPU。 |
| 用户主要目标 2 | 使用**原版官方 Faster-WAM LIBERO checkpoint**，零样本尝试 A3 饼干**单臂抓取**，保存过程、统计抓取成功率；如果低再考虑收集 A3 专家演示并微调。 |
| 已完成 LIBERO | 40 个 LIBERO 任务各 1 次，40/40 成功；这是**冒烟**而非 50-trial 基准；原结果与视频已备份 E 盘。 |
| A3 真实进度 | 三种动作适配器、视觉/状态适配、FK/IK/精度/夹爪多套测试和 MP4；曾获得一段 `32×7` 真实模型预测；常驻 GPU worker 曾推理约 2.48 秒；**闭环第一轮完成 3 个执行动作后，在下一次动作转换时因左夹爪控制 `+1.015625` 越界失败**，没有成功抓起饼干。 |
| A3 仿真环境 | [`E:\Innovative_practice\WAM\a3_dual_arm_sim`](https://github.com/Eter0109/a3_dual_arm_sim) 的 main（本地 checkout SHA **待查**）。`pyproject.toml` 要求 **Python >=3.12**、NumPy >=2。 |
| A3 桥接代码 | `E:\Innovative_practice\WAM\fasterwam_a3_bridge`；用户最新资源管理器截图**确证目录中存在** `.venv/`、`__pycache__/`、`outputs/` 及多个 `.py`，但解释器路径、版本、依赖和其余文件尚未核验；本交接包 `bridge_snapshot/` 保存**28 个去重后的用户上传 `.py` 文件**。 |
| Faster-WAM 推理环境 | 云端 `.venvs/libero` 采用 **Python 3.10.18**，Torch 2.7.1+cu128，LIBERO commit `8f1084e`。与 A3 的 3.12 环境**严格隔离**，用 NPZ 文件旗标 IPC 跨进程通信。 |
| 临时 Runpod 情况 | **第一轮旧 Pod 已停止**；2026-10-19 又开启过一个**新的** RTX A6000 Pod，`/workspace` 初始为空，曾重新安装源码环境、下载 Faster-WAM LIBERO 和额外 RoboTwin 权重、运行 A3 worker。用户说明每次使用后全部清空；**本次没有新控制台截图能核验所有 Pod 当前状态或费用**。不假设云端保留任何资源。 |
| 一键复现范围 | `scripts/preflight.py` 能**一条命令**完成 E 盘清点；`scripts/seed_batch.py` 能**一条命令**安排原测试的多 seed 批量执行（需本地 A3 3.12 环境）；`scripts/cloud_prepare_env.sh` 能尝试在已启动的临时 Pod 上自动安装与下载（**新脚本未端到端实测，需用户确认费用**）。它们**不能代替**配置 GPU、网络、权限、保存大 checkpoint 及验证真实抓取。 |
| 最新排错关键（2026-10-21） | 官方 LIBERO `configs/sim_libero.yaml` 当前默认 `EVALUATION.binarize_gripper: true`，官方 `eval_libero_single.py` 会在动作反归一化与 gripper 符号处理之后调用 `np.sign()`；但历史手写的 `infer_a3_worker.py` **漏掉这一项**。优先修复 worker 的官方后处理，而不是直接裁剪 A3 14 维动作。新的 `experimental/infer_a3_worker_official_gripper.py` 只补这几行，**待上 GPU 短闭环验证**。 |
| 最该避免的误判 | 越界向量 `[-0.025635,0.087891,-0.021973,0.205539,-0.023557,0.060376,**1.015625**, ...]` 中**第 7 个数、索引 6 才越界，属于左夹爪**；`0.205539` 是合理范围内的旋转，原 AI 的诊断是错的。 |
| 优先次序 | **本地清点 → seed=0 对照 → seed=1～5 控制精度验证 → 确认环境随机化 → 修复并测试夹爪越界（不动空间动作） → GPU 短闭环 → 3～5 个种子 → 20～50 个固定协议零样本 episode → 低成功率后考虑微调。** |

### 如果只给下一个 AI 一段简短指令

> 请将此 README 中的“最新总览”和 §18～§29 视为最优先事实，§0～§17 是原 LIBERO 复现手册。我的目标不是重跑 LIBERO，而是把 Faster-WAM 官方 LIBERO 检查点零样本接到 A3 机器人饼干抓取任务，并在本地先完成多 seed 机器人控制器精度复测。**A3 用 Python 3.12，Faster-WAM worker 用 Python 3.10，必须分开。** 本地代码在 E 盘 `WAM/`，Runpod 是不持久化的付费 GPU。先帮我检查本地文件和版本，只提供 1～2 条命令并等我回结果；不修改任何已经确认通过的核心控制器，不凭猜测声称一键运行成功。

---

## 原始 Faster-WAM / LIBERO 复现手册（初稿 + 针对新进展的必要修订）

## 0. 历史 LIBERO 交接摘要（2026-10-16；以最新总览为准）

1. **项目方向**：计算机系本科创新实践 / 科研，面向 MuJoCo 机械臂操作，基于 World Action Model（WAM）/ VLA，优先复现 **Faster-WAM**，后续探索**自适应未来特征融合**、**Dynamic SparseMoT** 等改进，目标包括可展示的机器人闭环操作、定量消融和潜在论文。早期曾讨论先从 Fast-WAM 入门，**实际落地并成功验证的是 Faster-WAM 官方 LIBERO 检查点**。
2. **截至 LIBERO 阶段的进度**：**未训练/微调模型，也未训练“专家演示轨迹”**；已经使用官方发布的 `step_021700.pt` 和 Wan 组件，在 RTX A6000 上完成真实 MuJoCo/LIBERO 闭环**推理评测**。四个 LIBERO 任务集共 **40 个任务，每任务 1 次，40/40 成功**；此前还成功完成一次单任务评测并检查视频。
3. **证据**：40 任务评测输出 `tasks=40 chunks=40 ... gpu_ids=['0']`；四个任务集各 `10/10`，总成功率 `100%`，报告的累计任务执行时间 `19m21s`。**这只是 smoke test，不是每任务 50 次的正式统计基准**，不能将 `100%` 当作论文结论。
4. **本地项目根目录 `[已验证]`**：Windows `E:\Innovative_practice\WAM\FasterWAM`；WSL2 Ubuntu `/mnt/e/Innovative_practice/WAM/FasterWAM`。本地已有官方 LIBERO checkpoint 和数据统计文件；评测的两个压缩备份包已经下载到 `experiment_backups/`。
5. **第一轮旧云端状态 `[已验证的历史]`**：此前的 Runpod Pod `FasterWAM-LIBERO`（RTX A6000 48GB）**已 Stop**，控制台最后截图显示 **`$0.00/hr`**；弹窗明确指出**没有配置 volume，Stop 后数据丢失**，停止后的闲置磁盘费为 `$0.000/hr`。**不要假设云端 `.venv`、Wan 下载文件、`/workspace/FasterWAM` 或原 SSH 入口仍存在**。历史 IP/端口仅作故障分析记录。
6. **如果用户当前只是继续本地研究**：先核验/解压 `experiment_backups/fasterwam_libero_40tasks_smoke.tar.gz`，查看视频、`summary.csv`、`task_success_rates.csv`、`summary.json`。过去对话只有下载成功和 `tar -tzf` 列表成功的明确记录；**没有用户确认解压已成功的记录**。
7. **如果用户需要重新租 GPU**：目前优先目标是 A3 饼干任务，不是重跑已有 LIBERO 40×1。应先做最新总览和 §18～§29 的本地检查；新建 Pod 后从本地源代码/权重重建环境，按第 7～11 节和 §25 操作；不要用历史 IP/端口盲连，也不要先跑 2000 次。
8. **交互偏好**：用户希望**一步一条或少量可复制命令、发输出再继续**；说明在哪个终端（Windows CMD / 本地 WSL / Runpod SSH）运行；优先控制 GPU 租赁费用，不要无谓重装和下载。

### 完成/未完成矩阵

| 事项 | 状态 | 证据 / 备注 |
|---|---|---|
| 确定研究方向、阅读 Fast-WAM / Faster-WAM | 已讨论 | 创新思路仍是候选，没有定稿实验设计 |
| Windows + WSL2、本地 E 盘项目 | 已使用 | 项目根目录见上 |
| 官方 Faster-WAM LIBERO 检查点 | 已持有并上传过 | `step_021700.pt`，传输日志 `11.12G` |
| 云端核心 Python 与 LIBERO 环境 | 曾搭建并验证成功 | **Stop 后不可视为保留** |
| MuJoCo EGL 离屏渲染 | 曾测试成功 | 渲染输出 `(64, 64, 3)` |
| 单任务闭环推理 | 已完成 | 1 次尝试、1 次成功；人工观看视频符合任务预期 |
| LIBERO 四套任务 × 10 任务 × 1 次 | **已完成** | `40/40`；总计 `19m21s`（统计中的任务时间） |
| 本地备份 40 任务结果 | **已完成** | `scp -O` 下载约 `8.5 MB`，`tar -tzf` 通过 |
| 本地解压并浏览 40 任务视频 | **未收到成功反馈** | 已给出命令，需下次自行核验 |
| LIBERO 每任务 10/50 次 | **未做** | 仅有计划与耗时估计 |
| LIBERO-Plus、RoboTwin | **未做** | 官方支持，不等于个人已验证 |
| 专家演示数据准备、训练/微调 | **未做** | 没有执行训练脚本 |
| 动态门控/改进模块、消融、论文 | **未做** | 下一研究阶段 |
| A3 饼干机器人接口测试 | **已做多项 seed=0 代码与视频实验** | V3 适配器、机器人状态和相机；详见 §18 起 |
| Faster-WAM → A3 单次推理 | **已完成至少一次** | 历史 `(32,7)` 动作输出、预测视频 |
| Faster-WAM → A3 完整闭环与抓取成功率 | **未完成** | Replan 1 已执行 3 步，下一次转换时左夹爪动作 `1.015625` 越界中止 |
| A3 多 seed 统计 | **未做** | 部分 seed 变化可能根本不随机化几何 |
| Runpod 停止 GPU 计费 | **已确认** | 最后截图 Pod 旁 `$0.00/hr` |

---

## 1. 课题背景与拟研究问题

- 场景：MuJoCo 仿真中的机械臂抓取、放置、开抽屉、开关炉灶、长时序多物体操作。官方基准选 **LIBERO**（含 `libero_spatial`、`libero_object`、`libero_goal`、`libero_10`）。LIBERO 基于 robosuite / MuJoCo。
- 官方基线：**Faster-WAM: Efficient Inference-Time Future Conditioning for Robust World Action Models**。它讨论如何在推理时有效利用未来表示并减少重复的视频-动作交互；仓库保留 FastWAM、JointWAM 基线。
- 参考基线：**Fast-WAM: Do World Action Models Need Test-time Future Imagination?**；两种路线为“推理阶段是否及如何使用未来表征”提供研究切入点。
- 拟题（**仅为早期建议，未正式定题**）：**《面向 MuJoCo 机械臂操作的自适应未来特征增强 World Action Model 研究》**，或《基于自适应未来表征的高效鲁棒机械臂控制方法研究与 MuJoCo 仿真验证》。
- 拟创新（**尚未实现**）：按当前视觉/状态、预测不确定性、动作阶段、场景变化等决定是否启用未来特征，选择性融合；探索动态 SparseMoT/门控、融合频率或 KV 缓存策略，在**成功率、鲁棒性、时延、显存/算力**之间折中。
- 科学问题：未来表征在什么任务阶段、何种 OOD 扰动下最有价值？是否能用更少的未来计算达到类似或更高成功率？
- **禁止过度宣称**：本项目尚未证实新方法有效，也未完成重训练或 50 次标准统计。后续研究必须设置同环境、同模型、同初始化/评测预算的公平对照。

### 官方资料（下次复现前建议重新检查版本）

- Faster-WAM 官方代码与 README：https://github.com/hustvl/FasterWAM
- Faster-WAM 发布检查点：https://huggingface.co/hustvl/FasterWAM
- Fast-WAM 官方代码：https://github.com/yuantianyuan01/FastWAM
- Wan2.2：https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
- Tokenizer 来源：https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
- 官方预处理 LIBERO 训练集（**本项目未下载用于训练**）：https://huggingface.co/datasets/yuanty/LIBERO-fastwam
- LIBERO 官方仓库：https://github.com/Lifelong-Robot-Learning/LIBERO
- Runpod 文档：https://docs.runpod.io/

**资料来源界限**：上述仓库 README 所述“可训练”“支持 LIBERO-Plus”等为**官方说明**；本文实验结果仅来自此前聊天内的终端日志与截图，未自行执行全量基准核验。

---

## 2. 关于“训练的是什么”：明确纠正一个容易误解的问题

1. **专家演示轨迹**是离线训练数据：机器人观测、动作、任务语言、环境状态等的序列；轨迹本身不是要“训练”的模型。
2. **完整模型训练/微调**需要读取训练集并反向传播、更新参数，通常由 `scripts/train_fasterwam_libero.sh` 等训练流程触发。
3. **目前实际做的事**是 `bash scripts/eval_fasterwam_libero.sh`：读取官方已训练的权重和统计信息，完成机器人闭环推理、渲染视频、计算成功率。**未发生自定义训练、参数优化、专家轨迹学习，也没有更新模型权重。**
4. 使用的是官方 Faster-WAM **完整 LIBERO 推理评测链路**（checkpoint、Wan VAE、T5 编码器、tokenizer、MuJoCo 环境），但不等于从零训练整个 Faster-WAM。
5. 正确成果描述：**“部署并初步复现 Faster-WAM 官方检查点在 LIBERO/MuJoCo 的推理评测；完成 40 任务 × 1 次的功能验证。”**

---

## 3. 机器、系统、路径与运行成本

### 3.1 本地机器 `[已验证]`

- 主机：Windows（用户终端为 `C:\Users\ASUS`），使用 **WSL2 Ubuntu**。
- 注意默认 WSL 发行版可能是 `docker-desktop`，在 **Windows CMD** 必须显式进入 Ubuntu：
  ```bat
  wsl -d Ubuntu
  ```
- Ubuntu 用户：`asus`；本地源代码和持久文件优先存 E 盘：
  ```bash
  cd /mnt/e/Innovative_practice/WAM/FasterWAM
  pwd
  ```
- Windows 对应路径：`E:\Innovative_practice\WAM\FasterWAM`。
- 本地 Runpod **私钥路径**：`/home/asus/.ssh/id_ed25519_runpod`，文件权限曾执行 `chmod 600`；该密钥设有 passphrase。**文档未收录任何密钥内容或口令；不得上传私钥到 GitHub/发给 AI。**
- 本地计算环境不是已验证的训练 GPU 环境；先前真正的推理运行在 Runpod。

### 3.2 曾使用的 Runpod `[历史配置；已停止]`

| 属性 | 历史值 / 实际结果 |
|---|---|
| Pod 名称 | `FasterWAM-LIBERO` |
| Pod ID | `1b454a5fx7wxiw` |
| 机型 | Secure Cloud；NVIDIA RTX A6000 ×1；48GB VRAM |
| 运行价格 | 页面显示约 `$0.61/hr`（早期记录 `$0.606/hr`） |
| 云端项目目录 | `/workspace/FasterWAM` |
| Container Disk | 120 GB；**没有配置持久化 Volume** |
| 旧容器提示符 | `root@9f302eb3c51f` |
| 旧 SSH 连接（仅历史） | `root@216.81.151.15 -p 14697` |
| 最后控制台状态 | **Stopped，`$0.00/hr`** |
| Stop 弹窗原文 | `You do not have a volume configured. ALL DATA will be lost!` |
| Stop 后闲置磁盘费用 | 弹窗明确显示 `$0.000/hr ($0.00/day)` |

**关键约束**：用户最后确实点击 **Stop Pod**，截图显示停止状态和 `$0.00/hr`。停止时没有 Volume，**不应假定之后可直接 `ssh` 返回此前工作区，也不应承诺云端软件/文件仍在**。以后重新租用时通常需要重新上传 checkpoint 和重新安装依赖。控制台显示的余额（当时截图约 `$7.16`）是账户余额，不等于仍有 GPU 在运行，也不代表自动退费。

**计费操作原则**：SSH `exit` / 关闭 CMD / 关闭 VS Code **不会**停止运行中的 Pod 计费。必须在 Runpod Pods 控制台 Stop；停止后检查显示 `$0.00/hr`，同时检查是否另有付费存储卷/其他运行中 Pod。若要永久删除 Pod 记录可以 Terminate，但该操作可能不可逆。启动新的 GPU 时确认当前价格、磁盘/Volume 配置；**不要凭历史价格估算当前账单**。

---

## 4. 核心目录布局与本地已保存资料

### 4.1 本地项目（建议检查的目标布局）

```text
E:\Innovative_practice\WAM\FasterWAM\
├── configs/                      # 官方 Hydra 配置（若本地同步完整）
│   ├── data/                   # 重要！之前云端缺失过
│   ├── model/fasterwam.yaml
│   ├── task/libero_fasterwam_2cam224_1e-4.yaml
│   └── sim_libero.yaml
├── scripts/
│   ├── eval_fasterwam_libero.sh
│   ├── setup/install_core.sh
│   └── setup/install_libero.sh
├── experiments/libero/
│   ├── run_libero_manager.py
│   ├── eval_libero_chunk.py
│   ├── eval_libero_single.py
│   └── summarize_results.py
├── src/fasterwam/
├── checkpoints/
│   └── fasterwam_release/libero/
│       ├── step_021700.pt          # 历史传输记录约 11.12GB
│       └── dataset_stats.json      # 约 40KB
├── experiment_backups/
│   ├── fasterwam_first_success.tar.gz          # 已下载；约 141KB
│   └── fasterwam_libero_40tasks_smoke.tar.gz   # 已下载；约 8.5MB
├── evaluate_results/
│   ├── libero_official_wan_test/  # 单任务压缩包解压后可得到
│   └── libero_40tasks_smoke/      # 40任务压缩包解压后可得到
└── README.md                       # 本交接文档，建议保存到这里
```

**注意“建议检查的目标布局”并非已通过当前文件系统逐项验证**。这份 README 由对话日志生成，AI 无法直接读取用户 E 盘；尤其 `evaluate_results/...` 是否已解压，需要重新确认。`experiment_backups` 下两份 tar.gz 的**远程下载成功**有明确历史记录。

### 4.2 两份实验档案 `[已验证传输，已验证 tar 列表]`

| 备份文件 | 来源 | 记录 |
|---|---|---|
| `experiment_backups/fasterwam_first_success.tar.gz` | `/workspace/fasterwam_first_success.tar.gz` | 一次 `libero_spatial_0` 推理，包含结果、任务配置与 MP4；通过 SCP 拉回本地，`tar -tzf` 已验证 |
| `experiment_backups/fasterwam_libero_40tasks_smoke.tar.gz` | `/workspace/fasterwam_libero_40tasks_smoke.tar.gz` | 四大套件全 40 任务结果，云端 `ls -lh` 8.5M，Windows WSL 的 `scp -O` 100% (`8684KB`)，`tar -tzf` 已正常列出汇总文件 |

**强烈建议长期保留 `.tar.gz` 原档案**。不要把解压后的结果目录当作唯一备份。云端没有持久化卷，因此后续恢复以本地文件为准。

### 4.3 本地解压和视频浏览（无需 GPU，也无需新 Pod）

在 **Windows CMD** 先进入本地 Ubuntu：

```bat
wsl -d Ubuntu
```

在 **本地 Ubuntu** 中运行（**`cd` 和 `tar` 各自保持一整行，不要在命令名和参数间误按 Enter**）：

```bash
cd /mnt/e/Innovative_practice/WAM/FasterWAM
ls -lh experiment_backups/*.tar.gz
tar -tzf experiment_backups/fasterwam_libero_40tasks_smoke.tar.gz | tail -5
tar -xzf experiment_backups/fasterwam_libero_40tasks_smoke.tar.gz
tar -xzf experiment_backups/fasterwam_first_success.tar.gz
find evaluate_results/libero_40tasks_smoke -type f -name '*.mp4' | head -20
```

用 Windows 资源管理器打开评测目录：

```bash
explorer.exe "$(wslpath -w "$PWD/evaluate_results/libero_40tasks_smoke")"
```

或直接在资源管理器地址栏输入：

```text
E:\Innovative_practice\WAM\FasterWAM\evaluate_results\libero_40tasks_smoke
```

各任务集的 `videos/` 中是 `.mp4`；用 Windows 视频播放器/VLC 即可。目录根部的 `summary.csv`、`summary.json`、`task_success_rates.csv` 可用 Excel / VS Code 查看；任务集下的 `gpu0_task*_results.json` 记录单任务结果；另有 `eval_config_gpu0_task*.yaml`、管理器配置与日志。注意有些文件/日志名因官方版本而变动，必要时用 `find` 实际搜索。

**一个真实的单任务视频例子**（在原云端生成，解压单任务包后路径相应变化）：

```text
evaluate_results/libero_official_wan_test/libero_spatial/videos/
2026_10_15-04_24_57--episode=task0_trial0--success=True--task=pick_up_the_black_bowl_between_the_plate_and_the_r.mp4
```

用户已亲自看过这段视频并确认机器人动作符合预期。

---

## 5. 已验证的软件栈和版本

| 组件 | 历史成功环境（不保证未来版本无变化） |
|---|---|
| 云端 Ubuntu / Python 系统默认 | 终端 `python3 --version` 曾为 Python `3.12.3`；**不是模型 venv 版本** |
| `uv` | `uv 0.9.0`，历史路径 `/usr/bin/uv` |
| Faster-WAM 核心 `.venv` | Python `3.10.18`；`torch 2.7.1+cu128` |
| LIBERO 独立 `.venvs/libero` | Python 3.10 环境；运行评测的 worker 使用 `.venvs/libero/bin/python` |
| MuJoCo | `3.3.2` |
| LIBERO | Git 固定到 commit `8f1084e`（历史确认的短 SHA） |
| GPU 头less 图形 | `MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`；安装了提供 `libEGL.so.1` 的系统库后可渲染 `(64,64,3)` |
| Faster-WAM 任务配置 | `libero_fasterwam_2cam224_1e-4`（两视角、224） |
| 官方 checkpoint | `checkpoints/fasterwam_release/libero/step_021700.pt` |
| 归一化统计 | `checkpoints/fasterwam_release/libero/dataset_stats.json` |

- 官方 Faster-WAM 使用 `uv.lock` 和独立 benchmark 环境隔离依赖；**历史过程中 `pyproject.toml` 与 `uv.lock` 一度被修改，后来已恢复为云端官方版本**，保留了 `*.cloud.bak`/`uv.lock.official.bak` 之类的历史备份。这些备份文件不应被当成官方源代码。
- `third_party/LIBERO` 曾因本地上传方式而不是合法 Git checkout，后改为真实 Git clone 固定 commit `8f1084e`，并有 `third_party/LIBERO.uploaded_backup` 历史残留。未来按官方 `scripts/setup/install_libero.sh` 的 pinned clone 机制处理，不要让旧备份目录冒充 Git 仓库。
- 模拟器必须能调用 EGL，不仅是 PyTorch 能看到 GPU；遇到 `libEGL.so.1` 错误要先补操作系统图形依赖。

---

## 6. 官方 Wan / Faster-WAM 模型文件与一个关键 401 问题

### 6.1 实际成功使用的模型来源

```text
checkpoints/
├── fasterwam_release/libero/
│   ├── step_021700.pt
│   └── dataset_stats.json
└── Wan-AI/
    ├── Wan2.2-TI2V-5B/
    │   ├── Wan2.2_VAE.pth                     # 官方 VAE，下载输出 2.82G
    │   └── models_t5_umt5-xxl-enc-bf16.pth   # 官方 T5 encoder，下载输出 11.4G
    └── Wan2.1-T2V-1.3B/
        └── google/umt5-xxl/
            ├── special_tokens_map.json
            ├── spiece.model
            ├── tokenizer.json
            └── tokenizer_config.json
```

上述 **Wan VAE/T5/tokenizer 曾在云端成功下载并使用；没有明确日志表明这几个大文件也存到了本地 E 盘**。旧 Pod 无 Volume，因 Stop 很可能需要重新下载。不要误把本地 11.12GB 的 Faster-WAM checkpoint 当作已经包含这两个额外 Wan 组件。

### 6.2 关键配置：`model.redirect_common_files=false`

历史 `configs/model/fasterwam.yaml` 中默认：

```yaml
model_id: Wan-AI/Wan2.2-TI2V-5B
tokenizer_model_id: Wan-AI/Wan2.1-T2V-1.3B
redirect_common_files: true
```

默认 `true` 时，历史下载器将一些通用模型文件重定向到 `DiffSynth-Studio/Wan-Series-Converted-Safetensors`，导致访问 VAE 时出现 **HTTP 401 Unauthorized**。最终成功的绕过办法是在**评测命令附加 Hydra override**：

```bash
model.redirect_common_files=false
```

它会改用 `Wan-AI` 官方的 `.pth` 文件。**这不是关闭 Faster-WAM 的未来表征功能**；只是调整共享组件的下载来源。此参数曾在 `manager_config.yaml` 中核实确实为 `false`。

### 6.3 下载额外的模型组件（在新云端的 LIBERO venv 中，需联网）

先完成第 7 节环境安装后，再运行：

```bash
cd /workspace/FasterWAM
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"
export HF_HUB_ENABLE_HF_TRANSFER=0

.venvs/libero/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Wan-AI/Wan2.2-TI2V-5B",
    allow_patterns=[
        "Wan2.2_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
    ],
    local_dir="checkpoints/Wan-AI/Wan2.2-TI2V-5B",
    max_workers=1,
)
snapshot_download(
    repo_id="Wan-AI/Wan2.1-T2V-1.3B",
    allow_patterns=["google/umt5-xxl/*"],
    local_dir="checkpoints/Wan-AI/Wan2.1-T2V-1.3B",
    max_workers=1,
)
PY
```

下载脚本是**依据已成功执行的三次 `snapshot_download` 组合成一个便利版本**；组合执行这一形式本身尚未再次实测。若自动寻址目录不正确，先检查 `DIFFSYNTH_MODEL_BASE_PATH` 和官方 `ModelConfig` 实际寻找位置，不要重复下载。

**重要坑 1：** 如果报 `HF_HUB_ENABLE_HF_TRANSFER=1 ... ModuleNotFoundError: No module named 'hf_transfer'`，在命令前使用 `HF_HUB_ENABLE_HF_TRANSFER=0`；该错误历史上出现过，随后禁用后下载成功。

**重要坑 2：** 手动在 `.venvs/libero/bin/python` 中 `from fasterwam...` 曾报 `ModuleNotFoundError: No module named 'fasterwam'`，从源码手动执行临时代码时用：

```bash
PYTHONPATH="$PWD/src:$PWD" .venvs/libero/bin/python -c 'import fasterwam; print("import OK")'
```

官方评测启动脚本本身能够正确设置加载环境；不要因手动测试进口失败就修改核心源码。

### 6.4 已验证的原始 Wan 模型哈希（历史校验，非 SHA-256）

- `Wan2.2_VAE.pth`：`hash_model_file` 实际与 `WAN22_MODEL_REGISTRY` 期望值一致，均为 `e1de6c02cdac79f8b739f4d3698cd216`。
- `models_t5_umt5-xxl-enc-bf16.pth`：同一加载器识别哈希 `9c8818c2cbea55eca56c7b447df170da`，匹配 `wan_video_text_encoder`。
- 这些是**仓库自定义 `hash_model_file` 的校验值**，不要把它们误记为标准 SHA-256；若文件版本改变，以当前代码注册表校验为准。

---

## 7. 如果旧 Pod 已丢失：从零恢复推理环境（分阶段，不要一口气执行）

> 本节由**官方安装说明 + 已成功历史配置**整理。**并非每一条都在新 Pod 上重新实测**。下一位 AI 应按用户习惯一次提供少量命令并检查输出；不要立即开启整批训练/下载。

### 阶段 A：新建 GPU 和检查

1. 选择适合的 GPU；历史成功机型 **A6000 48GB**，这是一个**成功实例**而非最低显存保证。先看当前总成本、磁盘可用空间、GPU 可用性。
2. 历史需要容纳 11GB Faster-WAM checkpoint + 2.82GB VAE + 11.4GB T5 + Python/CUDA/LIBERO 依赖；120GB 容器盘足够当次运行，但**若无 Volume，Stop 后数据会丢失**。若将来需要反复使用 GPU，应事先选择合理的持久化卷并衡量闲置费用。
3. 创建后获取**新** SSH 地址、端口、Pod ID；不要假定仍是 `216.81.151.15:14697`。
4. 在新云端确认 `nvidia-smi`、`df -h /workspace`、`python3 --version`、`uv --version`、系统图形依赖等。当前基础镜像与 CUDA 驱动可能改变。

### 阶段 B：同步/恢复源代码

方式 1：直接在新云端拉取官方源码（建议记录 Git commit）：

```bash
cd /workspace
git clone https://github.com/hustvl/FasterWAM.git
cd FasterWAM
git rev-parse HEAD
```

**复现精度注意**：此前成功运行的 Faster-WAM 主仓库完整 Git SHA **没有从聊天日志中得到明确记录**；当前 GitHub HEAD 可能更新。下次可先查看本地 E 盘 checkout 的 `git rev-parse HEAD`，优先复用相同版本以保证结果可比；切勿编造旧版本 commit。

方式 2：同步本地项目（如果需要保留自己的代码改动），由**本地 WSL**运行，按新 SSH 信息替换占位符。**避免直接上传 `.venv` / `.venvs` / 缓存/大结果视频**：

```bash
cd /mnt/e/Innovative_practice/WAM/FasterWAM
rsync -avhP --no-perms --no-owner --no-group \
  --exclude='.venv/' --exclude='.venvs/' --exclude='.git/' \
  --exclude='checkpoints/' --exclude='evaluate_results/' \
  --exclude='experiment_backups/' --exclude='/data/' \
  -e 'ssh -i ~/.ssh/id_ed25519_runpod -p <NEW_PORT>' \
  ./ root@<NEW_HOST>:/workspace/FasterWAM/
```

**上述 rsync 命令是模板，尖括号必须替换，不可直接复制执行。** 其中 `--exclude='/data/'` 的起始 `/` 用于限定排除同步根目录下的训练数据文件夹，**不要**改回 `--exclude='data/'`，否则可能连嵌套的 `configs/data/` 也一并过滤。

**已知严重坑：** `--exclude='data/'` 在先前同步时还意外过滤了 `configs/data/`，导致 Hydra 配置缺失。无论使用何种传输方式，都应在新云端确认：

```bash
cd /workspace/FasterWAM
ls configs/data
ls configs/model/fasterwam.yaml configs/task/libero_fasterwam_2cam224_1e-4.yaml
```

若 `configs/data` 缺失，从**本地 WSL**单独恢复（使用新连接信息）：

```bash
cd /mnt/e/Innovative_practice/WAM/FasterWAM
rsync -avhP --no-perms --no-owner --no-group \
  -e 'ssh -i ~/.ssh/id_ed25519_runpod -p <NEW_PORT>' \
  configs/data/ root@<NEW_HOST>:/workspace/FasterWAM/configs/data/
```

如只用官方 `git clone` 一般不应出现这个 rsync 排除问题。

### 阶段 C：安装依赖（云端）

官方说明要求先安装 `uv`，历史使用 `uv 0.9.0`；请优先使用官方安装指引：https://docs.astral.sh/uv/。然后从项目根目录：

```bash
cd /workspace/FasterWAM
bash scripts/setup/install_core.sh
bash scripts/setup/install_libero.sh
```

安装成功后检查：

```bash
.venv/bin/python -V
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
.venvs/libero/bin/python -V
.venvs/libero/bin/python -c 'import mujoco; print(mujoco.__version__)'
git -C third_party/LIBERO rev-parse --short HEAD
```

目标历史配置：核心 Python 3.10、torch 2.7.1+cu128；LIBERO commit `8f1084e`；MuJoCo 3.3.2。若 CUDA 版本/驱动不同，按当前仓库锁定依赖解决，不要机械地修改官方 `uv.lock`。

对 EGL 库的历史处理：曾需安装提供 `libEGL.so.1` 的系统包（Ubuntu 常见为 `libegl1`；具体包名依镜像确认），然后使用：

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

历史确认过离屏图像形状 `(64, 64, 3)`；重新部署请先重做 EGL 小测试再启动大模型。**不能保证仅一条 apt 命令解决所有可能镜像的 EGL 依赖。**

### 阶段 D：上传/checkpoint 和 Wan 模型

若本地 checkpoint 仍在，**本地 WSL**先检查：

```bash
cd /mnt/e/Innovative_practice/WAM/FasterWAM
ls -lh checkpoints/fasterwam_release/libero/step_021700.pt
ls -lh checkpoints/fasterwam_release/libero/dataset_stats.json
```

曾经成功使用的上传方式（将端口/IP换为新实例）：

```bash
rsync -avhP --no-perms --no-owner --no-group \
  -e 'ssh -i ~/.ssh/id_ed25519_runpod -p <NEW_PORT>' \
  /mnt/e/Innovative_practice/WAM/FasterWAM/checkpoints/fasterwam_release/libero/ \
  root@<NEW_HOST>:/workspace/FasterWAM/checkpoints/fasterwam_release/libero/
```

**曾经实际上传 11.12G 花费 34m48s、平均约 5.08 MB/s**，因此 GPU 已启动就可能在上传时产生租金。也可参考官方发布仓库直接从云端下载：

```bash
# [官方说明] 二选一：如果本地上传不划算，可在新云端使用 Hugging Face 下载
huggingface-cli download hustvl/FasterWAM \
  --local-dir checkpoints/fasterwam_release
```

这条官方命令默认可能同时下载 RoboTwin checkpoint，若只用 LIBERO 应考虑使用 `--include` 或对应的 HF Python API 选择文件，避免额外费用/磁盘。**上面这条“只下 LIBERO”优化还未在新环境测试**。

随后按第 6.3 节下载 Wan 2.2 VAE/T5 + Wan 2.1 tokenizer。设置：

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/workspace/FasterWAM/checkpoints
export HF_HUB_ENABLE_HF_TRANSFER=0
```

### 阶段 E：一条命令验证模型评测

使用第 8 节经过验证的单任务完整命令。先得到 **1/1** 成功以及 MP4；然后 `dry_run` 看 40 任务调度；最后才正式评测。

---

## 8. **已验证可用**：官方检查点单任务闭环推理命令

以下是在原云端 `/workspace/FasterWAM` 已成功运行的关键参数组合。执行环境为 **新建/恢复好的 Runpod Ubuntu 终端**，不是本地 WSL；要先确保第 6 节模型组件都可访问。

先构建任务文件（每行 `suite_name,task_id`）：

```bash
cd /workspace/FasterWAM
mkdir -p .runtime/eval_tests
printf 'libero_spatial,0\n' > .runtime/eval_tests/one_task.txt
```

启动评测：

```bash
cd /workspace/FasterWAM
HF_HUB_ENABLE_HF_TRANSFER=0 \
CKPT_PATH="$PWD/checkpoints/fasterwam_release/libero/step_021700.pt" \
DATASET_STATS_PATH="$PWD/checkpoints/fasterwam_release/libero/dataset_stats.json" \
NUM_GPUS=1 MAX_TASKS_PER_GPU=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
bash scripts/eval_fasterwam_libero.sh \
  MULTIRUN.task_file="$PWD/.runtime/eval_tests/one_task.txt" \
  MULTIRUN.create_only=false MULTIRUN.dry_run=false \
  model.redirect_common_files=false \
  EVALUATION.num_trials=1 \
  EVALUATION.visualize_future_video=false \
  EVALUATION.output_dir="$PWD/evaluate_results/libero_official_wan_test"
```

**历史成功输出**：

```text
Tasks completed: 1
Total attempts: 1
Successful attempts: 1
Success rate: 100%
Total time: approximately 37.23 seconds
```

任务指令：`pick up the black bowl between the plate and the ramekin and place it on the plate`。用户在本地浏览视频后明确确认行为符合预期。

**命名和覆盖风险**：下次评测若想与历史结果并存，给 `EVALUATION.output_dir` 换新的 run 名称，不要意外覆盖解压回来的数据。

---

## 9. **已验证可用**：四任务集 × 每集 10 任务的完整 smoke test

### 9.1 创建 40 任务文件

原云端执行过：

```bash
cd /workspace/FasterWAM
mkdir -p .runtime/eval_tests
for suite in libero_spatial libero_object libero_goal libero_10; do
  for i in $(seq 0 9); do
    echo "$suite,$i"
  done
done > .runtime/eval_tests/all_40_tasks.txt
wc -l .runtime/eval_tests/all_40_tasks.txt
head -3 .runtime/eval_tests/all_40_tasks.txt
tail -3 .runtime/eval_tests/all_40_tasks.txt
```

**实际输出**：`40 .runtime/eval_tests/all_40_tasks.txt`，从 `libero_spatial,0` 到 `libero_10,9`。

### 9.2 先 dry-run，不推理

```bash
cd /workspace/FasterWAM
HF_HUB_ENABLE_HF_TRANSFER=0 \
CKPT_PATH="$PWD/checkpoints/fasterwam_release/libero/step_021700.pt" \
DATASET_STATS_PATH="$PWD/checkpoints/fasterwam_release/libero/dataset_stats.json" \
NUM_GPUS=1 MAX_TASKS_PER_GPU=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
bash scripts/eval_fasterwam_libero.sh \
  MULTIRUN.task_file="$PWD/.runtime/eval_tests/all_40_tasks.txt" \
  MULTIRUN.chunk_size=1 \
  MULTIRUN.create_only=false MULTIRUN.dry_run=true \
  model.redirect_common_files=false \
  EVALUATION.num_trials=1 \
  EVALUATION.visualize_future_video=false \
  EVALUATION.output_dir="$PWD/evaluate_results/libero_40tasks_smoke" \
  > /workspace/libero_40tasks_dry_run.log 2>&1

grep -E 'Starting evaluation|chunk_size=|gpu_ids=|DRY_RUN|Traceback|Error' \
  /workspace/libero_40tasks_dry_run.log | head -20
```

**真实 dry-run 输出**：

```text
Starting evaluation: tasks=40 chunks=40 chunk_size=1 gpu_ids=['0']
DRY_RUN=true; first worker command:
/workspace/FasterWAM/.venvs/libero/bin/python \
  /workspace/FasterWAM/experiments/libero/eval_libero_chunk.py ... \
  --first-suite libero_spatial --first-task-id 0 --gpu-id 0 --num-trials 1 ...
```

特别说明 `MAX_TASKS_PER_GPU=1` 控制并行中的 chunk 数；主屏输出 `Launched ... (1/1)` 是**1 个并发槽位**，不等于只会跑 1 个任务。`MULTIRUN.chunk_size=1` 导致 40 个 chunk，一次调度一个。

### 9.3 正式运行 40 × 1 次

```bash
cd /workspace/FasterWAM
HF_HUB_ENABLE_HF_TRANSFER=0 \
CKPT_PATH="$PWD/checkpoints/fasterwam_release/libero/step_021700.pt" \
DATASET_STATS_PATH="$PWD/checkpoints/fasterwam_release/libero/dataset_stats.json" \
NUM_GPUS=1 MAX_TASKS_PER_GPU=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
bash scripts/eval_fasterwam_libero.sh \
  MULTIRUN.task_file="$PWD/.runtime/eval_tests/all_40_tasks.txt" \
  MULTIRUN.chunk_size=1 \
  MULTIRUN.create_only=false MULTIRUN.dry_run=false \
  model.redirect_common_files=false \
  EVALUATION.num_trials=1 \
  EVALUATION.visualize_future_video=false \
  EVALUATION.output_dir="$PWD/evaluate_results/libero_40tasks_smoke"
```

**这条命令真实执行完毕，不是“示例输出”。** 控制台从 `chunk_000000` 到 `chunk_000039` 均出现 `Completed ... on GPU 0`，并打印统计汇总。

### 9.4 真实结果（重要的、可复查的阶段性数据）

| 套件 | 已完成任务 | 尝试 | 成功 | 成功率 | 总任务时间 | 平均每任务 | 最大单任务时间 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `libero_spatial` | 10 | 10 | 10 | 100.00% | 04m14s | 25.39 s | 28.34 s |
| `libero_object` | 10 | 10 | 10 | 100.00% | 04m37s | 27.72 s | 31.83 s |
| `libero_goal` | 10 | 10 | 10 | 100.00% | 03m46s | 22.55 s | 30.07 s |
| `libero_10` | 10 | 10 | 10 | 100.00% | 06m45s | 40.46 s | 53.05 s |
| **总体** | **40** | **40** | **40** | **100.00%** | **19m21s** | **29.03 s** | **53.05 s** |

四个套件的时间逐项显示按分秒四舍五入，总和可能与 19m21s 存在约 1 秒的舍入偏差；以原始 `summary.json`/`summary.csv` 为准。

**统计学解释：** 40/40 指“40 个任务的**各一次**尝试成功”；每个任务记录的 `100%` 实际是 `1/1`，不能声称不同初始状态/随机种子下的稳健成功率是 100%，不能声称复现论文报告的正式基准数值。

### 9.5 真实输出文件

```text
/workspace/FasterWAM/evaluate_results/libero_40tasks_smoke/
├── summary.json
├── summary.csv
├── task_success_rates.csv
├── manager_config.yaml
├── task_chunks/
├── task_logs/                 # 如版本产生该目录
├── libero_spatial/
├── libero_object/
├── libero_goal/
└── libero_10/
```

各 `libero_*` 下包括 `videos/`、`gpu0_taskN_results.json`、`eval_config_gpu0_taskN.yaml` 等。历史备份 `tar -tzf ... | tail -5` 已证实存在 `libero_10/gpu0_task9_results.json`、`summary.csv`、`task_success_rates.csv`、`summary.json`。

### 9.6 监测方式

历史使用**Windows CMD 的 SSH 会话运行评测，VS Code 中第二个远程终端监测**：

```bash
cd /workspace/FasterWAM
nvidia-smi
ls -lt evaluate_results/libero_40tasks_smoke/task_logs/ | head -10
```

查看具体日志时使用 `tail -f <存在的日志文件路径>`；日志文件数量 **不等于**成功任务数。对于耗时任务推荐 `tmux`/`screen` 保持会话（**当时是否使用 tmux 无证据**）。

---

## 10. 两次成功结果如何从云端转移到了本地

### 10.1 单任务包

原云端打包：

```bash
cd /workspace/FasterWAM
tar -czf /workspace/fasterwam_first_success.tar.gz \
  -C /workspace/FasterWAM \
  evaluate_results/libero_official_wan_test \
  .runtime/eval_tests/one_task.txt
```

历史大小约 `141KB`；用户使用 `scp` 从旧 Pod 下载到 E 盘，`tar -tzf` 确认有任务结果/MP4。用户打开视频后反馈“符合预期”。

### 10.2 全 40 任务包

原云端打包（**已执行并成功**）：

```bash
cd /workspace/FasterWAM
tar -czf /workspace/fasterwam_libero_40tasks_smoke.tar.gz \
  -C /workspace/FasterWAM evaluate_results/libero_40tasks_smoke
ls -lh /workspace/fasterwam_libero_40tasks_smoke.tar.gz
# 历史结果：8.5M
```

首次从本地 WSL 使用默认 `scp` 显示 `0% ... stalled`，长时间没有数据传输；改成传统 SCP 协议 `-O` 成功。这是**已验证**的解决办法：

```bash
# 仅历史演示！Pod 已停止，下述旧地址现在不应直接使用。
cd /mnt/e/Innovative_practice/WAM/FasterWAM
scp -O \
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  -i ~/.ssh/id_ed25519_runpod -P 14697 \
  root@216.81.151.15:/workspace/fasterwam_libero_40tasks_smoke.tar.gz \
  experiment_backups/
```

实际输出：`100% 8684KB 883.9KB/s 00:09`，再执行：

```bash
ls -lh experiment_backups/fasterwam_libero_40tasks_smoke.tar.gz
tar -tzf experiment_backups/fasterwam_libero_40tasks_smoke.tar.gz | tail -5
```

实际验证成功，大小约 `8.5M`。`scp -O` 对应的是传统 SCP，属于本次有效的兼容性修复，不代表所有新云端都需要它。**绝不因为旧 SSH 不可连而重复跑实验：E 盘已有压缩包。**

### 10.3 用户最后的本地解压输入错误

用户曾在本地终端把 `cd` 与目录拆为两行、把 `tar` 与 `-xzf` 拆为两行，分别出现：

```text
-bash: /mnt/e/Innovative_practice/WAM/FasterWAM: Is a directory
tar: You must specify one of the '-Acdtrux' ... options
-xzf: command not found
```

助手随后给了逐行正确命令；**没有新的用户消息确认解压成功**。下次 AI 不要据此认定结果压缩包坏了。正确做法见第 4.3 节。

---

## 11. 典型报错、原因与已验证修复

| 故障表现 | 原因或证据 | 历史上有效的处理 |
|---|---|---|
| `wsl` 进入 `docker-desktop` | 默认发行版不是 Ubuntu | Windows CMD 用 `wsl -d Ubuntu` |
| Hydra 配置/缺少 `configs/data` | 同步时 `--exclude='data/'` 连 `configs/data` 一起排除 | 单独 rsync `configs/data/` 或完整官方 clone |
| `third_party/LIBERO` Git source revision 验证失败 | 原目录为上传拷贝，不是合格 Git checkout | 正式 Git clone 并 checkout commit `8f1084e` |
| `uv.lock` 与 `pyproject.toml` 非官方版本 | 云端安装排错期间曾修改这些文件 | 恢复官方版本再运行安装脚本，不把 debug 补丁当基线 |
| `libEGL.so.1` 缺失 / headless MuJoCo 渲染失败 | 容器缺系统 EGL 库 | 安装相应 EGL 系统包；`MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` |
| `401 Unauthorized` 访问 `DiffSynth-Studio/Wan-Series-Converted-Safetensors` | 默认 `redirect_common_files=true` 重定向至无法访问文件 | 评测传 `model.redirect_common_files=false`，使用官方 Wan `.pth` |
| `hf_transfer` 未安装 | 环境设置 `HF_HUB_ENABLE_HF_TRANSFER=1` | 下载/评测前设 `HF_HUB_ENABLE_HF_TRANSFER=0` |
| 手动导入 `fasterwam` 报 ModuleNotFoundError | LIBERO env 手动测试缺源码搜索路径 | `PYTHONPATH="$PWD/src:$PWD"` |
| `NUM_GPUS` 默认为 8 | 官方脚本为多 GPU 情况设计 | 单卡设置 `NUM_GPUS=1` |
| chunk 意外并发加载导致显存风险 | 默认 `MAX_TASKS_PER_GPU` 可能不是 1 | 单卡先设 `MAX_TASKS_PER_GPU=1`，`chunk_size=1` |
| `scp` 停在 `0% stalled` | 历史 SCP 默认传输模式卡住 | `scp -O`，加 SSH ServerAlive 参数 |
| `tar` / `cd` 报 Bash 命令行错误 | 用户把单条命令拆断 | 保持一行命令完整，如 `tar -xzf ...` |
| SSH 退出后仍在计费 | SSH session 与 Runpod Pod 生命周期不同 | 在网页 Stop/Terminate，核实 `$0.00/hr` |
| Stop 后原云环境无法恢复 | **没有配置持久化 Volume**，停止弹窗警告数据丢失 | 以本地 E 盘代码、checkpoint 和结果备份重建 |

**重要排错原则**：先保留原始报错、排查环境与路径，再考虑改核心代码；避免因一次 401 或 import error 直接重训练模型、重新下载所有权重。

---

## 12. 下一阶段的严谨评测计划（全部 `[待完成]`）

### 12.1 不要把 smoke test 当成论文级复现

- 当前只有 `40 × 1 = 40` 次机器人 episode。成功率 `100%` 是单次初始条件下的结果；成功不说明统计稳健性。
- 官方本地 `configs/sim_libero.yaml` 历史配置默认 `EVALUATION.num_trials: 50`，但**用户还没运行 50 次版本**。
- 原先建议：**先 40×10=400 次**确认稳定性，再 40×50=2000 次正式统计；也可在确认并行与恢复能力后规划直接完整测评，但应先完成下面的代码审查。
- 下一次在论文图表中应展示**每套件成功率、任务间变化、随机初始状态数量、总 attempts、失败案例、时延/峰值显存**；与官方论文数据对比前须匹配采样、环境版本、分辨率和评测协议。

### 12.2 随机/初始状态：已经从旧源码日志看到一个关键事实

原始代码搜索显示：

```python
# experiments/libero/eval_libero_chunk.py（历史代码的关键逻辑）
initial_states = task_suite.get_task_init_states(task_id)
while len(initial_states) < num_trials:
    initial_states.extend(initial_states[: (num_trials - len(initial_states))])
```

`eval_libero_single.py` 中也有同类扩展初始化状态的逻辑，并运行 `for trial_idx in range(int(cfg.EVALUATION.num_trials))`。

**推论（待实测核实）**：评测基于 LIBERO 提供的任务初始状态序列；若要求的 `num_trials` 超过提供的初始状态数量，代码将**重复已有初始状态**，因此“50 次”不必然等于 50 个独立随机场景。下阶段应核对每个任务初始状态数量、trial 使用顺序、是否需要固定随机种子及是否重复，确认官方协议后再写论文。

建议让下一位 AI 先检查：

```bash
cd /workspace/FasterWAM
grep -nE 'seed|init_state|initial_states|num_trials|range\(.*trial|resume|skip_existing' \
  experiments/libero/eval_libero_chunk.py \
  experiments/libero/eval_libero_single.py \
  experiments/libero/run_libero_manager.py | head -120
```

> 这一检查是前一轮聊天规划中的**下一步**，用户表示“暂时就做这么多”，因此并未继续正式验证 50 trial 的逐次初始状态。

### 12.3 分块、续跑和省钱

- 旧源码 `run_libero_manager.py` 有 `MULTIRUN.chunk_size`、`MULTIRUN.max_tasks_per_gpu`、`MULTIRUN.dry_run`、任务列表读取等逻辑；最初 `grep` 未发现 `resume`、`skip_existing` 等明确字段。**没有证据证明原管理器能安全自动断点续跑**。
- 曾成功用 `chunk_size=1` 和 `MAX_TASKS_PER_GPU=1` 串行执行所有 40 chunks。若要降低重复模型初始化开销，可以**考虑测试** `chunk_size=10` 或“每套件一个 task_file”等设计，但**尚未实测性能与结果完整性**。不能把其写成已实现的节约优化。
- 为保证可恢复性，建议按套件/任务组分别输出独立 `EVALUATION.output_dir`；每批完成立刻 `tar` 并同步到本地。聚合统计时使用各批独立结果，不要简单覆盖 `summary.csv`。
- 对长任务采用 `tmux`、设置失败策略、检查日志和输出文件；**目前仅有串行 40×1 测试的明确成功记录**。

### 12.4 仅供预算估计（不是承诺）

从 40×1 的 `19m21s`、平均每 episode 约 `29.03 s` 推算：

| 规模 | 推理次数 | 按测得累计任务时间线性估算（不含额外启动/开销） |
|---|---:|---:|
| 当前已做 | 40 | 19m21s |
| 10 trial/任务 | 400 | 约 3.23 h |
| 50 trial/任务 | 2000 | 约 16.13 h |

**这不是实际整趟墙钟时间。** 模型反复加载、环境初始化、不同任务/失败 episode 时长、IO、运行中的意外错误，均可能显著增加耗时；新租赁价格也需重新确认。历史云 GPU 价格约 `$0.61/hr`，仅参考。此前曾有粗略建议对正式任务保留比理论值更宽的 24～48 小时预算，**没有实测支持这个区间的准确性**。

---

## 13. 创新研究：下一位 AI 应如何推进（全部 `[待完成]`）

建议按以下依赖顺序，而不是马上实现大型新网络：

1. **基线冻结**：先归档源码 commit、checkpoint 哈希/尺寸、`manager_config.yaml`、Wan 组件版本、LIBERO commit 和 `mujoco` 版本；在同配置下至少重现单任务/40×1，确保后续可比较。
2. **代码理解**：阅读 `src/fasterwam/models/`、`src/fasterwam/runtime.py`、`configs/model/fasterwam.yaml`、`configs/task/libero_fasterwam_2cam224_1e-4.yaml`、推理路径里的 SparseMoT / future cache / Interval KV-Fusion，画出“观测 → 视频/未来表征 → 动作专家 → 机器人控制”数据流，并查每个张量 shape、更新频率、梯度流。
3. **廉价诊断实验**：先测推理平均时延、峰值显存、未来缓存调用次数/融合层数；再做 `future on/off`、融合间隔、融合层位、条件门控阈值等小消融。不要假设简单取消未来特征就是可比的 Fast-WAM；要确认结构/权重可比较。
4. **提出可验证假设**：例如任务简单或观测稳定时降低未来特征更新频率；遮挡/长时序/扰动场景增加未来条件化力度。明确门控输入（状态变化、置信度、视觉差异）以及可测开销。
5. **优先无需训练/小规模训练的可行实验**：先做推理时动态调用策略或冻结大骨干训练一个轻量门控；**这只是建议，尚未证明兼容，也未实现**。如果改结构，检查 checkpoint 权重映射、维度和 loss。
6. **公平定量对比**：官方 Faster-WAM、固定频率/固定层数消融、拟议动态方法；在 LIBERO 标准多初始状态，条件允许再扩到 LIBERO-Plus OOD、RoboTwin。记录 `success rate / wall clock / action inference latency / peak VRAM / FLOPs 或 forward-count`，至少重复稳定统计。
7. **论文与创新实践材料**：实验协议、可复现脚本、消融、失败分析、对比图、录屏（MP4）和工作日志。不能凭 40/40 smoke test 推断 OOD 泛化/新方法性能。

**数据集与训练说明**：如果需要训练，官方 README 提到将 `yuanty/LIBERO-fastwam` 四套预处理数据解压至 `data/libero_mujoco3.3.2/`，预计算文本嵌入 `python scripts/precompute_text_embeds.py task=libero_fasterwam_2cam224_1e-4`，从 Wan2.2 初始化 SparseActionDiT，再按官方训练脚本训练。**用户尚未做这些步骤；不要误以为训练数据已就绪**。官方示例使用多 GPU，不能因单卡评测成功就假设单卡 48GB 能完整训练。

---

## 14. 下一次与 AI 对话时的最佳开场白（可直接复制）

> 我在 Windows + WSL2 Ubuntu 的 `E:\Innovative_practice\WAM\FasterWAM` 开展 Faster-WAM / MuJoCo / LIBERO 机械臂科研。请先读我提供的 `README.md`。我**已**使用官方 `step_021700.pt` 在历史 Runpod A6000 48GB 上完成 LIBERO 四套共 40 个任务 × 每任务 1 次，结果 40/40，日志累计任务时间 19m21s；这**不是训练，也不是 50 trial 正式基准**。E 盘存有 checkpoint 和 `experiment_backups/fasterwam_libero_40tasks_smoke.tar.gz`（已校验）。历史 Pod 已 Stop、费用 `$0.00/hr`、无 Volume、云数据可能丢失。请不要让我反复做已成功的事情，也不要直接启动有 GPU 费用的任务。一次只给我一到两步命令，并告诉我在哪个终端运行，等我发输出后再继续。现在我希望先做：〔填入本次具体目标，如本地视频分析 / 代码理解 / 重新租 GPU / 正式 50 trial 设计 / 改进模块〕。

---

## 15. 复现实验与证据审计记录（避免下一位 AI 误读）

| 核对项 | 来源（此前对话中的原始证据） | 审计判断 |
|---|---|---|
| 项目本地路径 | 多次本地 WSL 终端 `cd /mnt/e/Innovative_practice/WAM/FasterWAM` | 已验证 |
| 11.12G 检查点上传 | `rsync -avhP` 日志 `step_021700.pt 11.12G 100%`，约 34m48s | 已验证当时上传；本地删除与否未重新检查 |
| LIBERO/YAML 环境 | 多次 `configs/sim_libero.yaml`、`experiments/libero/...` 源码输出；核心/独立环境成功执行 | 已验证旧云端；新云端需重建 |
| 401 和修复 | `redirect_common_files` 搜索、注册表下载来源与成功单任务命令 | 已验证 `false` 可运行 |
| VAE/T5 | `snapshot_download` 输出和 `hash_model_file` 校验结果 | 已验证旧云端；本地是否缓存未知 |
| 单任务 | `Tasks completed:1; Successful attempts:1`，MP4 用户主观验证 | 已验证 |
| 40 任务文件 | `wc -l` 输出 `40`，首尾正确 | 已验证 |
| 40 任务 dry-run | `Starting evaluation: tasks=40 chunks=40 chunk_size=1 gpu_ids=['0']` | 已验证 |
| 40 任务真实结果 | `chunk_000000` 至 `chunk_000039` 均完成；四套 10/10；总计 19m21s | 已验证、仅 1 trial/任务 |
| 两次云端打包 | `tar -czf` 命令与 `ls -lh` | 已验证 |
| 全 40 任务备份落地 E 盘 | `scp -O` 100% 8684KB；`ls -lh` 和 `tar -tzf` 成功 | 已验证；不是完整云端环境备份 |
| 本地解压评测 | 用户输入命令时发生换行错误，之后无最终确认 | **待核验** |
| 50 trial/任务、初始状态完全独立 | 仅观察到配置默认 50 与扩展初始状态的代码，未运行 2000 trials | **未验证** |
| 自动断点续跑 | 旧 manager grep 没有找到显式 `resume`/`skip_existing` | **不能保证** |
| 云端费用停止 | 末次 Pod 控制台截图 `FasterWAM-LIBERO ... $0.00/hr` | 已确认该 Pod 停止；其他资源需独立检查 |
| 云端保留情况 | Stop 确认弹窗 `no volume configured. ALL DATA will be lost!` | 必须按不可恢复对待 |
| 官方源码版本 commit | 聊天没有给出 Faster-WAM 主仓库完整 commit | **未知，勿编造** |
| 新创新模块和训练 | 没有训练命令或修改模块后实验记录 | **未开始** |

---

## 16. 未来更完整的产物清单（建议）

后续如果计划论文级交付，建议**另建实验日志和配置快照**（不要覆盖本 README）：

```text
project_notes/
  environment_YYYYMMDD.md       # 新 Pod 型号、费用、GPU driver、版本、git SHA
  experiment_registry.csv       # run_id, git_sha, checkpoint, seed, tasks, trials, timing, status
  architecture_notes.md         # SparseMoT / future fusion 结构和张量分析
  ablation_plan.md              # 假设、对照、预算、停止条件
experiment_backups/
  ...                            # 每次训练/评测结束都同步 tar.gz 到 E 盘
```

**安全、成本与数据治理**：不要公开 SSH 私钥、Runpod token、HF token、带签名的下载地址；不要把大 checkpoint/下载缓存提交 Git；不要在长任务运行前忽略数据持久化；完成后通过 Runpod 网页确认停机状态、费用与卷，退出 SSH 不等于关机。

---

## 17. 精简结论

**LIBERO 已经验证成功的是 Faster-WAM 官方 LIBERO 检查点推理评测；A3 当前只完成了推理到控制的短闭环原型，尚无真实饼干抓取成功率；从未自定义训练/微调模型。** 历史单张 RTX A6000 上，40 个 LIBERO 任务各测一次全部成功，总统计任务时间 19m21s；原始结果与视频已压缩并下载到 E 盘。Runpod Pod 已停止并显示 `$0.00/hr`，无持久化存储卷，不应依赖旧云端环境。下一阶段**先做本地 A3 不同 seed 的控制器复测与几何随机化核验，再验证夹爪动作修复和短闭环，之后做 Faster-WAM 在 A3 的零样本饼干抓取成功率，必要时才进行 A3 演示微调**；自适应未来表征 / Dynamic SparseMoT 仍为更后续的研究方向。**所有改进与大规模评测仍待开始。**


---

## 18. A3 饼干项目：最新任务、Python 版本与准确路径（2026-10-21 新增）

### 18.1 三套代码/环境绝对不要混淆

```text
Windows E:\Innovative_practice\WAM\
├── FasterWAM/                       # 官方 Faster-WAM 源码、LIBERO checkpoint、历史结果归档
├── a3_dual_arm_sim/                 # 同学维护的 A3 仿真，git main；本地 commit 暂未读到
├── fasterwam_a3_bridge/            # 用户此前手写的适配/测试代码、.venv、outputs/ 等
├── fasterwam_a3_bridge.zip          # 用户实际已有，197446732 字节；不是本交接包
└── FasterWAM_A3_Handoff_2026-10-21/ # 用户截图时尚不存在；正确交接 ZIP 解压后才出现
```

WSL 下对应 `/mnt/e/Innovative_practice/WAM/...`。A3 与 Faster-WAM 都是**各自独立的 Git 源码项目**；`fasterwam_a3_bridge` 是**用户定制适配层**，不应覆盖任一官方代码仓库。

- **A3**：当前公开 main 的 [`pyproject.toml`](https://github.com/Eter0109/a3_dual_arm_sim/blob/main/pyproject.toml) 明确 `requires-python = ">=3.12"`、`numpy>=2,<3`、`mujoco>=3.3,<4`；官方 README 推荐 Python 3.12。A3 可能在本地 Windows `.venv\Scripts\python.exe` 或 WSL `.venv/bin/python`；**尚未收到本地 `python -V`/venv 路径校验输出**。下一位 AI 先检查实际可用解释器，切勿猜。
- **Faster-WAM**：历史已成功 Python 3.10.18 的 `.venvs/libero` 以及核心 `.venv`，`torch 2.7.1+cu128`。**不要尝试把 A3 NumPy 2.x 和旧 LIBERO NumPy 1.26 强行装进一个解释器，也不应全局降级用户 Python 3.12。**
- **隔离**：云端曾使用 `FasterWAM/infer_a3_worker.py`（Python 3.10，GPU），`fasterwam_a3_bridge/run_a3_closed_loop.py`（A3 Python 3.12），交换 `FasterWAM/ipc/observation.npz` 和 `actions.npz`；通过 `request.flag` / `done.flag` 单请求握手。
- **物理模型**：A3 仿真是**双臂**，每臂 7 个关节及一套 Robotiq 类型夹爪，非 LIBERO Panda 原生同构机体；单臂 Faster-WAM checkpoint 无法原生共同控制双臂。**视觉和末端动作适配能解决软件集成，不保证零样本策略成功。**

### 18.2 请准确识别任务层级

- 同学的仓库内 `A3CookieTransferEnv` 是**80 块来源饼干、默认目标 10 块、2×5 装盒**的完整仿真任务；另有 2×10 / 20 块 benchmark 和专家演示接口。不要把“抓取一块”与“完成 10 块装盒”混称。
- 当前 Faster-WAM 提示语是 `Pick up the cookie with the left robot arm.`，历史更早一次输出 NPZ 使用 `Pick up the cookie with the left robot arm and transfer it to the right robot arm.`；提示语版本不一，比较成功率时必须固定。
- **第一阶段零样本科学目标**：左臂靠近 → 接触并夹住 → 抬起并保持**一块饼干**（必须制定新的独立抓取评分），不承诺自然完成左、右臂交接或 2×5 10 块装盒。**完整 A3 任务成功由 `info["success"]` / 真实 `cookies_in_target` 与官方槽位规则判定，不等于首次抓住一块**。
- 公共代码、契约文档：https://github.com/Eter0109/a3_dual_arm_sim ，`docs/contracts.md`、`docs/randomization.md`、`configs/cookie_batch.yaml` 和 `configs/randomization.yaml`。未经确证不要基于旧聊天发明双臂指令映射。

## 19. 本交接包的真实文件清单与是否可恢复

### 19.1 `bridge_snapshot/`：28 个无重复的 `.py`（全部来自用户上传）

```text
# 动作 + 观测/视觉桥
  action_adapter.py
  action_adapter_v2.py
  action_adapter_v3.py
  observation_adapter.py
  gripper_proprio_adapter.py
  vision_adapter.py
  vision_adapter_official.py
  export_fasterwam_a3_input.py
# 数值诊断
  diagnose_fk_v2.py
  diagnose_ik_reference.py
  diagnose_joint_tracking.py
  diagnose_precision.py
  diagnose_torque.py
  inspect_full_proprio.py
  inspect_proprio_normalization.py
# 演示/验证
  test_adapter.py
  test_adapter_v2.py
  test_cookie_bridge_video.py
  test_cookie_combined_v3.py
  test_cookie_gripper_video.py
  test_cookie_rotation.py
  test_cookie_visible_v3.py
  test_fk_consistency.py
  test_gain_sweep.py
  test_ik_diagnostic.py
  test_mujoco_motion.py
  test_observation.py
  test_orientation_consistency.py
```

用户后续上传时某些文件带 `(1)`、`(2)`、`(3)` 后缀；本包将**字节相同**的重复上传归一为原文件名，未覆盖修改后的内容。经过去重的 5 组有：`diagnose_precision`、`test_observation`、`test_orientation_consistency`、`vision_adapter`、`vision_adapter_official`。本包内 28 文件均可通过 Python `ast.parse` **语法检查**；这不等于 GPU 或 MuJoCo 物理执行成功。

### 19.2 `recovered_historical/`：从完整聊天原文精准恢复的、非直接上传脚本

- `infer_a3_worker.py`：历史用于模型加载、官方 Processor、加载 8D proprio、推理 `infer_action`、官方反归一化 + 夹爪变换、保存 `actions.npz`；原对话有两段仅输出文案标点不同的代码，这里保存较早的 ASCII 连字符版本。
- `run_a3_closed_loop.py`：历史用于启动 A3、不断观测/重推理、每块执行前 5 步、录像；硬编码 `seed=0`、`REPLAN_STEPS=5`、`MAX_STEPS=400`、`SETTLE_STEPS=20`，原始输出 `/workspace/a3_closed_loop_cookie_grasp.mp4`。
- **证据级别**：两份源代码已从 2026-10-20 聊天全文确实提取并通过语法解析，且历史终端有 worker ready 和 3 次动作步日志；**并未在本次沙盒的 CUDA/Runpod 上运行**。不要把这里的“源码恢复”当成现在的“已运行验证”。
- 这些历史脚本不包含在用户上传的 E 盘目录快照中（目前未证明 E 盘没有副本）；同步到用户项目时应先 diff、不要覆盖原有更晚版本。

### 19.3 `assets/outputs_original.zip`：本次实检的 12 个归档条目

```text
outputs/a3_grasp_test.mp4
outputs/a3_prediction.mp4
outputs/cookie_bridge_smoke.mp4
outputs/cookie_combined_v3.mp4
outputs/cookie_gripper_test.mp4
outputs/cookie_visible_v3.mp4
outputs/fasterwam_a3_input/observation.npz
outputs/fasterwam_a3_input/observation.png
outputs/fasterwam_camera_preview.png
outputs/fasterwam_official_input.png
# 另外有两个目录条目（outputs/ 与 outputs/fasterwam_a3_input/），本列表省略
```

**可验证结论**：6 个 MP4、1 个 NPZ 观测、3 张 PNG；`a3_prediction.mp4` 是未来**预测视频**，不是成功执行动作的视频。已检查 ZIP CRC；**未见 `a3_left_pickup_actions.npz`、闭环完整 `actions.npz` 日志、跨 seed 成功率、抓到/抬起饼干的视频证据**。如果用户在 E 盘有其他未上传结果，以本地新发现为准。

### 19.4 本包其他内容

- `docs/README_original_20261016.md`：原始 822 行 LIBERO 交接文档，**不丢弃历史**。
- `scripts/preflight.py`：纯标准库 E 盘路径、SHA/仓库存在性巡检，**不启动 GPU，不安装包，不修改模型**。
- `scripts/seed_batch.py`：复制临时版本把原测试的 `env.reset(seed=0)` 换成指定整数，在**独立 seed 输出目录**分别记录 stdout/stderr/返回码；**不修改项目原脚本**，也不保证 seed 引起几何变化。
- `scripts/seed_scene_check.py`：A3 3.12 按 seed 记录 cookie 位姿签名；可指定非零扰动。使用**仿真特权真值仅作评测诊断，不输入 Faster-WAM**。
- `scripts/offline_action_check.py`：读取未来如有的 `(T,7)` 动作 NPZ 做离线合法性检查（不驱动机器人）。
- `scripts/cloud_prepare_env.sh`：**新编写、未在新 Pod 全流程验证**的半自动部署脚本；必须手动确认付费，不包含在旧历史验证项中。
- `scripts/cloud_archive_results.sh`：云端结束前打包**代码、日志和视频但不包含大权重与虚拟环境**；归档后仍需**下载到 E 盘**才能保留。
- `experimental/infer_a3_worker_official_gripper.py`：**优先建议使用的候选**。从历史原版 worker 复制，仅在 `invert_gripper_action` 后添加与当前 Faster-WAM 官方 LIBERO 评测代码一致的 `EVALUATION.binarize_gripper` 条件 `np.sign()`；**比下游直接 clip 更接近官方正式协议**，当前尚无新的完整 A3 闭环验证。
- `experimental/action_adapter_v3_gripper_guard.py`：**次选的防御式候选**，只对轻微超界的模型夹爪动作饱和，并记录告警，禁止静默裁剪空间动作；离线单元测试通过，但尚未真实闭环验证。在官方二值化修复生效后通常不需要再依赖它。保持原 V3 不动，只有在审核通过后才替换实际控制路径。
- `MANIFEST_SHA256.txt` + `scripts/verify_bundle.py`：每个交接文件的 SHA256 与纯 Python 检查脚本，防止后续误读同名不同版本；**清单自身不包含在自己的哈希中**。

## 20. A3 动作/图像/本体状态接口：严格区分维度和语义

| 数据 | A3 | 当前 Faster-WAM LIBERO 适配假设 | 已确认到的程度 |
|---|---|---|---|
| 主要任务相机 | 前方、左右腕三路 RGB | 取前方 + **左腕**，各 256×256 缩放/裁成 224×224 后水平拼接至 `uint8[224,448,3]`、`float32[1,3,224,448]`，[-1,1] | 静态形状可用、已有 `observation.npz` 和可视化；相机外参差异仍是 OOD 风险 |
| 关节/夹爪状态 | `observation.state` 16 维，左夹爪在索引7，右在15；开度 `[0,1]` | 只取左末端 XYZ/rotvec + 近似 2 个 Panda 手指位置（组成8D） | 维度正确，不证明训练分布对齐 |
| 末端状态 | `observation.eef_pose` 14 维，两臂分别 `xyz(3)+quat(wxyz)(4)` | 四元数转轴角 + `0.04 × 左夹爪开度`、`-0.04 × 左夹爪开度` | 四元数表示数学一致性脚本存在；手指位置标定**只是近似** |
| LIBERO 策略动作 | 单臂 7D：末端 6D、夹爪 1D | `action_adapter_v3.py` 映射为双臂 14D cartesian_delta：左臂 7D，右臂不运动且保持夹爪开度 | 接口通道可映射，但原机器人并非同一机体，坐标与时间语义未全面独立验证 |
| A3 执行 | `cartesian_delta` 14D，所有通道必须 `[-1,1]` | 前 3 维线性动作每单位 `0.025m`，后三维旋转每单位 `0.1rad`（读 A3 controller config 核对） | 曾在小动作种子0测试获得很小控制误差；未知跨 seed 分布 |

`vision_adapter.py` 使用 OpenCV INTER_AREA 的直接缩放；`vision_adapter_official.py` 使用 PIL BILINEAR 中心裁剪并宣称匹配官方评测。两者不能无记载混用；先核对真实 Faster-WAM 评测的图像通路，不要只凭函数名 `official` 认定完全等价。

**动作适配版本演变**：

1. `action_adapter.py` **早期版本**：假设模型 XYZ 已是米，夹爪 `[0,1]`，转换 `2*g-1`；这与后来推理结果中 `g≈-1.0` 矛盾，不能直接用来接当前 checkpoint 的最终 LIBERO 执行动作。
2. `action_adapter_v2.py`：假设 LIBERO 平移 0.05m/单位、A3 0.025m/单位，旋转采用比例 0.5/0.1，夹爪仍假设 `[0,1]`；大量原有精度/控制测试通过它生成**人为指定**的动作，不等于模型真实动作适配通过。
3. `action_adapter_v3.py`：空间转换为 `x_A3=2*x_LIBERO`、`R_worldvec=0.5*rot_LIBERO`、`R_local=R_EEF^T@R_worldvec`、`rot_A3=R_local/0.1`，夹爪 `g_A3=-g_LIBERO`，右臂前六维为 0、右夹爪开放量保持；末尾 `abs(output)>1+1e-6` 抛错，允许微小浮点余量内 clip。**这是真实闭环实验使用的版本，但不代表物理坐标变换被严格证明正确。**

**官方动作处理不能重复做**：历史恢复的 worker 里 `pred['action'] → _denormalize_action(action, processor)[0] → g*2-1 → invert_gripper_action(action)`，随后输出 NPZ。**由 worker 输出的 `(T,7)` 是经过官方处理的 LIBERO 执行动作，而不是未经处理的模型原始归一化输出**。下游适配器不得再次反归一化或重复符号翻转。

## 21. A3 测试和“误差极低”的准确证据边界

- 上传文件 `test_fk_consistency.py`：拷贝 `qpos` 重建 MuJoCo FK 并断言坐标匹配 `atol=1e-7m`；**断言衡量数值正运动学一致性，不等于驱动机械臂的最终位置误差 <0.1µm**。
- `diagnose_precision.py`：分解**原始 IK 解算误差、关节限速、执行目标和实际末端的误差**。值得在 seed=0 对照及 seed=1～5 中重复；单次误差用户主观报告极低，但本批上传主要为**代码+视频而非全套打印输出**，没有可核对的跨 seed 平均/最大数值。
- `diagnose_ik_reference.py`：显式区分使用当前实际状态还是上次指令关节目标作为 IK 参考，原代码正确考虑 A3 `last_applied_action` 与 `qpos` 不同。
- `diagnose_joint_tracking.py`：100 个零笛卡尔动作让关节持位，记录实际关节与命令误差；零动作持位可能受重力影响。
- `test_cookie_rotation.py` 与 `test_orientation_consistency.py`：前者读取真实相对旋转、后者验证四元数↔轴角一致；不能仅从语法断言推定不同 seed 物理旋转误差均极小。
- `test_cookie_gripper_video.py`、`test_cookie_combined_v3.py`、`test_cookie_visible_v3.py`：含人为动作序列的场景录像，验证接口执行/可视化，**不是零样本抓取策略**。
- `test_gain_sweep.py`：在**临时创建的环境实例**里试 `1.0、1.5、2.0` 倍增益（并修改阻尼），未提交项目永久控制器增益；下一步**不应盲目修改增益**。
- **明确的小错误**：`test_ik_diagnostic.py` 用 V2 输入 0.05，映射成 A3 左 X 实际期望 **2.5mm**，但打印字样写 `Expected target X delta: 5.0 mm`。这是**测试打印错误**，不应认定机器人有 2.5mm 实际偏差。

## 22. 真实 Faster-WAM → A3 推理与闭环：最后一次卡点

### 22.1 2026-10-19 第二次 Runpod 历史日志

- 第二轮新的云容器提示符 `root@b9c03a3dd75e`，`nvidia-smi` 报 `NVIDIA RTX A6000, 49140 MiB`；`/workspace` 起初几乎为空。再次克隆 `https://github.com/hustvl/FasterWAM.git`。
- 安装 `uv 0.13.1` 至 `/root/.local/bin`，旧 AI 误叫 `source ~/.cargo/env` 报不存在；此时 `uv` 已存在或路径需用 `source ~/.local/bin/env`，不要无意义修 Cargo。
- 执行 `bash scripts/setup/install_core.sh` 和 `install_libero.sh` 成功，Python 3.10.18，torch 2.7.1+cu128，MuJoCo 3.3.2，LIBERO `8f1084e`。
- 使用 `huggingface-cli` 未加入 PATH 时出现 `command not found`；随后通过虚拟环境调用 HF CLI 成功下载 **Faster-WAM LIBERO 11.1GB 以及不需要的 RoboTwin 11.1GB**。这是一次额外占用空间与时间的误操作；**以后仅下载 HF 仓库根下 `libero/*`，不要下载 RoboTwin**。
- 后续 `snapshot_download` 的错误示例把 `allow_patterns` 写成 `fasterwam_release/libero/*` 并把 `local_dir` 设成 `checkpoints`，返回 `Fetching 0 files`。实际仓库根路径是 `libero/step_021700.pt` 和 `libero/dataset_stats.json`，正确的目标是 `local_dir='checkpoints/fasterwam_release'`。不要把 0 files 一概解释为已缓存；**本例更可能是匹配模式写错**。
- Wan 下载示例曾混入 Unicode 非 ASCII 连字符 `‑`，引发 `HFValidationError`；应全部使用仓库 ID `Wan-AI/Wan2.2-TI2V-5B` 等普通 ASCII `-`。
- 该环境后续确实运行了一次 LIBERO 单任务成功；聊天摘要报告约 32 秒，但完整结果的终端数值未在当前简化交接中逐条复核，不纳入 40×1 的原统计。
- 更进一步的 A3 worker 得到 `Faster-WAM worker ready` 和 `Request finished in 2.48s`（至少一次请求）；单次历史 `infer_a3.py` 输出 `(32,7)` 动作序列和 `a3_prediction.mp4`。A3 本地 3.12 环境也曾在新云实例独立运行。
- Runpod **在用户停止后会清空所有工作区**（反复实际观察）。本 README **不声称**新实例目前 Running/Stopped，也不复用历史 SSH 地址、端口。**如果用户此刻仍看到 Running，应立即先备份最新代码/日志再在网页停止 GPU 计费；不能由文档替用户停机。**

### 22.2 可复现的原始失败数据：索引 6 是左夹爪

```text
=== Replan 1 ===
Predicted chunk: 32 actions, executing first 5
step=001 chunk_step=1/5 reward=0.000 left_gripper_cmd=-0.992
step=002 chunk_step=2/5 reward=0.000 left_gripper_cmd=-0.992
step=003 chunk_step=3/5 reward=0.000 left_gripper_cmd=-1.000
ValueError: Converted action exceeds A3 limits:
[-0.025635  0.087891 -0.021973  0.205539 -0.023557  0.060376  1.015625
  0.        0.        0.        0.        0.        0.        1.      ]
```

- 前 6 个空间通道的最大幅值仅 `0.205539`，**没有超过 1**。
- 第 7 个通道（Python `action14[6]`）是**左夹爪** `+1.015625`，模型处理后 LIBERO 夹爪为 `-1.015625`，V3 符号翻转后轻微超限。此前 AI 错误地归因“第 4 个旋转分量 0.205539 超限”，并建议随意调小 `action_horizon`，这是**不成立**的诊断。
- 单次 `infer_a3.py` 历史输出夹爪最小约 `-1.03125`、最大约 `+0.984375`，说明需要检查**官方 Processor/模型数值范围与夹爪饱和处理约定**。
- **优先修正建议（2026-10-21 发现，更符合官方评测）**：先检查并补全 **Faster-WAM worker 遗漏的官方夹爪二值化**。当前公开 `configs/sim_libero.yaml` 有 `EVALUATION.binarize_gripper: true`，官方 `eval_libero_single.py` 在 `invert_gripper_action` 后执行 `if bool(cfg.EVALUATION.get("binarize_gripper", False)): action[..., -1] = np.sign(action[..., -1])`。历史手写 `infer_a3_worker.py` 没有这一步，因此输出 `-1.015625` 等浮点值，而**如果启用官方默认条件，本来会变成 `-1`，V3 翻转后刚好是 `+1`**。本包 `experimental/infer_a3_worker_official_gripper.py` 在历史 worker 中**仅插入相同 3 行逻辑**；源码 diff 和 Python 语法检查已通过，但**该候选还没有真实 A3 GPU 闭环验证**。要在目标 Git SHA 中先核对配置，再选择此修复。
- 备用下游方案（非首选）：只对**夹爪**轻微越界做有日志的边界饱和，对空间通道仍超界拒绝；`experimental/action_adapter_v3_gripper_guard.py` 有此候选并通过离线单元测试。它有 `0.125` 容许超界阈值，属人为保护假设，不是官方参数。优先修正官方处理缺漏，不要两条方案在论文 baseline 里混淆。
- 不应全通道 `np.clip` 后继续实验而不记录，否则物理动作被改变难以解释；**调 `action_horizon` 影响动作长度，不能简单当作减小单步动作幅度的数学修复**。
- 记录中 V3 的旋转从 world 到 local `R.T` 与比例系数属于待验证假设；旧数值测试未完全覆盖这条链路在全部姿态/seed 下的准确性。

### 22.3 来自官方源码的关键查漏：`binarize_gripper: true`

2026-10-21 直接核查 [`configs/sim_libero.yaml`](https://github.com/hustvl/FasterWAM/blob/main/configs/sim_libero.yaml) 与 [`experiments/libero/eval_libero_single.py`](https://github.com/hustvl/FasterWAM/blob/main/experiments/libero/eval_libero_single.py)：前者明确 `EVALUATION.binarize_gripper: true`；后者在 `_denormalize_action`、`g*2-1`、`invert_gripper_action` 之后仍有**可选的** `action[..., -1] = np.sign(action[..., -1])`。

```python
# 2026-10-21 官方评测源码（逻辑；实际仍需检查已 checkout Git SHA）
action = _denormalize_action(action, processor)[0]
action[..., -1] = action[..., -1] * 2 - 1
action = invert_gripper_action(action)
if bool(cfg.EVALUATION.get("binarize_gripper", False)):
    action[..., -1] = np.sign(action[..., -1])
```

**历史人工 worker 的确缺少最后两行。** 新版候选在 `experimental/infer_a3_worker_official_gripper.py`，保留原件 `recovered_historical/infer_a3_worker.py` 便于精确复现当时失败。除了这几行，候选 worker 不调整动作长度、关节比例、旋转坐标或模型权重。先对完整 checkpoint 配置检查，再开展短闭环，不应将“加上 `np.sign`”宣传成已经成功抓取。注意当前官方配置默认 `replan_steps: 10`，而历史 A3 控制器硬编码 **5**；这属于**另一个评测协议差异**，不构成这次夹爪越界的原因，成功率评测必须记录并统一。

## 23. 不花 GPU 的第一件事：一键本地盘点与多 seed 控制器复测

> **实测更正**：2026-10-21 用户 E 盘 `WAM/` 当时只有 FasterWAM、A3、bridge 和 `fasterwam_a3_bridge.zip`，**没有**本交接包目录。先按文首的“用户本地终端实测更正”把本次交接 ZIP 放到 WAM 根目录并解压；后续 A3 Python3.12 环境是否可用仍待核验。**命令逐步执行，不要一次跑几十个高耗时仿真。**

### 23.1 A3 目录和 Python 环境盘点（目前还缺用户实测输出）

**Windows CMD**：

```bat
wsl -d Ubuntu
```

**本地 WSL Ubuntu**（必须**先**按本文开头把正确交接 ZIP 解压到 WAM 根目录）：

```bash
cd /mnt/e/Innovative_practice/WAM
python3 FasterWAM_A3_Handoff_2026-10-21/scripts/preflight.py --wam-root "$PWD"
```

这个巡检**只读现有项目目录**，可报告缺失路径、旧结果/大权重以及 Git HEAD；缺少 WAN 大模型本地缓存是预期情况，**不表示原 LIBERO checkpoint 丢失**。下一位 AI 应根据输出判断 `.venv/bin/python` 与 Windows `Scripts/python.exe` 是否可用，绝不要假定 WSL 可以直接运行 Windows 安装的 `.venv`。A3 官方要求 Python ≥3.12，建议优先使用已有正确环境；若没有，**单独**创建 Linux 3.12 venv、安装 A3（详见 §25），无需 GPU。

### 23.2 不修改原代码的自动多 seed 驱动

先做**只计划不运行**的 1 个测试 × 2 个 seed：

```bash
cd /mnt/e/Innovative_practice/WAM
python3 FasterWAM_A3_Handoff_2026-10-21/scripts/seed_batch.py \
  --bridge "$PWD/fasterwam_a3_bridge" --python "$PWD/fasterwam_a3_bridge/.venv/bin/python" \
  --tests test_fk_consistency.py --seeds 0 1 --dry-run
```

如果本地的 A3 venv 实际路径不同，按 `preflight.py` 输出替换 `--python` 的解释器；以上 `.venv/bin/python` 只是**待核验的示例路径**。确认正确、依赖可导入后，先运行 seed0 再跑 seed1：

```bash
python3 FasterWAM_A3_Handoff_2026-10-21/scripts/seed_batch.py \
  --bridge "$PWD/fasterwam_a3_bridge" --python "$PWD/fasterwam_a3_bridge/.venv/bin/python" \
  --tests test_fk_consistency.py --seeds 0 1
```

脚本先检查每个待执行原文件**恰好含一次字面 `reset(seed=0)`**，再在 `seed_runs/<test>/seed_NNNN/` 复制临时版本并独立运行、保存 stdout/stderr/返回码；未直接编辑 E 盘原测试代码。这只是**可重复运行的工程工具**，目前用户尚未实测；如果原测试文件名和本包不同，先比较来源再复制。

后续选择 `diagnose_precision.py`、`test_cookie_rotation.py`、`test_cookie_gripper_video.py`；**先 0～5 六个 seed，再判断是否扩大到 20 个**。每项报告实际 XYZ/角度/夹爪目标误差、是否安全终止、耗时、有效 scene 变化、Git SHA。若 `seed=1...5` 几何不变，不能宣称不同初始布局下的可靠性。

### 23.3 为什么换 seed 可能只重复同一场景？

2026-10-21 查 A3 main：`CookieTransferTaskConfig.position_noise_m=0.0`、`yaw_noise_rad=0.0`，`reset` 默认 `randomize_boxes=False`；默认 `randomize_cookies=True` 但乘以零噪声，**位置和朝向几乎不因 seed 随机化**；外观随机化 profile 可能改变相机图像样式，必须单独分析。

若要构造**真正几何差异**的第二组测试，在 A3 Python 3.12 环境中调用：

```bash
cd /mnt/e/Innovative_practice/WAM/fasterwam_a3_bridge
.venv/bin/python ../FasterWAM_A3_Handoff_2026-10-21/scripts/seed_scene_check.py \
  --seeds 0 1 2 3 4 5 --position-noise 0.0003 --yaw-noise 0.015 \
  --output outputs/seed_scene_check_randomized.json
```

上面是**新编写脚本、还未在用户 A3 环境实测**，`0.0003m`、`0.015rad` 是**示例轻微扰动**而不是已经跑过的配置；它会报告不同种子的饼干坐标 SHA 指纹和头三块位置。如结果相同或受限，先检查 A3 的具体 reset 与 task_config，而不是直接扩大 seed 计数。

**科学上分两组**：A 组固定几何、改变随机源做数值稳定性重复；B 组明确非零扰动、记录扰动配置以验证泛化。不同 seed 的“控制器精度”与“模型任务成功率”是**完全不同的实验**。

## 24. A3 离线动作诊断、修复建议与验收门槛（GPU $0）

1. 首先在 E 盘搜索 `a3_left_pickup_actions.npz`、`actions.npz` 和 `observation.npz`。本上传的 `outputs.zip` **仅保存观测，不含历史单次生成的 `(32,7)` 动作 NPZ**。如果没有，不要声称已能离线回放完整 32 步；后续下一次 GPU 运行必须保存并下载动作文件。
2. 如找到了动作，**本地 A3 Python 或 NumPy 环境**用：

   ```bash
   python3 /mnt/e/Innovative_practice/WAM/FasterWAM_A3_Handoff_2026-10-21/scripts/offline_action_check.py /path/to/actions.npz
   ```

   如果系统 `python3` 无 NumPy，改用 A3 已安装 NumPy 的 `.venv/bin/python`。离线脚本只统计各通道幅值和夹爪超界，不执行仿真；拿到真实动作后再设计安全的短回放。
3. **首选在 worker 补官方 `binarize_gripper` 后处理**（源码位于 `experimental/infer_a3_worker_official_gripper.py`）；备用才是检查下游夹爪饱和的**数学逻辑**（NumPy 环境即可）：

   ```bash
   python3 /mnt/e/Innovative_practice/WAM/FasterWAM_A3_Handoff_2026-10-21/scripts/test_gripper_guard.py
   ```

   备用夹爪饱和单测已在本次交接沙盒运行成功，但未在 A3 真实闭环里运行。原 `action_adapter_v3.py` **保留不改**。若使用官方二值化修复，应优先沿用原 V3 并保存 worker 最终动作及真实 14D 控制输出，而不是同时叠加多个修补。
4. 控制器验收门槛建议：同一个脚本、相同 A3 commit / seeds、无安全停止；空间动作不静默饱和；视频清楚显示是否触饼干；每步日志包含原 7D、实际 14D、EEF 位姿、夹爪开度、`info`、`terminated/truncated`；给出相同场景重复性的统计。
5. **限制**：零样本成功/失败主要取决于策略和 OOD 机器人几何差异，极低 IK/FK 误差不能证明该模型已经学会用 A3 夹爪抓饼干，也不能证明用 LIBERO Panda 手指近似后的状态是有效归一化输入。

## 25. 新的付费 Runpod：准备好一切再开启，不保证无条件一键恢复

### 25.1 本地首先备妥（建议完成后再开 GPU）

- `preflight.py` 确认 E 盘 Git 仓库、checkpoint、历史压缩包及 bridge 存在；记录本地 `git -C FasterWAM rev-parse HEAD`、`git -C a3_dual_arm_sim rev-parse HEAD`。新云实例**先 checkout 到经核验的同版本**，不要只用持续变化的 main 生成虚假复现承诺。过去的 Faster-WAM 原 commit **仍未知**，不允许捏造。
- 用 Python 3.12 本地完成 seed0～5 控制器验收和 A3 专家 1 次标准任务 smoke（官方提供例如 `examples/benchmark_cookie_batch.py --task 2x5 --policy expert --profile basic --episodes 1 --seed-start 1000 --workers 1 --max-steps 500`；运行前用当前 repo 的 `--help` 确认）。专家成功不代表 Faster-WAM 成功。
- 承认单卡 48GB A6000 是历史**能完成评测/推理**的配置，不代表单卡足够做完整模型训练。核对当前 Runpod 计费与磁盘容量，避免无持久化导致反复下载。
- `FasterWAM_A3_Handoff_2026-10-21.zip` 解压后目录必须保持为此名称，或调整部署脚本的 `HANDOFF_BUNDLE`。用户需自行提供模型下载网络、GPU、git 和 Bash 运行权限，且自己确定不违法服务使用规则。

### 25.2 自愿的半自动临时 Pod 部署（尚未端到端执行）

在 **新 Runpod Cloud Terminal** 中：先通过 Git/SCP/上传把本交接包放到 `/workspace/FasterWAM_A3_Handoff_2026-10-21`。上传命令需要**新 Pod 实际 SSH Host/Port**；旧 `216.81.151.15` 和旧端口完全不能当作今天有效的地址。确认有 ≥90GiB 空间、收费和网络后，**仅在本人理解开 GPU 费用与 Stop 丢盘的情况下**运行：

```bash
cd /workspace/FasterWAM_A3_Handoff_2026-10-21
I_ACKNOWLEDGE_GPU_CHARGES=YES bash scripts/cloud_prepare_env.sh
```

**新编写脚本的意图（尚未新 Pod 全部跑通）**：必要时安装 `uv`/EGL → 克隆两个仓库（当前 HEAD，须再核对）→ 运行 Faster-WAM 官方 `install_core.sh`、`install_libero.sh` → 创建 A3 Python 3.12 venv → 在**正确**的 HF 相对目录只下载 Faster-WAM `libero/step_021700.pt`、`libero/dataset_stats.json`（不下载 RoboTwin）+ Wan VAE/T5/tokenizer → 检查模型文件 → 非覆盖地复制桥接、历史与修订的 worker、控制器 → 生成 `/workspace/a3_environment_manifest.txt`。

不能保证所有网络、当前 CUDA 镜像、下载 API、第三方依赖随时间不变。**若失败，应仅修实际失败环节，不重新装一切，也不要长时间让 GPU 在空耗时**。新 Pod 之前的 Python3.12 系统解释器不能取代旧 .venvs/libero 的 3.10；A3 venv 需要独立激活。

### 25.3 单任务加载验证再试 A3

- 首先使用历史 §8 的**真实成功过的 LIBERO 单任务命令**，再次验证 checkpoint、Wan 组件、EGL 正常。如果本次任务只是为了 A3，不必重复完整 40×1。
- **GPU worker 终端 1**（新云已部署后）：

  ```bash
  cd /workspace/FasterWAM
  PYTHONPATH="$PWD/third_party/LIBERO:$PWD/src:$PWD:$PWD/experiments/libero" \
  DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints" HF_HUB_ENABLE_HF_TRANSFER=0 \
  .venvs/libero/bin/python infer_a3_worker.py
  ```

  预期输出 `Faster-WAM worker ready`，进程常驻等待。**新部署脚本会优先放置只补官方 gripper 二值化的实验 worker；已存在的同名用户版本不会被强制覆盖，运行前必须 diff 确定最终代码含 `np.sign`。**它只验证加载成功，不意味着有抓取结果。
- **A3 控制器终端 2**（暂不直接运行几百步）：

  ```bash
  cd /workspace/fasterwam_a3_bridge
  MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  PYTHONPATH="/workspace/fasterwam_a3_bridge:/workspace/a3_dual_arm_sim/src" \
  /workspace/a3_venv/bin/python run_a3_closed_loop.py
  ```

  **警告：这份恢复的历史控制器硬编码 seed0 / 最多400步，且使用历史 V3，极可能复现第3步的夹爪越界错误。不要在未审核、未设置短回合停止条件和动作边界修复之前将此命令作为新零样本评估去直接长跑。** 推荐先复制一份控制器、锁定 1 个 replan/至多 3～5 个动作、留存所有原/转换动作并确认模型输出合法，随后再逐步开放步数。
- 新模型加载 / 短闭环成功后，记录 `cfg`、checkpoint 版本、A3 commit、评测 seed、时间、视频、`info` 和是否真实接触。**如果只有 reward=0 或“总运动了3步”，绝不可宣称抓取成功。**

### 25.4 每次 Stop 前的数据保全（强制）

在 **Runpod 云端终端**：

```bash
cd /workspace/FasterWAM_A3_Handoff_2026-10-21
bash scripts/cloud_archive_results.sh
```

此脚本创建 `/workspace/a3_session_YYYYMMDD...tar.gz`，过滤掉虚拟环境和模型大权重，只保留桥接源码、日志、IPC 和可能的视频。**重要：压缩包仍然在临时容器内**，必须先用**本地 WSL** `scp -O` / `rsync` 到 E 盘，再检查 `tar -tzf` 成功；然后 Runpod 网页真正 `Stop` 并确认费用。旧云端地址不能使用；每次都读新 SSH 页。

`cloud_archive_results.sh` **新编写未在新云 Pod 实测**，实际历史结果可能存放其他目录，应按文件清单人工补充再打包；不要以“脚本结束”误认为所有东西已备份。

## 26. 两个目标的可复现实验协议（成功率必须可解释）

### 26.1 目标一：多 seed 验证原有机械臂控制准确度

- 对照：**先复现 seed0**，再固定动作/参数/机器人初始约束跑 seeds 1～5。若源脚本实际上在 `A3DualArmEnv` 而不是饼干任务里跑，应在报告里标成“机器人沙盒控制”，不能写“饼干几何随机化稳定性”。
- 每个 seed 记录原始模型物理 `target_delta`、机器人实际 `EEF_delta`、误差范数（mm）、角误差（rad/deg）、夹爪开合目标与实际状态、右臂漂移、出现安全停机与否。所有误差必须与实际接入的动作版本关联，例如 V2/V3 不可混为一张表。
- 之后再给 `CookieTransferTaskConfig(position_noise_m>0,yaw_noise_rad>0)` 的随机化组，至少记录所有起始 cookie 坐标 SHA；不应该无证据说仅 seed 变化创建了独立场景。
- 5～6 个 seed 主要用于工程 smoke；如果确实准备论文稳定性评测，再根据计算成本扩大到 20 个/更多，并按均值、标准差、最大、失败率报告。

### 26.2 目标二：Faster-WAM 零样本 A3 饼干抓取

- **模型固定**：官方 Faster-WAM LIBERO `step_021700.pt` + 对应 `dataset_stats.json`，相同 Wan VAE/T5/tokenizer；**没有训练**；保留 worker 中已验证的官方 Processor/反归一化通路。
- **任务固定**：单臂左夹爪抓取一块饼干是**第一阶段子任务**；不等于双臂/十块装盒。设计一个不窥视真值的**策略输入**（仅图像/自我状态/文本），仿真特权真值只能在**独立评测器**里检查接触、是否离开盒面、提升距离和稳定保持时间。
- **动作固定**：使用明确版本的 V3 + **worker 与官方一致的夹爪 `binarize_gripper` 后处理**（若仍超界再审查备用防御），保存原 7D/执行 14D；空间动作越界则安全中止并计入“接口失败/unsafe”，不能偷偷裁剪后算一次成功。
- **闭环频率固定**：`REPLAN_STEPS=5`、当前模型每块 32 个动作、A3 控制 20Hz（依 A3 配置），换动作执行步数会改变有效实时控制与闭环响应，应作为独立超参数记录。
- **零样本成功率**：在同一固定设置先 3～5 次完整、短预算试跑，排除代码 crash、时限错误；锁定输入/后处理/提示词/初始随机化/最大步数后扩大到 20～50 个明确不同 seed。**按已实际跑出的成功数 / episode 总数报告**，附失败（定位失败、夹爪空闭、碰撞、掉落、超时、动作越界、软件错误）分类，不能只统计“程序不报错”或“首次夹爪接触”。
- **完整 2×5**：A3 官方 full task success 是要求将 10 块饼干置于目标槽位等条件；如果想测它，需要任务级策略和正式判分，现有单臂 LIBERO checkpoint 不具备直接解决的已验证能力。应分别报告**单块抓取成功率 / 部分转移数量 / 2×5 官方整回合成功率**，不能混为一谈。
- **微调决策**：先分类 zero-shot 失败。若主要是机器人形态、相机/夹爪动作语义不匹配，即使微调也要先统一表示；若接口正确但不识别/不能稳定抓取，则咨询同学 A3 专家脚本、采集与现有 LeRobot 数据集（A3 README 有录制/SmolVLA 接口，但与 Faster-WAM 训练数据的字段、分辨率、时间步和观测状态不一定兼容），研究数据转换后再小规模微调。**不要把 SmolVLA 公开数据直接声称能无改动训练 Faster-WAM。**

## 27. 技术坑核对：误差、夹爪、随机化、E 盘、下载、未来预测

| 坑/误判 | 本次核实与新策略 |
|---|---|
| Python3.12 vs 3.10 | A3 `>=3.12` 且 NumPy2；Faster-WAM LIBERO 的历史 Python3.10+NumPy1.26，两个 venv 分开，文件 IPC，不要全局更改 Python |
| `fasterwam` 手工 import 失败 | 是 src 布局路径问题，worker 启动设置 `PYTHONPATH`，别多次重装 torch |
| Runpod Stop 清空 | 第二次云 Pod 初始化 `/workspace` 空；小交接包 ≠ 完整 `.venv`/11GB 权重/额外 Wan 缓存；Stop 前手工核对备份 |
| HF 仓库根目录 | 只下载官方 `hustvl/FasterWAM` 的 `libero/*` 到 `checkpoints/fasterwam_release/`；原补救脚本中的 `fasterwam_release/libero/*` allow-pattern **不会匹配**，不能把 `Fetching 0 files` 直接解释成命中缓存 |
| HF CLI 的 PATH | 虚拟环境执行 `.venvs/libero/bin/python -m huggingface_hub.commands.huggingface_cli` 可用历史方式，直接 `huggingface-cli` 可能找不到 |
| Unicode 破折号 | GitHub/HF 模型 ID、仓库名中的所有 `-` 必须 ASCII 短横，而不是 `‑` |
| Wan 401 / hf-transfer | 采用 `model.redirect_common_files=false`，`HF_HUB_ENABLE_HF_TRANSFER=0`；历史成功 |
| 单位和旋转 | V3 的 world→local、0.05m/0.025m、0.5rad/0.1rad是当前适配假设；真实 A3 的 0.025/0.1 应再与 Git SHA 对应的 config 核验 |
| 夹爪 `>1` | 失败值索引6，超过的是夹爪；**历史 worker 漏了官方 `binarize_gripper` 的 `np.sign` 条件分支**；优先补齐，不要误以为旋转0.205539越界，更不应盲调动作 horizon |
| 0.04m Panda 手指代理 | 只是根据 A3 开度构造出来的代理状态，A3 模拟的 Robotiq 夹爪每侧行程与 Panda 并不相同，且 LIBERO 训练统计范围未全部实证对齐 |
| 不同 seed | A3 CookieTransferTaskConfig 默认位置/朝向 noise=0；只换 seed 不一定产生不同饼干位置；可先做 seed_scene_check |
| 成功率的含义 | LIBERO 40/40 是任务 ×1 的 smoke；A3 仅 worker 推理和3步动作；目前 A3 真抓取**没有统计成功率** |
| 录像的含义 | `a3_prediction.mp4` 是模型想象的未来视频，不是机械臂真实完成抓取的演示；`cookie_visible_v3.mp4` 是人为控制动作测试 |
| 真实模型输出文件缺失 | 上传 ZIP 中没有 `a3_left_pickup_actions.npz`，不能保证任何下一位 AI 能马上本地回放原 32 步；未来先保存再关机 |

## 28. 所有事件按时间顺序（避免新 AI 把不同时期混为一谈）

1. **2026-10-14～15：第一次 Runpod / LIBERO**：原始 `FasterWAM-LIBERO` Pod A6000 48GB，完成单任务 1/1、LIBERO 40×1 全成功；结果包 `fasterwam_first_success.tar.gz` / `fasterwam_libero_40tasks_smoke.tar.gz` 下载到 E 盘；之后 Pod Stop，无 Volume。
2. **2026-10-16：原始 822 行 README**：完整记录上述基线、安装、401/EGL/uv/rsync 坑与科研创新计划。即本包 `docs/README_original_20261016.md`；本版保留其主体 §0～§17。
3. **2026-10-18 左右：决定尝试同学 A3 双臂饼干场景**：选择 main 的 `a3_dual_arm_sim`，以左臂零样本抓取作为第一阶段，开始写 bridge；A3 安装要求 Python3.12。
4. **2026-10-19：第二次临时 Runpod（新容器）**：`/workspace` 初始为空，再克隆 Faster-WAM、安装两套 LIBERO/core venv；误下载 RoboTwin 额外权重；重新补 Wan 组件，做过模型载入、单次推理，得到 `actions (32,7)` 与未来预测视频；又用云端 A3 Python3.12 测桥接。
5. **2026-10-20：双进程闭环**：worker 进程完成一次约 2.48s 请求；控制器 Replan1 计划执行5步，第1、2、3步执行后，在第4次转换动作尝试时**左夹爪输出 +1.015625 越界而抛异常**（具体已执行到第3步，实际 fail 为下一动作转换）。没有抓取成功证明。原 AI 误读为旋转越界、提出全通道 clip / 减小 action_horizon，但未收到这些改动在真实闭环中成功的证据。2026-10-21 核查官方源码发现手写 worker 漏掉 `EVALUATION.binarize_gripper=true` 对应的 `np.sign`，疑为**更直接的根因**；候选修复仍待 GPU 验证。
6. **2026-10-20～21：用户提供剩余 bridge 和输出归档**：总计上传33份 `.py`，归一去重28份；另从原始聊天恢复2个实际执行文件；`outputs.zip` 保存6段视频、1份观测 NPZ、3张 PNG，没有全闭环成功率。明确目标：先本地换 seed，后真实模型零样本抓饼干，成功低再考虑微调。
7. **2026-10-21：本 README 和交接包**：新增可复核文件清单、关键错误纠偏、成本约束、单命令巡检/多 seed 工具、历史双进程脚本与未端到端测试的半自动云恢复方案。**本日没有用户新跑的 seed1～5 或 GPU A3 成功日志，不能谎称它们已完成。**

## 29. 最终证据审计矩阵 / 下一位 AI 的验收列表

| 项目 | 当前证据 | 可信等级 | 下一位 AI 下一动作 |
|---|---|---|---|
| LIBERO 40/40 smoke | 原始终端汇总、tar 包备份下载 | `[历史已运行]` | 不重跑；先打开原归档核对 |
| A3 `requires-python>=3.12`、NumPy2 | 当日 GitHub main `pyproject.toml` | `[当前官方源码]` | 固定本地 A3 SHA、查实际 Python3.12 venv |
| A3 机器人原生动作与评分 | 当日 GitHub `docs/contracts.md`、`src/.../actions.py`、`tasks/cookie_transfer.py` | `[当前官方源码]` | 对当前 E 盘 checkout 再 diff，不保证未来 main 不变 |
| 上传桥接代码 28 唯一 `.py` | 对本轮及前面上传文件逐一哈希、AST 解析 | `[源码静态核验]` | 与 E 盘真实 bridge diff，防止覆盖新版本 |
| 恢复 worker 和 controller | 原聊天记录中重复2次的 heredoc | `[源码恢复且静态解析]`；历史调用曾运行 | 本地备份原文件后比对，载入前复核 API |
| 原始 `outputs.zip` | 6MP4+1NPZ+3PNG，ZIP CRC | `[本轮文件核验]` | 视觉展示 ≠ 抓取成功；找动作 NPZ |
| A3 工程误差非常小 | 用户报告 + 相应诊断/视频，具体跨seed数值未全存 | `[局部历史证据；非统计结果]` | 先 seed0 对照、再1～5 执行并保存数字 |
| A3 多 seed 场景不同 | A3 当前默认扰动为0 | `[当前官方源码]` | 跑 `seed_scene_check.py` 的实际布局 SHA |
| A3 零样本 `(32,7)` 推理 | 原聊天输出数组范围、worker ready | `[历史已运行]` | 同 commit/模型重新验证，保存 NPZ |
| A3 真实抓取成功率 | 无完整抓取/抬升结果 | **`[未完成]`** | 先短闭环，再单块抓取标准，之后多 seed |
| A3 V3 夹爪越界具体原因 | 报错向量索引6为1.015625；恢复 worker 缺少官方 `binarize_gripper` 二值化 | `[历史已运行 + 当前官方源码直接对照]` | 优先测试仅补二值化的 worker，后真实短闭环；备用才是 gripper guard |
| 实验级部署“一键零错” | 新脚本仅静态检查和部分单元测试，远程 Git/HF/Pod 仍不可控 | **`[不保证]`** | 执行前确认 Pod、费用、磁盘与源码版本；失败只处理失败环节 |
| 微调、Dynamic SparseMoT | 没有训练或改进结果 | **`[未开始]`** | 先冻结 zero-shot baseline 并记录失败模式 |

### 无法从当前资料确认，严禁编造的内容

- **真正本地 E 盘**各 Git HEAD、A3 `.venv` 所在系统/版本、所有用户未上传文件（包括更新的 worker/controller）、Wan 大文件是否存在于本地、LIBERO 40 任务包是否已解压。
- 2026-10-19 第二次 Runpod 当前是否仍运行、第二 Pod 的真实 Pod ID / SSH Host / Port、当前实际租金。用户确认不持久化；**不能据此虚构已 Stop 的第二 Pod 控制台截图**。
- 所谓“极低误差”的每个 seed 真实数值、所有物理控制参数是否和用户 E 盘同学最新 main 完全一致；也不能保证在不同随机配置上稳定。
- V3 夹爪边界修复后能否真正抓起饼干；任何非零 A3 抓取成功率、正式 2×5 成功率以及模型微调收益。
- 动作转换/自我状态代理在分布偏移下的正确性，未来 video 是否被合理使用于跨机器人任务。需要评测，不要先写结论。

### 下一位 AI 最合适的第一步

**不要以交接包目录已在 E 盘为前提。** 第一轮让用户在本地 WSL 执行本 README 开头的 `find` / Python 版本和 A3 Git SHA 两条检查，再根据实际解释器复测原目录里的 seed0。可选的交接工具必须先解压或单独复制后才能调用。

如果仍找不到文件，先确认**本次修正 ZIP 已被正确复制并解压**，不要要求用户重装 Ubuntu；看完输出**最多再给一条**本地 A3 Python 3.12 版本检查。**不要此时要求运行/修改所有 28 个脚本，更不要让用户新买 GPU 后原地等候 AI 写环境代码。**

---

**本 README 更新截止 2026-10-21。** 文内“历史已验证”只代表以前那次运行及其日志，绝不代表当前机器的新环境自动通过；“本次静态测试通过”不代表用户已经多 seed 跑过或 A3 抓取成功。
