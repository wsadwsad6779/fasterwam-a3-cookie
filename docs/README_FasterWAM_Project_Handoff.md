# Faster-WAM × LIBERO / MuJoCo：项目总 README、复现手册与 AI 交接记录

> **用途**：将此文件直接交给下一位 AI 或项目成员，即使无法访问原对话，也能知道当前研究目标、**真正完成过什么**、本地资料在哪里、如何从零恢复已验证的 LIBERO 推理评测、曾经出现的错误与解决方法、下一步的科研决策。  
> **整理时间**：2026-10-16（基于 2026-10-14～15 的实验与后续对话记录）。  
> **重要声明**：本文把信息区分为 `[已验证]`、`[官方说明]`、`[待验证]`、`[历史配置]`。不得把计划、推测、官方默认配置，写成用户已经完成的实验。

---

## 0. 给下一位 AI：先读这一节（最高优先级）

1. **项目方向**：计算机系本科创新实践 / 科研，面向 MuJoCo 机械臂操作，基于 World Action Model（WAM）/ VLA，优先复现 **Faster-WAM**，后续探索**自适应未来特征融合**、**Dynamic SparseMoT** 等改进，目标包括可展示的机器人闭环操作、定量消融和潜在论文。早期曾讨论先从 Fast-WAM 入门，**实际落地并成功验证的是 Faster-WAM 官方 LIBERO 检查点**。
2. **当前真实进度**：**未训练/微调模型，也未训练“专家演示轨迹”**；已经使用官方发布的 `step_021700.pt` 和 Wan 组件，在 RTX A6000 上完成真实 MuJoCo/LIBERO 闭环**推理评测**。四个 LIBERO 任务集共 **40 个任务，每任务 1 次，40/40 成功**；此前还成功完成一次单任务评测并检查视频。
3. **证据**：40 任务评测输出 `tasks=40 chunks=40 ... gpu_ids=['0']`；四个任务集各 `10/10`，总成功率 `100%`，报告的累计任务执行时间 `19m21s`。**这只是 smoke test，不是每任务 50 次的正式统计基准**，不能将 `100%` 当作论文结论。
4. **本地项目根目录 `[已验证]`**：Windows `E:\Innovative_practice\WAM\FasterWAM`；WSL2 Ubuntu `/mnt/e/Innovative_practice/WAM/FasterWAM`。本地已有官方 LIBERO checkpoint 和数据统计文件；评测的两个压缩备份包已经下载到 `experiment_backups/`。
5. **云端状态 `[已验证]`**：此前的 Runpod Pod `FasterWAM-LIBERO`（RTX A6000 48GB）**已 Stop**，控制台最后截图显示 **`$0.00/hr`**；弹窗明确指出**没有配置 volume，Stop 后数据丢失**，停止后的闲置磁盘费为 `$0.000/hr`。**不要假设云端 `.venv`、Wan 下载文件、`/workspace/FasterWAM` 或原 SSH 入口仍存在**。历史 IP/端口仅作故障分析记录。
6. **如果用户当前只是继续本地研究**：先核验/解压 `experiment_backups/fasterwam_libero_40tasks_smoke.tar.gz`，查看视频、`summary.csv`、`task_success_rates.csv`、`summary.json`。过去对话只有下载成功和 `tar -tzf` 列表成功的明确记录；**没有用户确认解压已成功的记录**。
7. **如果用户需要重新租 GPU**：新建 Pod 后从本地源代码/权重重建环境；遵循本文第 7～11 节。不要使用历史 IP/端口盲连，不要先跑 2000 次；先做单任务与 40 任务配置检查。
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

**目前真正复现成功的，是 Faster-WAM 官方 LIBERO 检查点的推理/闭环评测，而不是演示数据训练或完整模型重训练。** 历史单张 RTX A6000 上，40 个 LIBERO 任务各测一次全部成功，总统计任务时间 19m21s；原始结果与视频已压缩并下载到 E 盘。Runpod Pod 已停止并显示 `$0.00/hr`，无持久化存储卷，不应依赖旧云端环境。下一阶段先查本地解压文件与实验配置、再做结构理解和多初始状态正式测评，最后尝试自适应未来表征 / Dynamic SparseMoT 改进和消融。**所有改进与大规模评测仍待开始。**
