# FasterWAM → A3 饼干抓取：云端复现 README

> 目标：在任何一台新开的 Radeon Cloud 实例上，**照抄本文件即可复现到当前进度**（零试错）。
> 最后更新：2026-10-26（本文件需在每完成一步后更新「当前进度」章节）

---

## 0. 当前进度（每次更新）

| 项 | 状态 | 说明 |
|---|---|---|
| 实例 + 访问 | ✅ | 新实例已开，JupyterLab / web terminal 可用 |
| pip 依赖 | ✅ | mujoco/robosuite/libero/bddl/robomimic/… 已装，`termcolor`/`pynvml`/`rich` 已补（§5） |
| 代码上传解压 | ✅ | 3 个 zip 已上传到 `/workspace` 并解压 |
| 梯度权重 | ✅ | `step_021700.pt` 11,117,679,929 B，sha256 `01be4efc…` 已校验 |
| dataset_stats.json | ✅ | 40,952 B，sha256 `c06dc67f…`，与本地 E: 备份一致 |
| A3 环境冒烟 | ✅ | `ENV OK`，80 个源饼干 / required 10 |
| worker 脚本 | ✅ | 已重建 + `SYNTAX_OK`，`termcolor` 已装 |
| zip 完整性 | ⚠️ | 旧 tar 打包误删 `configs/data/*.yaml`（2 个文件）→ 已在云端补回（§12.1） |
| Wan2.2 组件 | ✅ | VAE/T5/tokenizer 已预下载到 `/workspace/FasterWAM/checkpoints/Wan-AI/`（§6.1） |
| 首次推理 | ✅ | `run_20261026_035243`：400 步 / 40 replans / **cookies 0/10** / 171.9 s / success=false |
| 复现一致性 | ✅ | 与旧实例（`run_20261025_180753`，155.5 s，0/10）一致 → 管线可复现 |
| 成功率批量 | ✅ | `run_20261026_040101`：seed 21–25 共 5 个 episode，**全部 0/10、success=false** |
| 累计零样本 | ❌ **0/7** | seed 11、21–25、42 → 全部 `cookies_in_target=0`，无一饼干进目标盘 |
| 失败机理 | ✅ 已定位 | 抓取时机错位：seed 42 在 **第 6 步（离饼干 14.7 cm）就闭合**、保持闭合 84 步，此后 310 步**再未闭合**；最近点 2.8 mm 时夹爪是**张开**的（§9.1） |
| 包装器实验 | ❌ 无效 | `soft`（软闭合）min dist 4.6 cm；`soft_guard`（守门）触发 162 次后模型发出越限动作崩溃（§9.1） |
| oracle 探针 | ✅ 已跑 | `grasp_triggers 3`、`max_lifted_gt_2cm 2`、`cookies_in_target 0` → **动作/接近可用，搬运缺失**（§17） |
| 专家数据集下载 | ✅ | `Eter0109/a3-front-close-left-100` 4.1 GB @ `8fdd8a7c…` 已下到 `/workspace/a3_data/`；**无 videos/，图像以 PNG 内嵌 parquet**（§19.1） |
| 数据预处理 | ✅ | `prepare_a3_arrays.py` 跑通：42681 帧 → `frames.dat` 12.85 GB + `arrays.npz`（7D 动作/8D proprio）；sat 0%，夹爪 16–24 段/条（§19.3） |
| 动作重放校验 | ⏳ 等结果 | `make_stats.py` + `verify_a3_replay.py` 已发云端；通过后动作约定（0.05/0.5/±1）才算实锤 |
| 微调训练 | ⏳ 未开始 | 待写 `a3_video_dataset.py` + `configs/{data,task}/a3_*.yaml`（§19.4–19.5） |
| **产物下载** | ⏳ 待做 | 打包 `outputs/`（含 `outputs/guarded/`、`outputs/oracle/`）下载（§11） |

**seed 规则（2026-10-26 起）：统一用 `--seed 42`，不再换 seed。**
**下一次从这里继续：① 看 `verify_a3_replay.py` 输出（动作约定实锤）；② 写 `a3_video_dataset.py` + 训练配置（§19.4）；③ 微调。**

已完成校验（2026-10-26）：
- `step_021700.pt` 11,117,679,929 B，sha256 `01be4efc…` ✅
- `dataset_stats.json` 40,952 B，sha256 `c06dc67f…`（云端 = 本地 E: 备份）✅
- `infer_a3_worker.py`：`SYNTAX_OK` / `np.clip` 1 处 / `default=11` 1 处 ✅

---

## 1. 平台与访问约定

- Radeon Cloud，实例模板选 **Blank Notebook**（Python 3.12.3，`/opt/venv`）。
- 实例 ID 每次不同（本次 `rgapi-one-click-24316-57b6b870`）。
- JupyterLab：`https://jupyter.anruicloud.com/instances/<实例ID>/lab`
- web terminal 提示符形如 `root@j-xxxxxxxx:/workspace#`。
- **文件系统：JupyterLab 文件浏览器的根 `/` == 容器内的 `/workspace`。**
- ⚠️ **实例退出即自动删除，`/workspace` 全部丢失**（本次就是这么丢的，只剩 `lost+found`）。
  - 结束工作前必须把产物（NPZ/MP4/summary.json）下载回本地。
  - 想持久化：新建模板时启用 **Workspace Storage（PVC）**；模板表单里还有 **SSH Access (advanced)** 开关。
  - 时长约束：15 credits ≈ 15 GPU 小时，1 credit/GPU/小时；每账号同时只能 1 个实例。
- 容器内 **GitHub 被墙**（克隆/raw 均失败），所以代码必须从本地上传；HF 走 `hf-mirror.com` 很快（76–99 MB/s）。

**终端标签（非常重要）**：本文所有命令块都标了 `[云端 bash]` 或 `[本地 cmd]`，不要混。
本地 shell 是 cmd.exe，PowerShell 命令必须写成 `powershell -NoProfile -Command "..."` 整行。

---

## 2. 云端环境检查 `[云端 bash]`

```bash
python3 -V && python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())" && df -h /workspace | tail -1 && pip3 -V
```

期望：Python 3.12.3 / `2.9.1+gitff65f5b True` / 约 98G 可用 / pip 26.x。

---

## 3. 本地打包（只需做一次，zip 已在桌面） `[本地 cmd]`

源码位置：
- `E:\Innovative_practice\WAM\FasterWAM`
- `E:\Innovative_practice\WAM\a3_dual_arm_sim`
- `E:\Innovative_practice\WAM\fasterwam_a3_bridge`

⚠️ **不要用 `tar --exclude=FasterWAM\data` 这种写法**：Windows 自带 tar 的排除规则会匹配**任意路径段**，把 `FasterWAM/configs/data/`（hydra 必需，含 `libero_2cam.yaml`）一起删掉，导致 `MissingConfigException: In 'train': Could not find 'data/libero_2cam'`。用下面的确定性脚本，排除只作用于**顶层目录**：

`C:\Users\ASUS\Documents\Qoder\2026-10-24\c3f102f0\make_zips.py` `[本地 cmd]` `py make_zips.py`

```python
import os
import zipfile
from pathlib import Path

SRC_ROOT = Path(r"E:\Innovative_practice\WAM")
OUT = Path(r"C:\Users\ASUS\Desktop")

JOBS = {
    "FasterWAM_code.zip": ("FasterWAM", {
        ".venv", ".venvs", "checkpoints", "third_party", ".runtime",
        "evaluate_results", "experiment_backups", "runs", "results",
        "data", "code", "cache", "assets",
    }),
    "a3_dual_arm_sim.zip": ("a3_dual_arm_sim", {".venv"}),
    "fasterwam_a3_bridge.zip": ("fasterwam_a3_bridge", {"outputs", ".venv"}),
}

SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache"}

for name, (src, top_excl) in JOBS.items():
    src_dir = SRC_ROOT / src
    out_path = OUT / name
    count = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(src_dir):
            rel = Path(root).relative_to(SRC_ROOT)
            if rel.parts and rel.parts[0] in top_excl:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                if f.endswith(".pyc"):
                    continue
                full = Path(root) / f
                arc = (Path(src) / full.relative_to(src_dir)).as_posix()
                z.write(full, arc)
                count += 1
    print(f"{name}: {count} files, {out_path.stat().st_size} bytes")
```

产物（文件数对得上即内容完整）：

| 文件 | 文件数 | 大小（字节） |
|---|---|---|
| `FasterWAM_code.zip` | 234 | ≈1,743,000（含 `configs/data/*.yaml`） |
| `a3_dual_arm_sim.zip` | — | 22,570,720 |
| `fasterwam_a3_bridge.zip` | — | 1,510,825 |

**打包后必做自检**（`compare_zip.py`，`missing` 必须为 0）：

```bat
py "C:\Users\ASUS\Documents\Qoder\2026-10-24\c3f102f0\compare_zip.py"
```

```python
import os, zipfile
from pathlib import Path

ZIP = Path(r"C:\Users\ASUS\Desktop\FasterWAM_code.zip")
SRC = Path(r"E:\Innovative_practice\WAM\FasterWAM")
TOP_EXCL = {".venv", ".venvs", "checkpoints", "third_party", ".runtime",
            "evaluate_results", "experiment_backups", "runs", "results",
            "data", "code", "cache", "assets"}

in_zip = set(zipfile.ZipFile(ZIP).namelist())
missing = []
for root, dirs, files in os.walk(SRC):
    rel_root = Path(root).relative_to(SRC)
    if rel_root.parts and rel_root.parts[0] in TOP_EXCL:
        dirs[:] = []
        continue
    for f in files:
        arc = f"FasterWAM/{(rel_root / f).as_posix()}"
        if arc not in in_zip:
            missing.append(arc)
print("zip entries:", len(in_zip), "missing:", len(missing))
for a in missing:
    print("  MISSING", a)
```

> 旧 zip（用 tar 打的那版）实测缺这 2 个文件：`configs/data/libero_2cam.yaml`、`configs/data/robotwin.yaml`。若已用旧 zip 部署，见 §12 的应急补文件命令。

---

## 4. 上传 + 解压 `[云端]`

**上传**：平台自带的文件上传框是坏的（`Upload Error / Failed to fetch`），**必须用 JupyterLab 左侧 File Browser → ↑ Upload Files**，把 3 个 zip 传到根目录（= `/workspace`）。

**解压** `[云端 bash]`：

```bash
cd /workspace && python3 -c "import zipfile; [zipfile.ZipFile(z).extractall('.') for z in ['FasterWAM_code.zip','a3_dual_arm_sim.zip','fasterwam_a3_bridge.zip']]" && ls -d FasterWAM a3_dual_arm_sim fasterwam_a3_bridge
```

期望输出三个目录名。解压后目录结构：
```
/workspace/FasterWAM
/workspace/a3_dual_arm_sim
/workspace/fasterwam_a3_bridge
```

---

## 5. 安装依赖 `[云端 bash]`

**一次装完所有（Aliyun 镜像，分 3 条以免 `--no-deps` 混在批量里）**：

```bash
pip3 install -i https://mirrors.aliyun.com/pypi/simple/ mujoco==3.3.2 numpy==2.1.2 robosuite==1.4.0 gymnasium
pip3 install -i https://mirrors.aliyun.com/pypi/simple/ libero --no-deps && pip3 install -i https://mirrors.aliyun.com/pypi/simple/ bddl --no-deps && pip3 install -i https://mirrors.aliyun.com/pypi/simple/ robomimic --no-deps
pip3 install -i https://mirrors.aliyun.com/pypi/simple/ easydict av thop wandb jsonlines GitPython einops omegaconf hydra-core imageio imageio-ffmpeg scipy h5py matplotlib opencv-python numba pandas transformers diffusers accelerate huggingface_hub termcolor pynvml rich
```

说明：
- `gymnasium` 是 a3_dual_arm_sim 的硬依赖，漏了会 `ModuleNotFoundError: gymnasium`。
- **`termcolor` / `pynvml` / `rich` 是 FasterWAM import 链上的依赖**（`fasterwam/utils/misc.py` 顶部 import termcolor），漏了会在 worker 启动瞬间崩（本次就卡在这）。
- pip 走 Aliyun 约 300 kB/s，整套约 30–60 分钟，属正常。

---

## 6. 模型权重下载 `[云端 bash]`

仓库：**`hustvl/FasterWAM`**（HF 上是「子目录式」仓库，实际文件在 `libero/` 下；`jadechoghari/FasterWAM` 不存在）。
镜像：`https://hf-mirror.com`。

先列目录确认文件名与大小：

```bash
curl -s "https://hf-mirror.com/api/models/hustvl/FasterWAM/tree/main/libero" | python3 -c "import json,sys; [print(x['path'], x.get('size',0)) for x in json.load(sys.stdin)]"
```

下载（`-C -` 可断点续传）：

```bash
mkdir -p /workspace/FasterWAM/checkpoints/fasterwam_release/libero
cd /workspace/FasterWAM/checkpoints/fasterwam_release/libero
curl -L -C - -o step_021700.pt "https://hf-mirror.com/hustvl/FasterWAM/resolve/main/libero/step_021700.pt"
curl -L -o dataset_stats.json "https://hf-mirror.com/hustvl/FasterWAM/resolve/main/libero/dataset_stats.json"
sha256sum step_021700.pt dataset_stats.json
ls -l step_021700.pt dataset_stats.json
```

**必须核对（唯一 truth）：**

| 文件 | 大小 | sha256 |
|---|---|---|
| `step_021700.pt` | 11,117,679,929 B | `01be4efca565d1890fd8a7dc1507b01c00c8cc63c16ae252287492ba9ac7550d` |
| `dataset_stats.json` | 40,952 B | `c06dc67fcf819b04e2720c90e3d02e0bc70283af9bca3cc77a7a162432fd0910` |

- 本次第一次下载**大小完全正确但 sha256 错**（`9547bd0d…`），是 CDN 传输损坏；删除重下即好（约 2 分钟 / 99 MB/s）。**永不信大小，只信 sha256。**
- 本地备份（可直接对比）：`E:\Innovative_practice\WAM\FasterWAM\checkpoints\fasterwam_release\libero\` `[本地 cmd]`
  ```bat
  powershell -NoProfile -Command "(Get-FileHash 'E:\Innovative_practice\WAM\FasterWAM\checkpoints\fasterwam_release\libero\step_021700.pt' -Algorithm SHA256).Hash.ToLower(); (Get-FileHash 'E:\Innovative_practice\WAM\FasterWAM\checkpoints\fasterwam_release\libero\dataset_stats.json' -Algorithm SHA256).Hash.ToLower()"
  ```

---

## 6.1 预下载 Wan2.2 组件（绕开 hf_hub 429）`[云端 bash]`

模型加载时 `fasterwam/runtime.py → load_wan22_ti2v_5b_components()` 会通过 `ModelConfig` 拉 3 类文件：

| 用途 | repo | 文件 | 大小 |
|---|---|---|---|
| VAE | `Wan-AI/Wan2.2-TI2V-5B` | `Wan2.2_VAE.pth` | 2.82 GB |
| 文本编码器 | `Wan-AI/Wan2.2-TI2V-5B` | `models_t5_umt5-xxl-enc-bf16.pth` | 11.4 GB |
| tokenizer | `Wan-AI/Wan2.1-T2V-1.3B` | `google/umt5-xxl/{special_tokens_map.json,tokenizer_config.json,spiece.model,tokenizer.json}` | ~21 MB |

`ModelConfig.reset_local_model_path()` 的规则：目录 = `$DIFFSYNTH_MODEL_BASE_PATH/<repo_id>/`（默认 `./checkpoints/`）；**若目标文件已存在则不再下载**；`DIFFSYNTH_SKIP_DOWNLOAD=true` 可彻底禁用下载。

所以先按文件名直接放好（`curl -L` 走 CDN，不会触发 snapshot 的 resolver 429）：

```bash
B=/workspace/FasterWAM/checkpoints
VAE=$B/Wan-AI/Wan2.2-TI2V-5B
TOK=$B/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl
mkdir -p "$VAE" "$TOK"
curl -L -C - -o "$VAE/Wan2.2_VAE.pth" "https://hf-mirror.com/Wan-AI/Wan2.2-TI2V-5B/resolve/main/Wan2.2_VAE.pth"
curl -L -C - -o "$VAE/models_t5_umt5-xxl-enc-bf16.pth" "https://hf-mirror.com/Wan-AI/Wan2.2-TI2V-5B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth"
for f in special_tokens_map.json tokenizer_config.json spiece.model tokenizer.json; do curl -L -o "$TOK/$f" "https://hf-mirror.com/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/google/umt5-xxl/$f"; done
ls -l "$VAE" "$TOK"
```

运行 worker 时带上这两个环境变量（§9 已包含）：
`DIFFSYNTH_MODEL_BASE_PATH=/workspace/FasterWAM/checkpoints DIFFSYNTH_SKIP_DOWNLOAD=true`

---

## 7. A3 环境冒烟测试 `[云端 bash]`

```bash
cd /workspace && MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="/workspace/a3_dual_arm_sim/src:/workspace/fasterwam_a3_bridge" python3 -c "from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv; env=A3CookieTransferEnv(action_mode='cartesian_delta', render_cameras=True); obs,info=env.reset(seed=11); print('ENV OK keys:', sorted(obs.keys())); print('info:', info); env.close()"
```

期望：`ENV OK keys: [...]`，`info` 里 `cookies_in_source=80`、`required_cookies=10`。
（`libEGL warning` / `os_same_file_description` 警告无害。）

关键 obs 约定：
- 左夹爪开度 = `obs["observation.state"][7]`（会瞬时 >1，必须 `np.clip(…,0,1)`）
- 左 EEF = `obs["observation.eef_pose"]`（前 3 位置，后 4 四元数 **wxyz**）

---

## 8. 创建 worker 脚本 `[云端 bash]`

把下面整段粘进云端 bash（heredoc `'PYEOF'` 不展开变量，安全）。
> 本文件内的版本已包含两处修复：`np.clip` 夹爪、`--seed` 默认 11。
> 历史原始副本：`C:\Users\ASUS\Documents\Qoder\2026-10-24\c3f102f0\heredoc_extracts\worker_1853_2.txt`（需自行打这两个补丁）。

```bash
cat > /workspace/fasterwam_a3_bridge/infer_a3_worker.py <<'PYEOF'
#!/usr/bin/env python3
"""Real FasterWAM zero-shot inference worker for the A3 cookie task.

Mirrors eval_libero_single.py inference path exactly:
obs -> images/proprio -> model.infer_action -> denorm + gripper chain -> A3 adapter.
"""
import argparse
import inspect
import json
import os
import sys
import time
from pathlib import Path

import imageio
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

BRIDGE_DIR = Path(__file__).resolve().parent
FASTERWAM_ROOT = BRIDGE_DIR.parent / "FasterWAM"
A3_SRC = BRIDGE_DIR.parent / "a3_dual_arm_sim" / "src"

for p in (str(BRIDGE_DIR), str(A3_SRC), str(FASTERWAM_ROOT), str(FASTERWAM_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv
from action_adapter_v3 import FasterWAMA3ActionAdapterV3
from vision_adapter_official import prepare_fasterwam_images_official
from observation_adapter import extract_left_eef_state
from gripper_proprio_adapter import a3_opening_to_panda_fingers
from fasterwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fasterwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fasterwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fasterwam.utils.pytorch_utils import set_global_seed

CKPT = FASTERWAM_ROOT / "checkpoints/fasterwam_release/libero/step_021700.pt"
STATS = FASTERWAM_ROOT / "checkpoints/fasterwam_release/libero/dataset_stats.json"
INSTRUCTION = ("Pick up the cookie with the left robot arm "
               "and transfer it to the right robot arm.")


def load_cfg():
    with initialize_config_dir(version_base="1.3", config_dir=str(FASTERWAM_ROOT / "configs")):
        return compose(config_name="sim_libero.yaml",
                       overrides=["model.redirect_common_files=false"])


def quat_wxyz_to_matrix(quat_wxyz):
    from scipy.spatial.transform import Rotation
    q = np.asarray(quat_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q)
    xyzw = np.array([q[1], q[2], q[3], q[0]])
    return Rotation.from_quat(xyzw).as_matrix()


def normalize_proprio(proprio, processor):
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError("expected single merged state key in shape_meta['state']")
    state_key = state_meta[0]["key"]
    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def denormalize_action(action, processor):
    if action.ndim == 2:
        action = action.unsqueeze(0)
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("expected single merged action key in shape_meta['action']")
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    return normalizer.backward(action).numpy()


def build_proprio(obs):
    eef6 = extract_left_eef_state(obs)
    opening = float(np.clip(obs["observation.state"][7], 0.0, 1.0))
    fingers = a3_opening_to_panda_fingers(opening)
    return np.concatenate([eef6, fingers]).astype(np.float32)


def predict_action_chunk(obs, model, processor, cfg, seed, model_device):
    rgb, x_np = prepare_fasterwam_images_official(obs)
    image = torch.from_numpy(x_np).to(device=model_device, dtype=model.torch_dtype)
    proprio = normalize_proprio(build_proprio(obs), processor)

    action_horizon = int(cfg.data.train.num_frames) - 1
    prompt = DEFAULT_PROMPT.format(task=INSTRUCTION)

    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    num_inference_steps = (
        int(cfg.get("eval_num_inference_steps", 20))
        if num_inference_steps_cfg is None
        else int(num_inference_steps_cfg)
    )

    infer_kwargs = {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": seed,
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    action_infer_mode = str(cfg.EVALUATION.get("action_infer_mode", "joint_rollout")).strip().lower()
    infer_action_fn = model.infer_action
    if action_infer_mode == "one_pass_future_cache":
        infer_action_fn = getattr(model, "infer_action_one_pass_future_cache", None)
        if infer_action_fn is None:
            raise ValueError("model lacks infer_action_one_pass_future_cache")
    num_video_frames = (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1
    if "num_video_frames" in inspect.signature(infer_action_fn).parameters:
        infer_kwargs["num_video_frames"] = num_video_frames

    with torch.no_grad():
        pred = infer_action_fn(**infer_kwargs)

    action = pred["action"]
    action = denormalize_action(action, processor)[0]  # [T, 7]
    action[..., -1] = action[..., -1] * 2 - 1          # 0/1 -> -1=open, +1=close
    action[..., -1] = action[..., -1] * -1.0           # invert_gripper_action
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return rgb, action.astype(np.float64)


def run_episode(env, model, processor, cfg, adapter, out_dir, episode_idx, seed, max_steps):
    set_global_seed(int(seed), get_worker_init_fn=False)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))

    obs, info = env.reset(seed=seed)
    pending = []
    chunks_7d = []
    executed_14d = []
    front_frames = []
    merged_frames = []
    current_rgb = None
    t_start = time.time()

    for step in range(max_steps):
        if len(pending) == 0:
            current_rgb, chunk7 = predict_action_chunk(obs, model, processor, cfg, seed, model_device)
            if step == 0:
                print(f"[worker] first chunk shape={chunk7.shape} "
                      f"pos_range=[{chunk7[:, :3].min():.3f},{chunk7[:, :3].max():.3f}] "
                      f"rot_range=[{chunk7[:, 3:6].min():.3f},{chunk7[:, 3:6].max():.3f}] "
                      f"gripper_range=[{chunk7[:, 6].min():.3f},{chunk7[:, 6].max():.3f}]",
                      flush=True)
            chunks_7d.append(chunk7)
            R = quat_wxyz_to_matrix(np.asarray(obs["observation.eef_pose"][3:7], dtype=np.float64))
            pending = [adapter.convert(a, R) for a in chunk7[:replan_steps]]

        a3_action = np.asarray(pending.pop(0))
        obs, reward, terminated, truncated, info = env.step(a3_action)
        executed_14d.append(a3_action)
        front_frames.append(np.asarray(obs["observation.images.front"]).copy())
        merged_frames.append(current_rgb.copy())

        if (step + 1) % 25 == 0:
            print(f"[worker] ep{episode_idx} step={step + 1} "
                  f"cookies={info.get('cookies_in_target', 'N/A')}", flush=True)

        if terminated or truncated:
            break

    duration = time.time() - t_start
    success = bool(info.get("success", False))
    np.savez_compressed(
        out_dir / f"episode_{episode_idx}.npz",
        actions_7d=np.stack(chunks_7d),
        executed_14d=np.asarray(executed_14d),
    )
    imageio.mimsave(out_dir / f"episode_{episode_idx}_front.mp4", front_frames, fps=30)
    imageio.mimsave(out_dir / f"episode_{episode_idx}_model_input.mp4", merged_frames, fps=30)
    return {
        "episode": episode_idx,
        "seed": seed,
        "steps": step + 1,
        "replans": len(chunks_7d),
        "success": success,
        "cookies_in_target": int(info.get("cookies_in_target", -1)),
        "duration_s": round(duration, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max-steps", type=int, default=400)
    args = ap.parse_args()

    global model_device
    cfg = load_cfg()
    model_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[worker] device={model_device} torch={torch.__version__}", flush=True)
    print(f"[worker] video_size={list(cfg.data.train.video_size)} "
          f"num_frames={cfg.data.train.num_frames} "
          f"action_video_freq_ratio={cfg.data.train.action_video_freq_ratio}", flush=True)

    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=model_device)
    model.load_checkpoint(str(CKPT))
    model = model.to(model_device).eval()
    print("[worker] model + checkpoint loaded", flush=True)

    dataset_stats = load_dataset_stats_from_json(str(STATS))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    print("[worker] processor ready", flush=True)

    out_dir = BRIDGE_DIR / "outputs" / "real_inference" / time.strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[worker] output dir: {out_dir}", flush=True)

    adapter = FasterWAMA3ActionAdapterV3()
    env = A3CookieTransferEnv(action_mode="cartesian_delta", render_cameras=True)

    results = []
    try:
        for ep in range(args.episodes):
            seed = args.seed + ep
            r = run_episode(env, model, processor, cfg, adapter, out_dir, ep, seed, args.max_steps)
            results.append(r)
            print(f"[worker] episode {ep} done: {json.dumps(r)}", flush=True)
    finally:
        env.close()

    with (out_dir / "summary.json").open("w") as f:
        json.dump(results, f, indent=2)
    print(f"[worker] all done, outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
PYEOF
```

**校验** `[云端 bash]`：

```bash
python3 -m py_compile /workspace/fasterwam_a3_bridge/infer_a3_worker.py && echo SYNTAX_OK; grep -c "np.clip" /workspace/fasterwam_a3_bridge/infer_a3_worker.py; grep -c "default=11" /workspace/fasterwam_a3_bridge/infer_a3_worker.py
```

期望：`SYNTAX_OK` / `1` / `1`。

---

## 9. 运行推理 `[云端 bash]`

**单 episode 验证（第一步）**（seed 统一 42；下表中的历史 run 用的是 11/21–25）：

```bash
cd /workspace/fasterwam_a3_bridge && DIFFSYNTH_MODEL_BASE_PATH=/workspace/FasterWAM/checkpoints DIFFSYNTH_SKIP_DOWNLOAD=true MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="/workspace/FasterWAM/src:/workspace/FasterWAM:/workspace/a3_dual_arm_sim/src:/workspace/fasterwam_a3_bridge" python3 infer_a3_worker.py --episodes 1 --seed 42 --max-steps 400; d=$(ls -td /workspace/fasterwam_a3_bridge/outputs/real_inference/run_* 2>/dev/null | head -1); [ -n "$d" ] && ls -la "$d" && cat "$d/summary.json"
```

期望日志顺序：
1. `[worker] device=cuda torch=2.9.1+…`
2. `[worker] video_size=[224, 448] num_frames=33 action_video_freq_ratio=4`
3. `[worker] model + checkpoint loaded`（组件已在本地，**不再联网下载**；若无 §6.1 的文件则会去 HF 拉 VAE/T5 并可能 429）
4. `[worker] processor ready`
5. `[worker] output dir: …/outputs/real_inference/run_YYYYmmdd_HHMMSS`
6. `[worker] first chunk shape=(…, 7) pos_range=…`
7. 每 25 步一行 `cookies=…`
8. `episode 0 done: {...}` + `all done`

耗时：加载 ~90 s，单 episode 400 步约 2–4 分钟。

### 9.1 结果记录（零样本，seed 11）

| run 目录 | 实例 | 步数 | replans | success | cookies_in_target | 耗时 |
|---|---|---|---|---|---|---|
| `run_20261025_180753` | 旧实例（已销毁） | 400 | 40 | false | 0/10 | 155.5 s |
| `run_20261026_035243` | 重建实例 | 400 | 40 | false | 0/10 | 171.9 s |

两次独立实例结果一致 → 环境与管线可复现。首个 chunk 形状 `(32, 7)`，`pos_range=[-0.225, 0.207]`、`rot_range=[-0.061, 0.025]`、`gripper_range=[-1, 1]`（已二值化）。
已知行为（旧实例回放诊断）：夹爪约第 10 步闭合、第 56 步重开；左 EEF 从 `[0.099, 0.251, 0.917]` 漂到 `[-0.113, 0.866, 1.31]`（+y 0.6 m / +z 0.4 m，跑出画面），中途碰倒一片饼干、约第 66 步短暂夹住一个后丢失 → 域差距，非适配器 bug。
注意：该任务 `required_cookies=10`（源盘 80 个），成功需转移 10 个，难度远高于单次抓取。

#### 9.1.1 累计零样本结果（0/7）

| run 目录 | seed | 步数 | success | cookies | 耗时 | min 抓取点-饼干距离 |
|---|---|---|---|---|---|---|
| `run_20261025_180753` | 11 | 400 | false | 0/10 | 155.5 s | 2.7 mm |
| `run_20261026_035243` | 11 | 400 | false | 0/10 | 171.9 s | — |
| `run_20261026_040101` | 21–25 | 400×5 | false×5 | 0/10 ×5 | 33–199 s | 1.4–7.5 mm（ep4 除外：14.9 cm，手臂飞出画面）|
| `outputs/guarded/run_20261026_045356` | 42 | 400 | false | 0/10 | 202.5 s | **2.8 mm** |

#### 9.1.2 seed 42 三组对照（`run_guarded.py`）

同一 seed、同一起点，只改夹爪执行方式：

| 变体 | 改动 | 结果 | min dist | guard 触发 | 耗时 |
|---|---|---|---|---|---|
| `baseline` | 无（等同纯零样本） | 0/10 | 2.8 mm | 0 | 202.5 s |
| `soft` | `max_gripper_step` 0.08→0.03 | 0/10 | 46 mm | 0 | 49.9 s |
| `soft_guard` | 软闭合 + 距饼干 >3 cm 拒绝闭合 | **崩溃**（≈step 300） | ~50 mm | 162 | — |

`soft_guard` 崩溃原因：守门把夹爪强行撑开后，观测分布被推离训练分布，模型发出位置分量 −1.12 的动作；`action_adapter_v3.py:54` 对越限动作**抛异常而不是裁剪** → 整个 episode 中断。后续任何 wrapper 都必须在 convert 之后 `np.clip(a3, -1, 1)`。

#### 9.1.3 失败机理（`baseline_episode_42.npz` 逐帧数组）

```
step   0 cmd=+1 dist=0.1514 open=1.000 z=0.920
step   6 cmd=-1 dist=0.1470 open=0.905 z=0.915   <- 离饼干 14.7 cm 就闭合
step  90 cmd=+1 dist=0.1433 open=0.094 z=0.896   <- 保持闭合 84 步
min dist 0.0028 at step 169  cmd=+1  open=0.417   <- 最近点处夹爪是张开的
steps within 2cm: 21   min opening there: 0.416
opening overall min -0.008   first step open<0.2: 15
```

结论：整段 episode 只有 **2 次夹爪指令切换**（step 6 闭合、step 90 张开），闭合并未发生在接近阶段，接近阶段 310 步一直保持张开。这不是「稍微早一点闭合」，而是**抓取相位整体错位**——LIBERO 策略按 LIBERO 的场景/时序触发抓取，对 A3 几何没有响应。因此：

- 「软闭合」「闭合守门」这类只改夹爪执行参数/时序的包装**原理上无法修复**（模型根本不再发第二次闭合指令），实验也证实了这一点；
- 力矩偏大（峰值 270–650 N）是闭着夹爪在源盘里平移造成的挤压，属同一根因的下游现象。

**判定：零样本迁移在 A3 上为 0/7，且失败原因不是适配器 bug、不是到位精度，而是抓取相位错位 → 走「采 A3 演示 + 微调」。**

**确认只有 1 个进程（防 GPU OOM 把实例跑挂）**：

```bash
ps aux | grep infer_a3_worker | grep -v grep
```

## 10. 统计成功率的批量运行（后续）`[云端 bash]`

**必须串行、单进程**（此前 5 个 worker 并发 → GPU OOM → 实例死亡）：

```bash
cd /workspace/fasterwam_a3_bridge && DIFFSYNTH_MODEL_BASE_PATH=/workspace/FasterWAM/checkpoints DIFFSYNTH_SKIP_DOWNLOAD=true MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="/workspace/FasterWAM/src:/workspace/FasterWAM:/workspace/a3_dual_arm_sim/src:/workspace/fasterwam_a3_bridge" python3 infer_a3_worker.py --episodes 5 --seed 42 --max-steps 400; d=$(ls -td /workspace/fasterwam_a3_bridge/outputs/real_inference/run_* | head -1); echo "RUN=$d"; cat "$d/summary.json"
```

> **seed 统一 42**（2026-10-26 起不再换 seed）。worker 源码里的 `default=11` 只是历史默认值，实际运行一律显式传 `--seed 42`。

## 11. 取回产物（结束前必做）`[云端 bash]` + JupyterLab

JupyterLab 只能下载**文件**（下载整个目录需先压缩）。**打包整个 `outputs/`**：

```bash
cd /workspace/fasterwam_a3_bridge && python3 -c "import zipfile,os; src='outputs'; z=zipfile.ZipFile('outputs_all.zip','w',zipfile.ZIP_DEFLATED); [z.write(os.path.join(r,f), os.path.relpath(os.path.join(r,f),'.')) for r,_,fs in os.walk(src) for f in fs]; z.close(); print('outputs_all.zip', os.path.getsize('outputs_all.zip'))" && ls -la /workspace/fasterwam_a3_bridge/outputs_all.zip
```

然后在 **JupyterLab File Browser** 里进入 `fasterwam_a3_bridge/`，右键 `outputs_all.zip` → **Download**。
本地解到 `E:\Innovative_practice\WAM\fasterwam_a3_bridge\`。

只打包最新一次 run（更小的替代方案）：

```bash
d=$(ls -td /workspace/fasterwam_a3_bridge/outputs/real_inference/run_* | head -1); echo "$d"; cd /workspace/fasterwam_a3_bridge/outputs/real_inference && python3 -c "import zipfile,glob,os; p=sorted(glob.glob('run_*'))[-1]; z=zipfile.ZipFile(p+'.zip','w',zipfile.ZIP_DEFLATED); [z.write(os.path.join(p,f), os.path.join(p,f)) for f in os.listdir(p)]; z.close(); print(p+'.zip', os.path.getsize(p+'.zip'))"
```

---

## 12. 已知坑（逐条对应症状）

| 症状 | 原因 / 解决 |
|---|---|
| 实例重启后 `/workspace` 只剩 `lost+found` | 实例退出即删；新模板要开 Workspace Storage(PVC) |
| `Upload Error / Failed to fetch` | 平台上传框坏了，改用 JupyterLab File Browser 上传 |
| 下载大小对、sha256 错 | CDN 传输损坏；删掉重下，只认 sha256 |
| `ModuleNotFoundError: gymnasium` | `pip3 install gymnasium` |
| `ModuleNotFoundError: termcolor`（traceback 指向 `fasterwam/utils/misc.py`） | `pip3 install termcolor pynvml rich` |
| `AssertionError: Seed outside the np.uint32 bounds!` | `set_global_seed` 要求 seed>0；seed=0 必崩，用 11 |
| 夹爪开度 >1 导致归一化越界 | `build_proprio` 里 `np.clip(opening, 0, 1)` |
| 401 / common files 下载失败 | compose override 必须带 `model.redirect_common_files=false` |
| 实例卡死/挂掉 | 多 worker 并发 GPU OOM；严格串行，`ps aux \| grep infer_a3_worker` 检查 |
| `libEGL warning`、`os_same_file_description` | 无害，忽略 |
| 容器内 git clone GitHub 失败 | GitHub 被墙；代码只能本地上传 |
| 用 PowerShell 语法粘到云端 bash | 报 `syntax error near unexpected token '('`；先看命令标签 |
| 浏览器自动化往 Jupyter 注入代码被拦 | 安全分类器拦截；改用 heredoc 粘贴 |
| `420`/`429 Client Error … hf-mirror.com/Wan-AI/Wan2.2-TI2V-5B/...` 或 `LocalEntryNotFoundError` | 加载器用 `hf_snapshot_download` 拉组件，撞镜像限流；按 §6.1 用 `curl -L` 预下载到 `$DIFFSYNTH_MODEL_BASE_PATH`，运行时加 `DIFFSYNTH_SKIP_DOWNLOAD=true` |
| `hydra.errors.MissingConfigException: In 'train': Could not find 'data/libero_2cam'` | zip 打包时 `--exclude=…\data` 误删了 `FasterWAM/configs/data/`；用下面的应急命令补回，或按 §3 用 `make_zips.py` 重打包 |

### 12.1 应急：补回 `configs/data/`（2 个 YAML）`[云端 bash]`

```bash
mkdir -p /workspace/FasterWAM/configs/data
cat > /workspace/FasterWAM/configs/data/libero_2cam.yaml <<'YAML1'
train:
  _target_: fasterwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset
  dataset_dirs:
    - ./data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot
    - ./data/libero_mujoco3.3.2/libero_object_no_noops_lerobot
    - ./data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot
    - ./data/libero_mujoco3.3.2/libero_10_no_noops_lerobot
  shape_meta:
    images:
      - key: image
      # - key: observation.images.image
        raw_shape: [3, 512, 512]
        shape: [3, 224, 224]
      - key: wrist_image
      # - key: observation.images.wrist_image
        raw_shape: [3, 512, 512]
        shape: [3, 224, 224]
    action:
      - key: default
      # - key: action
        raw_shape: 7 # before transform
        shape: 7 # after transform
    state:
      - key: default
      # - key: observation.state
        raw_shape: 8 # before transform
        shape: 8 # after transform
  num_frames: 33
  global_sample_stride: 1
  action_video_freq_ratio: 4 # 32 action, 9 video frames
  video_size: [224, 448] # final resize video
  camera_key: null
  val_set_proportion: 0.0
  is_training_set: true
  skip_padding_as_possible: false
  concat_multi_camera: "horizontal"
  processor:
    _target_: fasterwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor
    shape_meta: ${data.train.shape_meta}
    num_obs_steps: ${data.train.num_frames}
    num_output_cameras: 2
    action_output_dim: 7 # eef_pose (6) + gripper (1)
    proprio_output_dim: 8 # eef_pose (6) + gripper (2)
    delta_action_dim_mask:
      default: [true, true, true, true, true, true, false] # eef poses are delta, gripper is not

    # action & state normalization
    action_state_transforms: null
    # action_state_transforms:
    #   - _target_: fasterwam.datasets.lerobot.transforms.libero.LiberoTransform
    use_stepwise_action_norm: False
    norm_default_mode: min/max
    norm_exception_mode: null

    action_state_merger:
      _target_: fasterwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign

    train_transforms:
      - _target_: fasterwam.datasets.lerobot.transforms.image.ToTensor
      - _target_: torchvision.transforms.Resize
        size: [224, 224]
    val_transforms:
      - _target_: fasterwam.datasets.lerobot.transforms.image.ToTensor
      - _target_: torchvision.transforms.Resize
        size: [224, 224]
  text_embedding_cache_dir: ./data/text_embeds_cache_fasterwam/libero
  context_len: 128
YAML1
cat > /workspace/FasterWAM/configs/data/robotwin.yaml <<'YAML2'
train:
  _target_: fasterwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset
  dataset_dirs:
    - ./data/robotwin2.0/robotwin2.0
  shape_meta:
    images:
      - key: cam_high
        raw_shape: [3, 480, 640]
        shape: [3, 240, 320]
      - key: cam_left_wrist
        raw_shape: [3, 480, 640]
        shape: [3, 240, 320]
      - key: cam_right_wrist
        raw_shape: [3, 480, 640]
        shape: [3, 240, 320]
    action:
      - key: default
        raw_shape: 14 # before transform
        shape: 14 # after transform
    state:
      - key: default
        raw_shape: 14 # before transform
        shape: 14 # after transform
  num_frames: 33
  global_sample_stride: 1
  action_video_freq_ratio: 4 # 32 action, 9 video frames
  video_size: [384, 320] # final resize video
  camera_key: null
  val_set_proportion: 0.01
  is_training_set: true
  pretrained_norm_stats: ./data/robotwin2.0/dataset_stats.json
  skip_padding_as_possible: false
  concat_multi_camera: "robotwin"
  processor:
    _target_: fasterwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor
    shape_meta: ${data.train.shape_meta}
    num_obs_steps: ${data.train.num_frames}
    num_output_cameras: 3
    action_output_dim: 14
    proprio_output_dim: 14

    # action & state normalization
    action_state_transforms: null
    # action_state_transforms:
    #   - _target_: fasterwam.datasets.lerobot.transforms.libero.LiberoTransform
    use_stepwise_action_norm: False
    norm_default_mode: "z-score"
    norm_exception_mode: null

    action_state_merger:
      _target_: fasterwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign

    train_transforms:
      - _target_: fasterwam.datasets.lerobot.transforms.image.ToTensor
      - _target_: torchvision.transforms.Resize
        size: [240, 320]
    val_transforms:
      - _target_: fasterwam.datasets.lerobot.transforms.image.ToTensor
      - _target_: torchvision.transforms.Resize
        size: [240, 320]
  text_embedding_cache_dir: ./data/text_embeds_cache_fasterwam/robotwin
  context_len: 128

val:
  _target_: fasterwam.datasets.lerobot.robot_video_dataset.RobotVideoDataset
  dataset_dirs:
    - ./data/robotwin2.0/robotwin2.0
  shape_meta:
    images:
      - key: cam_high
        raw_shape: [3, 480, 640]
        shape: [3, 240, 320]
      - key: cam_left_wrist
        raw_shape: [3, 480, 640]
        shape: [3, 240, 320]
      - key: cam_right_wrist
        raw_shape: [3, 480, 640]
        shape: [3, 240, 320]
    action:
      - key: default
        raw_shape: 14 # before transform
        shape: 14 # after transform
    state:
      - key: default
        raw_shape: 14 # before transform
        shape: 14 # after transform
  num_frames: 33
  global_sample_stride: 1
  action_video_freq_ratio: 4 # 32 action, 9 video frames
  video_size: [384, 320] # final resize video
  camera_key: null
  val_set_proportion: 0.01
  is_training_set: false
  pretrained_norm_stats: ./data/robotwin2.0/dataset_stats.json
  skip_padding_as_possible: false
  concat_multi_camera: "robotwin"
  processor:
    _target_: fasterwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor
    shape_meta: ${data.train.shape_meta}
    num_obs_steps: ${data.train.num_frames}
    num_output_cameras: 3
    action_output_dim: 14
    proprio_output_dim: 14

    # action & state normalization
    action_state_transforms: null
    # action_state_transforms:
    #   - _target_: fasterwam.datasets.lerobot.transforms.libero.LiberoTransform
    use_stepwise_action_norm: False
    norm_default_mode: "z-score"
    norm_exception_mode: null

    action_state_merger:
      _target_: fasterwam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign

    train_transforms:
      - _target_: fasterwam.datasets.lerobot.transforms.image.ToTensor
      - _target_: torchvision.transforms.Resize
        size: [240, 320]
    val_transforms:
      - _target_: fasterwam.datasets.lerobot.transforms.image.ToTensor
      - _target_: torchvision.transforms.Resize
        size: [240, 320]
  text_embedding_cache_dir: ./data/text_embeds_cache_fasterwam/robotwin
  context_len: 128
YAML2
ls -l /workspace/FasterWAM/configs/data
```

期望：列出 `libero_2cam.yaml`（2,508 B）和 `robotwin.yaml`（4,279 B）。

---

## 13. 关键路径速查

```
云端
  /workspace/FasterWAM                                            # 代码
  /workspace/FasterWAM/checkpoints/fasterwam_release/libero/step_021700.pt
  /workspace/FasterWAM/checkpoints/fasterwam_release/libero/dataset_stats.json
  /workspace/a3_dual_arm_sim                                      # 场景
  /workspace/fasterwam_a3_bridge                                  # 适配器 + worker
  /workspace/fasterwam_a3_bridge/infer_a3_worker.py
  /workspace/fasterwam_a3_bridge/outputs/real_inference/run_*/    # 产物

本地
  E:\Innovative_practice\WAM\{FasterWAM,a3_dual_arm_sim,fasterwam_a3_bridge}
  C:\Users\ASUS\Desktop\{FasterWAM_code.zip,a3_dual_arm_sim.zip,fasterwam_a3_bridge.zip}
  C:\Users\ASUS\Documents\Qoder\2026-10-24\c3f102f0\heredoc_extracts\worker_1853_2.txt
  C:\Users\ASUS\Documents\Qoder\2026-10-24\c3f102f0\replay_local.py   # 本地 GUI 回放 NPZ
  C:\Users\ASUS\Documents\Qoder\2026-10-24\c3f102f0\make_zips.py      # 确定性打包（替代 tar）
  C:\Users\ASUS\Documents\Qoder\2026-10-24\c3f102f0\compare_zip.py    # 打包完整性自检
```

## 14. 推理管线要点（防止后人改坏）

1. 观测 → `prepare_fasterwam_images_official(obs)` 得 `(1,3,224,448)` float[-1,1]；必须与 `cfg.data.train.video_size=[224,448]` 一致。
2. proprio 8D = 左 EEF 位姿 6D + `a3_opening_to_panda_fingers(clip(opening,0,1))`；先 `action_state_transform` 再 `normalizer.forward`。
3. 动作后处理顺序固定：反归一化 → `*2-1` → `*-1`(invert) → `np.sign`(binarize)，最终 **+1=close / -1=open**。
4. 7D→14D：`FasterWAMA3ActionAdapterV3().convert(a7, R)`，`R` 取当前左 EEF 四元数（**wxyz**）转矩阵，每次 replan 重算。
5. 每 5 步 replan（`replan_steps`）；动作缩放 pos ×2.0。
6. `seed`：episode 级 `set_global_seed(seed, get_worker_init_fn=False)`，同时传 `infer_action(seed=…)`。

## 15. 回放诊断（CPU，无需模型）`[云端 bash]`

把已存 `executed_14d` 重灌进 A3 仿真（确定性，同 seed 可复现轨迹）。**v2 脚本**通过环境属性 `env.cookie_positions`（= `data.xpos[cookie_*]`，80 个饼干中心）计算「EEF 到最近饼干的真实距离」，用来量化视频里看到的「没碰到就闭合」。

关键常量（判断阈值用）：
- 饼干半尺寸 `[0.025, 0.00317, 0.0125] m`（直径 5 cm，厚 6.3 mm）→ 水平距离 <2.5 cm 才算在饼干正上方
- 每步位移上限 = `cartesian_translation_scale_m(0.025) × adapter 2.0 = 0.05 m/单位`（与 LIBERO OSC_POSE 相同）
- 夹爪每步开度变化上限 = `max_gripper_step = 0.08`（从 1.0 全开合到 0 约 13 个控制步）

```bash
cat > /workspace/fasterwam_a3_bridge/replay_diagnostic.py <<'PYEOF'
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/workspace/a3_dual_arm_sim/src")
from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv

npz = Path(sys.argv[1])
seed = int(sys.argv[2])
d = np.load(npz)
acts = d["executed_14d"]
print(f"npz={npz.name} seed={seed} actions_7d={d['actions_7d'].shape} executed_14d={acts.shape}", flush=True)

env = A3CookieTransferEnv(action_mode="cartesian_delta", render_cameras=False)
obs, info = env.reset(seed=seed)
c0 = env.cookie_positions.copy()
eef0 = np.asarray(obs["observation.eef_pose"], dtype=float)[:3].copy()
eef = eef0
src0 = info.get("cookies_in_source")
print("cookies_in_source at reset:", src0, flush=True)

min_dist, min_step = np.inf, -1
max_force = 0.0
max_lifted = 0
events = []
prev_cmd = None
for i, a in enumerate(acts):
    obs, reward, terminated, truncated, info = env.step(a)
    eef = np.asarray(obs["observation.eef_pose"], dtype=float)[:3]
    opening = float(obs["observation.state"][7])
    cps = env.cookie_positions
    dists = np.linalg.norm(cps - eef[None, :], axis=1)
    j = int(np.argmin(dists))
    d3 = float(dists[j])
    dxy = float(np.linalg.norm((cps[j] - eef)[:2]))
    if d3 < min_dist:
        min_dist, min_step = d3, i
    f = np.atleast_1d(np.asarray(obs.get("observation.force", [0.0]), dtype=float))
    max_force = max(max_force, float(np.max(np.abs(f))))
    max_lifted = max(max_lifted, int(np.sum(cps[:, 2] - c0[:, 2] > 0.02)))
    cmd = round(float(a[6]), 3)
    if prev_cmd is None or cmd != prev_cmd:
        events.append((i, cmd, round(opening, 3), round(dxy, 4), round(d3, 4), j,
                       info.get("cookies_in_source"), info.get("cookies_in_target")))
    prev_cmd = cmd
    if i % 50 == 0:
        print(f"step {i:03d} eef={np.round(eef,3)} open={opening:.3f} cookie#{j} "
              f"dxy={dxy:.4f} d3d={d3:.4f} src={info.get('cookies_in_source')} "
              f"tgt={info.get('cookies_in_target')}", flush=True)
    if terminated or truncated:
        print("ended early at step", i, "terminated=", terminated, "truncated=", truncated, flush=True)
        break

print("eef start/end:", np.round(eef0, 3), np.round(eef, 3),
      "displacement:", round(float(np.linalg.norm(eef - eef0)), 3), flush=True)
print("min eef-cookie 3d dist:", round(min_dist, 4), "at step", min_step, flush=True)
print("cookies_in_source:", src0, "->", info.get("cookies_in_source"),
      "| target:", info.get("cookies_in_target"), "| success:", info.get("success"), flush=True)
print("max |force| (obs):", round(max_force, 3),
      "| env peak:", getattr(env, "_max_recorded_gripper_force", None),
      "| max cookies lifted >2cm:", max_lifted, flush=True)
print("gripper cmd changes (step, cmd, opening, dxy, d3d, cookie_idx, src, tgt):", flush=True)
for e in events:
    print("  ", e, flush=True)
env.close()
PYEOF
```

批量回放当前实例上所有 run（按目录名映射起始 seed）：

```bash
cd /workspace/fasterwam_a3_bridge && for run in outputs/real_inference/run_*; do base=$(basename "$run"); case "$base" in run_20261025_180753|run_20261026_035243) s0=11;; run_20261026_040101) s0=21;; *) s0=42;; esac; for f in "$run"/episode_*.npz; do [ -e "$f" ] || continue; e=$(basename "$f" .npz | sed 's/episode_//'); echo "=== $base ep$e seed=$((s0+e)) ==="; python3 replay_diagnostic.py "$f" $((s0+e)); done; done
```

## 16. 待办

- [x] 装 termcolor/pynvml/rich、补 configs/data、预下载 Wan2.2、单 episode 验证
- [x] 批量 5 episodes（seeds 21–25）→ 5/5 失败，累计 0/6
- [x] 回放诊断 7 个 NPZ（§15）→ 全部「远距离闭合 + 后续不再闭合」
- [x] **seed 42 三组对照**（`run_guarded.py`：baseline / soft / soft_guard）→ 全部 0/10，包装器路线判死（§9.1.2–9.1.3）
- [x] 用 **seed 42** 做后续验证（不再换 seed）
- [x] **oracle 抓取探针**（§17）：`grasp_triggers 3` / `max_lifted_gt_2cm 2` / 0 进目标 → 动作可用、搬运缺失
- [x] 下载专家数据集 `Eter0109/a3-front-close-left-100` 并确认 schema（PNG 内嵌 parquet、无 videos/）
- [x] 预处理 `prepare_a3_arrays.py` → `frames.dat` + `arrays.npz`（sat 0%）
- [ ] 看 `verify_a3_replay.py` 输出，实锤 7D 动作约定（0.05 m / 0.5 rad / 夹爪 ±1）
- [ ] 写 `finetune/a3_video_dataset.py` + `configs/data/a3_2cam.yaml` + `configs/task/a3_fasterwam.yaml`（§19.4）
- [ ] 微调训练（先量单步耗时，再决定 LoRA/全参；`resume=<官方 ckpt>`）（§19.5）
- [ ] 打包下载 `outputs/`（含 `outputs/guarded/`、`outputs/oracle/`）（§11）
- [ ] 把 `action_adapter_v3.py` 的越限 `raise` 改成 `np.clip(a, -1, 1)`（本次 `soft_guard` 崩溃的直接原因）
- [ ] （下次开实例）模板开 SSH Access + Workspace Storage(PVC)

---

## 17. oracle 抓取探针（判断「动作可用性」）`[云端 bash]`

目的：零样本的**动作**（位置/旋转增量）到底好不好？把「抓取触发」换成几何规则，其余完全交给模型：

- 抓取点与最近饼干距离 `< 2 cm` → 位置/旋转增量置 0，夹爪闭合 15 步（0.3 s），冷却 40 步
- 其它时候完全执行模型动作（`max_gripper_step` 恢复默认 0.08）
- 所有动作 `np.clip(-1, 1)`，避免 `soft_guard` 那种崩溃

这样标注为 **zero-shot motion + oracle grasp**（不是纯零样本），用于估计「动作上界」。

`run_oracle.py`（与 `infer_a3_worker.py`、`run_guarded.py` 同目录）：

```python
import argparse
import json
import time

import imageio
import numpy as np
import torch
from hydra.utils import instantiate

import infer_a3_worker as W
import run_guarded as G

CLOSE_M = 0.020
DWELL = 15
COOLDOWN = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=400)
    args = ap.parse_args()

    cfg = W.load_cfg()
    W.model_device = "cuda" if torch.cuda.is_available() else "cpu"
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=W.model_device)
    model.load_checkpoint(str(W.CKPT))
    model = model.to(W.model_device).eval()
    stats = W.load_dataset_stats_from_json(str(W.STATS))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    adapter = W.FasterWAMA3ActionAdapterV3()
    print(f"[oracle] model ready device={W.model_device}", flush=True)

    out_dir = W.BRIDGE_DIR / "outputs" / "oracle" / time.strftime("run_%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[oracle] out={out_dir}", flush=True)

    seed = args.seed
    W.set_global_seed(int(seed), get_worker_init_fn=False)
    replan = int(cfg.EVALUATION.get("replan_steps", 5))
    env = G.make_env(0.08)
    obs, info = env.reset(seed=seed)
    c0 = env.cookie_positions.copy()

    pending, chunks, executed = [], [], []
    frames, merged, eef_log, dist_log, open_log, hold_log = [], [], [], [], [], []
    current_rgb = None
    holding, cooldown, triggers, max_lifted = 0, 0, 0, 0
    t0 = time.time()
    step = -1

    for step in range(args.max_steps):
        if not pending:
            current_rgb, chunk7 = W.predict_action_chunk(
                obs, model, processor, cfg, seed, W.model_device)
            chunks.append(chunk7)
            R = W.quat_wxyz_to_matrix(
                np.asarray(obs["observation.eef_pose"][3:7], dtype=np.float64))
            pending = [adapter.convert(a, R) for a in chunk7[:replan]]

        a3 = np.asarray(pending.pop(0), dtype=np.float64).copy()
        dist = G.nearest_cookie_dist(env)
        hold = 0
        if holding > 0:
            a3[:6] = 0.0
            a3[6] = -1.0
            holding -= 1
            hold = 1
        elif cooldown > 0:
            cooldown -= 1
        elif dist < CLOSE_M:
            a3[:6] = 0.0
            a3[6] = -1.0
            holding = DWELL - 1
            cooldown = COOLDOWN
            triggers += 1
            hold = 1

        applied = np.clip(a3, -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(applied)
        executed.append(applied)
        eef_log.append(np.asarray(obs["observation.eef_pose"][:3], dtype=np.float64))
        dist_log.append(G.nearest_cookie_dist(env))
        open_log.append(float(obs["observation.state"][7]))
        hold_log.append(hold)
        max_lifted = max(max_lifted,
                         int(np.sum(env.cookie_positions[:, 2] - c0[:, 2] > 0.02)))
        frames.append(np.asarray(obs["observation.images.front"]).copy())
        merged.append(current_rgb.copy())

        if (step + 1) % 25 == 0:
            print(f"[oracle] step={step + 1} cookies={info.get('cookies_in_target')} "
                  f"dist={dist_log[-1]:.4f} open={open_log[-1]:.3f} "
                  f"grasps={triggers} src={info.get('cookies_in_source')}", flush=True)

        if terminated or truncated:
            break

    record = {
        "variant": "oracle_grasp",
        "seed": seed,
        "steps": step + 1,
        "replans": len(chunks),
        "success": bool(info.get("success", False)),
        "cookies_in_target": int(info.get("cookies_in_target", -1)),
        "cookies_in_source": int(info.get("cookies_in_source", -1)),
        "grasp_triggers": int(triggers),
        "max_lifted_gt_2cm": int(max_lifted),
        "min_cookie_dist": round(float(min(dist_log)), 4),
        "duration_s": round(time.time() - t0, 1),
    }
    np.savez_compressed(
        out_dir / f"oracle_episode_{seed}.npz",
        actions_7d=np.stack(chunks),
        executed_14d=np.asarray(executed),
        eef_xyz=np.asarray(eef_log),
        opening=np.asarray(open_log),
        cookie_dist=np.asarray(dist_log),
        hold=np.asarray(hold_log),
    )
    imageio.mimsave(out_dir / f"oracle_episode_{seed}_front.mp4", frames, fps=30)
    imageio.mimsave(out_dir / f"oracle_episode_{seed}_model_input.mp4", merged, fps=30)
    with (out_dir / "summary.json").open("w") as f:
        json.dump([record], f, indent=2)
    print(f"[oracle] done: {json.dumps(record)}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
```

运行（`run_guarded.py` 必须已在同目录，它提供 `make_env` / `nearest_cookie_dist`）：

```bash
cd /workspace/fasterwam_a3_bridge && python3 -m py_compile run_oracle.py && DIFFSYNTH_MODEL_BASE_PATH=/workspace/FasterWAM/checkpoints DIFFSYNTH_SKIP_DOWNLOAD=true MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH="/workspace/FasterWAM/src:/workspace/FasterWAM:/workspace/a3_dual_arm_sim/src:/workspace/fasterwam_a3_bridge" python3 run_oracle.py --seed 42 --max-steps 400
```

判读：

| 输出 | 含义 | 下一步 |
|---|---|---|
| `cookies_in_target` 增加、`max_lifted_gt_2cm` ≥ 1 | 零样本**动作可用**，只差抓取时机 | 微调只需修抓取相位，成功率高 |
| 有 lift 但进不了目标盘 | 抓取会了、搬运/交接不会 | 微调数据要覆盖「搬→放」段 |
| `grasp_triggers` 多但 `max_lifted=0` | 连几何规则都夹不住（指尖高度/开口宽度问题） | 先改夹爪几何/阈值，再做数据 |

---

## 18. 微调路线（探针之后启动）

1. **采数据**：`a3_dual_arm_sim` 自带专家控制器（`controllers/cookie_expert.py`、`cookie_2x10_expert.py`），用 `data/recording.py`（LeRobot v3 writer）录 N 条 episode，观测字段必须与 FasterWAM 训练输入一致（front + 双 wrist 相机、8D proprio、7D LIBERO 格式动作）。
2. **数据配置**：`configs/data/libero_2cam.yaml` 为模板改出 A3 版本（相机数、分辨率 224×448、`num_frames=33`、`action_video_freq_ratio=4`）。
3. **训练预算**：AMDu 实例 1 credit/GPU-hour、当前 15 credits → 微调前必须先算清「每 step 时间 × 计划步数」，建议先做 LoRA + 小步数跑通 1 条链路，再决定全参。
4. **评测**：同一 `seed 42`、同一 `run_guarded.py baseline` 变体，与 0/7 零样本基线直接对比。

> **2026-10-26 更新**：不再需要自采数据。官方仓库作者已发布同任务专家数据集 `Eter0109/a3-front-close-left-100`（100 条，`same_column` 专家），采集路径改为「下载 + 转换」，见 §19。

---

## 19. GitHub 发布：文件清单、目录结构与完整执行流程

> 目的：新建一个 GitHub 仓库（或分支），**只放本项目原创/恢复的文件**；三个上游仓库、模型权重、Wan 组件、演示数据集全部记录来源与下载命令，不进仓库。照 §19.6 的流程即可从零复现到「微调评测」。

### 19.1 资源来源总表（不进仓库的外部资源）

| 资源 | 来源 | 放置位置（云端） | 备注 |
|---|---|---|---|
| FasterWAM 源码 | github.com/hustvl/FasterWAM，commit **`83667817df0d4f823f39d90700e61ea2f432ac45`** | `/workspace/FasterWAM` | 云端 GitHub 被墙 → 本地 zip 上传（§3–4） |
| a3_dual_arm_sim 源码 | github.com/Eter0109/a3_dual_arm_sim（相机默认值与数据集对齐的 commit `ff552c4fe54fe344aeaff4c65270ee0fd2af4153`） | `/workspace/a3_dual_arm_sim` | 同上，zip 上传 |
| LIBERO / robosuite | pip：`libero==0.1.1 --no-deps`、`robosuite==1.4.0`、`bddl`、`robomimic --no-deps`（§5） | Python 3.12.3 环境 | 只有零样本 LIBERO 评测需要；A3 微调路径不碰 LIBERO 场景 |
| 官方 checkpoint | HF `hustvl/FasterWAM` → `libero/step_021700.pt`（11,117,679,929 B，sha256 `01be4efc…`）+ `libero/dataset_stats.json`（40,952 B，sha256 `c06dc67f…`） | `/workspace/FasterWAM/checkpoints/fasterwam_release/libero/` | 下载命令 §6；**微调训练用 `resume=` 从此 ckpt 起跑** |
| Wan2.2 VAE/T5 | HF `Wan-AI/Wan2.2-TI2V-5B`（`Wan2.2_VAE.pth` 2.82 GB、`models_t5_umt5-xxl-enc-bf16.pth` 11.4 GB） | `checkpoints/Wan-AI/Wan2.2-TI2V-5B/` | 预下载绕 429：§6.1 |
| Wan2.1 tokenizer | HF `Wan-AI/Wan2.1-T2V-1.3B` → `google/umt5-xxl/*`（4 个文件） | `checkpoints/Wan-AI/Wan2.1-T2V-1.3B/` | §6.1 |
| 专家演示数据集 | HF **dataset** `Eter0109/a3-front-close-left-100` @ `8fdd8a7caf8275eba8be8a83b53a99176a084062`（4.1 GB，100 eps / 42681 帧 / 20 Hz） | `/workspace/a3_data/a3-front-close-left-100/` | **无 videos/ 目录，图像以 PNG 字节内嵌 parquet**；下载必须带 `--repo-type dataset`（§19.6 阶段 6） |

### 19.2 本仓库文件清单（原创/恢复，进 GitHub）

| 仓库内路径 | 来源 | 说明 |
|---|---|---|
| `README.md` | 本文件 | 完整复现指南（照抄零试错） |
| `bridge/infer_a3_worker.py` | 本会话从零重建（§8 全文；**E 盘无此文件**，云端/§8 heredoc 是唯一权威副本） | 模型推理 worker，含两处修复：夹爪 opening `np.clip`、`--seed` 默认 11 |
| `bridge/action_adapter_v3.py` | E 盘原版 | 7D→14D 适配；**待改**：越限 `raise ValueError` → `np.clip(a, -1, 1)` |
| `bridge/observation_adapter.py` | E 盘原版 | `eef_pose`(14D)→左臂 6D 状态 + `quat_wxyz_to_axis_angle` |
| `bridge/gripper_proprio_adapter.py` | E 盘原版 | `a3_opening_to_panda_fingers`（`[0.04g, -0.04g]`） |
| `bridge/vision_adapter.py`、`vision_adapter_official.py` | E 盘原版 | 图像适配参考（worker 内嵌 `_center_crop_resize`，这两个文件不参与运行） |
| `bridge/run_guarded.py` | 本会话新增 | baseline / soft / soft_guard 三变体对照（§9.1） |
| `bridge/run_oracle.py` | 本会话新增 | oracle 抓取探针（§17 全文） |
| `bridge/test_closed_loop_a3.py` | E 盘原版 | 早期闭环测试（`R=np.eye(3)` 占位，仅历史参考） |
| `finetune/prepare_a3_arrays.py` | 本会话新增 | v3.0 parquet → `frames.dat` memmap + `arrays.npz`（7D 动作/8D proprio） |
| `finetune/make_stats.py` | 本会话新增 | 按**经验 min/max** 生成 `dataset_stats.json`（训练与推理共用） |
| `finetune/verify_a3_replay.py` | 本会话新增 | 用数据集内 `seed` 重放 env 校验动作约定 |
| `finetune/a3_video_dataset.py` | 待写 | 训练 Dataset（读 memmap，输出与 `RobotVideoDataset` 同构的 sample dict） |
| `finetune/configs/data/a3_2cam.yaml` | 待写 | 数据配置（以 `libero_2cam.yaml` 为模板，`_target_` 指向上者） |
| `finetune/configs/task/a3_fasterwam.yaml` | 待写 | 训练任务配置（`resume=<ckpt>`、小 batch、cosine） |
| `tools/replay_local.py` | 本会话新增 | 本地 GUI 回放 NPZ（无 GPU） |
| `tools/make_zips.py`、`compare_zip.py` | 本会话新增 | 确定性打包 + 清单比对（修复 tar `--exclude` 误删 `configs/data/` 的坑，§12） |
| `tools/extract_heredocs.py` | 本会话新增 | 从聊天记录提取 heredoc |
| `results/`（可选） | 运行产物 | 只放 `summary.json` 等文本；MP4/NPZ 大文件下载回本地即可，不进仓库 |

**不默认上传**：E 盘桥接目录里其余 ~20 个 `diagnose_*/test_*` 历史调试脚本、`outputs/`、`outputs.zip`、`__pycache__/`、`.venv/`。

### 19.3 新仓库目录树

```text
fasterwam-a3-cookie/                    # 新 GitHub 仓库
├── README.md                           # 本文件（19.6 为从零执行流程）
├── bridge/                             # 模型 ↔ A3 桥接（原创）
│   ├── infer_a3_worker.py
│   ├── action_adapter_v3.py
│   ├── observation_adapter.py
│   ├── gripper_proprio_adapter.py
│   ├── vision_adapter.py
│   ├── vision_adapter_official.py
│   ├── run_guarded.py
│   ├── run_oracle.py
│   └── test_closed_loop_a3.py
├── finetune/                           # 微调管线（原创）
│   ├── prepare_a3_arrays.py
│   ├── make_stats.py
│   ├── verify_a3_replay.py
│   ├── a3_video_dataset.py             # 待写
│   └── configs/
│       ├── data/a3_2cam.yaml           # 待写
│       └── task/a3_fasterwam.yaml      # 待写
├── tools/                              # 本地辅助
│   ├── replay_local.py
│   ├── make_zips.py
│   ├── compare_zip.py
│   └── extract_heredocs.py
└── results/                            # 可选：文本摘要（summary.json），不放大文件
```

### 19.4 动作约定（数据转换的依据，改动前必读）

微调数据的 7D 动作与运行时适配器 `action_adapter_v3` **互为逆映射**（§14）：

| 通道 | 训练数据（本仓库转换脚本产出） | 运行时适配器（执行方向） |
|---|---|---|
| 位置 0–2 | `(p_{t+1} − p_t) / 0.05`（世界系，LIBERO 单位） | `×2` 后 env 按 0.025 m/单位执行 ⇒ 1 单位 = 0.05 m |
| 旋转 3–5 | `rotvec(R_{t+1} R_tᵀ) / 0.5`（世界系，单位 0.5 rad） | `×0.5` 后 `R.T@`（世界→EEF 局部）再 `/0.1` |
| 夹爪 6 | opening 下降 = `+1`（闭），上升 = `−1`（开） | `−sign()` 得 A3 约定（−1 闭 / +1 开） |
| proprio 0–2 | TCP 位置（`L_eef` site，法兰沿接近轴 +0.145 m） | 与 worker 运行时一致 |
| proprio 3–5 | `quat_wxyz_to_axis_angle`（LIBERO 约定） | 同左 |
| proprio 6–7 | `[0.04·opening, −0.04·opening]`（Panda 手指近似） | 同左 |

动作归一化用**经验 min/max**（`make_stats.py`，min/max 模式），训练与推理共用同一份 `dataset_stats.json`；位置增量均值仅 0.005–0.03 单位，**不得**把区间写死成 [−1, 1]。

### 19.5 微调数据管道（已跑通部分）

```
a3-front-close-left-100 (HF, 4.1GB, PNG 内嵌 parquet)
        │  prepare_a3_arrays.py（已跑通：42681 帧，sat 0%，夹爪 16–24 段/条）
        ▼
/workspace/a3_data/a3_fw/
  frames.dat      uint8 [42681, 224, 448, 3]（front|left_wrist 拼接，12.85 GB）
  arrays.npz      action[42681,7]、proprio[42681,8]、opening、episode 表
        │  make_stats.py（已写；经验 min/max）
        ▼
  dataset_stats.json   ← 训练 Dataset 与推理 worker 共用
        │  verify_a3_replay.py（已写，等结果）：用 ep 的 seed 重置 env，
        ▼   7D 动作过 adapter 逐帧重放，对比真实 eef_pose
  通过 ⇒ 动作约定实锤 → a3_video_dataset.py（待写）→ 训练
```

### 19.6 完整执行流程（从零到微调评测）

每条命令的完整参数见对应章节；这里只给阶段顺序、关键参数与验收标准。终端标签：阶段 0–8 全部 `[云端 bash]`（本地准备在阶段 1 用 `[本地 cmd]`）。

**阶段 0：开实例**（§1）
- Blank Notebook 模板（Python 3.12.3）；实例退出即销毁，产物先下载。
- 验收：`python3 -c "import torch; print(torch.cuda.is_available())"` → `True`（§2）。

**阶段 1：代码就位**（§3–4，`[本地 cmd]` 打包 + JupyterLab 上传）
- 本地 zip：`FasterWAM_code.zip`（1.7 MB，排除 `.venv/.venvs/checkpoints/third_party/data/code/.runtime/…`）、`a3_dual_arm_sim.zip`（22.6 MB）、`fasterwam_a3_bridge.zip`（1.5 MB，排除 `outputs/`）。
- ⚠️ 打包必须用 `tools/make_zips.py`（`tar --exclude` 会静默吞掉 `configs/data/*.yaml`，§12 的坑）。
- 新仓库 zip（本 §19.2 文件）单独上传；`bridge/infer_a3_worker.py`、`finetune/*` 从仓库解压，**不要**用旧 zip 里的 worker。
- 验收：`python3 -c "import zipfile; ..."` 解压后三目录存在，`compare_zip.py` missing=0。

**阶段 2：依赖**（§5）
- `pip3 -i https://mirrors.aliyun.com/pypi/simple/` 依次：mujoco==3.3.2 → robosuite==1.4.0 → `libero==0.1.1 --no-deps` → `bddl --no-deps` → `robomimic --no-deps` → 一次性 bulk（easydict av thop wandb jsonlines GitPython einops omegaconf hydra-core imageio scipy h5py matplotlib opencv-python imageio-ffmpeg numba pandas transformers diffusers accelerate huggingface_hub gymnasium termcolor pynvml）。
- ⚠️ 已装好的别重复装；~300 kB/s 是带宽上限，不要中途换源。
- 验收：4 个关键包（transformers/diffusers/accelerate/huggingface_hub）可 import。

**阶段 3：模型文件**（§6 + §6.1）
- checkpoint + stats：curl 直连 `hf-mirror.com/hustvl/FasterWAM/resolve/main/libero/{step_021700.pt,dataset_stats.json}`（76–99 MB/s）。
- Wan2.2：同法预下载到 `checkpoints/Wan-AI/...`，运行设 `DIFFSYNTH_MODEL_BASE_PATH=/workspace/FasterWAM/checkpoints DIFFSYNTH_SKIP_DOWNLOAD=true`。
- 验收：sha256 与 §0 两值一致。

**阶段 4：A3 冒烟**（§7）
- 验收：`ENV OK`，80 源饼干 / required 10。

**阶段 5：（可选，已做过）零样本推理**（§8–9）
- 结论已固化：0/7、抓取相位错位；新实例无需重跑。若要复现：`infer_a3_worker.py --seed 42 --episodes 1 --max-steps 400`。

**阶段 6：数据集 + 预处理 + 校验**（§19.5，本次新增）
```bash
cd /workspace/a3_data && HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=0 \
huggingface-cli download Eter0109/a3-front-close-left-100 --repo-type dataset \
  --revision 8fdd8a7caf8275eba8be8a83b53a99176a084062 --local-dir a3-front-close-left-100
python3 /workspace/a3_data/prepare_a3_arrays.py     # ~3-6 分钟，产出 12.85 GB memmap
python3 /workspace/a3_data/make_stats.py
PYTHONPATH="/workspace/a3_dual_arm_sim/src:/workspace/fasterwam_a3_bridge" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python3 /workspace/a3_data/verify_a3_replay.py
```
- 验收：`sat_fraction ≈ 0`；`verify` 中 demo-act rel-RMSE ≪ zero-action（例如 <3 cm vs >20 cm）、`reset offset ≈ 0`。

**阶段 7：微调训练**（§19.4 配置就位后）
- 先量单步耗时（`max_steps=3` 试跑），再按 15 credits 预算决定 LoRA/全参。
- 入口：`python3 scripts/train.py task=a3_fasterwam ... resume=$PWD/checkpoints/fasterwam_release/libero/step_021700.pt output_dir=...`（不走 deepspeed 脚本，单卡 `accelerate` 内部路径即可；`batch_size 16` 改 1–2 + `gradient_accumulation_steps`）。
- 训练前预计算文本嵌入：`scripts/precompute_text_embeds.py`（指令 = `transfer 10 cookies into target box`，**先修 worker 的 "…to the right robot arm" 指令不匹配问题**）。
- 验收：loss 下降；save_every 产出 ckpt。

**阶段 8：评测**（§9 baseline 变体）
- 同一 seed 42、同一 `run_guarded.py baseline`，`processor.set_normalizer_from_stats` 换成 `a3_fw/dataset_stats.json`。
- 验收：与零样本 0/7 基线对比 `cookies_in_target`。

**阶段 9：产物下载**（§11，退出实例前必做）
- `python3 -c "import zipfile,os;..."` 打包 `outputs/` + 训练产物 → JupyterLab 文件浏览器下载 → 本地 `du` 核对 → 才退出实例。
