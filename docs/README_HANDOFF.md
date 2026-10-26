# FasterWAM → A3 饼干夹取：交接说明（2026-10-27）

> 本文件对应云端实例 `rgapi-one-click-24316-57b6b870`（anruicloud，Radeon 48 GB，容器内 `/workspace`）。
> 实例关闭后 `/workspace` 会整体销毁；本压缩包 + 两个大文件（见文末）是恢复现场所需的全部内容。
> Jupyter token 没有写进任何文件，如需继续请重新从 JupyterLab 获取。

---

## 1. 当前进度速览

| 项 | 状态 | 结果 |
|---|---|---|
| 环境 / 依赖 | ✅ | Python 3.12.3 + ROCm；mujoco 3.3.2、robosuite 1.4.0、libero/bddl/robomimic、transformers 4.57.6、accelerate 1.13.0 |
| A3 场景 | ✅ | `A3CookieTransferEnv`，80 块源饼干 / 目标盘 required 10 |
| 专家数据 | ✅ | HF dataset `Eter0109/a3-front-close-left-100`（42681 帧，20 Hz），已转成 `frames.dat` + `arrays.npz` |
| 微调 | ✅ 完成 | LoRA（392 层 q/k/v/o，30.7 M 参数），batch 6、2000 步、lr 2e-4，75 min，峰值显存 28.5/48 GB |
| 合并权重 | ✅ | `step_002000_merged.pt` 11,117,977,659 B（r26/r27 使用的就是这个）；已上传 HF `wsadwsad6779/fasterwam-a3-cookie-lora` |
| 裸模型评测 | ❌ 0/10 | 合并权重直接跑 400 步：`cookies_in_target=0`，65.4 s |
| 启发式包装器 r26 | ⚠️ 4/10 | 600 步，175.4 s；第二批插入时倾斜 18–22°，把第一批顶出盒沿 |
| 启发式包装器 **r27** | ⚠️ **5/10** | 600 步，326.8 s；第一批 5 块全进；第二批已释放但计数没增加，仍未 success |
| 距目标 | ❌ | 需要 10/10；当前最好 5/10 |

**一句话结论**：微调本身可复现且 loss 正常收敛，但裸模型仍然不会完成搬运；4–5/10 的成绩来自 `a3_wrap.py` 的“到达/闭合/搬运/放置”启发式包装器。r27 修掉了第二批被挤散的问题，但第二批释放后没有被计入目标盘，是当前唯一待解的关键点。

---

## 2. r26 / r27 到底改了什么

### r26（4/10）
- 第一批：5 块全部进目标盘（step 200 `cookies=5`）。
- 第二批：抓取时列内已有 6.9 mm 散布；插入阶段左端比内壁多约 1.5 mm，蹭壁后整叠倾斜 **18–22°**，把第一批顶到 +y 盒沿，计数从 5 → 2 → 4。
- 指标：`zfloor_steps=47`、`max_pad_force=38.1`、`min_cookie_dist=0.0038`。

### r27（5/10）
- 诊断发现：第一批取走后 **col0 剩下的前排 5 块仍然完全对齐（散布 0.00 mm）**，但 `nearest_cookie` / `batch_center` 里“排除已入箱饼干”的过滤用了 ±(半宽 + 60 mm)，把正好在盒子上方的整列 col0 也排除了，于是第二批只能去抓已被挤散的 col1。
- 修复 `patch_col0.py`：把排除边距 **60 mm → 8 mm**，只排除真正落在盒体积内的饼干；`patch_place2.py` 的“按实测批次中心在剩余空隙内居中”保留。
- 结果：第二批改为从 col0 抓取；`zfloor_steps=0`、`max_pad_force=17.1`（插入明显更干净），第一批也保持 5 块。
- **新问题**：第二批在 step 425 释放后，`cookies_in_target` 仍是 5（`placed=2` 只统计释放事件）。说明第二批没有进入“计数体积”：最可能是叠在第一批上方 / 落在盒沿外侧。

---

## 3. 关键参数（复现必须一致）

```text
bin position      = [0.095, 0.1, 0.749]
inner_half        = [0.057, 0.029]      # 盒内半宽 (m)
wall height       = 0.026
replan            = 10                  # run_wrapped.py --replan 10
instruction       = 'transfer 10 cookies into target box'
seed              = 42
max steps         = 600
```

---

## 4. 从零复现到 r27

### 4.1 实例与依赖
- 新建 Radeon Cloud「Blank Notebook」实例（Python 3.12.3）；`/workspace` 对应 JupyterLab 文件浏览器根目录。
- 安装依赖（Aliyun 镜像）：

```bash
pip3 install -i https://mirrors.aliyun.com/pypi/simple/ mujoco==3.3.2 numpy==2.1.2 robosuite==1.4.0 gymnasium
pip3 install -i https://mirrors.aliyun.com/pypi/simple/ libero --no-deps && pip3 install -i https://mirrors.aliyun.com/pypi/simple/ bddl --no-deps && pip3 install -i https://mirrors.aliyun.com/pypi/simple/ robomimic --no-deps
pip3 install -i https://mirrors.aliyun.com/pypi/simple/ easydict av thop wandb jsonlines GitPython einops omegaconf hydra-core imageio imageio-ffmpeg scipy h5py matplotlib opencv-python numba pandas transformers diffusers accelerate huggingface_hub termcolor pynvml rich
```

### 4.2 代码就位
- 上游代码（不进本仓库）：
  - `https://github.com/hustvl/FasterWAM` @ `83667817df0d4f823f39d90700e61ea2f432ac45` → `/workspace/FasterWAM`
  - `https://github.com/Eter0109/a3_dual_arm_sim` @ `ff552c4fe54fe344aeaff4c65270ee0fd2af4153` → `/workspace/a3_dual_arm_sim`
- 本项目原创代码：把本包的 `bridge/`、`finetune/` 放回 `/workspace/fasterwam_a3_bridge/`、`/workspace/FasterWAM/`（finetune 里的文件按包内相对路径覆盖/新增）。
- 容器内 GitHub 被墙，代码只能用本地上传（JupyterLab 拖拽即可）。

### 4.3 模型 / 数据下载
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download hustvl/FasterWAM --include "libero/step_021700.pt" "libero/dataset_stats.json" \
  --local-dir /workspace/FasterWAM/checkpoints/fasterwam_release
huggingface-cli download Eter0109/a3-front-close-left-100 --repo-type dataset \
  --revision 8fdd8a7caf8275eba8be8a83b53a99176a084062 \
  --local-dir /workspace/a3_data/a3-front-close-left-100
```
另外需要 Wan2.2 VAE/T5 与 Wan2.1 tokenizer（约 14 GB），**必须预先按文件名放好**，否则运行时 `fasterwam/runtime.py` 会走 snapshot resolver 触发 HF 429：

```bash
B=/workspace/FasterWAM/checkpoints
VAE=$B/Wan-AI/Wan2.2-TI2V-5B
TOK=$B/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl
mkdir -p "$VAE" "$TOK"
curl -L -C - -o "$VAE/Wan2.2_VAE.pth" "https://hf-mirror.com/Wan-AI/Wan2.2-TI2V-5B/resolve/main/Wan2.2_VAE.pth"
curl -L -C - -o "$VAE/models_t5_umt5-xxl-enc-bf16.pth" "https://hf-mirror.com/Wan-AI/Wan2.2-TI2V-5B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth"
for f in special_tokens_map.json tokenizer_config.json spiece.model tokenizer.json; do
  curl -L -o "$TOK/$f" "https://hf-mirror.com/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/google/umt5-xxl/$f"
done
```

完整历史复现记录（含每一步的验收输出、hash）见仓库 `docs/README_REBUILD_FASTERWAM_A3.md`（https://github.com/wsadwsad6779/fasterwam-a3-cookie/blob/a3-cookie-r27/docs/README_REBUILD_FASTERWAM_A3.md）；本机同目录另有一份，不进交接压缩包。

### 4.4 数据预处理（云上约 10 分钟）
```bash
cd /workspace/a3_data && python3 prepare_a3_arrays.py   # -> a3_fw/frames.dat + arrays.npz
python3 make_stats.py                                   # -> a3_fw/dataset_stats.json
```

### 4.5 LoRA 微调（75 分钟 / 48 GB 卡）
```bash
cd /workspace/FasterWAM
accelerate launch scripts/train.py data=a3_2cam task=a3_fasterwam \
  batch_size=6 max_steps=2000 learning_rate=2e-4 log_every=20 \
  output_dir=/workspace/a3_train_runs/a3_lora_r1 \
  resume=/workspace/FasterWAM/checkpoints/fasterwam_release/libero/step_021700.pt
```
> 命令由本次实际生成的 `a3_train_runs/a3_lora_r1/config.yaml` 反推，参数与日志一致。5.56B 全参在 48 GB 上会 OOM，LoRA 是必须的。

### 4.6 合并 LoRA（得到推理用 checkpoint）
```bash
cd /workspace/FasterWAM
python3 scripts/export_lora_merged.py data=a3_2cam task=a3_fasterwam \
  lora_ckpt=/workspace/a3_train_runs/a3_lora_r1/checkpoints/step_002000.pt \
  output=/workspace/a3_train_runs/a3_lora_r1/step_002000_merged.pt
```
> `lora_ckpt` 的官方路径是 `checkpoints/step_002000.pt`，本机实际落在
> `checkpoints/weights/step_002000.pt`（11,179,613,340 B，基座 + 未折叠 LoRA），按实际路径传参即可。
>
> **不想重训**：三个大文件都已上传到 HF 公开仓库 `wsadwsad6779/fasterwam-a3-cookie-lora`：
>
> ```bash
> cd /workspace/a3_train_runs/a3_lora_r1
> # 推理用（LoRA 已折叠，r26/r27 实际用的就是它，11.1 GB）
> hf download wsadwsad6779/fasterwam-a3-cookie-lora step_002000_merged.pt --local-dir .
> # 重新合并 / 换 scaling 用（基座 + 未折叠 LoRA，11.18 GB）
> hf download wsadwsad6779/fasterwam-a3-cookie-lora trainer/step_002000.pt --local-dir ckpt_from_hf
> # 精确续训用（优化器 123 MB + scheduler/RNG + 模型 safetensors 12.59 GB）
> hf download wsadwsad6779/fasterwam-a3-cookie-lora trainer/state/step_002000 --local-dir state_from_hf
> ```
>
> 直链前缀：`https://huggingface.co/wsadwsad6779/fasterwam-a3-cookie-lora/resolve/main/`
> （`step_002000_merged.pt`、`trainer/step_002000.pt`、`trainer/state/step_002000/*`、`data/frames.dat`）

### 4.7 裸模型评测（0/10，用于对照）
```bash
cd /workspace/fasterwam_a3_bridge
FASTERWAM_A3_CKPT=/workspace/a3_train_runs/a3_lora_r1/step_002000_merged.pt \
FASTERWAM_A3_INSTRUCTION='transfer 10 cookies into target box' \
DIFFSYNTH_MODEL_BASE_PATH=/workspace/FasterWAM/checkpoints DIFFSYNTH_SKIP_DOWNLOAD=true \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
PYTHONPATH="/workspace/FasterWAM/src:/workspace/FasterWAM:/workspace/a3_dual_arm_sim/src:/workspace/fasterwam_a3_bridge" \
python3 infer_a3_worker.py --episodes 1 --seed 42 --max-steps 400
```

### 4.8 r27 复现（当前最好结果 5/10）
```bash
cd /workspace/fasterwam_a3_bridge
FASTERWAM_A3_CKPT=/workspace/a3_train_runs/a3_lora_r1/step_002000_merged.pt \
FASTERWAM_A3_INSTRUCTION='transfer 10 cookies into target box' \
DIFFSYNTH_MODEL_BASE_PATH=/workspace/FasterWAM/checkpoints DIFFSYNTH_SKIP_DOWNLOAD=true \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
PYTHONPATH="/workspace/FasterWAM/src:/workspace/FasterWAM:/workspace/a3_dual_arm_sim/src:/workspace/fasterwam_a3_bridge" \
python3 run_wrapped.py --seed 42 --max-steps 600 --variants full --replan 10
```

---

## 5. 压缩包内容

```text
a3_handoff_20261027/
├── docs/                      README_HANDOFF.md（本文件）+ 三份历史 handoff 文档
│                              （README_FasterWAM_2026-10-23 / _A3_NATIVE_PATHS_2026-10-21 /
│                              _Project_Handoff）
├── bridge/                    a3_wrap.py（r27 当前版）、patch_col0.py、patch_place2.py、
│                              run_wrapped.py、infer_a3_worker.py、适配器、全部 r* 诊断脚本、
│                              a3_wrap.py.bak3–bak10 历史备份
├── finetune/                  a3_video_dataset.py、lora.py、export_lora_merged.py、
│                              prepare/make_stats/verify、configs、a3_lora_r1/config.yaml、
│                              训练/合并/评测日志（含 smoke/env/disk/rollout）、
│                              upstream_backups/（官方 fasterwam.yaml / pyproject / uv.lock 原件）
├── data/                      arrays.npz、dataset_stats.json、summary.json、info.json、
│                              a3_left_pickup_actions.npz、text_embeds_cache、数据集元信息、
│                              _smoke_v2.log / _smoke_eval.log
├── results/
│   ├── wrapped/               run_20261026_164924（r26）、run_20261026_173807（r27）等全部 30 次
│   ├── guarded/               baseline/soft 对照
│   ├── oracle/                oracle 抓取探针
│   ├── real_inference/        零样本评测
│   └── logs/                  全部顶层 r*.log / probe*.log / geo*.log / diag*.log
├── MANIFEST.txt               每个文件的字节数 + sha256
└── LARGE_FILES.md             未打包大文件清单
```

---

## 6. 未打包的大文件（关闭实例前处理）

全部 4 个（约 48 GB）都在 HF 公开仓库 `wsadwsad6779/fasterwam-a3-cookie-lora`，云端实例可以放心销毁。

| 云端路径 | 大小 (B) | HF 内路径 |
|---|---|---|
| `/workspace/a3_train_runs/a3_lora_r1/step_002000_merged.pt` | 11,117,977,659 | `step_002000_merged.pt`（推理用，LoRA 已折叠） |
| `/workspace/a3_train_runs/a3_lora_r1/checkpoints/weights/step_002000.pt` | 11,179,613,340 | `trainer/step_002000.pt`（基座 + 未折叠 LoRA，可重新合并） |
| `/workspace/a3_train_runs/a3_lora_r1/checkpoints/state/step_002000/` | 12,712,079,410 | `trainer/state/step_002000/`（`optimizer.bin` 123 MB + `model.safetensors` 12.59 GB + scheduler/RNG/state） |
| `/workspace/a3_data/a3_fw/frames.dat` | 12,849,371,136 | `data/frames.dat`（训练 memmap，也可用 parquet 重建） |

以下两个本来就不属于本工作，直接按 §7 的地址重下即可：

| 云端路径 | 大小 | 来源 |
|---|---|---|
| `/workspace/FasterWAM/checkpoints/fasterwam_release/libero/step_021700.pt` | 11,117,679,929 | HF `hustvl/FasterWAM`（sha256 `01be4efca565d1890fd8a7dc1507b01c00c8cc63c16ae252287492ba9ac7550d`） |
| `/workspace/a3_data/a3-front-close-left-100/` | 4.1 GB 目录 | HF dataset `Eter0109/a3-front-close-left-100` |

---

## 7. 外部资源地址（不重复上传）

| 资源 | 地址 / 版本 |
|---|---|
| FasterWAM 源码 | https://github.com/hustvl/FasterWAM @ `83667817df0d4f823f39d90700e61ea2f432ac45` |
| A3 仿真源码 | https://github.com/Eter0109/a3_dual_arm_sim @ `ff552c4fe54fe344aeaff4c65270ee0fd2af4153` |
| 官方基座权重 | HF `hustvl/FasterWAM` → `libero/step_021700.pt` + `libero/dataset_stats.json` |
| **本工作合并权重**（推理用） | HF **model** `wsadwsad6779/fasterwam-a3-cookie-lora` → `step_002000_merged.pt`（11,117,977,659 B，r26/r27 实际使用） |
| **本工作训练侧权重**（可重合并） | 同仓库 → `trainer/step_002000.pt`（11,179,613,340 B，基座 + 未折叠 LoRA） |
| **本工作优化器状态**（可续训） | 同仓库 → `trainer/state/step_002000/`（12,712,079,410 B，`model.safetensors` + `optimizer.bin` + scheduler/RNG） |
| **训练 memmap** | 同仓库 → `data/frames.dat`（12,849,371,136 B = 42681×224×448×3；也可由 parquet 重建） |
| 专家数据集 | HF **dataset** `Eter0109/a3-front-close-left-100` @ `8fdd8a7caf8275eba8be8a83b53a99176a084062`（下载必须带 `--repo-type dataset`） |
| Wan2.2 VAE/T5 | HF `Wan-AI/Wan2.2-TI2V-5B`（`Wan2.2_VAE.pth`、`models_t5_umt5-xxl-enc-bf16.pth`） |
| Wan2.1 tokenizer | HF `Wan-AI/Wan2.1-T2V-1.3B` → `google/umt5-xxl/*` |

---

## 8. 下一步（建议从这里继续）

1. **先看视频**：`results/wrapped/run_20261026_173807/full_episode_42_front.mp4`，确认第二批 5 块在释放后是“叠在第一批上”还是“落在盒沿外”。
2. **加一次释放后诊断**：在 `a3_wrap.apply` 的 done 分支后，记录离 TCP 最近的 5 块饼干的 xyz 与 `cookies_in_target`，跑 600 步即可判定计数体积的边界条件。
3. **若确认是叠放**：把第二批的落点从“同一条 x 槽位”改为“按实测 free gap 侧向放置”（`patch_place2.goal_tcp_x` 已有雏形），并把释放高度降到盒沿附近（当前 z≈0.798，盒沿 z≈0.775）。
4. **若确认是落点偏外**：用 `batch_offsets()` 的 min/max 直接对齐盒子内壁，而不是用 TCP 偏置。
5. 目标：先稳定 10/10，再回头评估是否需要第二轮数据/训练（当前包装器是主要贡献，微调增益尚未证明）。

---

## 9. GitHub 发布建议

- 仓库已建：https://github.com/wsadwsad6779/fasterwam-a3-cookie ，分支 `a3-cookie-r27`（默认分支）。
- **只上传原创内容**：`bridge/`、`finetune/`、`tools/`、`docs/`、`results/` 里的文本摘要；不要上传 `outputs/`、`.venv/`、`__pycache__/`、大模型和大数据。
- 上游代码、HF 权重、数据集全部用第 7 节的地址引用，不重复上传。
- 本包中的 MP4/NPZ 建议留在交接压缩包里，不放进 Git。
- 4 个大文件（合并权重 / 训练侧权重 / 优化器状态 / frames.dat，共约 48 GB）都在 HF 仓库 `wsadwsad6779/fasterwam-a3-cookie-lora`，GitHub 单文件上限 100 MB，不放。
- HF 仓库内已核对过的路径与字节数见 §6；下载命令见 §4.6。
