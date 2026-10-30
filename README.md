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

### 4.0 一键恢复（推荐；4.1–4.3 是它展开后的手工版本）

新实例上只跑一个脚本，代码从 GitHub 的固定 commit 取、权重从 Hugging Face 取：

```bash
HF_TOKEN=... bash deploy/bootstrap_pod.sh --smoke
source /workspace/activate_fasterwam_a3.sh
```

实例还没起来时可以挂一个每 60 秒重试、成功即退出的启动循环（不会自己开实例）：

```bash
POD_SSH="ssh -p 45339 root@1.2.3.4" bash deploy/retry_launch.sh
```

`deploy/README.md` 里列了全部环境变量（镜像、代理、`BRIDGE_SRC` 离线兜底）、磁盘占用和
两个已知的坑。脚本不写任何密钥，也不改 `~/.bashrc`。

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
  - `https://github.com/Eter0109/a3_dual_arm_sim` @ `08ebe8cb370856051d0b32b07e691cc9b056716b` → `/workspace/a3_dual_arm_sim`
    > **此处原来是 `ff552c4`，是错的。** `ff552c4`（2026-09-28 14:47）是重构前的扁平布局
    > （`src/a3_dual_arm_sim/cookie_transfer.py`），而 `bridge/` 里 import 的是
    > `a3_dual_arm_sim.tasks.cookie_transfer` / `.controllers.same_column_batch_expert`，
    > 这两个包从 `8088b31`（2026-09-28 21:44）起才存在，按 `ff552c4` 克隆会直接
    > `ModuleNotFoundError`。`08ebe8c`（2026-10-21）在 `origin/main` 上，是本地跑过
    > `bridge/_next/selftest.py` 和 `calibrate.py` 的那一版。详见 `bridge/_next/README.md` §2。
- 本项目原创代码：把本包的 `bridge/`、`finetune/` 放回 `/workspace/fasterwam_a3_bridge/`、`/workspace/FasterWAM/`（finetune 里的文件按包内相对路径覆盖/新增）。
- **`bridge/infer_a3_worker.py` 用 `Path(__file__).parent.parent / "FasterWAM"` 定位上游代码**，
  不读 `FASTERWAM_ROOT`。上游按上表放在 `/workspace/FasterWAM` 时，必须补一个链接，
  否则 hydra 会去 `/workspace/fasterwam_a3_bridge/FasterWAM/configs` 找配置并报
  `Primary config directory not found`：
  ```bash
  ln -sfn /workspace/FasterWAM /workspace/fasterwam_a3_bridge/FasterWAM
  ```
- 容器内 GitHub 被墙，代码只能用本地上传（JupyterLab 拖拽即可）。

### 4.3 模型 / 数据下载

要拿的是约 36 GB 的大文件。**不要用 `hf download` 单连接硬等**：2026-10-30 在 Radeon Cloud 实例上实测，
同一批文件用 `hf download`（hf-mirror 单流）只有 **2 MB/s**，全部下完要 4 小时以上。

同一时间、同一台机器、同一个 egress 代理，四种取法实测：

| 取法 | 实测速率 | 适用 |
| --- | --- | --- |
| ModelScope 直链（单流） | **36 MB/s** | 公开权重（基座 ckpt、Wan VAE/T5、tokenizer） |
| hf-mirror 6 路 Range 分段 | **26 MB/s** | 私有仓库（ModelScope 上没有） |
| hf-mirror 单流 `curl` | 5.6 MB/s | 兜底 |
| `huggingface.co` 直连 | 1.7 MB/s | 不用 |
| `hf download`（hf-mirror 单流） | 2 MB/s | 慢，只用来续传 |

四个源都返回 `206`，所以分段和续传都成立。换成上面前两行之后，36 GB 从 4 小时降到 **15 分钟**。

**(a) 公开权重：走 ModelScope**（`hustvl/FasterWAM` 与 `Wan-AI/*` 在 ModelScope 上都有，字节数与 HF 一致）

```bash
P=${https_proxy:-}   # 需要出网代理时带上 -x "$P"
B=/workspace/FasterWAM/checkpoints
VAE=$B/Wan-AI/Wan2.2-TI2V-5B
TOK=$B/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl
mkdir -p "$B/fasterwam_release/libero" "$VAE" "$TOK"

ms() {  # ms <owner/repo> <输出路径> <仓库内路径>
  curl -sL -x "$P" --retry 8 --retry-all-errors -C - -o "$2" \
    "https://modelscope.cn/api/v1/models/$1/repo?Revision=master&FilePath=$3"
  echo "MS_RC_$? $3 $(stat -c %s "$2" 2>/dev/null)"
}

ms hustvl/FasterWAM "$B/fasterwam_release/libero/step_021700.pt" libero/step_021700.pt
ms hustvl/FasterWAM "$B/fasterwam_release/libero/dataset_stats.json" libero/dataset_stats.json
ms Wan-AI/Wan2.2-TI2V-5B "$VAE/Wan2.2_VAE.pth" Wan2.2_VAE.pth
ms Wan-AI/Wan2.2-TI2V-5B "$VAE/models_t5_umt5-xxl-enc-bf16.pth" models_t5_umt5-xxl-enc-bf16.pth
for f in spiece.model tokenizer.json tokenizer_config.json special_tokens_map.json; do
  ms Wan-AI/Wan2.2-TI2V-5B "$TOK/$f" "google/umt5-xxl/$f"
done
```

**(b) 私有 LoRA（11.1 GB）：hf-mirror 6 路分段再拼接**

```bash
P=${https_proxy:-}
U=https://hf-mirror.com/wsadwsad6779/fasterwam-a3-cookie-lora/resolve/main/step_002000_merged.pt
D=/workspace/a3_train_runs/a3_lora_r1
SZ=11117977659; N=6; CH=$(( (SZ + N - 1) / N ))
cd "$D"
for i in $(seq 0 $((N-1))); do
  s=$((i*CH)); e=$((s+CH-1)); [ $e -ge $SZ ] && e=$((SZ-1))
  curl -sL -x "$P" --retry 10 --retry-all-errors -r $s-$e -o part$i "$U" &
done
wait
# 逐段核对字节数，全部对上才拼接；任何一段短了就是 LORA_PARTS_BAD，重下该段即可
ok=1
for i in $(seq 0 $((N-1))); do
  s=$((i*CH)); e=$((s+CH-1)); [ $e -ge $SZ ] && e=$((SZ-1))
  want=$((e-s+1)); got=$(stat -c %s part$i)
  echo "part$i want=$want got=$got"; [ "$want" != "$got" ] && ok=0
done
[ $ok = 1 ] && cat part0 part1 part2 part3 part4 part5 > step_002000_merged.pt && rm -f part?
```

> 分段并发别开太大：8 路以上会被 hf-mirror 限速到每路几百 KB/s，6 路是实测的拐点。
> 同一个文件名在 HF 和 ModelScope 上是同一份字节，所以从 `.incomplete` 续传、换源续传都安全 —— 但要按下面的
> sha256 核对。

**(c) 下载完成后核对**（HF 的 LFS oid 就是 sha256）

```bash
sha256sum /workspace/FasterWAM/checkpoints/fasterwam_release/libero/step_021700.pt \
          /workspace/a3_train_runs/a3_lora_r1/step_002000_merged.pt
```

| 文件 | 字节数 | sha256 |
| --- | --- | --- |
| `libero/step_021700.pt` | 11,117,679,929 | `01be4efca565d1890fd8a7dc1507b01c00c8cc63c16ae252287492ba9ac7550d` |
| `a3_lora_r1/step_002000_merged.pt` | 11,117,977,659 | `4091b02a23b3e35fb63b8bac38ea8e9c6d44439c052ed5aa5db151435a175601` |
| `Wan2.2-TI2V-5B/Wan2.2_VAE.pth` | 2,818,839,170 | 见 `Wan-AI/Wan2.2-TI2V-5B` 的 LFS oid |
| `Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth` | 11,361,920,418 | 见 `Wan-AI/Wan2.2-TI2V-5B` 的 LFS oid |

Wan2.2 VAE/T5 与 Wan2.1 tokenizer **必须预先按上面的文件名放好**，否则运行时 `fasterwam/runtime.py`
会走 snapshot resolver 触发 HF 429。

数据集（A3 相机数据，只有几百 MB，单流即可）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Eter0109/a3-front-close-left-100 --repo-type dataset \
  --revision 8fdd8a7caf8275eba8be8a83b53a99176a084062 \
  --local-dir /workspace/a3_data/a3-front-close-left-100
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
>
> `hf download` 在这个网络环境下只有 2 MB/s，11 GB 要一个多小时；用 §4.3(b) 的 6 路 Range
> 分段（hf-mirror，实测 26 MB/s）十几分钟就能拉完。

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
| A3 仿真源码 | https://github.com/Eter0109/a3_dual_arm_sim @ `08ebe8c`（**不是 `ff552c4`**，原因见 §4.2） |
| 官方基座权重 | HF `hustvl/FasterWAM` → `libero/step_021700.pt` + `libero/dataset_stats.json` |
| **本工作合并权重**（推理用） | HF **model** `wsadwsad6779/fasterwam-a3-cookie-lora` → `step_002000_merged.pt`（11,117,977,659 B，r26/r27 实际使用） |
| **本工作训练侧权重**（可重合并） | 同仓库 → `trainer/step_002000.pt`（11,179,613,340 B，基座 + 未折叠 LoRA） |
| **本工作优化器状态**（可续训） | 同仓库 → `trainer/state/step_002000/`（12,712,079,410 B，`model.safetensors` + `optimizer.bin` + scheduler/RNG） |
| **训练 memmap** | 同仓库 → `data/frames.dat`（12,849,371,136 B = 42681×224×448×3；也可由 parquet 重建） |
| 专家数据集 | HF **dataset** `Eter0109/a3-front-close-left-100` @ `8fdd8a7caf8275eba8be8a83b53a99176a084062`（下载必须带 `--repo-type dataset`） |
| Wan2.2 VAE/T5 | HF `Wan-AI/Wan2.2-TI2V-5B`（`Wan2.2_VAE.pth`、`models_t5_umt5-xxl-enc-bf16.pth`） |
| Wan2.1 tokenizer | HF `Wan-AI/Wan2.1-T2V-1.3B` → `google/umt5-xxl/*` |

---

## 8. 阶段 B/C：`bridge/_next/`（2026-10-29 加入）

`bridge/_next/` 是第二阶段的实验脚本，**不改 `bridge/` 里任何已有文件**，`run_wrapped.py`
的 npz 格式也保持原样。它要回答的问题换了：不再是"怎么把 5/10 推到 10/10"，而是
**模型自己到底学到了什么**。

2×2 对照：`{raw, lora}` 权重 × `{safety, full}` 控制层，同 seed、同一次 reset，
因而是配对的。`safety` 格只做硬件保护（力超限急停、单步位移/转角上限、空手压箱沿），
`full` 格才是 r27 的启发式包装器。判据写在 `bridge/_next/README.md` §4，先写下来是为了
避免事后解释。

| 文件 | 作用 |
|---|---|
| `selftest.py` | 本地自检，不需要权重 |
| `run_teacher_forcing.py` | 专家驱动、模型只被查询；含不需要权重的动作链自检（`--check-only`） |
| `run_matrix.py` | 闭环矩阵（`--controls C0,C1..C5` 按能力放权）+ 每步力/腕力/执行动作记录 |
| `b0_probe.py` | 夹爪方向实测：伺服语义、adapter 符号、checkpoint 实际输出 |
| `analyze_runs.py` | 汇总可控率、改写幅度、成功率、Oracle 读取；按「阶段 × 通道」给放权比例，并做 `C5 vs C0` 配对与动作流 trace check |
| `calibrate.py` | 验证噪声真的生效、核对力布局、量出专家自身峰值力 |
| `scene_audit.py` | 诊断扰动后的饼干重叠 / 倾斜 / 出盘；同时说明"逐块噪声会破坏场景"的证据 |
| `safety_wrap.py` / `oracle_probe.py` / `common.py` / `env_factory.py` | 安全层、真值读取计数、权重与 stats 注册、建环境、`batch_offset_m` 整批平移、`scene_sanity()` 有效性门 |
| `gripper_label_audit.py` | 用 `joint_cmd` 校验训练数据里的夹爪标签 |

放到 pod 上的最短路径（从 GitHub 直接取，不需要本地上传）：

```bash
git clone -b a3-cookie-r27 https://github.com/wsadwsad6779/fasterwam-a3-cookie /workspace/fasterwam-a3-cookie
mkdir -p /workspace/fasterwam_a3_bridge
cp -r /workspace/fasterwam-a3-cookie/bridge /workspace/fasterwam_a3_bridge/bridge
cd /workspace/fasterwam_a3_bridge/bridge
export PYTHONPATH=/workspace/fasterwam_a3_bridge/bridge/_next:$PYTHONPATH
python3 -m py_compile _next/*.py && python3 _next/selftest.py
```

**2026-10-29 定下的路线**（写在 `bridge/_next/README.md` §3）：A 自检 → B0 stats / 夹爪
配对 + 夹爪方向实测（`b0_probe.py`）→ **E0 无扰动低干预基线（C0）** → **按能力放权**，
一次只跑一级（`C1` 搬运 xyz → `C2` 接近 xyz → `C3` 抓取目标 → `C4` 下降与闭合 →
`C5` 释放/回收）→ `--batch-offset` 整批平移。
`C0` 是接口+安全的参照，`C5` 与它执行逻辑相同，所以必须逐步动作一致（泄漏检查），
**真实推理下要再确认一次**。某级放权失败就停在该级，用该级轨迹做 LoRA 微调后只重跑该级；
没有权重更新时 `C5` 不会比 `C0` 好，因此**成功判据是 `C5` 的真实指标明显高于初始 `C0`**，
"跑到了 C5"不算。
**跳过**完整 B1 teacher forcing 扫描、B2 的 2×2、B3，以及扰动场景下的专家成功率。
四档 `full/guard/force` 只留作历史对照。不用任务级 wrapper 规则替代模型。

每步都记录模型原始动作、wrapper 输出、仿真真正执行的 16 维动作、是否被改动、包装器阶段
（npz 的 `pre_14d` / `post_14d` / `applied_canonical` / `changed` / `phase` / 指垫力 / 腕力），
`analyze_runs.py` 按「状态机阶段 × 通道」给放权比例，并单独打印 `C5 vs C0` 的配对差值
和动作流 trace check——全程 0% 往往只是该阶段没走到，不能当实现错误。E0 每个种子都留完整
轨迹，至少一个种子带视频；0/10 也要留，用来定位卡在哪一段。`bridge/_next/README.md` §6 记了本地已验证 / 未验证的分界，§5 记了训练数据里
27.2% 的夹爪标签与 `joint_cmd` 矛盾这件事。

---

## 9. 下一步（建议从这里继续）

r26/r27 的"把 5/10 推到 10/10"不再是最优先级。现在的顺序是：

1. **先把 2×2 跑完**（`bridge/_next/README.md` §3 的 B0/B1/B2）。没有它，"5/10 是模型的功劳
   还是包装器的功劳"这个问题没有答案，任何继续调包装器的工作都无法证伪。
2. **stats 配对已经查清**：r27 存下来的 `actions_7d` 里 y 的范围是 `[-0.105, +0.310]`，
   与 A3 stats 的 `[-0.115, 0.299]` 吻合（LIBERO stats 是 ±0.9375 的对称范围），所以
   r26/r27 是「A3 权重配 A3 stats」，这一项可以从混淆清单里划掉。B0 仍然要跑，因为它
   顺带确认夹爪模式（LIBERO stats 的夹爪维是 `[0,1]`，A3 是 `[-1,1]`，两个模型不能共用
   同一个 `FASTERWAM_A3_GRIPPER_MODE`）。
3. **B1 看夹爪符号**：`FASTERWAM_A3_GRIPPER_MODE` 的 `pm1` / `libero01` 各跑一遍同一批
   查询，用 `grip.sign_agree_moving` 决定哪个对。用现有 checkpoint 就能验，不需要重训。
4. **只有当 B1 证明"模型学到的是错标签"时**，才按 `bridge/_next/README.md` §5 重建夹爪
   标签（`clip(-Δjoint_cmd/0.08, -1, 1)` + adapter 改 `-action[6]`）并重训。阶段 B 的 2×2
   必须继续用旧 adapter，否则两列不在同一基准上。
5. **C0 已在本地产出确定数字**：专家指垫峰值 1.46 N、左腕静载 10.7 N、
   `obs[12:14]` 与 `_sensor` 在 223 个接触步上误差 0.00 N。**C1 要先改设计再跑**：
   脚本专家在 1 mm 噪声下就只剩 7/10（`VERIFY_RELEASE` 失败），2 mm 时第 41 步撞上
   48.3 N 直接放弃 —— 所以"扰动下的成功率"没有可达到的参照，只能看降级曲线，或者
   干脆等 Stage B 的结论再说。
6. r26 那条老线索（第二批释放后没被计入目标盘）留着，但优先级在这五步之后 —— 它回答的
   是"包装器怎么改"，而 2×2 回答的是"还该不该把包装器当主线"。

---

## 10. GitHub 发布建议

- 仓库已建：https://github.com/wsadwsad6779/fasterwam-a3-cookie ，分支 `a3-cookie-r27`（默认分支）。
- **只上传原创内容**：`bridge/`、`finetune/`、`tools/`、`docs/`、`results/` 里的文本摘要；不要上传 `outputs/`、`.venv/`、`__pycache__/`、大模型和大数据。
- 上游代码、HF 权重、数据集全部用第 7 节的地址引用，不重复上传。
- 本包中的 MP4/NPZ 建议留在交接压缩包里，不放进 Git。
- 4 个大文件（合并权重 / 训练侧权重 / 优化器状态 / frames.dat，共约 48 GB）都在 HF 仓库 `wsadwsad6779/fasterwam-a3-cookie-lora`，GitHub 单文件上限 100 MB，不放。
- HF 仓库内已核对过的路径与字节数见 §6；下载命令见 §4.6。
