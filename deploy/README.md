# 一键恢复云端环境

`bootstrap_pod.sh` 在一个全新的 GPU 实例上依次做四件事，最后打印可以照抄的运行命令：

1. pip 装齐依赖（版本与 2026-10-27 那个实例一致）；
2. 从 GitHub 检出**固定 commit** 的三份代码（本仓库 + `a3_dual_arm_sim` + `FasterWAM`）；
3. 从 Hugging Face 下载权重和（可选）数据；
4. 生成 `$WORKSPACE/activate_fasterwam_a3.sh`，跑 `py_compile` + `selftest.py`，可选再跑一次 2 步冒烟。

## 用法

```bash
# 只恢复推理所需（代码 + 37 GB 权重）
HF_TOKEN=... bash deploy/bootstrap_pod.sh --smoke

# 连训练数据、trainer 权重、优化器状态一起恢复（再多约 37 GB）
HF_TOKEN=... bash deploy/bootstrap_pod.sh --with-training
```

放在 `tmux` 或 `nohup` 里跑：大文件下载要 20–40 分钟。

```bash
source /workspace/activate_fasterwam_a3.sh
python3 /workspace/fasterwam_a3_bridge/bridge/_next/selftest.py
```

## 环境变量（都有默认值）

| 变量 | 默认 | 说明 |
|---|---|---|
| `WORKSPACE` | `/workspace` | 所有代码和权重都落在它下面 |
| `HF_TOKEN` | 空 | 只在本进程内存里传给 `huggingface-cli`，不落盘、不打印 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 设为空字符串则走 `huggingface.co` |
| `PIP_INDEX` | 阿里云镜像 | |
| `GH_MIRROR` | 空 | 直连 GitHub 被墙时给一个前缀，例如 `https://ghfast.top` |
| `BRIDGE_SRC` | 空 | 不给 GitHub 也能恢复：指向本地上传的仓库副本，脚本改成 `cp -a` |
| `A3_COMMIT` | `08ebe8cb3708…` | 仿真代码的验证版本 |
| `FASTERWAM_COMMIT` | `83667817df0d…` | 推理框架的验证版本 |
| `BRIDGE_BRANCH` | `a3-cookie-r27` | |

Flag：`--skip-pip`、`--skip-weights`、`--with-training`、`--smoke`。

## 代码与权重的分界线

| 内容 | 位置 | 原因 |
|---|---|---|
| 本仓库（`bridge/`、`finetune/`、`docs/`、`deploy/`） | GitHub | 文本，可 diff |
| `a3_dual_arm_sim`、`FasterWAM` | GitHub 上游，按 commit 固定 | 不复制、不修改 |
| 基座权重 11.1 GB + LIBERO stats | HF `hustvl/FasterWAM` | 二进制，GitHub 放不下 |
| 合并 LoRA 11.1 GB | HF `wsadwsad6779/fasterwam-a3-cookie-lora` | 同上 |
| 训练侧权重 / 优化器状态 / `frames.dat` | 同一 HF 仓库 | 只在重训时需要 |
| 专家数据集 4.1 GB | HF `Eter0109/a3-front-close-left-100` | 同上 |
| Wan2.2 VAE/T5 + Wan2.1 tokenizer ≈14 GB | HF `Wan-AI/*` | 推理必需 |

GitHub 上不放：密钥、`HF_TOKEN`、虚拟环境、`outputs/`、视频（`*.mp4`）、`*.npz`、`*.pt`、`*.zip`。`.gitignore` 已经覆盖。

## 需要预留的空间

| 模式 | 磁盘 |
|---|---|
| 仅推理（`--skip-weights` 之外） | ≈ 40 GB（权重 37 + 仓库与缓存） |
| `--with-training` | ≈ 78 GB |

## 已知的两个坑

- **容器内 GitHub 经常不通。** 2026-10-27 那个实例就是这种情况，代码只能从 JupyterLab 上传。
  三条出路：给 `GH_MIRROR`、把仓库副本放好并设 `BRIDGE_SRC`，或者先上传交接压缩包再解压。
  上游两份代码同样受影响，`GH_MIRROR` 对它们一起生效。
- **上游 commit 必须钉死。** `a3_dual_arm_sim@ff552c4`（交接文档里旧的那条）是重构前的
  扁平布局，`tasks/`、`controllers/` 都不存在，`_next` 里的 import 会直接失败。脚本默认
  `08ebe8c`，那是本地验证过的版本。

## 脚本不做的事

- 不登录 HF（不写 `~/.cache/huggingface/token`），token 只作为 `--token` 参数传递；
- 不改 `~/.bashrc`，环境变量写在 `$WORKSPACE/activate_fasterwam_a3.sh`；
- 不覆盖已有的权重：文件在就跳过下载，大小不符只告警；
- 不启动任何评测，`--smoke` 只跑 2 步，用来确认权重和 normalizer 配对正确。
