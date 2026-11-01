# 在新 GPU 实例上恢复 FasterWAM / A3 工作集

对应 2026-11-01 退租归档。所有字节数与 sha256 以两份台账为准，不认别的：

```
archive/2026-11-01/p0p1_manifest.tsv      # 715 行：档、path_in_repo、pod 相对路径、字节、sha256
archive/2026-11-01/parts60gz_manifest.tsv # 532 行：档、分片 path_in_repo、相对路径、字节、sha256
```

两份台账在 HF 仓的 `MANIFEST/` 目录下也各有一份，内容逐字节相同。仓库里只有第一份；第二份如果不存在，脚本会自动从
`MANIFEST/parts60gz_manifest.tsv` 拉下来并核对固定 sha256（`PARTS_SHA`），所以不需要手工准备。

远端仓库：`https://huggingface.co/wsadwsad6779/fasterwam-a3-cookie-lora`（模型仓，公开读）。
任一文件只要 sha256 与台账不符，就不算恢复成功——文件大小相同不算。

## 0. 前提

- Linux + NVIDIA GPU 实例，Python 3.12.3（与旧 pod 一致），`/workspace` 至少 150 GB 空闲：
  26.8 GB gz 分片 + 33.4 GB 重装配后的三个整文件 + 数据与视频。
- 不要用 `HF_ENDPOINT=hf-mirror.com` 去**写**任何东西；本脚本只读。
- 取文件的默认端点写死 `https://huggingface.co`；`--mirror` 切换到 `https://hf-mirror.com`（实测更快，见第 4 节）。

## 1. 一键脚本

```bash
git clone -b archive-2026-11-01 https://github.com/wsadwsad6779/fasterwam-a3-cookie.git a3_archive
cd a3_archive
bash scripts/restore_new_instance.sh --help 2>/dev/null || sed -n '1,20p' scripts/restore_new_instance.sh
```

分步跑（推荐，先小后大）：

```bash
bash scripts/restore_new_instance.sh --code        # 上游两个仓库 + 软链
bash scripts/restore_new_instance.sh --deps        # mujoco/robosuite/libero/transformers...
bash scripts/restore_new_instance.sh --weights     # weights/**、trainer/**、libero/**
bash scripts/restore_new_instance.sh --rest        # eval/ videos/ replay/ runs/ tests/ research/
bash scripts/restore_new_instance.sh --data        # data/**（第三方数据集需 WITH_DATASET=1）
bash scripts/restore_new_instance.sh --parts       # 三个 >2 GiB 权重按分片重装配
bash scripts/restore_new_instance.sh --verify      # 逐文件 sha256 对账
```

`--all` 等于按上面顺序全部跑一遍。脚本不删除任何文件，不启动训练或评估，已存在且 sha256 相同的文件直接跳过。

**落盘路径一律用台账第 3 列（旧 pod 上的相对路径）**，不是从 HF 的 `path_in_repo` 拼出来的。两者不一样：HF 里 `weights/r3/full_state/step_003000.pt` 在旧 pod 上是 `a3_train_runs/a3_lora_r3/checkpoints/weights/step_003000.pt`，r4/r5 的轻量 ckpt 也带 `checkpoints/light/` 这一段。恢复后目录结构与旧 pod 逐级相同，训练/评估脚本里写死的相对路径才能直接用。

## 2. 代码与环境

上游两个仓库按 pin 的 commit 取，不要用 `master`：

| 仓库 | commit | 落在 |
|---|---|---|
| `https://github.com/hustvl/FasterWAM`（Apache-2.0） | `83667817df0d4f823f39d90700e61ea2f432ac45` | `/workspace/FasterWAM` |
| `https://github.com/Eter0109/a3_dual_arm_sim` | `08ebe8cb370856051d0b32b07e691cc9b056716b` | `/workspace/a3_dual_arm_sim` |

worker 路径里硬编码了这个软链，缺了 bridge 就 import 不到 FasterWAM：

```bash
ln -sfn /workspace/FasterWAM /workspace/fasterwam_a3_bridge/FasterWAM
```

依赖（旧 pod 上的实测集合，pip 走阿里云镜像一次装完）：

```
mujoco==3.3.2  robosuite==1.4.0  transformers==4.57.6  accelerate==1.13.0
robomimic  libero  bddl  numpy  opencv-python  h5py  imageio  imageio-ffmpeg
tqdm  pyyaml  hydra-core  omegaconf  safetensors  einops  timm
```

torch/ROCm 来自实例镜像，不要用 pip 覆盖。

## 3. 权重（小于 2 GiB 的部分）

`weights/r2|r3|r4|r5/**`、`trainer/**` 按台账逐个取回并校验，落盘到台账第 3 列的旧 pod 路径。
r4/r5 的轻量 ckpt 是 `step_000500…step_008000` 每档五个文件（adapter / optimizer / rng / scheduler / trainer_state），
落在 `a3_train_runs/a3_lora_r{4,5}/checkpoints/light/<step>/`。

**底座不在本归档里**：`FasterWAM/checkpoints/fasterwam_release/libero/step_021700.pt`（11,117,679,929 B）
与配对的 `dataset_stats.json` 来自上游 `hustvl/FasterWAM`，用 `deploy/bootstrap_pod.sh` 的 `hf_download` 取，
不要从本仓库找；台账里没有它们，`--verify` 也不会检查它们。

**恢复不了的东西**（台账里记了，别指望这次备份能补）：r4/r5 的轻量 checkpoint 只存了 784/786 个张量，被训练的 `proprio_encoder`（36,864 元素）从未落盘，也没有副本。详见 README 第 8 节。

## 4. 三个 >2 GiB 权重：60 MiB 分片（gzip）重装配

这三个文件在旧 pod 上传不上去：hf-mirror 的 Xet 上传器在服务端对某个 chunk 返回
`"result":"found"` 之后就静默挂死——没有 socket、`/proc/<pid>/io` 的 rchar 不动、CPU 接近 0、xet 日志冻结。
不是文件大小问题，也不是 64 MiB xorb 边界；同一目录里 dedup 查询返回 `not_found` 的分片都正常传完。
把每片用 `gzip -n -1` 重新压缩后内容全变，dedup 必然落空，上传恢复：实测 8 片（402 MB）一个 commit 约 40 s（10 MB/s），
回读 sha256 与台账一致。所以远端存的是 `part_NNN.gz`，不是裸字节。细节见 README 11.2。

| 整文件 `path_in_repo` | 档 | 字节 | sha256 | 分片数 |
|---|---|---|---|---|
| `weights/r3/full_state/step_003000.pt` | P0 | 11,179,613,340 | `e45110fd6742a72a998aa8533eb0bfb7cdd47513174b4e052962f1af47de7547` | 178 |
| `weights/r3/step_003000_merged.pt` | P0 | 11,117,977,659 | `1ac089bf0b4ec41441be274dc3b9d66d75aa31e55dec6751907c59955c4f1c90` | 177 |
| `weights/r2/step_003000_merged.pt` | P1 | 11,117,977,659 | `baafffcf9aacfc7a8a1e9276100fdfce4763e2dda0d513c7be7b303f0356d6a2` | 177 |

分片在远端的路径是 `parts60gz/<档>/<整文件 path_in_repo>/part_000.gz … part_NNN.gz`，按名字排序就是原始字节顺序。
gzip 是流式的、多 member 可以串联，所以「按序 cat 再 gunzip」得到的正是整文件。

脚本 `--parts` 做的事，等价于手动执行：

```bash
mkdir -p /workspace/_parts60gz/step_003000.pt /workspace/a3_train_runs/a3_lora_r3/checkpoints/weights
for i in $(seq -w 0 177); do            # seq -w 补零到最大值的宽度，这里是 3 位：000…177
  curl -sSL --fail -C - \
    "https://huggingface.co/wsadwsad6779/fasterwam-a3-cookie-lora/resolve/main/parts60gz/P0/weights/r3/full_state/step_003000.pt/part_$i.gz" \
    -o "/workspace/_parts60gz/step_003000.pt/part_$i.gz"
done
cat /workspace/_parts60gz/step_003000.pt/part_*.gz | gunzip > /workspace/a3_train_runs/a3_lora_r3/checkpoints/weights/step_003000.pt
sha256sum /workspace/a3_train_runs/a3_lora_r3/checkpoints/weights/step_003000.pt
# 必须是 e45110fd…4de7547
```

分片数不是整数页时最后一片的**原始切面**是短的（11,179,613,340 − 177 × 62,914,560 = 43,736,220 B；
另外两个文件的末片是 45,015,099 B），压缩后的字节数在台账的 `part_bytes` 列，必须逐片比对，不能只看 `part_*.gz` 通配。
`cat … | gunzip` 遇到任何一片截断或损坏都会报错并返回非零；脚本里也逐片校验 sha256。

取法实测速率（2026-11-01，旧 pod）：

| 取法 | 速率 |
|---|---|
| ModelScope | 36 MB/s |
| hf-mirror 6 路 Range 分段 | 26 MB/s |
| hf-mirror 单流 | 5.6 MB/s |
| huggingface.co 直连 | 1.7 MB/s |
| `hf download` 默认 | 2 MB/s |

单流够用时就直接 `curl`；11 GB 级别整文件（如果能拿到）用脚本里的 `fetch_range6`（6 路 Range + 最后 `cat`），六路叠加约 26 MB/s。

## 5. 数据

- `data/a3_fw/**`：原始 100 条专家示范转换后的数组与 `frames.dat`、`dataset_stats.json`。
- `data/a3_fw_dagger{,_near}/**`、`data/a3_fw_force/**`：新增 73 条低力与纠正示范（30 + 17 + 8 + 18）。
- `data/a3_fw/corpus_manifest_v1.json` 的 sha256 应为 `b679fa6bd07e12c3be25e02f1d5aa3d14e3d785fe726e04f98b1f3a391dd0e00`。
- 原始第三方数据集只引用不重传：`Eter0109/a3-front-close-left-100`，revision `8fdd8a7caf8275eba8be8a83b53a99176a084062`。需要时 `WITH_DATASET=1` 才会去拉；再分发前要拿到数据所有者的许可。

## 6. 评估数据与视频

`bash scripts/restore_new_instance.sh --rest` 会把这一类全部取回（共 619 行、约 5.8 GB）：

`eval/a3_eval/**`（B0/C0 的 `summary_lora.json`、`manifest_lora.json`、npz、打分日志）、`eval/bridge/**`（dagger 恢复件）、
`videos/a3_eval/**`、`videos/fasterwam_a3_bridge/**`、`replay/bridge/**`、`runs/**`（各轮 `trainer_state.json`、`lr_restart.json`）、
`tests/light_state_test{,2}/**`、`research/**`。

打分口径别搞混：`cookies_in_target` 有三套统计（READM 第 6 节），报告里的数字都标了口径。

## 7. 校验

```bash
bash scripts/restore_new_instance.sh --verify
```

期望输出：

```
verify rows=715 ok=715 missing=0 sha_bad=0 giants_missing=0 giants_bad=0
```

只要 `sha_bad` 或 `giants_bad` 不为 0，就先别启动任何训练。

单文件抽查（从远端真实回读一段字节，绕过本地缓存）：

```bash
curl -sSL -r 0-65535 "https://huggingface.co/wsadwsad6779/fasterwam-a3-cookie-lora/resolve/main/parts60gz/P0/weights/r3/full_state/step_003000.pt/part_000.gz" -o /tmp/p0.head.gz
sha256sum /tmp/p0.head.gz   # 与本地 part_000.gz 的前 64 KiB 相同即通过
gunzip -c /tmp/p0.head.gz > /tmp/p0.head 2>/dev/null || true   # 前 64 KiB 不是完整 gzip 流，截断解压报错是正常的
```

## 8. 重启（脚本不自动执行）

训练与评估的入口、随机种子、学习率重启点都在 `pod_snapshot/2026-11-01/` 里的原始脚本中：

- 续训：`pod_snapshot/2026-11-01/_r4_stage/run_r5.sh`（含 `lr_restart.json` 的接线：global step 8000、段 4000）。
- 评估：`pod_snapshot/2026-11-01/` 下的 `run_matrix.py`（B0/C0 两档）。
- 打分：`autonomy_metrics.py`。

r4/r5 的 `proprio_encoder` 缺失意味着**不能从轻量 ckpt 精确续训**——要继续训练必须从 r3 的 `full_state` 出发，或者接受随机初始化的 proprio 分支。

## 9. 已知不完整项

1. r4/r5 `proprio_encoder`（36,864 元素）永久缺失，本次备份无法恢复。
2. 8D → 14D 力感知（`FORCE_V6=[0,1,2,12,13,16]`）只在代码里接好，从未训练。
3. light_state v3 的真实模型一致性 smoke 未做（只有 1-step smoke）。
4. 三个 >2 GiB 权重按 60 MiB 分片（gzip）存档，上传后已在旧 pod 侧做过真实回读比对；但新实例上 `cat … | gunzip`
   重装配 + 整文件 sha256 的完整演练仍需在新实例上跑一次，未通过前视作「pod 唯一副本」的替代品。
