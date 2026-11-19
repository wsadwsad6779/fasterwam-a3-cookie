# 从归档恢复现场（GitHub + Hugging Face）

实例销毁后，按下面顺序即可重建训练/评估现场。所有凭据只放运行期文件，不入库。

## 1. 代码

```bash
git clone https://github.com/wsadwsad6779/fasterwam-a3-cookie.git
cd fasterwam-a3-cookie
git checkout main
# 主线脚本在 restore_20261117/reconstructed_normalizer/
```

## 2. 运行环境（与本次实例一致）

- Python 3.12（`/opt/venv`），torch ROCm 版（gfx942/MI300X 类），numpy、cv2（opencv-python）、Pillow；
- 上游代码：
  - `hustvl/FasterWAM` @ `83667817df0d4f823f39d90700e61ea2f432ac45`
  - `Eter0109/a3_dual_arm_sim` @ `08ebe8cb370856051d0b32b07e691cc9b056716b`
- 基座权重：HF `hustvl/FasterWAM` 的 `libero/step_021700.pt`（sha256 见 `restore_20261117/base_sha256.txt`）。

## 3. Hugging Face 拉取（私有仓需 token）

仓库：私有 `wsadwsad6779/fasterwam-a3-force-40k` + 公开 `wsadwsad6779/fasterwam-a3-archive-public`（两仓合计为完整现场；私有配额满后其余内容转存公开仓）。

| 远端路径 | 内容 |
|---|---|
| `v1_force_40k/nodes/step_005000 … step_040000/`（私有） | 40k 训练各节点（adapter/optimizer/scheduler/rng/trainer_state/manifest） |
| `v1_force_40k/nodes/step_040000/adapter.pt`（私有） | 40k 合并所需 delta（配合公开基座） |
| `v1_force_40k/code/`（私有） | 训练期代码快照（与 main 同源） |
| `v2_repair_20261119/evidence/rn_text_bundle.tar.gz`（私有） | reconstructed_normalizer 全部文本/证据/录像 |
| `v2_repair_20261119/evidence/archive_evidence.tar.gz`（私有） | 原始 100 条专家语料（arrays+joint_cmd） |
| `v2_repair_20261119/corpus/teacher_diverse_20261119/`（私有） | 多布局语料（arrays/poses/force/manifest + `frames.dat` 2.71 GB） |
| `v2_repair_20261119/checkpoints/action_expert_v1|v2|v3|v4/`（私有） | `last.pt`（全状态）+ `delta_step_*`（纯权重）+ `segments.jsonl` |
| `v2_repair_20261119/full/reconstructed_normalizer/`（私有） | rn 目录逐文件镜像（含 `model_repair_runs` 两个 23.9 GB 全状态） |
| `eval_restore_20261117/`（公开） | 完整树：base/merged 权重、deploy、archive_evidence、step_040000、全部评估产物 |
| `root_repair_runs/`、`root_quarantine_lora300/`、`a3_data/`（公开） | 训练产物镜像、损坏的 step_000200 全状态、文本嵌入缓存 |
| `workspace_top_files.tar.gz`（公开） | 归档脚本与顶层文件 |

示例（token 从文件读，不写进命令历史）：

```bash
export HF_HUB_DISABLE_XET=0           # 容器默认可能是 1；必须覆盖
export HF_ENDPOINT=https://hf-mirror.com
/opt/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('wsadwsad6779/fasterwam-a3-force-40k',
                  allow_patterns=['v2_repair_20261119/**'],
                  local_dir='/workspace/hf_restore',
                  token=open('/root/.cache/hf_token').read().strip())
PY
```

## 4. 现场拼装

1. 上游代码放到 `stage_40k/fw`（或恢复后的等价路径），`PYTHONPATH` 指向它；
2. `base_step_021700.pt` 放回 `step_040000/` 同级目录（训练器按 `base_weights.json` 校验 sha）；
3. 40k 合并：用 `stage_40k/export_merged*.py`（或 `v1_force_40k/code/export_merged_v3.py`）
   以公开基座 + `adapter.pt` 重建；
4. 语料：`teacher_diverse_20261119/` 解包到 `/workspace/eval_restore_20261117/reconstructed_normalizer/`
   同级（`frames.dat` 建议放本地大容量盘，其它文件小）；
5. 归一化统计：`RECONSTRUCTED_NORMALIZER_ACCEPTANCE.json` 记录重建口径；
   原 stats 若不可得，按 `reconstructed_normalizer` 内脚本重建并跑 CPU 契约验证。

## 5. 训练 / 评估复跑

```bash
# 续训（示例：从 v4 last.pt 继续，步数上限自己定）
cd /workspace/eval_restore_20261117/reconstructed_normalizer
/opt/venv/bin/python train_action_expert_v4_20261119.py --help   # 见脚本内环境变量

# C 类裸模型评估（必须带视频）
C_RESUME=1 C_OUT=c_class_eval_manual C_CKPT=/path/delta_step_016000.pt \
C_SEEDS=29,30 C_MAX_STEPS=600 C_VIDEO=1 \
/opt/venv/bin/python c_class_eval_20261119.py
```

## 6. 验收清单（恢复后自检）

- [ ] `c_class_eval_20261119.py` 能加载 C_CKPT 并输出 `paired_A_vs_C.json`；
- [ ] 双视角 mp4 可解码，帧数 = 观测帧数（601）；
- [ ] npz 内 `action` 为 14 维 A3（col6 −1=闭）；
- [ ] 语料 `data_sha256` 与 `segments.jsonl` 中记录一致（a8f1bc51…）；
- [ ] val 回合 [11,12,13,14] 从不进入训练采样。
