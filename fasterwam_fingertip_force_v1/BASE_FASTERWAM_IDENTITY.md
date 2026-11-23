# BASE_FASTERWAM_IDENTITY（第一阶段，2026-11-20）

状态：**部分确认**。代码级事实已核实；checkpoint 级事实有 1 项待验证；GPU 侧复核因无实例暂缺。

## 1. 已核实（源码/配置为证）

| 项 | 值 | 证据 |
|---|---|---|
| 优化目标模型 | FasterWAM（Wan2.2-TI2V-5B + SparseActionDiT） | `stage_40k/fw/src/fasterwam/models/wan22/fastwam.py` |
| 上游 pin | hustvl/FasterWAM `8366781…`（Apache-2.0）；Eter0109/a3_dual_arm_sim `08ebe8c…` | README / 历史台账 |
| 原始 A3 配置 | `stage_40k/fw/configs/data/a3_2cam.yaml` | 文件本身 |
| proprio（原始） | **8 维** | `shape_meta.state.raw_shape: 8`；`build_v1_dataset.py` 断言 base proprio 为 8 |
| proprio（旧力实验） | 14 = 8 + 6 力通道 | `a3_2cam_force.yaml` 注释与实现；**不是原始协议** |
| action | **7 维**，前 6 维 delta（mask true），第 7 维绝对量（mask false） | `delta_action_dim_mask: [true×6, false]` |
| 夹爪语义 | LIBERO：+1=闭，−1=开；A3 adapter `output[6] = −np.sign(a6)`（−1=闭，+1=开） | `bridge/action_adapter_v3.py` |
| 时序 | `num_frames=33` → `action_horizon=32`；`action_video_freq_ratio=4` → `num_video_frames=9` | `a3_2cam.yaml` + `a3_video_dataset.py` |
| 视觉 | 双相机 224×224，水平拼接 224×448 | `video_size: [224,448]`、`concat_multi_camera: horizontal` |
| 归一化 | min/max，统计文件 `/workspace/a3_data/a3_fw/dataset_stats.json` | `processor.pretrained_norm_stats` |
| 控制频率 | 20 Hz（步长 0.05 s） | 采集/评估协议 |
| 文本 | T5 缓存 `/workspace/a3_data/text_embeds_cache`，context_len 128 | `a3_2cam.yaml` |

## 2. 权重身份

| 资产 | 大小 | 状态 |
|---|---|---|
| 上游基座 `libero/step_021700.pt` | 11,117,679,929 B | 字节数/sha 见 `restore_20261117/base_sha256.txt`；公开仓可再取 |
| A3 40k 节点 `v1_force_40k/nodes/step_040000` | 211,090,223 B | adapter/optimizer/scheduler/rng/trainer_state/manifest；**待读 manifest 确认是否为无力（proprio 8、无 force conditioner）** |
| `step_040000_merged.pt` | 11,126,493,579 B | 已在公开 HF 仓备份 |

## 3. 未确认（不得当作已知）

1. 40k 节点/merged 是否含 force conditioner 与 14 维 proprio（前缀名 `v1_force_40k` 使其存疑）。
2. 原始 normalizer 是否为原始 100 条语料统计（归档中存在“重建 normalizer”，禁止混用）。
3. 指尖力传感器：左/右指垫通道、法向/合力、符号、单位、零载基线、接触阈值（需 MuJoCo XML + sensor 声明）。
4. 4372.9 N 异常通道来源。

## 4. 阻断与下一步

- **B1 无 GPU 实例**：旧实例已销毁，Radeon Cloud 资源池已满。短推理复现、MuJoCo 基线、传感器 XML 核验待实例可用。
- **B2 HF 私有仓配额满**：读 40k 节点 manifest 需 token；或改用公开仓同内容。
- 不依赖 GPU 的下一步：论文与源码审计（ForceVLA/TacFiLM/ImplicitRDP/MimicGen）、原始语料 arrays 与 dataset_stats 身份核对、夹爪标签 V2 规则设计与事件统计。
## 6. 基座锁定（2026-11-23 修订，覆盖此前安排）
- **唯一基座**：`hustvl/FasterWAM` 官方 `libero/step_021700.pt`（sha256 `01be4efc…`，11,117,679,929 B）。
- 直接证据：`proprio_encoder.weight=[4096,8]`、`mot.mixtures.action.head.weight=[7,1024]`，顶层键仅 `mot/step/torch_dtype/proprio_encoder`，无 force conditioner/head。
- **禁止**：寻找/下载/修复/加载/评估任何旧 A3 无力微调 checkpoint；禁止从旧 Force40k 或 C 类权重继续训练；原始权重只读。
- 官方权重只具备 LIBERO 预训练能力，不具备 A3 任务能力；F0/F1/F2 均从同一官方基座出发，同数据、同预算、同评估协议。
