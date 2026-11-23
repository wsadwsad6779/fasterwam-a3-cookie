# P8：8 维 proprio 表示冻结（2026-11-23）

## 决定
新项目 F0/F1/F2 统一使用 **8 维 proprio = [左 EEF 位置 xyz(3), 左 EEF 姿态四元数 wxyz(4), 左夹爪开度(1)]**，
全部来自数据集 `observation.eef_pose`（14 = 左 7 + 右 7）与 `action[7]`（左夹爪，归一化开度，1.0=张开）。
右臂 7 维不进入 proprio。

## 证据
- 原始 A3 配置 `stage_40k/fw/configs/data/a3_2cam.yaml`：`state.raw_shape: 8`；README 记载 8 维为“末端位置/姿态与夹爪开度”。
- 官方基座 checkpoint：`proprio_encoder.weight=[4096,8]`（P1 实测）。
- 数据集 `observation.eef_pose` = 14 维，`recording.py:84` 命名 `eef_pose_0..13`；左臂在前 7 列（P7 连续性核验通过）。
- 数据集 `observation.state` 是 8 维关节（左臂 7 关节+夹爪，见 STATE_MAPPING_16_TO_8.json）——**不采用**，因为与原始 8 维 proprio 语义不同。

## 后果与限制
- 夹爪开度映射：数据集为归一化开度（0.28–1.0，1.0=开）；用 V2 滞回（<=0.6 闭、>=0.8 开）转 pm1 后进第 8 维；阈值在训练前冻结。
- 姿态用四元数还是 rotvec 需与模型输入约定一致：随 wxyz 四元数直接进 8 维在数值上跨度小，但需在 P4 全模型 smoke 时用真实 forward 验证；若模型期望 rotvec，则把四元数转 rotvec（仍 3 维）后再拼。
- 该决定不改变 7D 动作定义（前 6 维 eef 增量/rotvec，第 7 维 pm1 夹爪）。

## 证据文件
`p_items/STATE_MAPPING_16_TO_8.json`、`p_items/PHASE1_BASE_CHECKPOINT_INSPECT.json`、`stage_40k/fw/configs/data/a3_2cam.yaml`
