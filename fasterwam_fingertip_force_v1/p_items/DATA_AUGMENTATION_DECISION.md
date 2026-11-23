# DATA_AUGMENTATION_DECISION（P7，2026-11-23 依据 P6 实据）

- 数据源：Eter0109/a3-cookie-multi-tasks（LeRobot v3.0）
- 规模：845 episodes / 459,177 frames / 20 fps；749 条 2x5 + 96 条 2x10；complete_demonstrations=true；left arm
- 去重：845 条动作轨迹哈希全不相同（0 个完全重复组）；状态哈希同样分属各回合
- 接触覆盖：左指垫 ch12/13 接触帧占比 ≈50.9%/50.9%；右指垫 0%（本任务由左臂执行）
- 维度：action=8（joint_position：L_q1..q7+gripper）、state=8、force=18

## 判定：不再因“数量不足”补采
845 条独立轨迹、覆盖 2x5 与 2x10 两种目标布局、含真实指尖力与双相机视频，已满足 F0/F1/F2 的训练规模需求。

## 仍需注意（不得当作已解决）
1. 布局多样性只来自 seed 70000/80000 两段的随机化档（basic），元数据未给出逐回合相机外参/光照/摩擦 —— 标 **未取到**。
2. 数据存储动作是 joint_position，进入 FasterWAM 7D 监督前必须转换。**首选**：用 parquet 已有的 `observation.eef_pose` 直接做 7D 差分/rotvec（与旧教师管线的 pose→7D 口径一致），第 7 维用 joint gripper（pm1）；**备选**：关节角经仿真 FK 得 EEF 位姿后再差分。禁止把 8D 关节直接当 7D。
3. 若后续评估显示跨布局泛化差，再按缺失维度（相机/光照）定向补采；当前不盲目扩采。
