# FINGERTIP_SENSOR_AUDIT（第一阶段，2026-11-20）

来源（均为固定版本源码，不是口头约定）：
- MuJoCo 模型：`Eter0109/a3_dual_arm_sim @ 08ebe8c` 的 `models/a3_cookie_transfer_2x10.xml`
- 观测拼装：同 commit `src/a3_dual_arm_sim/sim/env.py`（`FORCE: Box(..., (18,))` 与 `force = np.r_[...]`）

## 1. 18 通道逐列身份（已核实）

| 通道 | 来源 | 维数 | 单位/语义 |
|---|---|---|---|
| 0–2 | `L_wrist_force` | 3 | N，左腕力，法兰坐标系（非世界系） |
| 3–5 | `L_wrist_torque` | 3 | N·m，左腕力矩 |
| 6–8 | `R_wrist_force` | 3 | N，右腕力 |
| 9–11 | `R_wrist_torque` | 3 | N·m，右腕力矩 |
| **12** | `L_finger_inner_touch_sensor` | 1 | N，左指垫内侧 **法向接触力（标量、非负）** |
| **13** | `L_finger_outer_touch_sensor` | 1 | N，左指垫外侧 法向接触力 |
| **14** | `R_finger_inner_touch_sensor` | 1 | N，右指垫内侧 法向接触力 |
| **15** | `R_finger_outer_touch_sensor` | 1 | N，右指垫外侧 法向接触力 |
| 16 | `sum(actuator_force[L fingers])` | 1 | N，左夹爪**驱动合力**（不是指垫接触！） |
| 17 | `sum(actuator_force[R fingers])` | 1 | N，右夹爪驱动合力 |

结论：
- **真正的指尖接触只有 12/13（左）与 14/15（右）**；ch16/ch17 是驱动合力，禁止当作第三个指尖力。
- MuJoCo `touch` 传感器是标量法向力，**无方向、非负**；不存在力矩通道。
- 旧 force6 `[0,1,2,12,13,16]` = 左腕三轴 + 左指垫两个 + 左驱动；右侧指垫 14/15 被排除的理由（旧审计“右臂恒定/任务用左臂”）需要在数据上复核，而不是继承。

## 2. 与旧结论的差异
- 旧 `force_calib_v1.json` 的 channel_note 与本次 XML/env 逐列一致（左腕 0–2、左垫 12–13、左驱动 16），但**没有覆盖右侧指垫 14/15**。
- 新项目的指尖力定义应为 `[12,13,14,15]`（左右各内外），或者按任务执行臂做带 mask 的子集；ch16/17 仅可作为可选辅助量并单独标注，不作为指尖力。

## 3. 待验证（不得写成已知）
1. 零载基线与接触阈值：需在同一实例上读 sensordata（或从归档 force18 的静置段统计）。
2. 4372.9 N 异常尖峰的通道来源：需逐通道定位（历史 `force18` 已归档，可离线重查）。
3. 右指垫是否在任务中真的恒为零：用归档语料逐通道统计接触段。
4. 采样时间与控制步关系：env.py 显示传感器读取发生在 mj_step 之后（注释“one physics substep behind”），需在实例上用同一状态核验一次。

## 4. 与 FasterWAM 原始协议的关系
原始 A3 配置（`a3_2cam.yaml`）**不含任何力输入**；18 通道力只存在于环境观测。新 F1/F2 的力输入必须由独立 `FingertipForceProcessor` 产生（ch12–15），不得回填到原始 8 维 proprio，也不得复用旧 14 维 proprio。
