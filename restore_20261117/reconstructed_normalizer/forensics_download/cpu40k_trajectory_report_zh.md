# CPU 40k 执行轨迹取证分析

## 身份与证据边界
- NORMALIZER_IDENTITY：`RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL`；original_stats_sha256：`UNKNOWN`。
- 仅描述可追溯重建 normalizer 条件下的 40k，不冒充原始 normalizer 或原始评估结果。NumPy 本地 CPU 分析，无 GPU、无重新推理。
- 新 NPZ 是真实 observation 的独立记录；与旧记录逐项对齐校验，不以局部一致推断记录器无副作用或完整实验身份等价。
- 原始预测、实际执行命令、IK/限速 opening 目标、实测 opening 分离。开/闭仅由实测 opening 差定义，命令翻转不等于实际开合反转。
- wrist 指 force 的 0:3 三列范数，并非模型全部六个力通道；pads 只用 12:16。不能把包含力矩的 force18 直接整体求牛顿范数。

## 统计口径
- frame0 为 reset；命令 i（0-based）完成后为 frame i+1。翻转记在较新命令的完成帧，初始命令没有前一条执行命令，不计机会。
- 块内/边界率=翻转数/对应相邻执行命令对数，不用翻转构成比代替机会率；边界计入新 chunk。
- 首次 maxabs(pads)>1N 为 c；互斥 pre：f<c-10、附近：c-10≤f≤c+10、post：f>c+10；未接触单列。按完成帧分层，不表示命令执行前已接触。
- raw 小：两端 abs≤0.1；大：两端 abs≥0.3；其他混合。所有机会均分类，JSON 包含交叉分层及幅度 min/median/mean/p95/max。
- seed12 仅执行前10行；32行计划的后22行不计机会、flip 或轨迹。
- rail 是 handed/requested 归一化命令 abs≥1−1e−6 的描述，分别统计 XYZ 和前6维；不把 gripper 的 ±1 当异常，也不称实测 XYZ 越界。

## 执行翻转结果

|seed|执行命令|flip|块内 flip/机会（率）|边界 flip/机会（率）|首次接触帧|对齐|
|---|---:|---:|---|---|---:|---|
|11|600|38|32/540 (5.93%)|6/59 (10.17%)|40|PASS|
|12|10|2|2/9 (22.22%)|0/0（不可估）|None|PASS|
|17|600|96|75/540 (13.89%)|21/59 (35.59%)|64|PASS|

### seed 11
- 实测开合命令步计数：{"closing": 220, "opening": 190, "stationary_eps1e-4": 190}
- 分层机会率：

|分层|flip|机会|率|
|---|---:|---:|---:|
|phase=pre|5|28|17.86%|
|phase=near_pm10|5|21|23.81%|
|phase=post|28|550|5.09%|
|phase=no_contact_observed|0|0|不可估|
|raw_class=small|2|2|100.00%|
|raw_class=large|12|507|2.37%|
|raw_class=mixed|24|90|26.67%|

- 力阈值首次超出帧（严格 >）：

|N|pads maxabs 首帧|wrist 0:3 norm 首帧|
|---|---:|---:|
|1|40|0|
|10|40|0|
|50|None|None|
|100|None|None|
|300|None|None|

- handed rail：{"XYZ": {"first_command_frame": 332, "count": 1}, "first6": {"first_command_frame": 332, "count": 1}}
- 对齐检查：{"frame_mapping": true, "actual_opening_matches_recorded": true, "actual_pads_match_recorded": true, "adapter_equals_handed": true, "handed_equals_requested": true, "raw_denorm_gripper_exact": true, "pm1_threshold_matches_raw": true, "adapter_gripper_negative_pm1": true, "all_raw_flips_match_executed_gripper_flips": true}
- flip 前后实测 opening/运动、wrist、pads、左右 EEF 相对 reset 姿态角与 XYZ、±10帧快照/窗口在逐 flip CSV 和 JSON 中；时间线全部超阈值区间、rail 全帧列表在 JSON。
- 每 chunk 原始幅度、机会率、开合净差、姿态和 XYZ、rail 数量在逐 chunk CSV。

### seed 12
- 实测开合命令步计数：{"closing": 7, "opening": 3}
- 分层机会率：

|分层|flip|机会|率|
|---|---:|---:|---:|
|phase=pre|0|0|不可估|
|phase=near_pm10|0|0|不可估|
|phase=post|0|0|不可估|
|phase=no_contact_observed|2|9|22.22%|
|raw_class=small|0|0|不可估|
|raw_class=large|2|7|28.57%|
|raw_class=mixed|0|2|0.00%|

- 力阈值首次超出帧（严格 >）：

|N|pads maxabs 首帧|wrist 0:3 norm 首帧|
|---|---:|---:|
|1|None|0|
|10|None|0|
|50|None|None|
|100|None|None|
|300|None|None|

- handed rail：缺失完整 XYZ/前6维执行命令，不估算。
- 对齐检查：{"plan_first10_equals_attributed_raw": true, "actual_opening_matches_attribution": true, "actual_opening_before_matches_attribution": true, "all_raw_flips_match_executed_gripper_flips": true}
- flip 前后实测 opening/运动、wrist、pads、左右 EEF 相对 reset 姿态角与 XYZ、±10帧快照/窗口在逐 flip CSV 和 JSON 中；时间线全部超阈值区间、rail 全帧列表在 JSON。
- 每 chunk 原始幅度、机会率、开合净差、姿态和 XYZ、rail 数量在逐 chunk CSV。

### seed 17
- 实测开合命令步计数：{"closing": 249, "opening": 256, "stationary_eps1e-4": 95}
- 分层机会率：

|分层|flip|机会|率|
|---|---:|---:|---:|
|phase=pre|16|52|30.77%|
|phase=near_pm10|4|21|19.05%|
|phase=post|76|526|14.45%|
|phase=no_contact_observed|0|0|不可估|
|raw_class=small|5|8|62.50%|
|raw_class=large|57|510|11.18%|
|raw_class=mixed|34|81|41.98%|

- 力阈值首次超出帧（严格 >）：

|N|pads maxabs 首帧|wrist 0:3 norm 首帧|
|---|---:|---:|
|1|64|0|
|10|129|0|
|50|143|143|
|100|157|153|
|300|159|185|

- handed rail：{"XYZ": {"first_command_frame": 151, "count": 248}, "first6": {"first_command_frame": 151, "count": 248}}
- 对齐检查：{"frame_mapping": true, "actual_opening_matches_recorded": true, "actual_pads_match_recorded": true, "adapter_equals_handed": true, "handed_equals_requested": true, "raw_denorm_gripper_exact": true, "pm1_threshold_matches_raw": true, "adapter_gripper_negative_pm1": true, "all_raw_flips_match_executed_gripper_flips": true}
- flip 前后实测 opening/运动、wrist、pads、左右 EEF 相对 reset 姿态角与 XYZ、±10帧快照/窗口在逐 flip CSV 和 JSON 中；时间线全部超阈值区间、rail 全帧列表在 JSON。
- 每 chunk 原始幅度、机会率、开合净差、姿态和 XYZ、rail 数量在逐 chunk CSV。

## 缺失与视频估计（不混入实测时间线）
- seed12 缺少完整 XYZ/前6维执行链数组，不能据 gripper JSON 恢复 rail；其余真实 observation 按11帧使用。
- 无结构化视频估计事件输入，本次未重新做视觉判读；视频事件列表为空不代表视频里没有事件。抓取、掉落、碰撞等视觉估计不得由力阈值替代或伪造。
- None 表示观察范围内未超阈值或字段不可用，具体以 missing 和对应字段定义区分；短 seed12 的未观察到不代表长期不发生。

## 解释限制
- 翻转、接触、姿态漂移、rail 的时序共现只构成相关描述，不证明任何一项导致失败；阈值不是任务失败标准。
- 独立 CPU observation 与已有 executed 链匹配，只对已校验字段有效；不外推模型六通道力、未执行预测或记录器无副作用。

## 复核与产物
- 脚本：`C:\Users\ASUS\Documents\Qoder\2026-10-31\b462d38d\_clone_default\restore_20261117\reconstructed_normalizer\cpu40k_trajectory_analysis.py`
- 输入绝对路径及 SHA256、身份原文、计算定义和全量结果均在 JSON；无随机采样。
- json：`C:\Users\ASUS\Documents\Qoder\2026-10-31\b462d38d\_clone_default\restore_20261117\reconstructed_normalizer\forensics_download\cpu40k_trajectory_analysis.json`
- flips：`C:\Users\ASUS\Documents\Qoder\2026-10-31\b462d38d\_clone_default\restore_20261117\reconstructed_normalizer\forensics_download\cpu40k_trajectory_flips.csv`
- chunks：`C:\Users\ASUS\Documents\Qoder\2026-10-31\b462d38d\_clone_default\restore_20261117\reconstructed_normalizer\forensics_download\cpu40k_trajectory_chunks.csv`
- md：`C:\Users\ASUS\Documents\Qoder\2026-10-31\b462d38d\_clone_default\restore_20261117\reconstructed_normalizer\forensics_download\cpu40k_trajectory_report_zh.md`
