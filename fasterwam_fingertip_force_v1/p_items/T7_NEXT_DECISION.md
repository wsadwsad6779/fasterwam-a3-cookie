# T7 下一步最小实验决策（2026-11-26）

## 当前已核事实（本轮）
- P0 scheduler=`WanContinuousFlowMatchScheduler` shift=5.0, train_timesteps=1000；**scheduler_steps 随 K = 5/10/20**（K 真生效），输出恒 [32,7]。
- P0 起点(approach)固定观测：K=5/10/20 执行夹爪 <0.30 均 0（min≈0.44–0.45，median≈0.55–0.58，K↑median 略升）。**approach 本就该张开，故此点不能判"去噪不足"**。
- 旧"12k 3/6块"=**NOT_VERIFIED**；可核是 A3 LoRA/pm1 `in_target_final 8/2 success:false`（倒着进框）；旧无加力12k ckpt **缺失**。
- 现 F0/F2 全里程碑 strict=0、exact=0；F0 15k relaxed 也=0。

## 假设 × 证据 × 反证（不认定唯一根因）
| 假设 | 支持证据 | 反证/漏洞 | 下一个最小实验 |
|---|---|---|---|
| **H1 去噪步数不足(5)** | 仅 5 步可能解不干净多模态；K 确实改变迭代数 | approach 帧 K 增大不降开度；未在接触帧测 | **P1**：CONTACT_CLOSE 专家帧上 K=5/10/20 配对，测闭合召回/误闭合 |
| **H2 监督/输出表示(速度MSE→半开)** | flow-velocity MSE 7维等权；teacher-forced chunk 后段闭合衰减(B2)；执行<0.30 5% vs 标签49% | 未证明=条件均值；需看 teacher-forced 分布与去噪轨迹 | teacher-forced 逐维分布 + denoise 轨迹；夹爪分类头 3k 试点（需批预算） |
| **H3 闭环状态分布偏移** | teacher-forced(专家观测)有闭合，自主罕见；C-20261124-05 开爪擦碰→空闭 | 需按阶段量化自主 vs 专家状态 | F0 9k/12k/15k 同 seed v2 重跑→phase 分类/状态覆盖审计 |
| **H4 力负迁移** | （曾被怀疑） | **F0(无加力)也全 0；B3/B5 力≈无影响** → 强反证 | 不列为优先 |
| **H5 旧实验协议不同/回归** | ≥5 处同时不同(标签/场景/谱系/数据/评估链) | 旧 ckpt 缺失 | 找回旧 checkpoint/adapter→同场景评估；否则 NOT_VERIFIED |

## 推荐顺序
1. **P1（免训练，最便宜）**：F2 15k 在 CONTACT_CLOSE/CARRY 专家固定观测上 K=5/10/20（3 噪声配对），看闭合召回/误闭合是否随 K 改善 —— 直接验/否 H1。
2. **P3 续**（CPU）：GitHub/HF 归档找旧 checkpoint/适配器；找不到则 H5=NOT_VERIFIED。
3. **H2/H3**：teacher-forced 分布 + 同 seed v2 重跑 phase 分类；夹爪头需预算。
- 不盲跑全部 P2；P1 无正信号则转向 H2/H3，不把"加去噪步数"当默认解。
