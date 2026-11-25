# B2 覆盖说明（部分覆盖，不可称六阶段全覆盖）

B2（`B2_SIXPHASE_TEACHER.json`）采用 V7 六阶段命名 + 显式映射，**实测只覆盖到其中 4 个阶段**：

| V7 阶段 | n | 备注 |
|---|---|---|
| APPROACH_OPEN | 1 | open & 无力 |
| CONTACT_CLOSE | 5 | mid & 无力 |
| DESCEND_CLOSED | 2 | closed & 无力 |
| CARRY_CLOSED | 16 | closed & 有接触力 |
| RELEASE_OPEN | **0** | mid & 有接触力 → 本批无样本 |
| RESET_OPEN | **0** | 仅凭 (label,force) 不可分，且无样本 |

- 样本：n=24，跨 **11 个独立 episode**（episode 145,214,23,318,347,374,451,483,50,680,7）。
- 结论必须写成"**部分覆盖（4/6 阶段有样本；RELEASE_OPEN、RESET_OPEN n=0）**"，**不得**写成"六阶段全覆盖"。
- 所有结论为 teacher-forced 观测（val 专家观测，32 步 infer，5 去噪），0 优化步，非能力结论。
