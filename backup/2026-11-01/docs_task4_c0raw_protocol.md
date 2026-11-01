# 任务四：自主抓放指标与 C0-raw 诊断协议

代码：`/workspace/_r4_stage/autonomy_metrics.py`（sha256 `a9096bed0ad055fe069951ace6eee30b10752781dd5c1af1b3c1bb25b1cc3c40`，本地同名副本）。
评估脚本：`/workspace/_r4_stage/up/run_matrix.new.py`。

## 1. C0-raw 是什么，落在哪条代码路径上

C0-raw 就是已经跑通的 B0 档，不另开一套实现，避免两条路径慢慢走偏。`resolve_control()` 里
`B0` 返回 `(控制名="B0", wrapped=False, profile=full, released=空集)`，`run_one()` 里
`bare = control_name == "B0"`，执行的动作就是 `post = pre.copy()`，也就是适配层输出本身。

保留的部分（这些是"必要动作适配"和"异常接触下的实验终止"，不是任务级辅助）：

| 保留项 | 位置 | 说明 |
| --- | --- | --- |
| 帧适配 + 夹爪二值化 | `policy.adapted(chunk7, obs)` | 14 维模型输出→仿真动作约定，pm1 夹爪门限。所有档位都过这一层 |
| 动作合法性校验与限速 | `sim/env.py` `validate_action` / `_limit_joint_action` | 在仿真环境内部，不属于 safety_wrap |
| 物理异常急停 | `sim/env.py:270-274` | 子步内 qpos/qvel 出现非有限值、机器人关节角速度超过 40、任意物体速度超过 1000 三种情况触发 `emergency_stop`，`terminated=True` 并写 `info["safety_reason"]` |
| 步数上界 | `--max-steps 600` 与环境 horizon 6000 | 到点 `truncated` |
| 非有限末端位姿即中止 | `run_matrix.new.py:213-219` | 记 `nan_abort_step`，不再往下污染日志 |
| 初始场景体检 | `env_factory.scene_sanity` | 饼干重叠或越界的场景直接拒评，避免把场景问题算成策略失败 |

去掉的部分（只在 C0 里生效，B0 里只影子计数不执行）：`safety_wrap.apply()` 的 25 N 软阈值＋连续 3 帧＋抬 z 0.06、40 N 硬截断、盒 footprint 限制、单步位移裁剪。

B0 的产物里额外写了 `safety_would_have_changed_pct`、`safety_would_have_hard`、
`safety_would_have_binfloor`：影子状态跑一遍 safety 层但丢弃输出，只统计"如果开了 Safety 会改多少步"。
所以 B0/C0 配对能直接回答"这 112 步改写到底救了什么、坏了什么"。

## 2. 九项自主抓放指标

`autonomy_metrics.py` 只读录制的 npz（`cookie_xyz (T,N,3)`、`eef_xyz`、`opening`、
`pad_force_inner/outer`、`wrist_force`、`changed`），不重新仿真，因此同一份录制可以反复改判据。

判据的几何常量取自场景本身：盒心 (0.095, 0.100)，内壁半宽 (0.057, 0.029)，盒底上表面 z=0.775，
饼干半尺寸 [0.025, 0.00317, 0.0125]。

1. **真实单块抓取次数**：把"饼干相对末端刚性贴合 + 距离 < 60 mm + 夹爪张开度 < 0.42"记为
   `attached`；一次夹取事件是一段连续有任意饼干 attached 的帧区间，事件内 attached 数量的最大值
   就是这次夹了几块，等于 1 才计入单块抓取。
2. **稳定夹持时间**：事件长度（帧，20 fps）。
3. **实际搬运距离**：事件内被夹饼干集合的质心位移（首帧均值到末帧均值）。
4. **正确释放次数**：事件结束 10 帧后该饼干已不再 attached，且 xy 落在盒内壁投影内。
5. **释放后稳定入盒数量**：末帧同时满足"在盒内投影、贴近盒底、尾部 50 帧内位移 < 8 mm、
   未被夹、夹爪已张开"。
6. **同时夹住或拖拽多块**：事件内 attached 最大值 ≥ 2 记一次连带。
7. **指垫力/腕部力的峰值与分布**：`max(pad_inner, pad_outer)` 与 `‖wrist_force‖` 的 p50/p95/max，
   外加超过 25 N（软阈值）和 40 N（硬阈值）的帧数。
8. **接触力过大导致的失败**：>40 N 的持续帧数与事件重叠情况；C0 档另报被改写步数。
9. **夹爪命令反复翻转**：观测张开度符号翻转次数与"实际下发"张开度符号翻转次数分开统计，
   前者是模型意图抖动，后者是真正打到执行器的抖动。

### 2.1 `cookies_in_target` 的三套口径（2026-10-31 钉死，r5 结论必须按这一节读）

同一个 seed-22 录制上同时存在三个"入盒数"，之前混用过，现在分开：

| 口径 | 定义 | 出处 | C0/22 | B0/22 |
| --- | --- | --- | --- | --- |
| **官方** | `_cookie_inside_target()`：盒体局部系下**带朝向的完整 footprint** 进内壁（+4 mm 容差）、底面在 `[盒底上沿−3 mm, 墙高)`、`立正`（倾斜 ≤ `max_tilt_rad`）、`静止`（线速 ≤ max(0.09, …)）；**`cookie_transfer.py:29` 是 `require_released=False`，所以不要求松手** | `tasks/cookie_transfer.py:434-472`，落到 `info['cookies_in_target']`，由评估 harness 写进 `summary_lora.json` | **3** | **6** |
| **自算松口径** | 只看饼干**中心** xy 落进盒内壁投影，不查 z、不查朝向、不查是否静止、不查是否松手 | `autonomy_metrics.py` 打的 `cookies_in_target(末)` | 6 | 6 |
| **任务口径（报告用）** | 九项里第 4/5 项：末帧在盒内投影 + 贴近盒底 + 尾部 50 帧位移 < 8 mm + **已松手** + 夹爪已张开 | 同上，字段 `rel_in_bin` / `rest_in_bin` | **0** | **0** |

三条结论：

1. **官方 3 与自算 0 不矛盾，是两个问题。** 官方那一栏按构造就不要求松手、也不要求"落在盒底"，
   它数的是"此刻这块饼干的整个身位处在盒壁围成的体积里"。录制里这些饼干终帧 z 在
   0.789–0.807，而盒底上沿是 0.775，饼干半厚 0.00317 —— 平放该在 0.778 附近，
   也就是它们是**在盒子里叠了 2–3 层/或斜搭着**：中心 xy 全进投影（6），带朝向 footprint
   + 立正 + 静止筛掉 3 个（官方 3），再要求"落在盒底且已松手静止"一个都不剩（任务口径 0）。
2. **我这边原来的记录是口径标错，不是数据错。** §3 表里 C0/22 那行写的
   `cookies_in_target(末)=6` 其实是自算松口径的值，而 harness 官方值是 3；
   B0/22 两者都等于 6 把这个错误盖住了。从现在起这一列只写自算值并改名为
   `xy_projection(末)`，官方值一律从 `summary_lora.json` 的 `cookies_in_target` 读，
   **报告"成功入盒几块"一律用 `rest_in_bin`**。
3. `mech_eval.py` / npz 自算**不能**替代官方口径下任何"官方说了多少"的结论，反过来也一样：
   评估 npz 里根本没有 `cookies_in_target`、`cookies_in_target_mask`、饼干四元数、`qvel`
   这些字段（只有 `cookie_xyz (600,80,3)`、`eef_xyz`、`opening`、力通道），
   所以官方的朝向/静止条件**离线复现不了**，只能读 harness 当场写的 json。
   要逐帧对齐官方，得在 recorder 里加一个 `cookies_in_target_mask` 键（只是多存一个 info 字段，
   不动模型/Wrapper/Safety/仿真执行逻辑），这一条等 r5 对照跑完再决定要不要加。

另外把 B0/22 的读数钉死，防止以后被当成成功：同一次 npz 复算
`pad_force_inner` 峰值 **859.170 N**、>2 N 帧 **542/600**（§3 记的 865.43 N 是
`max(pad_inner, pad_outer)` 两通道取大，两者不冲突但必须写清通道）；
终帧 6 块中心 xy 在盒内投影、但 z 仍停在源高度 0.768 附近、`opening≈0.37`，
即**挂着/推着饼干整体压过去**，正确释放 0 次、稳定入盒 0 块。
任何报告里这条只作为"碾压/挂爪"证据，不得计入自主抓放成功。

一个必须记住的尺度问题：第 1 项的 `opening < CLOSED_OPENING` 用的是策略的 pm1 张开度刻度
（默认 0.42）。专家夹爪在单块低力档位合到 0.011、只开到 0.192，用 0.42 判会把"夹着"读成
"张开"。所以 2026-10-31 起这个阈值改成可读环境变量：`CLOSED_OPENING`，默认仍是 0.42，
打策略录制用默认值，打单块专家录制用 0.12。同一份 npz、同一个打分器，只是刻度不同。

## 3. r4 在 C0-raw(B0) 与 C0 下的读数（历史协议，同一权重、同一 seed、同一动作流）

```
lora_B0_22  events=2  single=0  single_lift=1  bundle=2  held_no_release=2
            rel_in_bin=0  rest_in_bin=0  dur_med=254帧  car_med=72mm
            pad p50=38.15 p95=87.12 max=865.43 N | >25N 452帧 >40N 250帧 | wrist p50=11.21 max=333.05
            grip flips obs=244 cmd=66 | 改写步数=0 | cookies_in_target(末)=6
lora_C0_22  events=1  single=0  single_lift=1  bundle=1  held_no_release=1
            rel_in_bin=0  rest_in_bin=0  dur_med=340帧  car_med=130mm
            pad p50=18.92 p95=26.92 max=49.79 N | >25N 80帧 >40N 7帧 | wrist p50=11.10 max=97.64
            grip flips obs=257 cmd=68 | 改写步数=112 | cookies_in_target(末)=6
```

三点解读：

- B0 裸模型在真实接触里把指垫力顶到 865 N、250 帧超过 40 N 硬阈值，说明力不是偶尔尖峰，
  而是持续挤压；这正是任务二/任务三要解决的。
- C0 把峰值压到 49.79 N，代价是 112/600 步动作被改写，两种档位最终都没有"释放入盒"，
  入盒数 6 全部来自夹在闭合的夹爪上。Safety 只能限力，不能让模型学会放。
- 腕部六维力在两种档位都接近 11 N 的重力基线，峰值差主要来自指垫，说明后续把力作为观测时，
  指垫通道信息量明显大于腕部通道（详见任务三设计）。

异常终止这一项也要落个实数：六个录制（B0/C0 × seed 11/21/22）全部 `steps=600`、`replans=60`、
`nan_abort_step=-1`，也就是环境内部的物理急停（非有限 qpos/qvel、机器人关节角速度 >40、
任意物体速度 >1000）和非有限末端位姿中止都没有触发过，回合是被 `--max-steps 600` 截断的。
C0 seed 22 的 safety 层内部计数是 hard=7、binfloor=79；同一动作流在 B0 下影子计数是
`safety_would_have_hard=249`、改写 75.83 % 的步。结论是：裸模型跑满 600 步不会把仿真跑崩，
它崩的是饼干本身（865 N、250 帧超 40 N）；保留的终止机制是保险丝，在这次评估里没派上用场。

## 4. 后续 r5 评估怎么接

r5 的对照跑沿用同一 `run_matrix.new.py`、同一 seeds（11/21/22）、`--controls B0,C0`、
`--replan 10 --max-steps 600 --gripper-mode pm1 --video`，输出目录新建，不写回
`r4_c0_frozen` 或 `r4_b0_c0_paired`；跑完直接对同一套 npz 调 `autonomy_metrics.py`，
保证 r4/r5 的九项指标可以逐项对齐比较。

执行状态（2026-10-31）：r5 训练 20:31 退出后，`r5_watch.sh` 自动接 `r5_eval.sh`，
20:34 已在 `/workspace/a3_eval/r5_b0_c0_paired/` 写下 `manifest_lora.json` 并开始逐条评估；
r4 的对照目录只有 3.1 MB（npz 不存图像），当时 `/workspace` 余量 3.42 GB，磁盘不是瓶颈。
r5 的九项读数出来后，**入盒相关的一律按 §2.1 的三套口径分栏写**：
`rest_in_bin`（任务口径，报告用的成功数）、`xy_projection(末)`（自算松口径）、
`summary_lora.json` 里的官方 `cookies_in_target`，三者不得互相顶替。
