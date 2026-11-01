# 任务二：低力专家示范数据

代码：`/workspace/_r5_stage/replay_forces.py`。同一份逻辑的三个版本 sha256（结论按版本可追溯）：
验收用的冻结版 `a71cbd26a46cf34f9f47ab25020b972c1afa3b2fdc7cbbe222dd273fd4fb4865`；
加变体选择（源列/行/力档）之后 `f60a69cb124bf58ae74f639336930ecc0eced47b1ece9e89748a7b427150ad07`
（`variant_index=0` 与冻结版逐帧同结果，见 `demos_postpatch_smoke`）；
列限制为可达的两列后再变（`patch_columns2.py`，sha 记在该版运行日志里）。
数据准备脚本：`/workspace/_r5_stage/build_force_arrays.py`
（sha256 `4c4cbad32fee214aec3dd6738ad959dc59a140948636275909aef7c3827f6a73`）。
语料：`/workspace/data/a3_front_close_left_100`（LeRobot v3，100 条示范）。
回放产物：`/workspace/a3_eval/replay_single_gate10/`、`replay_single_gate/`、`replay_bundle_dist/`。

## 0 结论

原始 100 条专家示范**本来就带完整的力学通道**，指垫力、腕部六维力/力矩、时间戳都在，
且与图像、状态逐行对齐。所以"重新采集同步力学信息"这件事不需要重做——需要的是
把已被训练数组丢掉的那一列读回来，并用确定性回放补齐分阶段（接触/闭合/保持/搬运/释放）
的力标注。分阶段标注语料里没有，只有回放能给出。

真实专家数据给出的合理接触力区间（不是我事先定的目标值）：单指垫 p50 0.938 N、p95 1.445 N、
全语料最大 1.54 N，42681 帧里 0 帧超过 2 N。r4 裸模型把同一个指垫顶到 865 N，
差三个数量级，说明高力不是场景本身的属性，而是策略的输出问题。

新增示范按**一集一次完整抓放**采集：10 seeds 的单块低力回放 10/10 通过独立验收；
把 3 次夹取塞进一集就会在后续夹次出现 6.9–12.6 N 的尖峰并让 VERIFY_RELEASE 失败，
原因是盒内槽位行距 6.333 mm 对饼干厚度 6.34 mm，几乎零间隙。

## 1 原始示范的力数据审计

`meta/info.json` 的 feature 表（逐条读出来，不是推测）：

| 字段 | dtype | shape | 说明 |
| --- | --- | --- | --- |
| `observation.images.front/left_wrist/right_wrist` | image | — | 三路图像 |
| `observation.state` | float32 | [8] | 本体状态（训练用的 8 维 proprio 就是它） |
| `observation.velocity` | float32 | [16] | |
| `observation.eef_pose` | float32 | [14] | |
| **`observation.force`** | float32 | **[18]** | 腕部六维力/力矩 ×2 + 指垫 ×4 + 夹爪驱动力 ×2 |
| `action` | float32 | [8] | |
| `timestamp` / `frame_index` / `episode_index` / `index` / `task_index` | | | |

实际读全部 50 个 parquet 汇总：

- 行数 `(42681, 18)`，100 个 episode，`NaN` 计数 0；42681 与 `a3_fw/arrays.npz` 的帧数逐帧相等。
- `timestamp` 每个 episode 从 0.0 起、单调递增，相邻帧间隔中位数 `0.05000019073486328` s
  → 严格 20 Hz，与图像/状态同一行号，不需要额外对齐或插值。
- 18 维读数（单位：力 N，力矩 N·m；`in/out` 指左侧指垫内/外两路）：

```
L_fx   p50 -0.003  p95  0.064  max  3.563  min -1.408
L_fy   p50 -10.855 p95 -10.112 max -9.449  min -14.268   ← 左臂重力基线
L_fz   p50 -0.010  p95  0.166  max  0.454  min -1.380
L_tx/ty/tz  |max| ≤ 0.066 N·m                            ← 腕部力矩几乎不携带信息
R_fx 0.0  R_fy 10.834(常数)  R_fz 0.0  R_tx/ty/tz 0.0    ← 右臂通道在语料里是死的
L_in p50 0.938 p95 1.445 max 1.540
L_out p50 0.938 p95 1.454 max 1.518
R_in 0.0  R_out 0.0                                      ← 右指垫恒零
L_act p50 1.880 p95 2.897 max 2.952 min -1.238           ← 夹爪驱动力合计
```

- 逐帧（42681 帧全样本，含未接触帧）的 `max(L_in, L_out)`：p50 0.955 N、p95 1.455 N、最大 1.540 N；
  逐 episode 峰值的中位 1.45 N、最小 1.343 N、最大 1.54 N。只统计夹持期的话中位在 1.39 N 量级
  （与语料控制器的 `target_grip_force=1.3 / stable_grip_force=1.0` 一致）。
- 阈值统计：>2 N **0 帧**，>25 N 0 帧，>40 N 0 帧（左右两侧都是 0）。

训练数组 `/workspace/a3_data/a3_fw/arrays.npz` 只有
`action (42681,7)`、`proprio (42681,8)`、`opening`、`joint_cmd`、`episode_index`、
`ep_start/ep_len/ep_ids`——**没有 force 列**。力数据是在准备训练数组时被有意裁掉的，
不是仿真没存。所以 V1 力感知（任务三）不需要重采数据，只需要重做数组。

两个额外事实直接影响编码设计：右臂 6 维里只有 `R_fy` 有值且恒为 10.834 N（常数），
右指垫两路恒零；左臂重力投影在 `L_fy`（约 −10.86 N），右臂同号为正，说明两侧传感器
装反方向，归一化时必须按通道各自处理，不能共用一个尺度。

## 2 确定性回放能不能代表真实力学

回放用的专家控制器和采集这 100 条语料的是同一套（`A3SameColumnBatchExpert`，
`batch_size=5` 的捆绑档），同一 MuJoCo 模型、同一 20 Hz。所以可以用语料本身验收回放：

| 量 | 语料实测 | 捆绑档回放（seeds 7031/7032，噪声 0.3 mm） |
| --- | --- | --- |
| 夹持期间指垫力 | p50 0.938 / p95 1.445 N | hold_mean 1.339–1.377 N |
| 单次事件峰值 | max 1.54 N | closure_peak 1.354–1.398 N |
| >2 N 帧数 | 0 | 0 |
| 腕部重力基线 | L_fy p50 −10.855 N | wrist_p50 10.72–10.73 N（取模长） |
| 一次夹走几块 | —（语料按列成批） | `max_cookies_at_grip` = 5，`cookies_raised_20mm` = 5 |

回放的力落在语料分布的右上角（略高、但同一量级、同一"零超 2 N"性质），
6 条里 5 条（seeds 7031/7032/7033/7035/7036）把 10 块饼干全部立着放进了盒里
（`in_target_final=10`、`moved_into_bin_upright=10`、`moved_into_bin_tilted=0`、`frames_over_2n=0`、
指垫峰值 1.330–1.538 N）。

第 6 条 seed 7034 是同一套脚本专家自己的失败样本，很有诊断价值：
`steps=414`、`stop=expert_failed`、最终只有 5 块在盒内、6 块直立、**5 块倾斜**，
指垫峰值 2.384 N、2 帧超过 2 N（其余 5 条都是 0 帧）。
它说明在 0.3 mm 起始噪声下，捆绑档的失败模式就是"放不下第二块 → 继续压 → 压歪已放好的块"，
和 §5 的零间隙槽位分析对得上。这种轨迹只能作为纠正示范的反例来源，不能当成功示范。

## 3 成功抓放的力阶梯（分阶段实测）

`replay_forces.py` 按专家阶段把力切成六段，全部是实测：
`first_contact`（指垫第一次 >0.05 N）、`closure_peak`（CLOSE+VERIFY_GRASP 段最大）、
`hold_mean`（稳定夹持段均值）、`carry_peak`（LIFT/VERIFY_LIFT/MOVE_TO_SLOT/DESCEND_TO_PLACE 段最大）、
`pre_release`（OPEN 前 10 帧）、`post_release`（OPEN 后）。

单块低力档（`--mode single --clamps 1 --noise 0.0003`，seeds 7041–7050，10 集）：

```
first_contact  0.059 / 0.189 / 0.330   N   (min / med / max, n=10)
closure_peak   0.469 / 0.544 / 0.582
hold_mean      0.468 / 0.540 / 0.578
carry_peak     0.471 / 0.545 / 0.584
pre_release    0.468 / 0.540 / 0.578
post_release   0.009 / 0.010 / 0.011
wrist_p50      10.735 N（重力基线，搬运全程没有额外分量）
一次夹走 1 块；抬升 81.4 mm；搬运距离 147.4–147.8 mm；
落点在盒内壁投影内 10/10，落定后直立 10/10；>2 N 帧数 0
独立验收 accept = true 10/10
```

捆绑档（5 块一次夹走，seeds 7031/7032 展开的分阶段读数）：
`first_contact 0.055–0.303 → closure_peak 1.354–1.398 → hold 1.339–1.377 → carry 1.360–1.398 →
pre_release 1.337–1.368 → post 0.034–0.047 N`，
5 块同时离手、5 块抬升 >20 mm、抬升 81.2–81.3 mm、搬运 147.8–199.3 mm，10 块全部直立入盒。
其余 3 条成功 episode（7033/7035/7036）的夹持期均值在 1.297–1.450 N、峰值不超过 1.538 N，
同一量级。

对照 r4 的模型侧读数（任务四，同一指垫通道）：B0 seed 22 `pad p50 38.15 / p95 87.12 /
max 865.43 N`，250 帧超过 40 N；C0 压到 `max 49.79 N` 仍有 7 帧 >40 N。
**"释放前后力应该归零"这件事专家做到了（0.01 N 级），模型一次都没做到。**

由此定的验收带宽（来自实测，不是假设）：闭合峰值 ∈ [0.3, 1.6] N，
保持均值 ∈ [0.3, 1.6] N，搬运峰值 ≤2 N，全 episode 0 帧 >2 N，释放后 ≤0.05 N。

## 4 独立验收规则

`accept` 四个条件同时成立才算一条有效抓取，写死在 `clamp_stats()` 里：

1. `max_cookies_at_grip == 1`——25 mm xy 半径内、且相对 reset 时刻抬升 >20 mm 的饼干只有一块。
   这条专门用来排除"靠挤压整堆饼干成功"：捆绑档 10 块都进了盒，但因为一次夹走 5 块，判 `false`。
2. `carried_lift_mm >= 60.0`——真的抬起来了（实测 81.4 mm），不是贴着盘面拖。
3. `carry_peak_pad_n <= 2.0`——搬运段不超过语料的绝对上界（语料 max 1.54 N）。
4. `placed_in_bin_xy and placed_upright`——落点在盒内壁 (0.057, 0.029) 半宽投影内，
   且夹取段结束时相对首次接触高度变化 |Δz| < 4 mm，即放平而不是立着倒或斜靠。

episode 级再补 `frames_over_2n == 0` 和 `moved_into_bin_tilted == 0`。
这套判据只读录制的 `cookie_xyz/eef_xyz/opening/pad_force/wrist_force`，不重仿真，
和任务四的 `autonomy_metrics.py` 一样可以事后反复改判据。

## 5 为什么一条示范只做一次夹取

同一套单块低力参数，把每集夹取次数从 1 改成 3（`--clamps 3`，seeds 7011–7020）：

```
10 集里 9 集 stop = expert_failed: released Cookies not upright, settled and contained: [1]
每集第 1 次夹取正常：closure 0.47–0.58 N，carry 0.47–0.58 N，抬升 81.4 mm
第 2、3 次夹取出现孤立尖峰：6.850 / 6.892 / 7.023 / 7.055 / 7.587 / 7.593 / 7.613 /
                             12.513 / 12.610 N（各 1 帧，>2 N 帧数 0–2）
最终 in_target_final 只有 2–3，moved_into_bin_upright 3
```

原因在场景几何里：目标盒 `target_slots_local_m` 是 2 列 × 5 行，行距 6.333 mm，
而饼干半厚 0.00317 m（全厚 6.34 mm）——**行间距与饼干厚度相等，理论间隙约 0.007 mm**。
第一块放下后，第二块要挤进同一行，指垫就顶到已放饼干上产生 7–12 N 尖峰；
专家自己的 `VERIFY_RELEASE`（要求所有已放饼干仍直立且在盒内）随即失败。

结论：示范按"一集一次完整抓取—搬运—释放"设计，不要一集抽干整盘。
另有一组"用行感知槽位、每集尽量多夹"的极限测试（`replay_single_10clamp`，seed 7001，
目标是抽干 10 块）实际只走完了 3 次夹取就 `expert_failed`（702 步）：

```
clamp 0: closure 0.584 hold 0.578 carry 0.583 post 0.014 N, 1 块, 抬升 81.4 mm, 落点 xy 1 up 1, accept 1
clamp 1: closure 0.585 hold 0.579 carry 0.647 post 0.011 N, 1 块, 抬升 81.4 mm, 落点 xy 1 up 1, accept 1
clamp 2: closure 0.590 hold 0.634 carry 9.278 post 0.007 N, 1 块, 抬升 81.4 mm, 落点 xy 1 up 1, accept 0
episode: 最终 2 块在盒内、3 块直立、>2 N 帧数 1、指垫峰值 9.278 N
```

行感知放置确实修掉了"每次都打同一行"的问题（三次落点分别在不同行，xy/upright 判定都通过），
但第三次夹取仍然顶到了已放好的饼干，产生 9.278 N 的单帧尖峰——
**每集两次夹取是可接受的上限，三次及以上会把纠正压力转成挤压力。**
所以新增示范按一集一次（最多两次）夹取设计。

另外要如实记一句：多夹次档里第 2/3 次夹取的 `carried_index` 归因不可靠
（实测 `carried_lift_mm = -0.5`、`placed_in_bin_xy = 0`，而 episode 级确有 3 块直立入盒），
因为判据按"相对 reset 时刻抬升最多的一块"来认，后续夹次里被推倒的旧饼干会干扰归因。
一集一夹的档位没有这个问题（10 seeds 全部一致），所以验收只在单夹次档上生效。

## 6 新增示范设计

### 6.1 成功示范（低力单块）

| 项 | 设定 | 依据 |
| --- | --- | --- |
| 每集内容 | 1 次抓取 + 1 次搬运 + 1 次释放（`--mode single --clamps 1`） | §5 |
| 数量 | 首批 30 条，扩到 100 条再谈重训 | 任务三 §V1 的对照需要覆盖 3 种起始列 × 5 种噪声 |
| 种子 | 7101–7130（与验收用的 7041–7050 不同段，避免用挑出来的 seed） | |
| 位置噪声 | `--noise 0.0003`（0.3 mm），另配 0.0 / 0.0006 两档各 1/3 | 已验证 0.3 mm 下 10/10 通过 |
| 夹持力 | `target_grip_force 0.45`、`stable_grip_force 0.35`（回放内部量） | 实测落在 0.47–0.58 N，位于语料 p50 之下 |
| 释放槽位 | 行感知 `_place_pose`（按列内行偏移，避免固定打在同一行） | 已在脚本内 |
| 验收 | §4 四条 + episode 级两条，逐条打印，`accept=false` 一律不入库 | |
| 落地 | 每条转成与 `a3_fw` 同构的 `action/proprio/opening/episode_*` **外加 force18** 数组 | 任务三 V1 直接可用 |

**首批 30 条已交付并全部通过验收**（目录 `/workspace/a3_eval/demos_single_v30b`，
seeds 7301–7330，noise 0.0003，30/30 `accept=true`，30 条都只含 1 次夹取）：

| 指标 | 实测范围 | 验收线 |
| --- | --- | --- |
| 指垫峰值 `pad_max_n` | 0.453 – 0.587 N | 闭合 ∈ [0.3, 1.6]，搬运 ≤ 2 N |
| `frames_over_2n` | 30 条全为 0 | 0 |
| 入盒且直立 `moved_into_bin_upright` | 每条 1 块，tilted 0 | 1 |
| `stop` | 30 条全为 `expert_done` | 非 `expert_failed` |
| 力档轮换 | `target_grip_force` 0.45 / 0.50 / 0.55 各 10 条 | §6.1 |
| 抬升/搬运 | 抬升 81.7 mm 左右，搬运距离 156–200 mm（随目标行变化） | — |

代码身份：采集脚本 `replay_forces.py` 第三版 sha256 `b7938901f0121747e952006ec25136dab119776e6b18b21f6e73dd0b62532af7`
（= 第二版 `f60a69cb…` 经 `patch_columns2.py` 把源列限制为前两列、行限制为前 10 行）。
同批 4 列版先导采集（`demos_single_v30`，seeds 7201–7230）作为对照保留，
30 条中 accept 16 条、14 条为列 2/3 的真实失败，
**只作场景来源，不进成功集**。

### 6.2 纠正示范

覆盖任务要求的"高力下压或错误闭合之前的纠正状态"。都是**新写的控制器状态，
不改模型、不改 Wrapper、不改 Safety、不改仿真执行逻辑**，只在回放脚本内加阶段：

| 编号 | 场景 | 触发 | 纠正动作 | 判定成功 |
| --- | --- | --- | --- | --- |
| C-1 | 下压过深 | `DESCEND` 段 `max(pad) > 1.6 N`（语料绝对上界） | 停止下降、抬回 `z0+0.02`、重新对准再下压 | 纠正后 `carry_peak ≤ 2 N` 且正常释放 |
| C-2 | 错误闭合（夹到 2 块） | `CLOSE` 段 `attached` 计数 ≥2 | 张回到 `release_opening`、侧移 8 mm、重新对准单块 | 第二次闭合 `max_cookies_at_grip == 1` |
| C-3 | 空夹（未夹住） | `VERIFY_GRASP` 段 `max(pad) < 0.05 N` 且张开度未下降 | 抬升回 `APPROACH`，重新下压 | 复夹后 hold ∈ [0.3, 1.6] N |
| C-4 | 搬运途中力抬头 | `LIFT/MOVE` 段 `max(pad) > 2 N` | 减速—保持—重新对准槽位 | 释放后 `placed_upright` |
| C-5 | 释放高度不足 | `DESCEND_TO_PLACE` 末段 `eef_z` 高于安全释放高度 | 不上抬、在原高度直接张开 | `post_release ≤ 0.05 N` 且不撞已放饼干 |

每条纠正示范的验收要多一条：**纠正前必须真的观测到过异常**（例如 C-1 的 >1.6 N 帧存在），
否则就是把普通示范贴了个"纠正"标签。纠正前的异常段保留在数据里，不裁掉——
模型要学的正是"看到力抬头该怎么办"，把异常段删掉就退化成普通成功示范。

**恢复段也要验收（2026-10-31 由实测加上的判据）**。§4 的逐夹判据只看抓取段，
一次纠正后重新抓放的那一段力是否还在语料包络内，它管不着。所以对每条纠正示范，
按 `retry_log[*].at_step` 把整条轨迹切成 `异常半段` / `恢复半段`，两条都要求：

| 段 | 条件 | 含义 |
| --- | --- | --- |
| 异常半段 | `max(pad) > 1.6 N` 且 `>1.6 N 的帧数 ≥ 3` | 场景真的把异常暴露给了控制器（不是贴标签） |
| 恢复半段 | `max(pad) ≤ 1.6 N` 且 `>2 N 的帧数 == 0` | 纠正之后回到语料包络（语料 42681 帧绝对上界 1.540 N） |
| 恢复段夹取 | §4 逐夹判据 accept，且 `max_cookies_at_grip == 1`、`placed_upright` | 纠正真的把任务做完了 |

判据由只读工具 `adjudicate_correct.py`（pod：`/workspace/_r5_stage/adjudicate_correct.py`，
sha256 `bfd7c54cdf703793be7eb3c7c2e17d32da14412b1aea57c98c9c9b42dd6c30f5`，本地同名文件与它逐字节一致）
计算并打印每段的 `peak / over_1p6 / over_2p0`，以及恢复段里每个 >2 N 尖峰的**帧号、距切点的帧数、
周围 9 帧的 pad/阶段/张开度**——这样"边界伪影"和"恢复途中真的又顶了一下"可以分开判，不用肉眼猜。

### 6.3 采集流程与磁盘

录制前查空间（本轮实测 `/workspace` 5 GB 可用、已用 96 %）；一条单块示范 npz 约 120 KB
（`expert_single_*.npz` 10 条 + `summary.json` ≈ 1.3 MB），30 条 + 纠正示范 20 条
预计 < 10 MB，不构成压力。输出目录一律新建，不写回
`replay_single_gate10`、`r4_c0_frozen`、`r4_b0_c0_paired`。
`replay_forces.py` 自己就会拒绝非空输出目录和 <1 GB 的剩余空间。

### 6.4 变体采集的实测可达性（`replay_forces.py` 第二版 f60a69cb + 第三版列限制）

`--seeds` 的顺序号同时是变体号，决定"取源盘哪一块 + 用哪一档夹持力"。
第一批 30 个变体（`demos_single_v30`，noise 0.0003）跑出来的可达性很干净：

| 变体 | 源列 | 行 | 结果 |
| --- | --- | --- | --- |
| 0,4,8,12（列 0） | 0 | 0,1,2,3 | accept，closure 0.577–0.583 N，抬升 81.4 mm |
| 1,5,9,13（列 1） | 1 | 0,1,2,3 | accept，closure 0.467–0.577 N，抬升 81.6 mm |
| 2,6,10,14（列 2） | 2 | 0,1,2,3 | **全部 `expert_failed: batch phase…` / `insertion b…`** |
| 3,7,11（列 3） | 3 | 0,1,2 | **全部失败**，其中 seed 7204/7208 出现 353/354 帧 >2 N |

也就是说脚本专家的单块抓取只在源盘前两列可达（与 `_place_pose` 默认
`demonstration_column_order=(0,1)` 一致），后两列会卡在插入阶段；
失败样本不是"轻微没夹好"，而是**长时间顶撞**（353 帧超过语料绝对上界 2 N）。
这两个特征直接决定了示范采集的两件事：

1. 成功示范只在列 0/1 的 10 行里取样，变体号改成 `列 = 变体 % 2`、`行 = (变体 // 2) % 10`，
   共 20 个可达目标，夹持力三档轮换（`target_grip_force` 0.45 / 0.50 / 0.55），
   凑够 30 条时允许同一目标在不同力档下重复。
2. 列 2/3 的失败模式正是任务要求收录的"高力下压前的纠正状态"，
   纠正示范 C-1（下压过深→回退）就在这些目标上采，而不是人造异常。

失败样本一律不进成功集（§4 判据自动拒），只作为纠正示范的场景来源。

### 6.5 列 2/3 的 14 条真实失败按异常族分类（`demos_single_v30` 的 14 条 reject）

逐条实测（变体号 = `--seeds` 顺序号，`padmax` 为该集指垫峰值，`over2n` 为 >2 N 帧数）：

| 变体 | seed | padmax (N) | over2n | 失败原因（脚本专家自己打印的） | 属于哪一类纠正场景 |
| --- | --- | --- | --- | --- | --- |
| 3 | 7204 | 5.024 | 353 | `batch phase timeout; contacts=[0.0, 0.0]` | C-1 持续下压 |
| 7 | 7208 | 5.786 | 354 | 同上 | C-1 持续下压 |
| 27 | 7228 | 3.680 | 349 | 同上 | C-1 持续下压 |
| 14 | 7215 | 23.627 | 16 | `insertion blocked; pad forces=…` | C-1 高尖峰（卡顿） |
| 18 | 7219 | 22.735 | 14 | 同上 | C-1 高尖峰 |
| 22 | 7223 | 35.714 | 11 | 同上 | C-1 高尖峰 |
| 6 | 7207 | 3.373 | 3 | `batch phase timeout` | 边界（>1.6 N 但帧数少） |
| 10 | 7211 | 5.329 | 3 | 同上 | 边界 |
| 26 | 7227 | 5.021 | 4 | 同上 | 边界 |
| 2 | 7203 | 0.050 | 0 | 同上 | C-3 空夹（根本没接触） |
| 11 | 7212 | 0.718 | 0 | 同上 | C-3 未持住 |
| 23 | 7224 | 0.718 | 0 | 同上 | C-3 未持住 |
| 15 | 7216 | 0.000 | 0 | 同上 | C-3 空夹 |
| 19 | 7220 | 0.000 | 0 | 同上 | C-3 空夹 |

三条异常族都在真实场景里出现，不需要人造注入：
持续下压（数百帧 3.7–5.8 N）、卡顿尖峰（11–16 帧 22–36 N，已超过 r4 裸模型 C0 的 49.79 N 量级的一半）、
以及空夹/未持住（力始终 ≤0.72 N）。`contacts=[0.0, 0.0]` 是脚本专家自己报的接触计数，
不是 Safety 层，也不是特权观测（只进诊断日志，不进模型输入）。

纠正示范驱动 `replay_correct.py`（pod：`/workspace/_r5_stage/replay_correct.py`，
sha256 `754102b66ba467534d6d92fb064071de2b278648aef07da0b55778242a27314c`，
基于策略脚本 `b7938901…`；pod 上的这份是权威版本，本地同名文件是草稿）：
目标是异常族，脚本专家报 `expert.failed` 后在同一 MuJoCo 场景里重新 `reset()`
并把变体号换成列 0/1 的已知可达目标重规划（`--recover-variants` 给重规划链，
`0` = 列 0 行 0，`37` = 列 1 行 9，`5` = 列 1 行 1；`--retries` 给重规划预算）。
验收要求**纠正前观测到异常**并按异常来源打标 `anomaly_kind`
（`excessive_pad_force` = 峰值 > 1.6 N 且 ≥3 帧；`empty_grasp` = 峰值 < 0.3 N；
`stuck_no_force` = 两者之间），**且最后一个夹取段通过 §4**。

踩过的坑（已修）：驱动里 `expert.finished` 判在 `expert.failed` 之前，
而控制器把 `finished` 定义成属性 `phase in (DONE, FAILED)`
（`controllers/cookie_expert.py:458-460`，`_fail()` 在 `:731` 把阶段推进到 `CookiePhase.FAILED`），
所以失败时 `finished` 也是 True。结果第一版把真实失败报成了 `expert_done`，
`retry_log` 空、`anomaly_observed=false`；改成先判 `failed`（与
`replay_forces.run_episode` 的 `elif` 顺序一致）后才对得上。
顺带确认：`reset()` 会把阶段拉回 `SELECT_COOKIE`（属 `IDLE_PHASES`），
所以异常段和纠正后的成功段在 `segments()` 里天然分开，
验收只作用于 `clamps[-1]`（即纠正后那一次夹取）不会把 5 N 的异常峰算进成功段。

### 6.6 这批示范目前缺什么才能直接进训练

回放脚本录的是 `observation.state / eef_pose / force` 加特权几何（饼干 xyz、夹爪开合、
阶段标签），**没有相机帧**：`replay_forces.py` 里出现过的观测键只有
`observation.eef_pose`、`observation.force`、`observation.state` 三个，而
`make_env(..., render_cameras=False)`。仿真本身是出图的——`run_matrix.py:239` 和
`dagger_recover.py:29-30` 用的键是 `observation.images.front / left_wrist / right_wrist`。

所以这批数据现在的用途是：**力分布的测量、验收判据的标定、纠正路径的可行性验证**，
不能直接当 BC 语料。要进训练需要一次带图重采（同一 seed、同一 noise、同一变体号，
确定性回放已验证可复现同一条轨迹，见 §2）。

带图的体量**按现有训练集实测的每帧字节 × 语料里每条的真实帧数**算，不按 JPEG 估算
（这条之前估小了一个量级，也之前把单条长度猜成了 450–700 帧）：
`/workspace/a3_data/a3_fw/summary.json` 记的是 `num_episodes=100`、`num_frames=42681`、`fps=20`，
同目录 `frames.dat` 是 12 GB 的**非压缩帧流**（`A3VideoDataset` 直接读这个流，不是读 JPEG）。
每帧字节用**实测反推**而不是除法近似：

```
frames.dat  = 12,849,371,136 B
num_frames  = 42,681
12,849,371,136 / 42,681 = 301,056 B/帧   （整除，无余数）
```

也就是 **301,056 B/帧 = 294.0 KB/帧**；我之前写的"≈281 KB/帧"是把 12 GB 当成
十进制 12,000,000,000 去除得到的，偏小 7 %，本节的体积一律按 301,056 B/帧重算。
下面每条的帧数取自
`corpus_manifest_v1.json` 里逐条的 `steps`（§6.11），不是区间猜测：

| 语料 | 条数（严格合格） | 实测帧数 | 单条均帧 | 带图体积（301,056 B/帧） |
| --- | --- | --- | --- | --- |
| 成功示范（`demos_single_v30b`） | 30 | 6824 | 227.5 | 2,054,406,144 B = **2.05 GB** |
| C-1 下压过深纠正（c1..c5） | 17 | 11387 | 669.8 | 3,428,124,672 B = **3.43 GB** |
| C-2 错误闭合纠正（v9 6 + v8 2） | 8 | 2542 | 317.8 | 765,284,352 B = **0.77 GB** |
| C-3 空夹/未夹住纠正（c1..c7） | 18 | 12065 | 670.3 | 3,632,240,640 B = **3.63 GB** |
| 合计 | **73** | **32818** | 449.6 | 9,880,055,808 B = **9.88 GB**（9.20 GiB） |

单条平均 135.34 MB。四项帧数相加 6824+11387+2542+12065 = 32,818，与合计一致。

单条帧数范围 215–672：纠正族一定要长，因为一条里同时含"异常那段 + 重规划那段"，
切点在 450 帧前后，后半段才是纠正动作；单块成功示范只有搬运一夹，220–230 帧。
（`replay_single_gate10` 那 10 条是确定性回放的校验门，不与 v30b 重复计入语料。）

而 2026-10-31 21:36 实测 `/workspace` 只剩 **3.41 GB**，22:05 c7 收完再量是
**3.407 GB**（105 GB 用了 97%），
`a3_train_runs` 里 `a3_lora_r3` 占 21 G、`a3_lora_r1` / `a3_lora_r2` 各 11 G（大头是
`step_*_merged.pt` 合并权重，每轮真正要留的轻量 LoRA 态只有 1.4 G）。

所以带图重采**在当前格式下放不进去**，两条路都留给任务五一起定：
1. 归档或删除 r1/r2/r3 的 merged.pt 腾出 40 GB 级空间——这是删共享数据的动作，
   **必须用户批准，我不会自行执行**，执行前会先逐文件列清单（大小 + mtime + 是否还有脚本引用）。
   清单已经做完，结论是**这份"40 GB"靠不住**：r1 与 r3 那两份分别被环境默认指针和
   r4/r5 的评估基座在用，能动的只有 r2 那一份 11.12 GB。见 `docs_task5_decision.md` §5.2。
2. 带图落盘改成压缩存法——但 `A3VideoDataset` 读的是非压缩 `frames.dat`，
   加压缩等于改输入管线/数据加载器，和本轮"不改输入结构"的约束冲突，
   只能和任务三的 V1 接线放在一起做。

这一步之前不动任何文件、不删任何目录。

**2026-11-01 追加（任务四的第 4 小步）**：全量 9.88 GB 放不下，但
"先做 2–3 条带图转换 + 训练加载 smoke"这一步放得下，且不需要任何删除动作：
3 条 × 450 帧 × 301,056 B = 406,425,600 B ≈ 0.41 GB，剩 3.0 GB 缓冲；
最坏情况取纠正族最长的 672 帧 × 3 条 = 606,931,968 B ≈ 0.61 GB，仍然放得下。
所以 smoke 的边界是**先转换、再让 `A3VideoDataset` 真读一次**，读完确认字节数与
帧数对得上再决定是否谈全量。

### 6.7 C-1 纠正示范批次的严格裁定结果（`demos_correct_c1`）

批次命令（pod，tag `cc1`，2026-10-31 19:47 起）：驱动 sha `754102b6…`，
`--seeds 7204 7208 7228 7207 7211 7227 7215 7219 --variants 3 7 27 6 10 26 14 18
--noise 0.0003 --retries 2 --recover-variants 0 37 5 --max-steps 2400`。
目标全是列 2/3 的真实不可达单块，异常由场景自己产生；恢复目标固定用列 0 行 0
（变体 0），预算不够再退到列 1 行 9（变体 37）、列 1 行 1（变体 5）。

逐条按 §6.2 的严格判据裁定（`adjudicate_correct.py` 打印的原始分段统计）：

| seed | 变体 | 异常半段 | 恢复半段 peak / >2 N 帧 | 尖峰位置 | 逐夹 §4 | 严格判定 |
| --- | --- | --- | --- | --- | --- | --- |
| 7204 | 3（列 3） | 451 帧，peak 5.024，354 帧 >1.6 N | 218 帧，**0.581** / 0 | 无 | closure 0.581 / carry 0.580 / post_release 0.011，抬升 81.4 mm，正放 | **accept** |
| 7208 | 7（列 3） | 451 帧，peak 5.786，354 帧 >1.6 N | 218 帧，**0.588** / 0 | 无 | closure 0.587 / carry 0.588 / 0.011，抬升 81.4 mm，正放 | **accept** |
| 7228 | 27（列 3） | 451 帧，peak 3.680，349 帧 >1.6 N | 217 帧，**4.344** / 1 | 第 496 帧（切点后 45） | accept | **reject** |
| 7207 | 6（列 2） | 451 帧，peak 3.373，仅 3 帧 >1.6 N | 220 帧，**2.331** / 1 | 第 453 帧（切点后 2） | closure 0.491 / carry 0.490 accept | **reject** |
| 7211 | 10（列 2） | 451 帧，peak 5.329，3 帧 >1.6 N | 218 帧，**7.543** / 1 | 第 549 帧（切点后 49） | accept | **reject** |
| 7227 | 26（列 2） | 451 帧，peak 5.021，4 帧 >1.6 N | **17.045** / 1 | 第 494 帧（切点后 43） | closure 0.590 / carry 0.589 / 0.011 accept | **reject** |
| 7215 | 14（列 2，卡顿族） | 77 帧，peak 23.627，16 帧 >2 N；重规划后 51 帧再卡 **56.936** | 294 帧，**9.668** / 2 | 第 128 帧（切点后 0，即 FAILED 帧本身，属边界伪影）+ 第 176 帧（+48，真实） | accept | **reject** |
| 7219 | 18（列 2，卡顿族） | 76 帧，peak 22.735，14 帧 >2 N | 217 帧，**16.818** / 2 | 第 120/121 帧（+44/+45） | closure 0.579 / carry 0.607 accept | **reject** |

批次结论：**loose 8/8，strict 2/8，被恢复段拒掉 6 条**（`adjudicate_correct.py` 的
`tally` 原话：`{"dir": ".../demos_correct_c1", "n": 8, "loose": 8, "strict": 2,
"rejected_by_recovery": 6}`）。

两个 reject 的尖峰都做了归因，不是切点边界的伪影：

- **7228**：切点 451，尖峰在第 496 帧（切点后 45 帧）。第 446–450 帧的 pad 已经是
  `3.313 ×5`，第 451 帧起连续 11 帧 `0.0`，也就是下压在失败点前一帧就已经完全卸力；
  496 帧的 4.344 N 处在恢复段自己的 `DESCEND` 里，张开度 0.031–0.033（近乎闭合），
  是重新下压时又顶到东西。**是真实异常，必须拒**。
- **7207**：切点 451，尖峰在第 453 帧（切点后 2 帧），值 2.331，随后 454 帧 1.909，再归零。
  阶段序列是 `DESCEND(446–450) → FAILED(452) → APPROACH(453+)`，张开度从 454 帧起
  0.122 → 0.61 递增。也就是异常半段的力在切点前 2 帧就归零了，这一对 2.3/1.9 N
  是重规划后**闭合夹爪在 APPROACH 途中拖过饼干堆**碰出来的，同样算真实异常。

顺带得到的机制性结论（决定后面还能不能扩 C-1 批量）：

1. **`reset()` 只是控制器复位，不抬臂。** 卡顿族（7215/7219/7223，peak 22–36 N）失败时
   臂还深埋在源盘里，`expert.reset()` 后立刻重新下压会再次顶撞——7215 第一次卡顿
   23.627 N，重规划到变体 0 后 56.936 N 再卡，`stop=correction_budget`。
   所以卡顿族要变成合格纠正示范，必须在重规划前插入一段"抬回 `z0+0.02`"的动作，
   而这需要新的控制器阶段（本轮不改模型/Wrapper/Safety/仿真执行逻辑，
   回放脚本内加阶段是允许的，但它是下一步工作，不是现在能白拿的）。
2. **持续下压族（数百帧 3.7–5.8 N）是能用的**：它失败时力已经卸到 0，臂在源盘上方，
   重规划到列 0 行 0 之后能干净地抓—搬—放，7204/7208 就是这样拿到 accept 的。
3. 严格判据把 loose→strict 的差别缩小到"恢复段自己有没有再超 1.6/2 N"，
   这条比 §4 的逐夹判据更严，因为它看的是整段恢复轨迹而不是抓取窗口。
4. **被拒的 6 条有一个共同形状**：尖峰都落在恢复段自己的 `DESCEND` 里，
   距切点 **+43 ~ +49 帧**（7207 的 +2 是唯一的例外，它是重规划后闭爪拖过饼干堆），
   值是 1–2 帧的 4.3/7.5/17.0/16.8/9.7 N 冲击，前后帧都是 `0.0`，
   张开度 0.031–0.044（近乎闭合）。也就是"重新下压时爪尖撞到了东西"，
   属于瞬态撞击而不是持续挤压。它依然是不该教给模型的状态（语料 42681 帧的绝对上界是
   1.540 N），所以按同一判据拒；但它说明 C-1 的产出率由这一步撞击决定，
   **不是靠改判据能提上去的**，只能多采。
   据此已经追加一批纯持续下压族（tag `cc2`，`--seeds 7241–7247 --variants 3 7 11 15 19 23 27`，
   全是列 3 的不同行，其余参数与 cc1 相同，跑完立刻用同一工具裁定），
   目标是把严格合格的 C-1 纠正示范做到 8–10 条。

**`cc2` 批次结果**（2026-10-31 20:15–20:35，输出 `demos_correct_c2`，驱动 sha `754102b6…`）。
裁定原话：`{"dir": "/workspace/a3_eval/demos_correct_c2", "n": 7, "loose": 4, "strict": 3,
"rejected_by_recovery": 1}`，逐条 `strict_accept` 计数 3 true / 4 false。

| seed | 变体 | 异常半段（0–451） | 恢复半段 peak / >2 N | 严格判定 |
| --- | --- | --- | --- | --- |
| 7241 | 3（列 3） | peak 5.965，355 帧 >1.6 N，354 帧 >2 N | **0.581** / 0，无尖峰 | **accept** |
| 7242 | 7（列 3） | peak 6.464，20 帧 >1.6 N，17 帧 >2 N | **0.514** / 0，无尖峰 | **accept** |
| 7243 | 11（列 3） | peak 4.045，`stuck_no_force` 族，2 帧 >2 N | **0.563** / 0，无尖峰 | **accept** |
| 7244 | 15 | 异常半段 peak **0.0**，分类 `empty_grasp` | 0.553 / 0 | reject（异常根本没发生） |
| 7245 | 19 | 未触发异常，`stop=expert_done`，peak 0.506 | — | reject（同上） |
| 7246 | 23 | 未触发异常，`stop=expert_done`，peak 0.676 | — | reject（同上） |
| 7247 | 27 | peak 5.456，350 帧 >2 N | 有恢复段尖峰 | reject（loose 4 条里被恢复段拒掉的那 1 条） |

两个可直接用的结论：

1. **异常能不能发生由变体决定，不由 seed 决定。** 列 3 的 15/19/23 这三行在 seed 7244–7246
   下老老实实一次夹一块、0.5–0.68 N 干净放盒，压根没产生异常——这三条本质上是又三条
   成功示范（`demos_single_v30b` 已经有 30 条），对 C-1 是空耗。
   3/7/11/27 四个变体才是稳定产出异常的。
2. 于是 C-1 的实际产出率是 **严格 3/7 ≈ 43%**，其中 3 条空耗、1 条被恢复段撞击拒掉。
   已按这个结论启动第三批（tag `cc3`，`--seeds 7248–7253 --variants 3 7 11 27 3 7`，
   只在会出异常的变体上换 seed），输出 `demos_correct_c3`，跑完自动接 `adjudicate_correct.py`。
   C-1 累计严格合格：`demos_correct_c1` 2 条 + `demos_correct_c2` 3 条 = **5 条**，
   `cc3` 若按 43% 命中可再到 2–3 条。

**`cc3` 批次结果**（2026-10-31 20:35–21:00 前后，输出 `demos_correct_c3`）。
裁定原话：`{"dir": "/workspace/a3_eval/demos_correct_c3", "n": 6, "loose": 6, "strict": 6,
"rejected_by_recovery": 0}` ——**6/6 全部严格合格**，43% 那个预期被推翻，
原因是这一批完全放弃了 15/19/23 三个不产生异常的变体（上一条结论 1 的直接兑现）：

| seed | 变体 | 切点 | 异常半段 peak / >2 N | 恢复半段 peak | 严格判定 |
| --- | --- | --- | --- | --- | --- |
| 7248 | 3 | 451 | 5.989 / 354 | 0.471 / 0 帧 | accept |
| 7249 | 7 | 451 | 5.492 / 39 | 0.580 / 0 | accept |
| 7250 | 11 | 451 | 2.276 / 2（`stuck_no_force` 族） | 0.581 / 0 | accept |
| 7251 | 27 | 450 | 3.887 / 349 | 0.583 / 0 | accept |
| 7252 | 3 | 450 | 6.047 / 354 | 0.497 / 0 | accept |
| 7253 | 7 | 450 | 2.172 / 2 | 0.584 / 0 | accept |

恢复段全部是 0.47–0.58 N 的单块抓取、`>1.6 N` 与 `>2 N` 都是 0 帧、没有恢复段撞击；
6 条的异常族都在切点前（`excessive_pad_force` 4 条 + `stuck_no_force` 2 条），
纠正之后峰降一个数量级。**C-1 累计严格合格 = 2 + 3 + 6 = 11 条**
（`demos_correct_c1` 8 采 2 合、`c2` 7 采 3 合、`c3` 6 采 6 合，总 21 采 11 合）。
按 `cc3` 这条"只在会出异常的变体上换 seed"的配方，C-1 的产出率现在是 100%，
剩下的量只受算力与磁盘约束。

C-1 的三个目录也已和 C-2 一样落成自描述文件（`adjudicate_correct.py` 加了 `--write`，
新版 sha `50d86e69…`、`py_compile` 通过、只新增文件、已存在则 `REFUSE overwrite`）：

| 目录 | 文件 | 字节 | sha 前缀 | 内容 |
| --- | --- | --- | --- | --- |
| `demos_correct_c1` | `adjudication_correct.json` | 14096 | `90390fa2019a24c2` | n8 loose8 **strict2** rejected_by_recovery6 |
| `demos_correct_c2` | `adjudication_correct.json` | 7446 | `6e649e883133af79` | n7 loose4 **strict3** rejected_by_recovery1 |
| `demos_correct_c3` | `adjudication_correct.json` | 5845 | `558679a8741485cf` | n6 loose6 **strict6** rejected_by_recovery0 |

加它的理由和 C-2 一样：`demos_correct_c1` 的 `summary.json` 里 8 条 `accept` 全是 `true`
（逐夹松口径），只读 summary 的下游会把 6 条恢复段还在超时的也当合格样本收进训练集。

**`cc4` / `cc5` 两批**（2026-10-31 21:19–21:35，输出 `demos_correct_c4` / `demos_correct_c5`，
各 6 枚新 seed（7254–7265），变体只用会出异常的 3/7/11/27，驱动仍是 `754102b6…`）：

| 目录 | seed | 变体 | 驱动标的异常族 | 异常半段 peak / >1.6 N 帧 | 恢复半段 peak | 判定 |
| --- | --- | --- | --- | --- | --- | --- |
| c4 | 7254 | 3 | excessive_pad_force | 7.286 / 355 | 0.583 | C-1 accept |
| c4 | 7255 | 7 | excessive_pad_force | 6.588 / 354 | 0.517 | C-1 accept |
| c4 | 7256 | 11 | stuck_no_force | 1.078 / 0 | 0.584 | C-1 reject（没有高力段）→ C-3 |
| c4 | 7257 | 27 | excessive_pad_force | 3.809 / 350 | **5.250** | reject（恢复段撞击） |
| c4 | 7258 | 3 | excessive_pad_force | 6.088 / 354 | 0.580 | C-1 accept |
| c4 | 7259 | 7 | excessive_pad_force | 5.599 / 40 | 0.539 | C-1 accept |
| c5 | 7260 | 11 | stuck_no_force | 1.694 / 1 | 0.525 | C-1 reject → C-3 |
| c5 | 7261 | 27 | excessive_pad_force | 3.970 / 350 | 0.580 | C-1 accept |
| c5 | 7262 | 3 | excessive_pad_force | 6.098 / 354 | 0.583 | C-1 accept |
| c5 | 7263 | 7 | excessive_pad_force | 5.404 / 353 | 0.590 | C-1 accept |
| c5 | 7264 | 11 | stuck_no_force | 0.786 / 0 | 0.514 | C-1 reject → C-3 |
| c5 | 7265 | 27 | excessive_pad_force | 3.834 / 350 | 0.582 | C-1 accept |

stdout 里的原始 tally 是 `c4 {n6 loose5 strict4 rejected_by_recovery1}`、
`c5 {n6 loose5 strict5 rejected_by_recovery0}`。这两批的自动裁定我启动时漏写了 `--dirs`
（那一步直接报错退出，录制本身不受影响），补跑了手工
`adjudicate_correct.py --dirs … --write`，落成
`demos_correct_c4/adjudication_correct.json`（6589 字节，`9bb208a535c372c2`）和
`demos_correct_c5/adjudication_correct.json`（5845 字节，`49eac43b24298105`）。

**复核时发现裁定工具的异常半段判据写错了，已改并重跑五批。** 原来那一行是
`over_1p6 >= ANOMALY_FRAMES or peak > ANOMALY_PAD_N`——用了 `or`，
"至少 3 帧"这个条件永远不会起作用，半段里只有 1–2 帧 >1.6 N 也算"异常真的发生了"。
§6.2 定的判据是"peak > 1.6 N **且** >1.6 N 帧数 ≥ 3"。改成合取之后
（pod 权威版 sha `d88d3607dbb11e337fae7e45bf4719796fc8f0dc92f271c121d3e4fc95cc9331`、
6749 字节、`py_compile` 通过；顺带加 `--suffix`，让新结果写进
`adjudication_correct_v2.json`，**已经写出的 JSON 一个都没被覆盖**），五批的 strict 是
`c1 2 / c2 2 / c3 5 / c4 4 / c5 4 = 17`。

| 目录 | `adjudication_correct_v2.json` 字节 / sha 前缀 | n | loose | strict（修正后判据） |
| --- | --- | --- | --- | --- |
| c1 | 14140 / `80def0ca840b460b` | 8 | 8 | 2 |
| c2 | 7492 / `b78e44e39b7965e8` | 7 | 4 | 2 |
| c3 | 5891 / `22679fbfecdc7fa2` | 6 | 6 | 5 |
| c4 | 6633 / `86e805752a1d74e1` | 6 | 5 | 4 |
| c5 | 5891 / `53507bd474191e65` | 6 | 5 | 4 |

从 C-1 掉出来的正好 3 条：7243（peak 4.045 但只 2 帧 >1.6 N）、7250（2.276 / 2 帧）、
7260（1.694 / 1 帧）。它们都是驱动自己标成 `stuck_no_force` 的那一族——
"闭了爪、压到了、但一块都没带起来"，本来就不是"下压过深"，而是 §6.2 的 **C-3**（见 §6.9）。
**分家用的键就是驱动写进 `retry_log[0].anomaly_kind` 的那个标签**，
所以 C-1 与 C-3 两份名单的种子号不重叠，不存在一条示范被两个族重复计数。

**C-1 累计 17 条严格合格**：33 条采集里 `excessive_pad_force` 族 25 条，严格合格 17 条（68%），
按批是 c1 2/8、c2 2/3、c3 5/5、c4 4/5、c5 4/4。cc3 之后这条配方（只用 3/7/11/27
四个会出异常的变体、只换 seed）命中率稳定在八成以上，
剩下的拒绝全部来自同一件事——重新下压时那 1–2 帧撞击，改判据改不掉，只能多采。

### 6.8 C-2 错误闭合纠正示范（`demos_wrongclosure_v6`，2026-10-31 20:23–20:33）

**为什么必须从捆绑档来。** 单块档 70 条里 `max_cookies_at_grip` 只有 0 和 1，一次夹到
≥2 块这个状态在单块控制器下**根本不会出现**；而捆绑档 6 条实测每次夹取抬走的都是
源位 0–4（列 0 前 5 行），`max_cookies_at_grip=5`，闭合峰 1.32–1.54 N——力是低的，
错的是"一次夹了 5 块"。所以 C-2 的异常态只能借捆绑档的场景产生，再在携带阶段切回
单块低力控制器做纠正。

驱动：`/workspace/_r5_stage/replay_wrongclosure.py`，sha `6e841f5b…`，
被回放的策略 `replay_forces.py` sha `b7938901…`（未改动）。触发条件是
`phase ∈ CARRY_PHASES 且 cookies_at_grip ≥ 2`，命中后再走 10 帧把"挂着 5 块"的状态录进轨迹，
然后 `R.SingleCookieLowForce(env)` + `variant_index=1` + `reset()` 重新规划。
**恢复目标刻意选列 1**：被夹走又放回的 5 块就在列 0，若仍按默认变体 0 去抓，纠正段会去抓
一个已经被搬空的位置——第一批（tag `wc1`/冒烟）就是这么在 7032 上跑到 `max_steps` 的。

批次命令（pod，tag `wc3`）：
`--seeds 7031 7032 7033 7034 7035 7036 --noise 0.0003 --recover-variants 1 --max-steps 800`。

7032 的逐帧时间线（`inspect_wc.py` 原话，阶段相同的连续帧合并打印）：

```
 50-  80 CLOSE            grip=0 padmax=1.340 open=0.310..0.448
 93- 103 LIFT             grip=5 padmax=1.352 open=0.309..0.309
104- 104 APPROACH         grip=5 padmax=0.000 open=0.370..0.370   ← 触发纠正，控制器复位
105- 115 APPROACH         grip=0 padmax=0.000 open=0.456..1.012   ← 开度拉到全开，5 块当场放下
162- 169 CLOSE            grip=0 padmax=0.576 open=0.011..0.033   ← 重新单块闭合
182- 203 LIFT             grip=1 padmax=0.578
322- 322 DONE             grip=0
moved 1 inbin 1 both 1 / moved_idx [20]
```

也就是说这条轨迹同时含"错着夹住 5 块（1.35 N）→ 松手放回 → 重新单块抓（0.576 N）→
抬升 81.6 mm → 放盒"，正是要教给模型的纠正结构；放回的 5 块终态离名义位不足 5 cm，
没有污染入盒计数（`moved_idx` 只有 20）。

**驱动自带的 `accept` 字段全部为 False，是判据的缺陷不是数据的缺陷。**
`segments()` 只在相位进入 IDLE（`SUPPORT_BOX/SELECT_COOKIE/DONE`）时才收尾，
控制器复位后相位直接从 `LIFT` 跳回 `APPROACH`、中间没有 IDLE，于是整条轨迹被并成
**一个** clamp，而这个合并段里 `max_cookies_at_grip=5`，`== 1` 永不成立。
改法是只读复裁判器 `adjudicate_wc.py`（`/workspace/_r5_stage/`）：在 `cut`（异常发生帧）
之后切片再 `segments` + `clamp_stats`——切片以 `APPROACH` 开、以 `DONE` 收，段边界就对了。
（顺带确认 `clamp_stats` 的第 9 个实参 `carried_rest_z` 在函数体内没有任何引用，传 None 即可。）

| seed | 异常半段（cut 前） | 纠正后峰 / >2 N | 恢复 clamp（cut 后重算） | 严格判定 |
| --- | --- | --- | --- | --- |
| 7031 | 105 帧，grip=5，peak 1.393 N | — | 无：481 帧 `contacts=[0.0, 0.0]`，从未接触目标块，phase timeout | **reject** |
| 7032 | 104 帧，grip=5，peak 1.378 N | 0.578 / 0 | closure 0.576 / carry 0.578，grip=1，抬升 81.6 mm，入盒正放，carried_index 20 | **accept** |
| 7033 | 105 帧，grip=5，peak 1.393 N | 0.563 / 0 | closure 0.559 / carry 0.563，grip=1，81.6 mm，入盒正放 | **accept** |
| 7034 | 105 帧，grip=5，peak 1.378 N | 纠正段自身闭合峰 **2.993 N** / 1 | 逐夹 accept（carry 0.51，81.6 mm 入盒），但闭合瞬间超了 2 N | **reject** |
| 7035 | 104 帧，grip=5，peak 1.377 N | 0.535 / 0 | closure 0.534 / carry 0.535，grip=1，81.6 mm，入盒正放 | **accept** |
| 7036 | 105 帧，grip=5，peak 1.378 N | 0.579 / 0 | closure 0.571 / carry 0.579，grip=1，81.6 mm，入盒正放 | **accept** |

原话 tally：`{"n": 6, "strict": 4, "abnormal": 6, "recovery": 5, "grasp": 5}`
——异常态 6/6 都真实录到了，5 条纠正段力干净，5 条恢复抓取有效，交起来 4 条严格合格。

已按同一参数追加 tag `wc4`（`--seeds 7037–7042 --recover-variants 2`，输出
`demos_wrongclosure_v6b`，跑完自动接 `adjudicate_wc.py`）：这一批换到列 2 抓，
因为 `wc3` 4 条合格样本的 `carried_index` 全是 20（同一块饼干），只靠 seed 抖动提供变化，
抓取目标这个维度太窄。

**`wc4` 第一条（seed 7037）就暴露了采集器的一个真实缺陷，已修。** 原版驱动只在异常命中后
继续走 10 帧把"挂着 5 块"的状态录进轨迹，然后直接 `SingleCookieLowForce(env)` + `reset()`，
**指望复位后的头几条指令自己把夹爪张开**——`wc3` 里 7032/7033/7035/7036 确实是这样自然松手的
（开度 0.37 → 1.012）。但 7037 复位后 5 块一直挂在闭合的爪里，控制器按自己的计划朝列 2 下压，
于是 `recovery_half` 出现 **398 帧 >2 N、峰值 9.898 N** 的顶压，最后
`batch phase timeout; contacts=[0.114, 0.0]` 超时失败（`stop=bundle_failed_before_correction`）。
也就是说"松手"这一步在原版里是**碰运气**，不是被保证的纠正动作。

修法只动回放驱动这一个动作流（模型/Wrapper/Safety/仿真执行逻辑一行没改，
环境照旧执行驱动递给它的那个动作）：异常命中后的 `OBSERVE_FRAMES` 观察窗里，
把夹爪开度命令钉成 `RELEASE_OPEN = 1.0`，让挂着的批次**必须**离开夹爪，再交给
`SingleCookieLowForce` 重新规划；同时在 `retry_log` 里加一个自证字段
`extras_released_in_correction`（`min(at_grip[seen:]) <= 1`）。

**第一次修法钉错了通道，而且被自己的自证字段抓到了。** 那版把 `RELEASE_OPEN` 写进了
动作第 7 维（sha `c9e214a1…`、11311 字节）。验证批 `wc5a`（`--seeds 7043 7044
--recover-variants 1`，输出 `demos_wrongclosure_v7`）两条都报
`extras_released_in_correction: false`，于是用只读探针 `probe_release.py`
逐帧对 `applied_opening` 与 `cookies_at_grip`：纠正窗（帧 95–104）里
`applied_opening` **恒定 0.280**、开度纹丝不动，5 块一直挂着；
松手实际发生在帧 105——正好是重新规划后的控制器接管的第一帧，开度才从 0.308 起步
以每帧 0.08 的限速拉到 1.012。查 `batch_expert._servo` 得到通道定义：
控制器交给环境的是 **14 维 delta-TCP 动作**，
`action[0:3]`=位移、`action[3:6]`=姿态、`action[6]`=夹爪（编码为 `clip(2*open-1,-1,1)`）、
`action[7:13]`=右臂、`action[13]`=右夹爪。而录制里读的
`obs['observation.state'][7]` / `info['applied_action'][7]` 是**环境关节动作向量的第 7 维**
（自然开度刻度），两者不是同一个通道。也就是说第 7 维是**空闲右臂的 Δx**，
钉它既开不了夹爪，又白给了右臂 10 帧满幅位移指令。

正确的第二版（本地与 pod 逐字节一致，sha `b2c3ff0b3da77609e737670322b4e23046730016853b9b96a8c6194d95b5c0e8`、
11911 字节，`py_compile` 通过）：`GRIPPER_ACT_DIM = 6`、
`RELEASE_ACT_DIM6 = 2*RELEASE_OPEN-1`，并把观察窗从 10 帧加到 16 帧——
开度命令有限速 0.08/帧，0.31→1.00 需要 9 帧再加指节滞后，10 帧刚好不够看清松手。

验证批 `wc5b`（同两枚 seed，输出新目录 `demos_wrongclosure_v8`）：
**两条的 `extras_released_in_correction` 都变成 true**，触发帧 82（`cut=98`），
异常半段峰 1.467 / 1.480 N，纠正半段峰 0.583 / 0.550 N、`>1.6 N` 与 `>2 N` 都是 0 帧，
终态 `in_target_final=1`。自述文件
`/workspace/a3_eval/demos_wrongclosure_v8/adjudication_wc.json`（2462 字节，
sha 前缀 `f2204ec6dee2c739`），
`tally = {"dir": ".../demos_wrongclosure_v8", "n": 2, "abnormal": 2, "recovery_clean": 2,
"grasp_accept": 2, "strict_accept": 2}`。

`demos_wrongclosure_v7` 那两条**不进语料**：它们的"松手"不是被指令放开的，
而是接管瞬间的抖动把批次甩出夹爪（`grip` 在开度只到 0.367 时就从 5 跳到 0），
而且纠正窗里右臂通道被喂了 10 帧满幅 Δx。这两条只留作"错误通道"的证据。

`wc4`（列 2，`demos_wrongclosure_v6b`）四条已完成的 seed（7037–7040）
**全部 `stop=bundle_failed_before_correction`**，指垫峰 8.978 / 11.717 / 29.693 / 9.898 N，
`>2 N` 分别 16 / 109 / 6 / 398 帧，一条纠正示范都没产出：换到列 2 之后捆绑控制器
自己就在携带途中卡住（`batch phase timeout; contacts=…`），
而驱动的分支顺序是"先判 `expert.failed` 再判错误闭合"，于是异常命中路径根本没走到。
**结论：C-2 的恢复目标列 1 是当前唯一稳定可采的档位**，列 2 不是"另一种纠正"而是"采不到纠正"。
已排队 tag `wc6fix`：用第二版驱动按 `wc3` 的原参数重录
`--seeds 7031 7032 7033 7034 7035 7036 --recover-variants 1`，输出新目录
`demos_wrongclosure_v9`，跑完自动出 `adjudication_wc.json`（`v6` 原目录不动，
两版可以逐条对比"自然松手"与"指令松手"）。

**`wc6fix` 跑完：6/6 全部严格合格**（2026-10-31 21:26–21:33，驱动 sha `b2c3ff0b…`）。
裁定原话：`{"dir": "/workspace/a3_eval/demos_wrongclosure_v9", "n": 6, "abnormal": 6,
"recovery_clean": 6, "grasp_accept": 6, "strict_accept": 6}`；
`adjudication_wc.json` 是采集链自己写出的，我补跑那次被 `REFUSING to overwrite` 挡回去，
行为符合预期。逐条（切点全部在 97/98 帧，异常半段峰 1.318–1.434 N、`max_at_grip=5`）：

| seed | 纠正后 closure / carry N | grip | 抬升 mm | 正放入盒 | 对比 `v6` |
| --- | --- | --- | --- | --- | --- |
| 7031 | 0.760 / 0.602 | 1 | 81.3 | 是 | v6 **reject**（纠正段 481 帧从未接触目标块） |
| 7032 | 0.578 / 0.580 | 1 | 81.6 | 是 | v6 accept |
| 7033 | 0.560 / 0.564 | 1 | 81.6 | 是 | v6 accept |
| 7034 | 0.547 / 0.548 | 1 | 81.6 | 是 | v6 **reject**（纠正段闭合瞬间 2.993 N） |
| 7035 | 0.563 / 0.599 | 1 | 81.4 | 是 | v6 accept |
| 7036 | 0.577 / 0.581 | 1 | 81.6 | 是 | v6 accept |

与 `v6` 只有驱动这一个变量（同 seed、同 noise、同变体、同恢复列），
v6 被拒的两条这一批都过了。松手是被命令放开的，7031 的逐帧读数：

```
帧            90  91  92  93  94  95  96 ... 103 104 ... 110 111
applied_open .281 .281 .281 .281 .281 .361 .441 ... 1.00 1.00 ... 1.00 .92
opening      .308 .308 .308 .308 .308 .367 .454 ... 1.012 1.002 ... 1.00 .91
cookies_at_grip  0   0   0   0   5   0   0  ...  0   0  ...  0   0
```

命令从触发那一帧起按限速 0.08/帧拉到全开（9 帧到 1.0），挂着的 5 块在第一帧就离开夹爪；
`retry_log[0].extras_released_in_correction` 六条全为 `true`（这个字段在 `retry_log` 里，
不在行顶层，第一次按行顶层去读会读到 `null`）。

**C-2 语料定为 8 条**：`demos_wrongclosure_v9` 6 条 + `demos_wrongclosure_v8` 2 条
（seed 7043/7044，同一版驱动，`adjudication_wc.json` sha 前缀 `f2204ec6…`）。
`v6` 那 4 条同 seed 的已经被 `v9` 逐条取代，只留作"自然松手不可复现"的证据；
`v7` 那 2 条（通道钉错）本来就不进。所有目录都保持原样，没有覆盖、没有删除。

**裁定结果已经落成自描述文件**（不再只存在 stdout）：
`/workspace/a3_eval/demos_wrongclosure_v6/adjudication_wc.json`，6252 字节，
sha 前缀 `ccd1a5cd35a57a14`，内含 `note`（说明为什么驱动自带的 `accept` 字段不可用）、
`thresholds`、`tally` 和每条的 `recovery_clamp` 明细；
`tally = {"dir": ".../demos_wrongclosure_v6", "n": 6, "abnormal": 6,
"recovery_clean": 5, "grasp_accept": 5, "strict_accept": 4}`。
加这个文件的原因很实际：`summary.json` 里这 4 条合格样本的 `accept` 仍然是 `false`，
下游只读 summary 会把整个语料误判成全军覆没。

C-2 这批和 §6.6 说的一样**没有相机帧**（`make_env(..., render_cameras=False)`），
当前用途是"错误闭合这个状态到底长什么样、纠正路径走不走得通、验收判据怎么定"，
进训练仍需带图重采。

### 6.9 C-3「夹了但没拿住」：从 C-1 的废料里长出来的第三个族

**为什么需要单独一个判据文件。** §6.7 的 C-1 裁定器要求的异常段是"指垫力 >1.6 N 且连续 ≥3 帧"，
而 C-3 这个异常（夹爪闭合动作完成了，手里其实没有块）在力的表现上是**反过来的**——
要么整个闭合过程测不到任何接触（峰 0.0 N），要么只擦到 0.6–1.4 N 的一下。
这两种都不会触发 C-1 的异常定义，所以采集链（`replay_correct.py`，同一个驱动器）
录回来的这些条在 C-1 眼里一律是"异常没观测到 → 不合格"。
`demos_correct_c6`/`c7` 这 12 条的 `anomaly_observed` **全是 `false`**，
就是这个结构性错配的直接证据：驱动器自己认为没有异常，但状态确实错了。
如果不去重新判据，这 12 条就是纯废料；给了自己的判据，它们是 10 条严格合格的纠正示范。

**分族靠驱动器自己写进 `retry_log[0].anomaly_kind` 的键**，不是靠后处理再猜，
所以一个 seed 只可能落进一个族，C-1 与 C-3 之间不存在重复计数。
这条在 §6.7 的规则修复里被真正用到过一次：seed 7243 / 7250 / 7260 三条原先被
宽松的 `or` 判据算进 C-1（20 条），改成 `and` 之后 C-1 掉到 17 条，
这三条按 `anomaly_kind = stuck_no_force` 落进 C-3——实测确认它们确实在
C-3 的合格名单里，也就是说那 3 条只是换了族，没有被算两次、也没有丢。

**工具**：`/workspace/_r5_stage/adjudicate_c3.py`，8506 字节，
sha256 `9bd31c5ac858a87b24db63b746bc487fac0cd1556dc4cbcaf34e89df66567985`，
`py_compile` 通过；只读 `summary.json` + 各条 npz，输出一条新文件
`<dir>/adjudication_c3.json`，已存在则 `REFUSE overwrite`
（fin2 链里对 c6 的那次重复调用就是被这个守卫挡回去的，行为符合预期）。
阈值常量：`HOLD_MIN_N=0.3`、`ANOMALY_PAD_N=1.6`、`HARD_CAP_N=2.0`、
`EMPTY_PEAK_N=0.05`、`CLOSED_THROUGH_AIR=0.10`、`CARRY_LIFT_MM=60.0`、
`NO_CONTACT="contacts=[0.0, 0.0]"`。严格合格要同时满足：

```
expert_reports_no_contact  异常那一夹传感器自己报的是 contacts=[0.0, 0.0]
held_nothing               异常夹 max_cookies_at_grip==0 且 carried_lift_mm<60
recovery_clean             纠正段 over_2p0==0 且 peak<=1.6 N
grasp_accept               纠正段那一夹本身满足单块低力验收
hold_in_range              纠正段夹住块后的持有力在 0.3–1.6 N
```

第一条是关键：不是"我看它没夹住"，而是**专家策略自己在异常那一夹报告了零接触**，
这样才立得住 §6.2 那条"纠正前必须真的观测到过异常"的原则。

**c6（6 条，变体 15/19/23，seed 7266–7271）逐条**：

| seed | 变体 | kind | 异常夹峰 N | 切点 | 纠正段峰 N | >2N 帧 | 真空闭合 | 严格 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7266 | 15 | empty_grasp | 0.000 | 451 | 0.582 | 0 | 是 | **过** |
| 7267 | 19 | empty_grasp | 0.000 | 451 | 0.522 | 0 | 是 | **过** |
| 7268 | 23 | stuck_no_force | 0.722 | 450 | **9.388** | 1 | 否 | 不过 |
| 7269 | 15 | empty_grasp | 0.000 | 451 | 0.587 | 0 | 是 | **过** |
| 7270 | 19 | empty_grasp | 0.000 | 450 | 0.560 | 0 | 是 | **过** |
| 7271 | 23 | empty_grasp | 0.000 | 450 | **5.257** | 1 | 是 | 不过 |

`TALLY {"n": 6, "true_empty": 5, "closed_through_air": 5, "expert_reports_no_contact": 6,
"held_nothing": 6, "recovery_clean": 4, "grasp_accept": 6, "hold_in_range": 6, "strict_accept": 4}`，
`adjudication_c3.json` 11535 字节，sha 前缀 `b99ab5bfd4d7aa8b`。
两条被拒都只因为**纠正段自己下压过深**（5.3 N / 9.4 N 的单帧冲击），
和 §6.7 说的 C-1 唯一拒因是同一个物理成因——重下压时爪尖撞东西那 1–2 帧。

**c7（6 条，变体 11 ×6，seed 7272–7277）逐条**：

| seed | 变体 | kind | 异常夹峰 N | 切点 | 纠正段峰 N | >2N 帧 | 严格 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 7272 | 11 | stuck_no_force | 0.628 | 450 | 0.547 | 0 | **过** |
| 7273 | 11 | stuck_no_force | 1.436 | 451 | 0.488 | 0 | **过** |
| 7274 | 11 | stuck_no_force | 0.776 | 451 | 0.580 | 0 | **过** |
| 7275 | 11 | stuck_no_force | 1.048 | 450 | 0.582 | 0 | **过** |
| 7276 | 11 | stuck_no_force | 0.816 | 451 | 0.568 | 0 | **过** |
| 7277 | 11 | stuck_no_force | 0.897 | 450 | 0.586 | 0 | **过** |

`TALLY {"n": 6, "true_empty": 0, "closed_through_air": 0, "expert_reports_no_contact": 6,
"held_nothing": 6, "recovery_clean": 6, "grasp_accept": 6, "hold_in_range": 6, "strict_accept": 6}`，
文件 11550 字节，sha 前缀 `96551e00f7f89dcf`。6/6 全过，因为变体 11 的异常本身就是
"擦到一下（0.63–1.44 N，都在 1.6 N 线以下）但没夹住"，
重下压时块还在原位、爪尖没有直接撞桌面，所以没有出现 c6 那种纠正段冲击。
`c7` 与 `c6` 的 `adjudication_correct_v2.json`（C-1 复扫）分别 5872 / 7322 字节
（sha 前缀 `0383efba…` / `63daa7d7…`），两族计数都是 `strict 0`——
这 12 条里没有一条是 C-1 型异常，**C-1 停在 17 条**。

**C-3 语料最终 18 条**：c1 0 + c2 4 + c3 1 + c4 1 + c5 2 + c6 4 + c7 6（按 `corpus_manifest_v1.json` 逐条数）。
两个子形态都有，而且训练上都需要：
- **真空闭合** 6 条（`closed_through_air`，异常夹峰 0.0 N）：夹爪在空气里合到底，
  专家自己报零接触——模型要学的是"闭上了但什么都没夹到"这个观测模式；
- **擦到/压到但没夹住** 12 条（异常夹峰 0.628–4.045 N，中位 0.97 N）：合爪过程有接触、
  有力读数，但块最终没在手里（`max_cookies_at_grip==0`、抬升 <60 mm）——
  这是更接近真实失败的那一支，因为传感器是有读数的，
  没有力通道的模型（现在的 V0）**根本区分不了这两支**。
  这 12 条里峰超过 1.6 N 的正好是 4.045 / 2.276 / 1.694 三条（seed 7243 / 7250 / 7260），
  也就是 §6.7 规则修复从 C-1 掉出来的那三条：它们既是"下压过深"也是"没拿住"，
  按 `anomaly_kind` 归 C-3，**不再同时计入 C-1**，所以 17 + 18 = 35 这个数没有重复。

累计：纠正族一共采 45 条（c1..c5 的 33 + c6/c7 的 12），严格合格 17 C-1 + 18 C-3 = 35 条，
拒 10 条，拒因全部是纠正段的 1–2 帧冲击，没有一条是"异常没真发生"。

### 6.10 C-4 / C-5 在现有场景里没有自然样本，不硬造（2026-11-01 只读扫描）

§6.2 一共设计了五个纠正族。C-1/C-2/C-3 的异常都是场景自己产生的（§6.5 那三族），
C-4（搬运途中力抬头 >2 N）与 C-5（释放高度不足、碰到已放好的饼干）**扫遍现有全部专家录制都没有出现过**。
对 `/workspace/a3_eval/demos_*/summary.json` 的每一夹逐条读
`clamps[*].carry_peak_pad_n` 与 `post_release_pad_n`（脚本只读，不改任何文件）：

```
episodes_with_multiple_clamps 0
CARRY_GT_2N                   0
POST_RELEASE_GT_0.05N         0
```

第二条和第三条的含义分别是：**没有任何一夹在搬运途中超过 2 N**，
**没有任何一夹在松手后指垫还测得到力**。第一条说明这批录制全是单夹回合，
所以"往已经堆了饼干的盒里再放一块"这个 C-5 前提本身就没被触发过。

原始 100 条专家语料也是同一个结论的另一半证据：`docs_task3` §9 那张表里
`L_pad_inner` / `L_pad_outer` 去零载后的 |残差 max| 是 **1.5398 / 1.5180 N**，
也就是整个语料（42681 帧）指垫就没超过 1.54 N，C-4 的 >2 N 触发条件
在专家数据里是一个**空集**。

所以 C-4/C-5 只有两条路：在驱动里给动作流注入扰动（把搬运途中硬掰一下），
或者改场景（先往盒里放几块、或把多夹回合打开）。**这两条都不做**，理由是
§6.2 自己定的验收原则——"纠正前必须真的观测到过异常，否则就是把普通示范贴了个纠正标签"，
人造注入拿到的正是这种标签；而模型现在缺的不是"更多类型的异常剧本"，
是**根本没有力这个输入通道**（任务五 §1 证据 (1)+(3)）。
B 分支的可做量因此就是 C-1/C-2/C-3 三族，采 45 条、严格合格 **17 C-1 + 8 C-2 + 18 C-3 = 43 条纠正示范**，
外加 30 条单块低力成功示范（§6.11 把这三族一起钉成一份带哈希的清单）。

### 6.11 验收完的语料落成一份带哈希的清单（`corpus_manifest_v1.json`）

前面每个族各自有一份 `adjudication_*.json`，下游要用还得逐个目录去拼，而且拼错了也发现不了。
所以补一个只读聚合器 `/workspace/_r5_stage/build_corpus_manifest.py`
（9756 字节，sha256 `a4c9678d39d6323b652795cae9f16d928051f3981017def603dc2da68f553531`，
本地与 pod 逐字节一致、`py_compile` 通过；pod 上第一次落地时是 9754 字节，
逐行长度对拍定位到我抄写时把 `adjudicate_c3.json` 少打了一个字符，已按精确替换修回并复核哈希）。

命令（只新建一个文件，已存在即 `REFUSE overwrite`，不改任何源目录）：

```
python3 /workspace/_r5_stage/build_corpus_manifest.py \
  --correct demos_correct_c1 ... demos_correct_c7 \
  --wc demos_wrongclosure_v8 demos_wrongclosure_v9 \
  --single demos_single_v30b
```

产物 `/workspace/a3_eval/corpus_manifest_v1.json`：41118 字节，
sha256 `b679fa6bd07e12c3be25e02f1d5aa3d14e3d785fe726e04f98b1f3a391dd0e00`，
顶层键 `note / generated_utc / tally / sources / entries`，
每条 entry 带 `family / dir / seed / variant / anomaly_kind / cut / 异常与纠正段的力峰值 /
steps / npz / npz_bytes / npz_sha256`。

`tally = {"C1_press_too_deep": 17, "C3_closed_but_empty": 18,
"C2_wrong_closure": 8, "S1_single_lowforce": 30}`，合计 **73 条**。
复核（`/workspace/_r5_stage/verify_manifest.py`，只读）：

```
n 73
family {'C1_press_too_deep': 17, 'C2_wrong_closure': 8, 'C3_closed_but_empty': 18, 'S1_single_lowforce': 30}
dupes []                 # 同一 (dir, npz) 出现两次——没有
missing_npz 0 []         # 每条都指向真实存在的 npz
total_bytes 7657304      # 73 个无相机 npz 合计 7.65 MB
sha_ok 73 of 73         # 每条都带 npz_sha256
```

清单里最值得记的一条实现坑（已写进代码注释）：`replay_forces.py` 系列写出的
`summary.json` 行**根本没有行顶层 `accept` 字段**（只有 `replay_correct.py` 会写），
而它的行顶层 `success` 是环境"整个任务完成"的标志——单块回合一夹放完任务并没完成，
所以 `success` 恒为 `false`。用 `success` 当验收会把这 30 条全判成失败；
真正可用的判据是 `clamps[-1]["accept"]`（`demos_single_v30b` 实测 30 行、
`clamp_accept 30`、`success 0`）。

这份清单是后面任何一步的输入边界：带图重采按它的 73 条来（§6.6 的 9.88 GB，
按 301,056 B/帧实测；smoke 用 0.41–0.61 GB），
任务五 B/C 分支谁先上、V1 训练集里放哪些条，都以它的 sha 为准。

## 7 明确排除的数据

r4 的 B0/C0 模型跑出来的高力捆绑搬运轨迹（865 N、250 帧 >40 N、0 次正确释放、
6 块全部挂在闭合夹爪上）**不作为成功专家数据**，一条都不进。
它们只保留作诊断证据（任务四已归档）。
捆绑档回放虽然能把 10 块都放进盒，但一次夹走 5 块、峰值 1.4 N 属于"能成功但不是
目标行为"，只用作 §2 的回放自校验，同样不进成功示范集。

## 8 复现命令（pod；`bash /workspace/_r5_stage/*.sh` 由 agent 执行）

```
python3 /workspace/_r5_stage/replay_forces.py --mode single --clamps 1 --noise 0.0003 \
  --seeds 7041 7042 7043 7044 7045 7046 7047 7048 7049 7050 \
  --out /workspace/a3_eval/replay_single_gate10      # 10/10 accept，已完成

python3 /workspace/_r5_stage/replay_forces.py --mode bundle --noise 0.0003 \
  --max-steps 1400 --seeds 7031 7032 7033 7034 7035 7036 \
  --out /workspace/a3_eval/replay_bundle_dist        # 语料交叉校验，进行中
```

`export PYTHONPATH=/workspace/a3_dual_arm_sim/src` 是前置条件。
