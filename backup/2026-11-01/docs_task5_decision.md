# 任务五：r5 之后的路线决策（2026-10-31）

决策规则（来自任务书，逐条对应）：

- **A** r5 若在**不恶化接触力**的前提下改善单抓/搬运/释放 → 考虑把续训推到 20000 步；
- **B** 若多训只是换来更多挤压/拖拽 → 优先把低力 + 纠正示范做厚；
- **C** 若训练收益本身有限 → 开 V1 力感知输入路线。

下面先摆已经落地的证据，九项指标一到就填 §3 的表并给结论。

## 1 已经确定的证据（不依赖 r5 评估）

**(1) r5 这 4000 步没把动作分支训下去。** 分位均值（`docs_task1` §4）：
`loss_action` 0.02759（4020–4800）→ 0.02882（7220–8000），中间分位 4 最差 0.03449；
降的是 `loss_video`（0.11294 → 0.10488）。数据覆盖是干净的（24054 窗口全不同、零重复，
r4 同期有 1440 个窗口被重放两次），所以这不是"重复喂同样数据"造成的假平。
**这条直接指向 C（训练收益有限），而不是 A。**

**(2) r5 相对 r4 不是纯训练量实验。** 同时改了 lr 形状：`FW_LR_RESTART=1` 把 lr 从 r4
末段的重加热重新爬到 5e-05 再退火（`lr_restart.json` 留了 restored 快照）。
任何 r5−r4 的差值都要写成"续训 + 重加热"的联合效应。

**(3) 力和专家行为差两个数量级。** 专家单块低力档：闭合/搬运峰 0.45–0.59 N、
42681 帧语料绝对上界 1.540 N；r4 裸模型 seed 22：`pad_force_inner` 峰 **859.170 N**、
542/600 帧 >2 N、正确释放 0 次。这不是"再训 4000 步能自己收敛"的量级差。

**(4) 纠正示范这条路已经能产出，而且判据经得起复核（2026-11-01 收口）。**
- C-1（高力持续下压 → 重规划单块）：五批 `demos_correct_c1..c5` 共 33 采，
  按 §6.2 的严格判据（异常半段 >1.6 N 且 ≥3 帧）**严格 17 条**，
  `excessive_pad_force` 族单算命中率 17/25 ≈ 68%。
  复核时发现裁定工具把"≥3 帧"写成了 `or`，改回合取后掉出的 3 条
  （7243 / 7250 / 7260）都是 `stuck_no_force` 族，本来属于 C-3（`docs_task2` §6.7）。
  剩下的拒绝全部来自"重新下压时爪尖撞东西的 1–2 帧冲击"，
  **改判据提不上去，只能多采**（`docs_task2` §6.7 结论 4）。
- C-2（一次夹 5 块的低力错误闭合 → 松手放回 → 重抓单块）：修好驱动的动作通道
  （14 维 delta-TCP 的第 6 维，不是第 7 维）以后，`demos_wrongclosure_v9`
  **严格 6/6**、`demos_wrongclosure_v8` 2/2，合计 **8 条**；
  松手是被命令放开的，`extras_released_in_correction` 8/8 为 `true`（`docs_task2` §6.8）。
  `v6` 那批（自然松手 4/6）已被同 seed 的 `v9` 逐条取代，只留证据不进语料。
- C-3（闭了爪但一块没带起来 → 重新对准单块）：这批不是新采的，
  是把 C-1 判据拒掉的 `empty_grasp` / `stuck_no_force` 段用 `adjudicate_c3.py`
  重新裁定。c1..c5 里 **8 条**（c2 4 + c3 1 + c4 1 + c5 2），
  补采的 `demos_correct_c6`（seed 7266–7271）4/6、`demos_correct_c7`（7272–7277）**6/6**，
  合计 **18 条严格合格**（`docs_task2` §6.9）。按清单逐条数：其中 **6 条**是"完全没碰到东西"
  （异常半段峰 0.0 N、开度一路拉到全开）的真·空夹，另外 **12 条**闭合过程是有接触、
  有 0.628–4.045 N 的力读数，但块最终没在手里（`max_cookies_at_grip==0`、抬升 <60 mm）——
  **后一支更接近真实失败，而且它有力读数，
  没有力通道的 V0 模型区分不了这两支**。
  c6/c7 这 12 条的驱动器自带 `anomaly_observed` 全是 `false`：
  C-1 那套">1.6 N 且 ≥3 帧"的异常定义天生抓不到"夹空"，
  所以这不是把普通示范贴了纠正标签——专家策略在异常那一夹自己报告了 `contacts=[0.0, 0.0]`
  （或 ≤1.44 N 的擦碰），这个字段是判据的第一条。
  c6 被拒的 2 条（7268 纠正段 9.388 N、7271 纠正段 5.257 N）与 C-1 唯一的拒因同源。
  C-1 复扫（`adjudication_correct_v2.json`）在 c6/c7 两目录都是 `strict 0`，**C-1 停在 17 条**。
- 关键结构性发现：单块控制器**天然产生不了** ≥2 块的错误闭合，
  捆绑控制器每次都是 5 块；所以 C-2 只能从捆绑档场景来，且纠正目标必须换列
  （被夹走的 5 块就在源列 0，原位重抓会抓空——第一批就是这么烧掉一条 seed 的）。

**(5) 带图重采在当前格式下放不下。** 语料已经收口并且落成清单
（`/workspace/a3_eval/corpus_manifest_v1.json`，41118 字节，sha256 `b679fa6b…`，
**73 条严格合格**：30 单块 + 17 C-1 + 8 C-2 + 18 C-3，`docs_task2` §6.11）。
体积按清单里每条的真实帧数乘实测每帧字节算，不按区间猜：
训练集 `frames.dat` = 12,849,371,136 B，`summary.json` 记 `num_frames=42681`，
两者**整除**：`12,849,371,136 / 42,681 = 301,056 B/帧`（= 294.0 KB/帧）。
我之前写的"≈281 KB/帧"是把 12 GB 当十进制除法得到的，偏小 7 %，本节全部按
**301,056 B/帧**重算（2026-11-01 用户指正）：
新语料合计 **32818 帧**（单条 215–672，纠正族因为要含"异常段 + 纠正段"所以都是 ~670 帧，
单块成功示范只有 ~228 帧），带图 **9,880,055,808 B = 9.88 GB**（9.20 GiB）；
而 `/workspace` 21:36 量得 3.41 GB、22:05 c7 收完再量 **3.407 GB**（`docs_task2` §6.6）。
**缺口是 9.88 − 3.407 = 6.47 GB**（9,880,055,808 − 3,406,704,640 = 6,473,351,168 B），
不是"勉强塞得下"。
但**"先转 2–3 条带图 + 训练加载 smoke"这一步不需要任何删除**：3 条 × 450 帧 = 0.41 GB，
最坏取 672 帧 × 3 条 = 0.61 GB，都在 3.407 GB 里（`docs_task2` §6.6 追加段）。
全量要么腾空间（删 r1/r2/r3 的 merged.pt，需要批准），要么改压缩存法（动数据加载器，
和"不改输入结构"冲突，只能并进 V1 一起做）。

**(6) 纠正示范这条路有天花板，B 分支不能无限加码。** §6.2 设计了 C-1..C-5 五个族，
实际只有三个能在现有场景里**自然发生**。对全部 `demos_*/summary.json` 的每一夹做只读扫描
（2026-11-01，脚本逐条读 `clamps[*].carry_peak_pad_n` / `post_release_pad_n`）：

```
episodes_with_multiple_clamps 0
CARRY_GT_2N            0
POST_RELEASE_GT_0.05N  0
```

也就是**没有任何一条专家轨迹出现"搬运途中力抬头"（C-4）或"释放时碰到已放好的饼干"（C-5）**，
原始 100 条专家语料也从来没有：`docs_task3` §9 那张表里指垫两条通道去零载后的
|残差 max| 是 1.5398 / 1.5180 N，全语料就没超过 1.54 N。
所以 C-4/C-5 只能靠人造注入（在驱动里给动作流加扰动）或改场景（盒里先放饼干、
多夹一轮）才拿得到，而这与"异常必须是场景自己产生的、不是贴标签"这条验收原则冲突。
**结论：B 分支的可做量就是 C-1/C-2/C-3 三族，已经做到 33 采 17 合 + 8 + 8，
继续加采的边际收益是"同一族的 seed 多样性"，不是"覆盖更多异常类型"** ——
这正好是 C 分支（V1 力输入）该上场的理由。

## 2 我的倾向（§3 指标回来后已在 §4 落章）

证据 (1)+(3) 已经构成 **B+C 组合**的理由：继续在无力感知的输入上延长训练，
最可能的结果是"video 分支继续降、action 分支不动"，而力这个维度模型现在**根本没有输入通道**，
它不可能凭视觉把 859 N 压到 0.5 N。所以：

- 若 §3 显示 r5 的单抓/搬运/释放**有**改善且力没恶化 → 走 A，但要同时把 V1 排进下一轮；
- 若 §3 显示力恶化或改善只来自更多次"挂爪搬运" → 走 B+C，20000 步不启动；
- 无论哪条，**纠正示范的低力语料都要继续加厚**（这是 B 的内容，也已经是事实上的主任务）。

## 3 r5 vs r4 九项指标对照

**对照公平性先落账**：两次评估的 `run_matrix.py` sha 完全一致（`4ddeb5ea…`），
底座同为 `a3_lora_r3/step_003000_merged.pt`，同为 `--seeds 11 21 22 --replan 10
--max-steps 600 --gripper-mode pm1 --position-noise 0 --yaw-noise 0 --batch-offset 0 --video`、
同一 `dataset_stats.json`、同一 instruction，**唯一变量是 adapter**
（r4 `step_004000/adapter.pt` vs r5 `step_008000/adapter.pt`）。
但 lr 形状这个第二变量仍然存在（§1 证据 (2)），所以结论要写成"续训 + 重加热"的联合效应。

### 3.1 裸模型 B0（官方字段，取自各自 `summary_lora.json`）

| seed | run | `success` | 官方 `cookies_in_target` | `max_pad_force` N | `max_wrist_force` N | verbatim % |
| --- | --- | --- | --- | --- | --- | --- |
| 11 | r4 | false | 0 | 25.526 | 13.63 | 100 |
| 11 | r5 | false | 0 | **30.972** | 18.04 | 100 |
| 21 | r4 | false | 0 | 25.955 | 93.99 | 100 |
| 21 | r5 | false | 0 | **37.178** | 95.02 | 100 |
| 22 | r4 | false | **6** | **865.429** | **333.05** | 100 |
| 22 | r5 | false | **2** | **27.726** | 97.86 | 100 |

r4 B0/22 同一动作流下 Safety 的影子计数是"会改写 75.83 % 的步、会硬停 249 步"；
r5 三条 B0 的 `estop_steps` 与 `binfloor_steps` 全为 0，`safety_would_have_hard=0`。

三个读数（B0 侧）：

1. **那个 865 N 的碾压机质在 r5 消失了**：seed 22 的指垫峰从 865.429 N 降到 27.726 N，
   腕部从 333.05 N 降到 97.86 N，三条 seed 现在都落在 25–37 N 的同一档位。
   这是 r5 相对 r4 最实在的一项变化。
2. **官方 `cookies_in_target` 从 6 掉到 2 不是"技能退化"**：r4 那 6 块是挂在闭合夹爪上、
   z 还停在源高度的碾压/拖拽态（`docs_task4` §2.1 已拆开算过），
   按任务口径 `rest_in_bin` 两侧都是 0，所以这条只能记成"病态行为减少"，不能记成"入盒变差"。
3. **另外两条 seed 的力反而略升**（11：25.5→31.0，21：26.0→37.2 N）。
   所以"不恶化接触力"这个 A 分支的前提**不是无条件成立**的：
   整体分布往低力靠拢（灾难性尖峰消失），但常态峰值小幅上移。
   判 A 的时候必须把这两件事分开说，不能只报 865→27.7 这一半。

### 3.2 接口 + Safety C0（官方字段，取自各自 `summary_lora.json`）

| seed | run | `success` | 官方 `cookies_in_target` | `max_pad_force` N | `max_wrist_force` N | verbatim % | `estop_steps` | `binfloor_steps` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 | r4 | false | 0 | 25.526 | 13.63 | 100.0 | 0 | 0 |
| 11 | r5 | false | 0 | 30.972 | 18.04 | 99.83 | 0 | 0 |
| 21 | r4 | false | 0 | 25.955 | 87.16 | 99.0 | 0 | 0 |
| 21 | r5 | false | 0 | **49.167** | 89.76 | 98.17 | **1** | 0 |
| 22 | r4 | false | **3** | 49.793 | 97.64 | 81.33 | **7** | **79** |
| 22 | r5 | false | **1** | 26.423 | 93.55 | 91.83 | 0 | **37** |

C0 侧最值得记的一条：r4 的硬停集中在 seed 22（7 步 estop、79 步 bin-floor），
r5 的硬停换到了 seed 21（1 步 estop）且该条的指垫峰 **49.167 N 反而高于同条 r4 的 25.955 N**
——Safety 仍要在模型自己顶到 25 N 以上时接管，接管点的分布随权重变了，但没有变成"不用接管"。

### 3.3 九项指标（同一打分器 `autonomy_metrics.py` 重跑两侧，逐条verbatim）

`single` = 通过验收的单块抓—搬—放；`rel_in_bin` = 在盒内释放；`rest_in_bin` = 松手后留在盒内。

```
r5 (dir=/workspace/a3_eval/r5_b0_c0_paired, files=6)
lora_B0_11  events=0 single=0 single_lift=0 bundle=0 held_no_release=0 rel_in_bin=0 rest_in_bin=0 dur_med=0   car_med=0mm
            pad p50=22.46 p95=23.27 max=30.97 | >25N 14帧 >40N 0 | wrist p50=11.10 max=18.04 | flips obs=256 cmd=48 | 改写=0  | cit_end=0
lora_B0_21  events=1 single=0 single_lift=0 bundle=1 held_no_release=0 rel_in_bin=0 rest_in_bin=0 dur_med=1   car_med=0mm
            pad p50=22.11 p95=25.35 max=37.18 | >25N 88帧 >40N 0 | wrist p50=16.39 max=95.02 | flips obs=181 cmd=68 | 改写=0  | cit_end=0
lora_B0_22  events=3 single=0 single_lift=1 bundle=3 held_no_release=0 rel_in_bin=0 rest_in_bin=0 dur_med=8   car_med=2mm
            pad p50=18.00 p95=26.02 max=27.73 | >25N 116帧 >40N 0 | wrist p50=16.02 max=97.86 | flips obs=180 cmd=48 | 改写=0  | cit_end=4 (官方 2)
lora_C0_11  events=0 single=0 single_lift=0 bundle=0 held_no_release=0 rel_in_bin=0 rest_in_bin=0 dur_med=0   car_med=0mm
            pad p50=22.46 p95=23.07 max=30.97 | >25N 6帧  >40N 0 | wrist p50=11.12 max=18.04 | flips obs=193 cmd=56 | 改写=1  | cit_end=0
lora_C0_21  events=4 single=0 single_lift=0 bundle=4 held_no_release=1 rel_in_bin=0 rest_in_bin=0 dur_med=8   car_med=1mm
            pad p50=0.00  p95=25.11 max=49.17 | >25N 36帧 >40N 1 | wrist p50=10.96 max=89.76 | flips obs=185 cmd=104 | 改写=11 | cit_end=0
lora_C0_22  events=2 single=0 single_lift=1 bundle=2 held_no_release=1 rel_in_bin=0 rest_in_bin=0 dur_med=176 car_med=76mm
            pad p50=7.99  p95=25.30 max=26.42 | >25N 36帧 >40N 0 | wrist p50=11.01 max=93.55 | flips obs=220 cmd=76 | 改写=49 | cit_end=3 (官方 1)

r4 (dir=/workspace/a3_eval/r4_b0_c0_paired, files=6，同一打分器重跑)
lora_B0_11  events=0 single=0 single_lift=0 bundle=0 held_no_release=0 rel_in_bin=0 rest_in_bin=0 dur_med=0   car_med=0mm
            pad p50=22.48 p95=23.07 max=25.53 | >25N 1帧  >40N 0 | wrist p50=10.88 max=13.63 | flips obs=226 cmd=14 | 改写=0  | cit_end=0
lora_B0_21  events=4 single=0 single_lift=0 bundle=4 held_no_release=1 rel_in_bin=0 rest_in_bin=0 dur_med=2   car_med=0mm
            pad p50=4.93  p95=25.64 max=25.96 | >25N 67帧 >40N 0 | wrist p50=11.18 max=93.99 | flips obs=195 cmd=107 | 改写=0  | cit_end=0
lora_B0_22  events=2 single=0 single_lift=1 bundle=2 held_no_release=2 rel_in_bin=0 rest_in_bin=0 dur_med=254 car_med=72mm
            pad p50=38.15 p95=87.12 max=865.43 | >25N 452帧 >40N 250 | wrist p50=11.21 max=333.05 | flips obs=244 cmd=66 | 改写=0 | cit_end=6 (官方 6)
lora_C0_11  events=0 single=0 single_lift=0 bundle=0 held_no_release=0 rel_in_bin=0 rest_in_bin=0 dur_med=0   car_med=0mm
            pad p50=22.48 p95=23.07 max=25.53 | >25N 1帧  >40N 0 | wrist p50=10.88 max=13.63 | flips obs=226 cmd=14 | 改写=0  | cit_end=0
lora_C0_21  events=3 single=0 single_lift=0 bundle=3 held_no_release=0 rel_in_bin=0 rest_in_bin=0 dur_med=1   car_med=0mm
            pad p50=4.02  p95=24.87 max=25.96 | >25N 19帧 >40N 0 | wrist p50=11.26 max=87.16 | flips obs=207 cmd=107 | 改写=6  | cit_end=0
lora_C0_22  events=1 single=0 single_lift=1 bundle=1 held_no_release=1 rel_in_bin=0 rest_in_bin=0 dur_med=340 car_med=130mm
            pad p50=18.92 p95=26.92 max=49.79 | >25N 80帧 >40N 7 | wrist p50=11.10 max=97.64 | flips obs=257 cmd=68 | 改写=112 | cit_end=6 (官方 3)
```

`cit_end` 是自算"末帧中心在盒内 xy"的松口径，官方 `cookies_in_target` 见 §3.1/§3.2；
两者在 r5 B0/22（4 vs 2）、r5 C0/22（3 vs 1）、r4 C0/22（6 vs 3）三处不一致，
差的都是挂在闭合夹爪上、z 还停在源高度的那些块（`docs_task4` §2.1）。

## 4 决策落章：不启动 20000 步续训，走 B + C

按开头的三条判据逐条判：

- **A 不成立**。A 的前提是"不恶化接触力**且**改善单抓/搬运/释放"。
  后半句直接被 §3.3 否掉：`single`、`rel_in_bin`、`rest_in_bin` 三项在 **r4 与 r5 的
  全部 12 条录制里都是 0**，`success` 12/12 为 false；
  前半句也不干净——seed 22 的灾难尖峰确实消失（865.43→27.73 N），
  但常态峰值上移（11：25.53→30.97，21：25.96→37.18 N），
  C0 侧 seed 21 的峰值还从 25.96 N 涨到 49.17 N。
  唯一的"进步"是搬运距离类的读数（r5 B0/22 `dur_med=8`、`car_med=2mm`，
  r4 是 254 帧、72 mm）——那是"抓得更久、搬得更少"，不是技能。
- **B 成立且已经是事实上的主任务**。低力 + 纠正示范是能产出、能被独立复核、
  且量级差（0.45–0.59 N vs 25–49 N）明确指向"缺示范不缺步数"的那条路。
- **C 成立**。模型现在没有力的输入通道（`docs_task3` V0/V1），
  在无力输入上继续加步数，最可能的结果仍是一条已经观测到的曲线：
  video 分位降、action 分位不动（§1 证据 (1)）。

所以：**20000 步不启动**；下一步按顺序是 (a) 把 C-1/C-2 纠正示范与单块低力示范的产量
凑到可训练的规模，(b) 腾出带图重采的空间（等批准），(c) 在这一轮就把 V1 力输入接进训练，
并把轻量态漏 `proprio_encoder` 的坑先补掉（`docs_task1` §7），否则 V1 的新增输入仍然存不进 checkpoint。

r4→r5 的对照到此关闭；`r5_autonomy_metrics.txt` / `r4_autonomy_metrics_rerun.txt` 是原始产出，
上面两表逐条来自这两个文件与两份 `summary_lora.json`，没有手工合并过。

## 5 收口状态与需要批准的三件事（2026-11-01 22:15，全部只读测得）

### 5.1 B 分支的产量已经定数，落成一份清单

`/workspace/a3_eval/corpus_manifest_v1.json`，41118 字节，
sha256 `b679fa6bd07e12c3be25e02f1d5aa3d14e3d785fe726e04f98b1f3a391dd0e00`：

```
tally = {C1_press_too_deep: 17, C3_closed_but_empty: 18,
         C2_wrong_closure: 8,  S1_single_lowforce: 30}      # 合计 73 条
dupes [] / missing_npz 0 / sha_ok 73 of 73 / total_bytes 7657304
```

采集侧一共 45 条纠正族录制（c1..c5 的 33 + c6/c7 的 12），
严格合格 17 C-1 + 18 C-3 = 35，拒 10，**拒因全是纠正段那 1–2 帧冲击**，
没有一条是"异常其实没发生"。C-4/C-5 已证明在现有场景里没有自然样本，不硬造（§1 证据 (6)）。
也就是说 B 分支在"不加相机帧"这个前提下已经做到头了，
剩下能加的是**条数**，不是**族**；而条数已经够到 73 条这个量级。

### 5.2 磁盘：9.88 GB 的需求 vs 3.41 GB 的空闲，缺口 6.47 GB

需求按清单里每条的真实帧数乘实测每帧 **301,056 B** 算（`docs_task2` §6.6），不是区间估计；
这一列原来用的 281 KB/帧偏小 7 %，所以需求从 8.79 GB 修正为 **9,880,055,808 B**。
空闲 `df -B1` 22:14 实测 **3,406,897,152 B**。可回收的大头只有三个文件，
逐个列在下面（大小字节精确、mtime 实测、**以及现在还有谁引用它**——这条是关键，
前一次评估环境就是靠 r1 那份合并权重起家的）：

| 文件 | 字节 | mtime | 谁在用 |
| --- | --- | --- | --- |
| `a3_train_runs/a3_lora_r1/step_002000_merged.pt` | 11,117,977,659 | 2026-10-30 13:00 | **`activate_fasterwam_a3.sh:4` 把它 export 成 `FASTERWAM_A3_CKPT`（默认权重指针）**；`deploy/bootstrap_pod.sh:220/241`（带 size 校验）、`selftest_harness.py:49`、`_selftest/{old,new}/manifest_lora.json`、`_r4_stage/up/common.new.py:43`、`bridge/_next/common.py:43` |
| `a3_train_runs/a3_lora_r2/step_003000_merged.pt` | 11,117,977,659 | 2026-10-30 16:44 | **只有历史记录**：`a3_lora_r3/config.yaml` 的 `resume` 指它；`watch_r2.sh:6`、`run_r3.sh:6`、`_r4_stage/mktab2.py:20` —— 无现役入口 |
| `a3_train_runs/a3_lora_r3/step_003000_merged.pt` | 11,117,977,659 | 2026-10-30 19:20 | **现役两处**：`run_r5.sh` 的 `FW_BASE_CKPT`（缺文件直接 `ABORT`）；`_r5_stage/r5_eval.sh:20` 评估 r4/r5 adapter 时的 `--ckpt` 基座。删了 r4/r5 评估不可复现 |

顺带说清各目录的构成，免得按 `du` 的 11 G/21 G 误判：
`a3_lora_r1` 和 `a3_lora_r2` 这两个 11 G 目录**基本就是那一个 merged.pt**
（其余是 8 KB 的 config、12 KB 的 checkpoints）；
`a3_lora_r3` 的 21 G = merged 11 G + `checkpoints` 11 G。
`a3_lora_r4` / `a3_lora_r5` 各只有 1.4 G（纯轻量 LoRA 态）。
`/workspace/a3_data/a3_fw/frames.dat` 12,849,371,136 B 是原始语料的帧流，不动。

**我的建议：只归档 `a3_lora_r2` 那一份 merged.pt，而且这是唯一安全的一份。**
把"删 r1/r2/r3 腾 40 GB"这个想法收回——上一版这里写得不够准，逐个查引用之后，
三份里有两份是**现在还在用的**：

- `a3_lora_r1` 那份是环境的默认权重指针（`FASTERWAM_A3_CKPT`），删了以后所有
  依赖这个变量的入口（bridge 的 `infer_a3_worker.py:48`、`bootstrap_pod.sh` 的
  size 校验、selftest）会静默拿到不存在的基座。
- `a3_lora_r3` 那份**既是 r5 的 `FW_BASE_CKPT`**（`run_r5.sh:13` 的
  `[ -f "$BASE" ] || ABORT` 是硬前置），**也是 r4/r5 评估的基座**：
  `_r5_stage/r5_eval.sh:20` 就是 `--ckpt .../a3_lora_r3/step_003000_merged.pt --adapter <r5 light>`。
  删了它，r4/r5 的 adapter 从此没法再跑第二次——评估不可复现，比不可复现训练严重得多。
- 只有 `a3_lora_r2` 那份的全部引用都是"当年怎么造出 r3"的历史记录
  （`a3_lora_r3/config.yaml` 的 `resume`、`run_r3.sh:6`、`watch_r2.sh:6`、
  `_r4_stage/mktab2.py:20`），没有任何现役入口读它。

腾出 **11.12 GB**（一份 = 11,117,977,659 B），需求 9.88 GB，**余 1.24 GB**
（11,117,977,659 − 9,880,055,808 = 1,237,921,851 B）——
带图重采**刚好**放得下，余量比上一版算出来的 2.3 GB 少将近一半，
双臂 V0/V1 训练的检查点（每项 ~0.77 GB/次保存）会立刻顶到天花板，
所以真正利索的做法还是把 r1、r2 两份一起**移到 pod 外**（22.24 GB），
而不是在盘上删。r3 那份无论如何要留在原地。

这一步之前**不动任何文件、不删任何目录**（`docs_task2` §6.6 的约束在这里仍然生效）；
移走 r1/r2 的任何一份都需要先批准，我不会自行执行。

### 5.4 顺带查出来的一件事：V1 的补丁必须打在活的副本上

仓库里 `light_state.py` / `a3_video_dataset.py` / `run_matrix.py` 都各有两份，
打在错的那份上就是静默无效——这正是监督要求要避免的那种返工。实测：

```
md5  FasterWAM/src/fasterwam/light_state.py            == fasterwam-a3-cookie/finetune/light_state.py
md5  FasterWAM/src/fasterwam/datasets/lerobot/a3_video_dataset.py == fasterwam-a3-cookie/finetune/a3_video_dataset.py
md5  fasterwam_a3_bridge/bridge/_next/run_matrix.py    89d93c9a…  !=  fasterwam-a3-cookie/bridge/_next/run_matrix.py  fcb12a96…
```

- `activate_fasterwam_a3.sh` 的 `PYTHONPATH` 顺序是
  `fasterwam_a3_bridge/bridge/_next : a3_dual_arm_sim/src : FasterWAM/src : FasterWAM : fasterwam_a3_bridge/bridge`，
  所以**活的** `fasterwam` 包是 `/workspace/FasterWAM/src/fasterwam/`，
  活的 `run_matrix.py` 是 `/workspace/fasterwam_a3_bridge/bridge/_next/`（r5_eval 的 `cd` 也证实了这点）；
- `fasterwam-a3-cookie/**` 那份 checkout 目前对前两个文件还和活的逐字节一致，
  但 `run_matrix.py` **已经不一致了**——它就是不能被当成参照；
- V1 的三处扩列落点：`FasterWAM/src/fasterwam/light_state.py:35`（保存集）、
  `FasterWAM/src/fasterwam/datasets/lerobot/a3_video_dataset.py:37`（`A3VideoDataset`）、
  `fasterwam_a3_bridge/bridge/_next/run_matrix.py`（推理侧 `build_proprio` 在
  `fasterwam_a3_bridge/bridge/infer_a3_worker.py:95`）；
  数组准备脚本是 `fasterwam-a3-cookie/finetune/prepare_a3_arrays.py`（这个只有这一份）。
- 力的 18 维出处复核：`/workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/sim/env.py:375-386`
  （注意路径里有 `sim/`，`docs_task3` §12 之前只写了 `env.py`）。
  切片 `[0,1,2,12,13,16]` 对得上：`L_wrist_force` 3 维 + `L_wrist_torque` 3 + `R_*` 6 +
  四个指垫 touch 各 1 + 两个 `actuator_force` 求和各 1 = 18。

所以批准之后的动作顺序是：改上述三处（活副本）→ 逐文件 `py_compile` →
用一次 `--max-steps 2` 的 dry run 对拍 `state` 宽度与 `dataset_stats.json` → 才谈正式训练。
现在这些都还没执行。

### 5.3 需要批准的清单（我不自行执行）

1. **腾空间**：批准归档/删除上面哪一份（或哪几份）merged.pt；
   或者批准"带图重采改为分小批、每批先删后采"这种更慢但不动历史权重的走法。
2. **`light_state.py:35` 保存集**（`docs_task3` §11 第 1 项）：V1 的
   `proprio_encoder` 是 36,864 参数、正好被 `"lora_" in name` 这个过滤器**对称地**
   在存和读两侧丢掉——不修这个，V1 训完的新增输入根本进不了 checkpoint。
3. **V1 接线三处扩列 + `a3_fw_force/` 的 `frames.dat` 与 14 维 `dataset_stats.json`**
   （§11 第 2、3 项），归一化规格已在 §12 用实测钉死（除数 = `residual_std`、
   零载参考 = 全局向量），并附一条只读重算 `arrays_globalref.npz`（新文件名，不覆盖现有那份）。
4. **V0/V1 双臂各 2000 步的新训练**（§11 第 4 项）：这是 r5 之后第二个训练，按监督要求必须批。

2/3/4 是同一件事的三步，缺一步这套对照就不成立；1 是它们的物理前提。
