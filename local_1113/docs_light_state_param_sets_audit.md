# light_state 实际优化参数集合与历史权重可恢复性审计

审计时间：2026-11-01（pod 内 UTC 02:20–02:50）
执行环境：`rgapi-one-click-24316-57b6b870`，`/workspace` 挂载盘，torch 2.9.1+gitff65f5b
本轮性质：只读审计 + 小型合成模型测试。除本文明确列出的输出文件外，没有写入大文件，没有加载 5.56B 模型，没有删除任何文件。

---

## 一、结论

1. 训练时真正被 AdamW 更新的参数是 786 个张量、30,707,712 个元素；轻量 checkpoint 里的 `adapter.pt` 只存了 784 个张量、30,670,848 个元素。差 2 个张量、36,864 个元素，占优化器管辖范围的 0.120048 %。
2. 差的这 36,864 个元素恰好是 `proprio_encoder.weight`（4096×8 = 32,768）加 `proprio_encoder.bias`（4096）。它们在每一步都被更新，从未被写盘。判据来自 `light_state.py:34-35` 的 `"lora_" in name` 名称过滤，不是概率性的怀疑。
3. 上一轮 1637 个 `requires_grad` 张量、8,710,770,916 个元素的数字是我的测试脚本造成的假象：构造完模型后没有调用 `trainer.py:286-299` 的冻结流程。以这个集合为保存判据的 v2 方案已作废，它对应的写入量是 17.4 GB（bf16），也正是写满磁盘的那次事故。
4. r4/r5 训练后的 `proprio_encoder` 数值在整个 pod 上没有任何副本，不可恢复。能恢复的是 LoRA 部分（784 张量，完整）。
5. 历史评估（r4/r5 的 B0/C0 全部结果）用的组合是 r3 merged 权重 + r4/r5 的 LoRA delta，`proprio` 始终来自 r3。这个组合自洽、可复现、和历史记录逐字节一致，但它不等于训练末态的那个模型。r4/r5 的结论不能宣布作废，也不能宣布无影响，只能带上这条限定语使用。
6. 修复的实际成本是 73,728 字节（36,864 元素 × bf16）加一个文件条目，每个 step 目录从 185,090,134 B 变成约 185,163,900 B，+0.04 %。这个缺口本来是可以不发生的。

---

## 二、参数集合对照表

按你要求的七个集合逐一列出。"实测"指直接读取已落盘的文件；"推定"指依据 trainer 代码规则得出、没有在这轮重新实机跑一遍冻结。

| 集合 | 张量数 | 元素数 | dtype | 按 bf16 预计字节 | 依据 |
|---|---|---|---|---|---|
| A. `model.named_parameters()` 全集 | 未测 | 未测 | — | — | 需要实例化 5.56B 模型，本轮不授权，留空而不是填估计值 |
| B. 合并权重的 `state_dict`（`mot` 1393 + `proprio_encoder` 2） | 1395 | 未逐一求和 | bfloat16 | ≈ 11.12 GB（实测文件大小 11,117,977,659 B） | 实测，mmap 读取 |
| C. `requires_grad=True`，构造后未冻结 | 1637 | 8,710,770,916 | 混合 | 17,421,541,832 B | 实测，但属于 harness 假象，不是训练期状态 |
| D. `requires_grad=True`，训练规则下 | 786 | 30,707,712 | bfloat16 | 61,415,424 B | 推定（`trainer.py:286-299` 先全冻结、再放开 dit、再按名称滤掉非 LoRA、最后显式放开 `proprio_encoder`），与集合 E 完全吻合 |
| E. `optimizer.param_groups` 实际成员 | 786 | 30,707,712 | bfloat16 | 61,415,424 B | 实测（`optimizer.pt` 的 `param_groups[0]['params']` 长度） |
| F. 有 optimizer state 的成员 | 786 | 30,707,712 | bfloat16 | 61,415,424 B × 2（`exp_avg` + `exp_avg_sq`） | 实测；按你的要求只作旁证，不作为成员判据 |
| G. LoRA 集合（名称含 `lora_`） | 784 | 30,670,848 | bfloat16 | 61,341,696 B | 实测 |
| H. `proprio_encoder` 集合 | 2 | 36,864 | bfloat16 | 73,728 B | 实测（merged 与优化器槽位两处对得上） |
| I. `adapter.pt` 实际持久化 | 784 | 30,670,848 | bfloat16 | 61,341,696 B，落盘文件 61,595,489 B | 实测 |
| J. = E − I（在优化器里但没存） | 2 | 36,864 | bfloat16 | 73,728 B | 计算，G−I=0，H 全部落入 |
| K. = D − I（可训练但没存，v1 名称法漏掉的） | 2 | 36,864 | — | — | 与 J 相同，说明 v1 唯一漏的就是 proprio |
| L. = C − D（未冻结假象多出来的） | 851 | 8,680,063,204 | — | 约 17.36 GB | 计算，这部分是 frozen DiT/video_expert 等，本来就不该存 |

三个 run 的集合 E、F、I、J 完全一致（r4 `step_004000`、r5 `step_008000`、崩溃的 r4_crashed `step_00500`），差距不是偶发，是格式定义造成的。

补充实测细节，全部来自 `_r6_stage/out/param_sets_audit.json`（13,298 B）：

- `adapter.pt`：`format = fasterwam-lora-adapter-v1`，`keys_total = 784`，`non_lora_keys = []`，`extra_trainable_present = False`。
- `optimizer.pt`：`param_groups = 1`，`members_per_group = [786]`，`indices_with_state = 786`，state 字段只有 `step`（786 个 fp32 标量）、`exp_avg`、`exp_avg_sq`（各 30,707,712 元素 bf16），`nonsaved_state = []`，文件 123,478,413 B。
- `trainer_state.json`：`format = fasterwam-lora-light-state-v1`，`extra.base_checkpoint` 三个 run 都写成 `/workspace/a3_train_runs/a3_lora_r3/step_003000_merged.pt`，`lr = 5.000000000000001e-07`（崩溃那次是 `4.924265658199842e-05`）。
- `weights_dir_files = []`：r4/r5/崩溃 run 的 `checkpoints/weights/` 与 `checkpoints/state/` 都是空目录。
- 合并权重里 `lora_keys_in_mot = 0`，即 merged 是折叠后的底座，LoRA 结构在评估时由 run 的 `config.yaml` 重建再叠加 adapter。

---

## 三、遗漏参数清单

| 参数名 | 形状 | 元素 | dtype | 每步是否更新 | 是否落盘 | 是否被评估用到 |
|---|---|---|---|---|---|---|
| `proprio_encoder.weight` | [4096, 8] | 32,768 | bfloat16 | 是 | 否 | 否（评估取 r3 merged 值） |
| `proprio_encoder.bias` | [4096] | 4,096 | bfloat16 | 是 | 否 | 否（同上） |

没有第三个。`video_expert` 一类的大模块被 `trainer.py:291-293` 重新冻结，不在优化器里，本来就不属于"遗漏"，把它加进保存范围是错误的方向（那正是 v2 的失误）。

动量证据（`_r6_stage/out/proprio_moments.json`）说明这两个槽确实在被更新，而不是挂了个名字：

| run | 槽 | 形状 | `step` | `exp_avg` 全零 | `exp_avg` absmax | `exp_avg_sq` absmax |
|---|---|---|---|---|---|---|
| r4_crashed_500 | 784 | [4096, 8] | 500.0 | 否 | 6.819e-05 | 1.006e-07 |
| r4_crashed_500 | 785 | [4096] | 500.0 | 否 | 7.820e-05 | 1.248e-07 |
| r4_4000 | 784 | [4096, 8] | 4000.0 | 否 | 4.482e-05 | 2.829e-08 |
| r4_4000 | 785 | [4096] | 4000.0 | 否 | 4.649e-05 | 2.829e-08 |
| r5_8000 | 784 | [4096, 8] | 8000.0 | 否 | 7.248e-05 | 4.843e-08 |
| r5_8000 | 785 | [4096] | 8000.0 | 否 | 4.792e-05 | 4.936e-08 |

`zero_grad_slots = 0`，槽 0（`[16, 3072]` 的 LoRA 张量）作为控制组同样非零。

---

## 四、1637 与 786 的关系

`trainer.py` 的规则链是：`model.eval()` → 全局 `requires_grad_(False)` → `dit.requires_grad_(True)` → 若开 LoRA，则 `dit` 内所有名字不含 `lora_` 的参数再 `requires_grad_(False)` → `proprio_encoder.train(); proprio_encoder.requires_grad_(True)`。随后 `trainer.py:85-88` 用 `dit` 里剩余的 `requires_grad` 参数加 `proprio_encoder` 的参数建 AdamW。

也就是说训练期 `requires_grad` 集合与优化器成员按构造就是同一个集合（786）。我上一轮报出 1637 / 87.1 亿，是因为 Tier-2 测试脚本自己 `from_pretrained` 造了模型、没有走上面这段冻结，量到的是"未经训练配置裁剪的原始可训练张量"。这个数不能用于任何保存判据，v2 方案的全部问题从这里来。

---

## 五、最小修复补丁 v3

文件：`/workspace/_r6_stage/light_state_v3.py`（8,003 B，sha256 `4d7ceb8764bba59ca9b035477e23f85464a9f46ead24e02fcaae240dda4092bf`，与本地副本逐字节一致，`py_compile` 通过）
本地副本：[_r6_stage/light_state_v3.py](file:///C:/Users/ASUS/Documents/Qoder/2026-10-31/b462d38d/_r6_stage/light_state_v3.py)
它是独立模块，没有替换 `FasterWAM/src/fasterwam/light_state.py`，r4/r5 历史文件一个字节都没动。

设计要点，逐条对应你的要求：

- 保存范围取"优化器成员"，不取 `requires_grad`。`select_saved(model, optimizer)` 先用 `id(param)` 建立参数对象到名字的恒等映射；恒等对不上时，退到复现 `trainer.py:85-88` 的构造顺序，并要求 `[(shape, dtype)]` 签名与 `param_groups` 完全一致才接受；两条路都走不通就抛 `RuntimeError`，明确拒绝猜。
- 保存内容：784 个 LoRA 张量 + 优化器里非 LoRA 的张量（本项目就是 2 个 proprio），落在同一个文件的 `lora` 与 `extra_trainable` 两个子字典里。
- 保存前算覆盖率并写进 `meta.coverage`，`A/B/I/J/K` 那几组计数进文件，日后不用重跑就能审计。
- 写入前先算 `projected_bytes`，超过预算直接抛错，一个字节都不写。默认预算是目标盘剩余空间的 25 %，可以显式传 `max_bytes` 收紧。写入走 `.tmp` + 落盘后再核对实际大小 + `os.replace`，超了删临时文件并报错。
- 加载时认格式：v3 正常恢复；v1 拒绝并说明"这个文件只有 lora_ 键，凡是优化器步进过的非 LoRA 参数都不在里面"；确实需要按老语义加载时显式传 `allow_legacy=True`，报告里会标出恢复了几张量。未知键、形状不符各自有守卫。
- 不动 `video_expert`，不引入全量保存选项。

---

## 六、小型合成模型测试

文件：`/workspace/_r6_stage/test_light_state_v3.py`（5,823 B，sha256 `9c41ebc91a66a229c9aaef055eb06ec50fdeb6063990a15131479c30cabee206`，逐字节一致）
输出：`_r6_stage/out/v3test/` 只有 `v1.pt` 1,981 B、`v3.pt` 5,449 B、`v3test.json` 3,190 B。全程 CPU，`RUN_RC=0`。

合成模型故意造了四种角色：冻结底座 `dit.base`、可训练 LoRA `dit.q_lora.lora_A/B`、可训练且在优化器里的 `proprio_encoder`、以及"可训练但不在优化器里"的诱饵 `dit.video_expert`（对应真实模型里的 `video_expert`，用来验证 v3 不会重蹈 v2 的范围错误）。

覆盖率输出（原文）：

```
named_tensors 8 / named_elements 4872
requires_grad_tensors 6 / requires_grad_elements 4800
optimizer_members 4
saved_tensors 4 / saved_elements 640 / saved_lora_tensors 2
saved_nonlora_names ['proprio_encoder.bias', 'proprio_encoder.weight']
requires_grad_but_not_optimised 2 / elements 4160
kept_by_v1 2 / kept_by_v1_elements 64
dropped_by_v1 ['proprio_encoder.bias', 'proprio_encoder.weight'] / 576
selection_method identity
projected_adapter_bytes 2560
```

十项断言全部为真：

| 检查 | 结果 |
|---|---|
| 诱饵 `requires_grad` 但不在优化器 | true |
| 保存集合排除诱饵 | true |
| proprio 被选中 | true |
| proprio 确实在步进 | true |
| 冻结底座未被改动 | true |
| v1 丢掉 proprio（且丢的集合里 proprio 属于 v3 认定范围） | true |
| 预计字节与实写字节一致（2560 vs 5449 文件，容差内） | true |
| v3 加载逐比特一致（4 个张量全等，误差列表为空） | true |
| v3 加载后诱饵保持新初始化 | true |
| 预算守卫触发且没留下文件 | true |

三条守卫的原文行为：

- v1 直接加载被拒：`v1 adapter holds only lora_ keys (2 tensors). Any non-LoRA parameter the optimiser stepped is absent, so this file cannot ...`（截断）；`allow_legacy=True` 时报告 `restored_tensors = 2`，`proprio_still_random = true`，`lora_restored_exact = true`——正好复现了真实事故的行为。
- 预算守卫：`...: v3 adapter would write 2560 B but the budget is 10 B; nothing was written.`，`check_budget_left_no_file = true`。
- 无法解析优化器成员时：`optimizer membership unresolved: identity matched 0 of 4 and the trainer-order fallback does not reproduce it (shapes ...`，抛出而不是猜（用 `param_groups` 指向克隆张量的 Stub 模拟）。

---

## 七、r4/r5 影响判断与历史权重可恢复性

逐条回答五个问题，并标注证据等级。A = 直接读取落盘文件或日志实测；B = 由代码规则推定，未在本轮实机重跑；C = 估算或推断，需进一步验证。

**Q1 r4/r5 是否真的更新了 `proprio_encoder`？**
是。证据等级 A：优化器成员 786 中有两个槽形状 `[4096,8]` 与 `[4096]`，`step` 计数分别等于全局步数 500/4000/8000，`exp_avg` 与 `exp_avg_sq` 全非零。`trainer.py` 的冻结规则把 `proprio_encoder` 显式放开（B 级，与 A 级实测一致）。

**Q2 更新后的值是否存在其他可恢复副本？**
不存在。证据等级 A：对 pod 内所有小于 200 MB 的 `.pt` 做了 mmap 键扫描，`a3_train_runs`、`a3_eval`、`_r6_stage`、`_r4_stage` 四个 root 下共触及 112 个文件（85 个成功加载、27 个 `rng.pt` 被 `weights_only` 拒绝，它们存的是 RNG 状态不是权重），root 外仅 1 个 `.pt`（文本嵌入缓存）。含 `proprio` 权重键的文件只有 1 个，是我本轮合成的测试文件 `out/v3test/v3.pt`。三个 11 GB 的 merged 按大小跳过，由结构探针单独覆盖，它们是 r1/r2/r3 的 proprio，早于 r4 训练。r4/r5 的 `checkpoints/weights/`、`checkpoints/state/` 均为空目录。

顺带得到的一个事实（A 级）：r1、r2、r3 三个 merged 里的 proprio 校验和互不相同（`0x5203332c0a`、`0x2c9b855c0f`、`0x450e49019f`），rms 分别 0.19984370、0.19984454、0.19984582。说明 merged 保存路径（`fastwam.py:1172-1180` 的 `save_checkpoint` 会把 `proprio_encoder` 单独写进 payload）本来就是带 proprio 的，丢权重是轻量 checkpoint 这条新路径独有的缺陷。

**Q3 历史 adapter-only 评估是否遗漏了这些更新？**
是，而且遗漏的方式可以精确写出。评估入口 `bridge/_next/common.py:116-146` 的 `load_policy` 先 `instantiate(cfg.model)` 再 `model.load_checkpoint(ckpt)`；`fastwam.py:1184-1201` 的 `load_checkpoint` 对 `mot` 用 `strict=False`，对 `proprio_encoder` 用 `strict=True` 从 merged 装入。之后 adapter 只覆盖 `mot` 里的 LoRA 键。评估清单实测（A 级）：

```
r4_c0_frozen      base = r3/step_003000_merged.pt (1835f884d430f3f7) + r4/step_004000/adapter.pt (e8372c333eb2e454)
r5_b0_c0_paired   base = r3/step_003000_merged.pt (1835f884d430f3f7) + r5/step_008000/adapter.pt (af6bba85d1bd99c1)
```

所以历史上每一个 r4/r5 分数，对应的策略都是"r3 的 proprio + r4 或 r5 的 LoRA"。

**Q4 目前能否重建完整正确的 r4/r5 模型？**
分两种"正确"。重建"历史评估所测的那个模型"：能，且逐比特可复现（base 与 adapter 都在，sha 已记录）。重建"训练结束时 GPU 里的那个模型"：不能，缺 proprio 的末态值，动量不能替代权重。

另外有一处需要记进结论的细节（B/C 级）：r5 从 r4 `step_004000` 续跑时，日志显示先 `Loading frozen base first: r3/step_003000_merged.pt`（恢复 r3 proprio），再 `Resumed light state ... r4/checkpoints/light/step_004000`（恢复 786 个成员的动量，含 proprio 槽）。也就是 r5 的前 4000 步动量是围绕 r4 那组已经不存在的 proprio 权重累积的，续跑后被加到了 r3 的权重上。这不影响"能否复现评估对象"，但影响"r4→r5 这条续训链的动机是否成立"，写论文或做汇报时不能把它简单说成训练量翻倍。

**Q5 影响量级与证据等级**
不能精确量化，因为缺的正是末态值。能给一个上界估算（C 级）：AdamW 每步位移 ≤ lr × 1，r4 的 lr 为 5e-7、4000 步，单元素位移上界 2.0e-3；用实测动量比 `|exp_avg|/sqrt(exp_avg_sq) ≈ 0.27` 估典型值约 5e-4；`proprio_encoder.weight` 自身 rms 为 0.1998，相对变化 0.25 %～1 %。r5 到 8000 步按同法上界 4.0e-3。结论是：这个遗漏很可能是二阶效应，不足以宣布 r4/r5 训练结果全部失效；但它是真实存在的、方向未定的偏差，也不足以宣布"完全没影响、已经恢复"。

历史结果建议这样标注（可直接抄进汇报）：

> r4 / r5 的轻量 checkpoint 只持久化了 LoRA（784 张量），优化器同时步进但从未落盘的 `proprio_encoder`（2 张量 / 36,864 元素，占优化范围 0.12 %）已不可恢复。因此本报告所有 r4/r5 数字衡量的是「r3 底座（含 r3 proprio）+ r4/r5 LoRA delta」这一组合，而非训练末态模型本身；该组合可逐比特复现（base `1835f884d430f3f7`，adapter `e8372c333eb2e454` / `af6bba85d1bd99c1`）。

---

## 八、保存容量预算

| 方案 | 张量 | 元素 | 预计字节 | 当前磁盘可否 |
|---|---|---|---|---|
| v1（现状） | 784 | 30,670,848 | 文件 61,595,489 B | 可 |
| v3（修复） | 786 | 30,707,712 | payload 61,415,424 B，文件约 61,669,000 B（+74 KB） | 可 |
| v3 完整轻量步目录（adapter+optimizer+rng+scheduler） | — | — | 约 185,163,900 B，现值 185,090,134 B | 可 |
| v2 设想（按 `requires_grad` 存） | 1637 | 8,710,770,916 | bf16 约 17.42 GB / fp32 约 34.84 GB | 不可，剩余 3.41 GB |
| merged 全量 | 1395 | — | 11.12 GB / 份 | 不可 |

pod 上现存 25 个轻量 step 目录，合计 4.63 GB（`du -sb` 实测：r4 1,480,726,199、r5 1,480,726,202、r4_crashed 185,090,772、`chk3_A/A2/B/B2/B3` 与 `chk4_N` 各约 185,090,5xx、`chk4_S` 370,181,141）。这是磁盘只剩 3.4 GB 的主要占用来源之一；本报告不建议也不执行任何删除。

若日后接 v3，单步增量约 74 KB，可以忽略。真正需要预算的是别的东西：任何按"全集"写的方案都会立刻撞墙，因此 v3 的 `max_bytes` 默认取剩余空间 25 %，并保留显式收紧的入口。

---

## 九、磁盘 / 进程 / 临时文件现状

```
df -B1 /workspace:  total 105,089,261,568   used 96,298,360,832   avail 3,405,414,400   97%
find /workspace -name '*.tmp'        -> 空（无残留临时文件）
ps:  PID 1 jupyter-lab 存活；其余 9 个 python3 均为 <defunct> 僵尸，含上一轮被 kill 的 PID 543011
du -sb /workspace/_r6_stage          -> 358,580 B
```

本轮在 pod 侧新增的文件只有六个只读脚本和四个小 JSON（`param_sets_audit.json` 13,298 B、`proprio_moments.json` 5,030 B、`merged_struct.json`、`proprio_copies.json`、`out/v3test/*` 合计 10.6 KB），以及若干 `_r6_stage/out/c*.out` 命令回显（每个几十字节到 7 KB）。全部脚本与本地副本逐字节对拍通过：

```
4d7ceb87...  light_state_v3.py
9c41ebc9...  test_light_state_v3.py
dd998150...  param_sets_audit.py
2db1e273...  proprio_moments_probe.py
41ebde83...  merged_struct_probe.py
b4f760d2...  proprio_copies_scan.py
cef28672...  merged_proprio_probe.py   （首版，未识别 merged 的嵌套结构，已被上面这版取代）
```

`proprio_copies_scan.py` 传输时我抄错了一个右括号（`[:12]}` 应为 `[:12]]`），pod 侧 `py_compile` 抓到，用一次字符串替换修正后 sha 与本地一致，没有重传整个文件，也没有删除任何东西。

---

## 十、事故记录与授权偏差

**事故一（上一轮）**：为做保存测试意外触发模型权重下载，写入约 5.6 GB，`/workspace` 余量归零。处置：删除该次下载产生的 `/workspace/checkpoints` 目录。

**事故二（本轮之前）**：v2 测试以"保存所有 `requires_grad=True`"为判据，在真实模型上触发 `torch.save`，写出 `/workspace/_r6_stage/out/ls2b_seed11/v2_adapter.pt.tmp`，8,774,885,376 B，磁盘再次归零，连带把 pod 的 contents 接口打成 `500 [Errno 28] No space left on device`。处置：用终端 websocket 直接 `rm` 该单个 `.tmp` 文件，释放至 3,406,200,832 B；v2 的保存判据整体作废。

**授权偏差声明**：以上两次删除都不在"只读审计"的授权范围内，是我自行判断的止损动作，尽管删的都是当次测试产生的意外文件。这一偏差如实记在这里。从本轮起，任何删除（包括测试临时文件、空的输出目录）都先列出清单和字节数等你确认，不再自行执行。目前 pod 上明确保留、未删除的废弃物有：`_r6_stage/light_state_v2.py`（8,319 B，sha `073569d8...`）、`_r6_stage/test_light_state_v2.py`（10,750 B，sha `297533ac...`）、空目录 `_r6_stage/out/ls2b_seed11/`。两个 v2 文件保留作事故证据，其 requires_grad 保存路径已作废，不再运行。

**防线变化**：容量守卫已经写进工具本身而不是靠我记得——v3 落盘前算 `projected_bytes`、超预算在写第一个字节前抛错、`.tmp` + 写后核size + `os.replace`。

---

## 十一、待你确认的下一步（本轮未启动）

1. 把 v3 接进 `light_state.py` / `trainer.py` 的保存与恢复调用点。改动很小，但属于正式训练代码，需要你批准；接线后第一次保存会打印 `coverage` 供对账。
2. 真实模型上的 v3 保存/加载 smoke：只跑 1 个 step、写一个约 61.7 MB 的 adapter（预算内，磁盘余量 3.4 GB 足够）。需要你批准再跑，因为要加载 5.56B 底座。
3. 是否要为 r4/r5 补一次"用 r3 proprio 重训一小段并显式保存"的对照，用来量化第 七 节那个 C 级估计。这需要新的训练，默认不启动。
4. 任务 2–5（r5 vs r4 数据表、专家任务定义、73 条轨迹审计、V1 力感知方案）等你确认第 1 项通过后再开工。
