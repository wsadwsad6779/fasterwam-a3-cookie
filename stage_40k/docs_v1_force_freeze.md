# V1 力感知架构冻结记录（2026-11-10）

这是 40,000 步正式训练**启动前**的冻结点。此后任何改动核心结构、力通道、窗口长度
或归一化常量的决定，都等于作废这次训练，必须重新走一遍本文列出的验证。

实验性组件，不是官方 FasterWAM 或 FAWAM 的实现：视觉世界模型、SparseMoT、
Lite KV-Fusion、调度器与动作专家几何全部沿用上游，只在动作专家上新给力通路。

## 1. 力通道（实测决定，推翻了审计报告里我自己的建议）

`channels = [0, 1, 2, 12, 13, 16]`，18 通道表里的：

| 列 | 名称 | 单位 | 参考系 |
|---|---|---|---|
| 0,1,2 | 左腕三轴力 | N | 法兰坐标系（不是世界系） |
| 12,13 | 左指内/外触点法向力 | N | 触点法向 |
| 16 | 左夹爪作动力合力 | N | 标量 |

- **没有力矩通道**，因此模型描述一律写 "force"，不写 "wrench"。力矩在 3,4,5（左）/
  9,10,11（右）。
- 审计曾建议改用 `[0,1,2,3,4,5]` 取"完整六轴"。18 通道实测把它否掉了：左腕力矩
  对接触几乎不区分（残差 std 相对零载偏置太小），右臂通道在 100 条里是常数，
  而指垫法向力是抓取接触最直接的信号。换成力矩等于用两个死通道换掉两个有效通道。
- 未做重力补偿；`make_force_calib.py` 记录的 mean 吸收了静态偏置（含 `L_wrist_fy`
  实测 −10.72 N 的重力分量），所以模型看到的是去偏后的量。

## 2. 两路力输入，各自的所有权

| 路径 | 内容 | 归一化 | 所有者 |
|---|---|---|---|
| proprio token | `concat(state8, force6)` = 14 维 | processor 的 14 项 state normalizer（min/max） | 数据侧 `a3_fw_v1g/dataset_stats.json` |
| 力流 P2/P3 | 历史 16 帧 + 未来 32 帧原始牛顿值 | `ForceHistoryEncoder.normalize`（mean/std） | 模型 buffer，来自 `force_calib_v1.json` |

14 维 proprio 是 §三-A 的要求。`base_load_guard` 负责把上游 `[4096, 8]` 的
`proprio_encoder` 扩到 `[4096, 14]`：前 8 列逐元素相等、后 6 列**严格为 0**，
三点都在装载后当场断言，所以第 0 步的 proprio token 与 8 维模型 bit 相同。

## 3. 注入点（按实际计算图定，不照搬 FAWAM）

- **P2 力历史编码**：2 层因果 self-attention（`hidden=256`, `heads=4`），窗口
  `hist_len=16`（0.8 s @20 Hz），**左填充**，mask 标记尾部有效段；行内偏移
  embedding 按"距当前步的绝对距离"取表尾，所以短窗口的最后一行永远是当前帧。
- 历史 token 与（加噪后的）未来力 token 拼成 48 个 key，经
  `ForceTokenCrossAttention` **加性**注入 32 个动作 token。
  不用 AdaLN-Zero 的原因：`SparseActionDiT.pre_dit` 的 `t_mod` 形状是
  `[B, 6, D]`，没有时间轴，逐步力无法搭这条调制路径。token 注入会经
  self-attention 传到全部 30 层（含 22 个不读 context 的层），且不改
  `action_seq_len`、RoPE 缓存和 `video_kv_fusion` 的 mask。
- **P3 未来力预测**：MoT 之后第二次注入（避免力条件被 30 层自注意力稀释），
  再接 `ForceVelocityHead` 输出 `[B, 32, 6]` 的速度场。训练目标是
  `train_action_scheduler.training_target(force, noise, t)` —— 和动作**同一个**
  flow-matching 速度目标、同一个共享的 `t`、两份独立噪声。不是外挂 MSE。
- **零初始化**：`inject.attn.out_proj`（weight+bias）与力头最后一层
  `Linear(hidden,6,bias=False)` 置零。推论（梯度探针按此检查第 1 步而非第 0 步）：
  第 0 步只有这两个投影拿到梯度，编码器从第 1 步开始才有。
- **零初始化的第二个推论（本轮实测发现）**：注入门在 `pre_dit` 里，它的梯度只能
  来自动作损失。力损失那条路是 `tokens → 注入 → 力头(末层为 0)`，对末层之前的
  任何张量求导都要乘一个 0，所以只算 `loss_force` 时 `out_proj` 的梯度**恒等于 0**，
  门永远开不了。两个初始化点串联的结果是编码器/门的可训练性依赖联合损失：一旦有人
  把动作损失 detach（或对力分支单独反传），整条历史编码器会在 40k 步里静默冻结，
  而 `loss_force` 仍在下降，看起来一切正常。因此 T2/R4 的梯度阶梯用
  `λ_a·loss_action + λ_f·loss_force` 联合反传，并把"只用力损失时门梯度必须恰好为 0"
  写成负例控制。

## 4. 落盘与恢复

力模块挂在 `action_expert.force_conditioner` 上，即 `mot.mixtures.action` 内部，
因此 `save_checkpoint` 的 `mot.state_dict()` 天然包含它和三个校准 buffer
（`force_mean`/`force_std`/`stats_are_set`），发布权重自包含。
轻量状态按**优化器成员**挑选，训练器在建 optimizer 前有硬闸：任何力张量没进
optimizer 就直接报错（r4/r5 丢 `proprio_encoder` 的同类故障）。
buffer 不是优化器成员，所以 `light_state_v3` 在 `trainer_state.json` 里存
calibration 指纹，恢复时与当前 config 构建出的 buffer 比对，不一致就拒绝对接。

**采样器偏移的量纲（本轮 R9 逼出来的不变量）**：`resume_batch_offset` 存的是
**批数**，恢复时按 `offset * batch_size * num_processes` 换算成样本数跳过。这条换算
只在"当前 pass 里前 offset 批全是满批"时精确。`drop_last=False` 意味着每个 pass
最后一批可能是短批（39481 个窗口 / 6 = 6580 满批 + 1 个 1 样本短批），所以
`batch_in_epoch` 必须在 **roll 到新 pass 时**才清零，而不是在发出短批时清零
（trainer.py:701-706 就是这个顺序）。按此顺序：短批计入当前 pass 的批数，下一次
`next()` 才触发 epoch++/offset 清零，跳过量最多多算出 `indices[>len] = []`，
即"本 pass 已到尾，直接开下一 pass"，与不接断点完全一致。若在发短批时就清零，
短批那 1 个样本会被重复训练。测试替身必须照抄这个顺序，否则是测试自己的假故障。

**力输入 API 的一个不对称（记录，不是 bug）**：`_prepare_force_inputs`（训练侧）
要求 `[B,T,6]`，传未打包的 `[T,6]` 会直接报错；`_build_force_infer_state`（推理侧）
对 2 维输入自动 `unsqueeze(0)`。所以 R7 里同一条码路两条入口的入参形状不同。

## 5. 状态分类（按 §十五 的三档）

**已实现并测试通过** —— 在 pod 上跑过数值验证、且验证本身可复现：14 维 proprio
语料（`a3_fw_v1g`，float32 往返 1e-7）、全局零点替代前 20 帧估计、文本嵌入缓存，
以及 §6 表里全部 8 行（P2 编码器、P3 联合流与两阶段注入、mask 极性、
泄漏闸门、训练/推理同一条码路、light-state v3 往返与指纹）。第 6、8 行原来只剩
"必须真机才能确认"的那块，已由 task #65 的 6 步实机演练补齐，见 §6 末尾。
`base_load_guard` 也不再是"只在真实 `train_seg.py` 启动时才走"的未验证代码：演练里
它在真实训练（两段）、真实导出和真实部署重载上都触发过，每一次都 `absent=0`、
`shape_mismatch=0`、`value_verified=True` 且逐元素比过（C4a/C4b/C4c）。

**已实现但尚未充分验证**：P4 `force_feedback` 钩子（默认 None、不接管主路径，
尚无真实修正标签可用）。它不碰主训练路径，因此不阻塞 40k。

**因数据条件不足而未实现** —— P5 在线修正器的监督训练：语料里没有
`(policy_action, corrected_action)` 配对，DAgger 数组连 force 列都没有。
按 §三-D 不伪造标签。P4 只做接口：`force_feedback` 形参**默认 None**，
返回原样动作；传了 callable 才校验形状/有限性后替换 `out["action"]`，
同时保留 `out["action_raw"]` 和 `force_feedback_applied`。

## 6. 冻结前必须全绿（§九准入子集）—— 2026-11-10 实测结果

跑在实例 `rgapi-one-click-24316-57b6b870` 上，解释器 `/opt/venv/bin/python` 3.12，
代码由 `PYTHONPATH=/workspace/_v1_stage/fw/src` 提供。两层：
`smoke_force_v1.py`（T1–T6，小模型）与 `smoke_force_v1_real.py`（R1–R10，真实
18 维力表、真实 39481 窗口语料、真实 30 层动作专家）。报告
`smoke_force_v1_real_report.json`（11 passed / 0 failed）。

| # | 准入项 | 结论 | 实测证据 |
|---|---|---|---|
| 1 | 第 0 步等价（零初始化） | ✅ | R3：`pre_dit` 缓存张量、30 层真实前向、post-DiT 注入三处**逐元素相等**；把 `out_proj` 改成非零后输出立刻变化（证明不是死代码） |
| 2 | 新模块拿到有效梯度 | ✅ | R4：第 0 步只有 `inject.attn.out_proj.{weight,bias}` + `force_head.linear.2.weight` 动；3 步联合反传后 42 个力张量（4,258,304 元素）全部非零梯度；**只算力损失时门梯度恰好为 0**（负例） |
| 3 | 未来力标签无泄漏 | ✅ | R2：真实窗口 256 步与独立读路径 `force_rows()` 差 `max_abs_diff = 0.0`；标签左移一帧被拒；掩码尾部塞进非零值被拒。T4/R2b 另证零点用的是全局参考（残差 7.4e-08）而非逐回合中位数（6.2e-02，第 0 步不可能有） |
| 4 | mask 极性 | ✅ | T3 + R1：600 个抽样窗口里 20 个左垫、0 个右垫，全部符合"历史=尾部有效后缀 / 未来=头部有效前缀"，垫位严格为 0；T6 在 3 个窗口位置校验与数据集切片同构，反转掩码被编码器拒绝 |
| 5 | 冻结/优化器一致 | ✅ | R8 负例：把一个基座张量置为 `requires_grad` 而不放进 optimizer，`coverage` 必须数到它（就是 r4/r5 丢 `proprio_encoder` 的捕手）；本轮另按真实策略先冻结再断言为 0 |
| 6 | 落盘往返 | ✅ | R8 + 6 步实机演练（stage 4/5，11 项全过）。`export_merged_v3.py` 从 v3 adapter 出发，按部署同一条路径先加载冻结基座、再叠 adapter、最后合并：392 个被 LoRA 包住的矩阵逐个验证 `W + (B@A)*scaling` **逐位相等**，另外 1001 个非 LoRA 张量和 45 个力张量逐个验证**在合并中未变**；发布文件 11,126,493,579 B / 1438 张量 / 无 LoRA 残留 / step=12。全新进程用 `load_checkpoint(strict=False)` 重载：1438/1438 **逐张量等于文件**，42 个力参数**等于轻量状态**（先把 `action_expert.*` 映射到 `mixtures.action.*`，42 映到 42、无碰撞），3 个 calibration buffer **等于冻结 JSON 在模型 dtype 下的表示**。负例：改名张量被拒、力 buffer 值与文件声明的指纹不一致被拒、缺声明被拒、错声明被拒；正例：一份完整子集照常加载，证明闸门不是一律拒绝。 |
| 7 | 训练/推理归一化同一条码路 | ✅ | R7：同一原始窗口下 `_prepare_force_inputs` 与 `_build_force_infer_state` 的 history **逐元素相等**，反归一化误差 2.4e-07 N；并证明 proprio 的 min/max 流与力流的 z-score 是两套、没有互相套用。演练补上部署侧：发布文件自带 buffer，`normalize()` 只读模型自己的 buffer，配置里 `norm_stats.file` 只有一处且就是被冻结的那份 JSON |
| 8 | 八段 = 一条 40k 曲线 + 正确续训 | ✅ | 真实 `train_seg.py` 跑 0→6，再从轻量状态续 6→12（stage 1/2 RC 0，17 项全过）：两段都正好停在 `FW_STOP_STEP`（`hit_stop`），账本里 `scheduler_total` 只有 40000 一个值；段 2 的 `segment_start.lr` 与段 1 的 `segment_end.lr` **逐位相等**（3.498500000000001e-07）且 `prev_segment_end_step=6`；账本 LR 与用真实 `_build_scheduler` 走出的 40,000 点参考序列在 step 6/12 **逐位相等**；保存的 scheduler 就是那条曲线（`last_epoch=6`、`_milestones=[2000]`、cosine `T_max=38000`、`eta_min=1e-6`），不是重建的短 cosine；`global_step/epoch/batch_in_epoch` 连续；采样器段 2 从段 1 置换的第 36 个窗口原位续上（比对 102 个位置，覆盖全部 36 个已训练窗口），跨续训 0 个窗口重复；`loss_force` 12 步全部存在、有限、互不相同，`force_valid_step_frac=1.0`。八个边界 LR 严格递减：5k 9.8485e-05 → 40k 1.0000e-06 |

**6 步实机演练（task #65）实测数字** —— 2026-11-10 在同一实例上跑
`rehearsal65.sh`，真实 11 GB 基座、真实 39,481 窗口语料、真实 30 层动作专家，
`FW_STOP_STEP` 6/12 两段，发布再重载，最后由 `verify_rehearsal65.py` 核对账本：
stage 5 **11 项 11 过**、stage 6 **17 项 17 过**（`rehearsal65/rehearsal_report.json`
与 `rehearsal65/publish_verify.json`，`failed` 都是空）。

- 力模块确实学到东西：模型 2417 个命名参数、优化器 828 个成员（784 个 LoRA +
  44 个非 LoRA，其中 42 个是力张量）。`trainer_state.json` 现在记
  `optimizer_member_order`，按这个下标回到 `optimizer.pt` 的 `exp_avg`，828 个槽位
  全部**先验形状再读矩**（`shape_verified_slots: 828`、`shape_mismatched: []`），
  42 个力张量的 `exp_avg` 与 `exp_avg_sq` absmax 全部 > 0。样例：
  `force_head.linear.2.weight` 的 `exp_avg` absmax 1.59e-02、`force_head.linear.0.
  weight` 1.44e-07、`linear.0.bias` 2.64e-08 —— 跨度很大，但没有一个是 0，
  `force_grad_zero_tensors` 是空表，两次保存（step 6 与 step 12）都带 42 个力张量。
  这是"非零有效梯度"的直接证据；权重增量不能当证据，见下条。
- 两个必须写下来的 dtype 事实。(1) warmup 期 step 6 的 LR 是 3.5e-07，比一个
  bf16 ulp 还小，6 步内 `proprio_encoder` 那种 4096 宽的行在权重上**动不出来**，
  所以"练过"只能看优化器矩，不能看权重差。(2) 力的 `force_mean`/`force_std`
  buffer 是**模型 dtype（bf16）**，于是生效常量是冻结 JSON 舍到 bf16 的结果：
  mean 最大偏 0.0255 N（absmax 10.94），std 偏 1.9e-03。训练和推理读同一个
  buffer，所以两侧一致；而指纹哈希的是舍入后的值，这让它顺带成了 dtype 闸门
  —— 用 float32 建的部署模型加载 bf16 发布的权重，指纹不等，直接拒
  （演练里 `perturbed_force_buffer` 就是这个机制的实测负例）。
- 两类 checkpoint 实测体积：轻量状态 210,967,829 B/节点（adapter 70,257,165 +
  optimizer 140,644,485 + scheduler 1,675 + rng 14,557 + 状态 JSON 49,944），
  八个节点 1,687,742,632 B；合并发布 11,126,493,579 B，八个 89,011,948,632 B。
  当时 /workspace 剩 47,019,446,272 B：**轻量状态八份全留装得下，合并权重八份
  装不下**。所以正式跑的驱动 `drive40k.py` 一个合并文件都不写，合并与上传是
  每节点单独的动作，必须"上传→回读→能加载→才删本地"。
- 发布链路本身：`export_lora_merged.py` 读 `payload["mot"]`，而 v3 adapter 根本
  没有 `mot` 键，所以力模型的发布物此前不存在。新增
  `fw/scripts/export_merged_v3.py` 承担这条路（顺序是硬的：载基座 → 叠 adapter
  → 合并；先叠后载会把训练好的张量盖掉）。
- 正式启动前的预检 `drive40k.py --plan` 10 项全过，八个节点的边界 LR 与演练
  里那条参考曲线逐项对上（5k 9.84853132e-05 … 40k 1e-06），`resume` 串成一条
  链：段 1 从基座起，段 n 只能从 `step_(n-1)*5000` 的轻量状态起。

**本轮改动过的东西（必须一起记账）**：
- 真 bug 1 个：`A3VideoDataset.force_rows()` 把行下标数组和通道下标数组同时喂给
  `[]`，NumPy 按逐元素高级索引要求两者可广播，`[N]×[6]` 直接 `IndexError` ——
  也就是说泄漏闸门的独立读路径从来没跑通过。已改为先取行再选列（R2 转绿）。
- 真 bug 2 个（上一轮接线留下的半截代码）：
  (1) `FastWAM` 里被调用的 `_force_calibration_audit` 从来没被写出来，部署重载
  一走就是 `AttributeError`；补齐后它承担"声明的力单位 ≠ 本模型标定"的硬拒，
  并且只在真恢复出力权重时才跑，不误伤纯基座加载。
  (2) `load_adapter_v3` / 数据侧的采样字段直通：`robot_video_dataset` 与
  `a3_video_dataset` 原来会把 `_extra_sample_fields()` 生成的力窗口丢掉，训练侧
  于是永远看不到力，`loss_force` 也不会出现。已加直通钩子，并由
  `verify_dataset_force.py`（stage 0b，62 项全过）在真实数据集实例上把这条缝钉住。
- 记账缺口 1 个：`optimizer.pt` 的 `load_state_dict` 是按**下标**而不是按名字复位
  状态的，所以力通道/LoRA 目标一旦变化，Adam 矩会被静默安到别的张量上。现在
  `trainer_state.json` 记 `optimizer_member_order`，续训时不一致就拒（恢复路径
  `_check_member_order`，`drive40k` 的 `--next/--all` 也独立比一次）。
- 测试替身的错 10 处（改的全是测试，没有一处放宽判据）：R1 反推历史左端点少算
  一帧（`lo = s-nl+1`，含当前帧）；R5 复用了被 R4 步进过的缓存模型；R7 给训练侧
  入口传了未打包的 `[T,6]`；R8 忘了按真实策略冻结基座，于是数到 558 个"可训练但
  不在 optimizer"；R9 的 `_SimLoop` 让一个 batch 跨过 pass 边界、且在构造时就急切
  建迭代器；R10 拿"下一步的 LR"和"上一步的 LR"比，差一步；演练 harness 里
  C6 按行扫日志（rich 会把一条记录折行，`loss_force`/`lr` 全丢），C1 用
  `lr_climbed` 当判据（warmup 期恒真，等于没测），C4 数了别的阶段写的
  base-load 事件（导出也会加载基座），C5b 拿 `(pass, index)` 对纯 index 比、
  又比了两段不等长的尾巴；`verify_publish65.py` 的 P3a 直接跨名字域比张量、
  P3b 要求 bf16 buffer 与 float32 JSON 在 1e-5 内相等。
- 判据改写 1 处，需要说清楚：原来 P6 断言"把某个基座张量的值改一下，
  `base_load_guard` 必须报错"。这条断言本身是错的——guard 比的是"模型 vs 文件"
  的保真度，一个自洽的错误文件本来就该被忠实加载。改成守它真正承诺的两件事
  （无处安放的重命名键、力 buffer 值与声明指纹不符），并把"错误基座值会被忠实
  加载"作为 P6d 显式记录：能发现它的是 P2/P3a 那种"和独立来源比"的检查。
- 结论：8 项准入全部按 §十五 第一档落地。第 6、8 项剩下的那块已经用真实
  trainer、真实基座、真实发布文件补完，不再是替身。

第 1、2 条没有通过之前，本文不允许被引用为"已冻结"。

