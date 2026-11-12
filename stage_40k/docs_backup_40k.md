# 40k 节点备份与恢复验证记录（2026-11-10）

对象是 `/workspace/_v1_stage/run40k/checkpoints/light/step_0NNNNN` 这 8 个节点。实例是租的，
所以只存在于 `/workspace` 的节点等于不存在的节点。上传本身不难，难的是三件 HTTP 200 证明不
了的事：

- **V** 远端字节就是本地字节；
- **W** 远端那一套是完整恢复所需的一切，不只是权重；
- **R** 一个全新进程能从远端下回来的字节里接着训练。

工具是 `fw/scripts/backup40k.py`（一次一个节点）+ `fw/scripts/watch_backup40k.py`（值守）。
落点：`/workspace/_v1_stage/backup40k/`（`ledger.jsonl` 台账、`watch.jsonl`/`watch_heartbeat.json`
值守心跳、`report_step*_deep.json` 逐节点报告、`write_probe4.json` 传输层测量）。仓库
`wsadwsad6779/fasterwam-a3-force-40k`，private=true，前缀 `v1_force_40k/`；演练用的
rehearsal 节点走 `v1_force_rehearsal/`，不进生产命名空间。

## 1. 传输层：先测，再定方案

这台实例只有 egress 代理 `http://notebook-egress-proxy.amd-oneclick-lablab.svc.cluster.local:3128`
可用（`--noproxy` 直连全部不通）。代理对不同上传目的地的策略是硬差别：

| host | curl 结果 | 含义 |
|---|---|---|
| `cas-server.xethub.hf.co` | 404（1.26 s） | 隧道通，路径问题 |
| `transfer.xethub.hf.co` | 403（1.28 s） | 隧道通 |
| `hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com` | CONNECT 403 / 000 | **传统 LFS 数据面被堵死** |
| `cdn-lfs-us-1.hf.co` | 403（0.19 s） | 下载 CDN 也不通 |
| `cdn-lfs.hf-mirror.com` | 403（5.12 s） | 同上 |

服务端对二进制扩展名一律强制走大文件通道，把 `.pt` 改名成 `.datx` 试过，仍然落到 S3 那条被堵
的路。所以这台机器上唯一可用的 bulk 数据面是 **Xet**。逐案例、每案例一个进程实测（因为
`HF_ENDPOINT` 与 `HF_HUB_DISABLE_XET` 在 `huggingface_hub` import 时就冻结）：

| 案例 | 结果 |
|---|---|
| Xet 8 MB | 23.05 s |
| Xet 70 MB | 39.23 s（1.87 MB/s） |
| regular commit 64 KB | 2.23 s |
| LFS（走 S3）任何体积 | 403，失败 |
| 真实节点 211,260,295 B / 22 文件（首传） | 48.0 s（4.4 MB/s） |
| 同一节点内容未变的再传 | 14.5 s（命中 dedup，14.5 MB/s） |

体积账（单节点 v3 轻量状态）：`adapter.pt` 70,257,165 / `optimizer.pt` 140,644,485 /
`scheduler.pt` 1,675 / `rng.pt` 14,557 / `trainer_state.json` 49,947，节点合计 210,967,829 B；
8 个节点 1,687,742,632 B ≈ 1.69 GB。加代码、配置、校准、清单后每次上传 22 个文件、
211,260,898 B。**结论：全量 8 节点对外备份约 1.7 GB，是可行体积；真正的大件是 11 GB 基座
权重，它不需要重复上传，见 §5。**

旧实例上 hf-mirror 的 Xet 在 dedup 返回 `found` 之后会静默冻结（见记忆「HF 上传 dedup 挂死」）。
值守脚本因此给每次备份 `ATTEMPT_TIMEOUT_SECONDS = 3600` 硬超时，超时记 `rc=124` 并重试，绝不
因为超时去动本地文件。

## 2. 校验链：每个判据对应一个具体断言

| 档 | 判据 | 断言的是什么 |
|---|---|---|
| U1–U5 | 节点完整、与冻结契约一致、记录 lr 等于曲线、基座已钉、上传树就绪 | 上传前本地是干净的 |
| U7 | 校准文件 sha256 ↔ 节点指纹 ↔ dtype 舍入闭合 | 力归一化的来源能自证（唯一命中 bfloat16） |
| U6 | 仓库存在且 private | 权限 |
| U9 | 逐个文本文件 grep token | 凭证不落进任何会被上传的文件 |
| V1 | `upload_folder` 被接受 + 实测吞吐 | — |
| V2 | 远端树逐文件哈希（LFS 用 `lfs.sha256`，普通文件用 git blob sha1）与本地一致 | 远端就是这份 |
| W1 | 全部文件重新下载并逐字节 sha256 比对 | 出得去也回得来 |
| R1–R6 | v3 格式与步号、adapter 覆盖 828 张量含 42 个力张量、优化器动量齐、计数与 sampler 偏移、校准指纹 | 恢复的不是半成品 |
| W2 | 基座权重在别处以 sha256+字节数命中 | 换机器也能把 8 个模型合出来 |

R7 是 R1–R6 的汇总门禁。只有 `--deep`（W1+R）全绿的节点，才允许进 §6 的清理候选清单。

## 3. 本轮踩到的坑（现象 → 根因 → 现在的写法）

1. `upload_file(path_or_fileobj=...)`，不是 `path_or_file`。第一次探测因此只是 TypeError，
   关于传输层的结论一个都没测到。
2. `requests` 打 `huggingface.co` 拿到 503/超时，而同一时刻 `curl` 是通的。控制面 API 固定走
   `https://hf-mirror.com`。
3. 容器环境导出 `HF_HUB_DISABLE_XET=1`，client 于是选了被代理封死的 S3 路；只在脚本里
   `del` 掉这一行不够（进程继承 env），必须显式设成 `"0"` 且在 import `huggingface_hub` 之前。
4. huggingface_hub 0.36.2 里 `RepoFile.type` 是 `None`。按 `type == "file"` 过滤远端树，结果
   是 22 个文件全被判成"远端缺失"。类型要看类名，git blob 在 `blob_id`，LFS 摘要在
   `lfs.sha256`（不是 `oid`）。`expand=True` 单独加并不能救。
5. `hf_hub_download(local_dir=...)` 会把仓库前缀留在落盘路径里（`dest/v1_force_40k/nodes/…`）。
   自己拼 `dest / rel` 会 FileNotFoundError，然后把这个错算到 checkpoint 头上。现在直接用
   client 返回的路径。
6. 预签名 S3 URL（带 `X-Amz-Signature`）会出现在 transport 的异常文本里。`mask()` 现在抹掉
   token 字面量和 URL query；报告与台账写盘前再各 grep 一次，命中就拒绝写。
7. 根目录 `README.md` 不是节点载荷，`rel.split("/")[-2]` 在它上面 IndexError。改用
   `_parent_dir()` / `_is_node_payload()`。
8. **差点误报为训练阻断的那条：「`scheduler.pt` 恢复后 lr 掉回 warmup 底部」是测试脚本错，不是
   trainer 错。**
   - 现象：R2 在 step 13 报 `5.3846e-08`，曲线是 `6.99675e-07`。
   - 根因：`load_light_state` 的顺序是先恢复 optimizer、再 `scheduler.load_state_dict`。
     `Optimizer.load_state_dict` 会把保存的 `param_groups` 里除 `params` 外的键全部写回活动
     组，**lr 就在其中**；scheduler 文件只负责让后续步进接上曲线。旧 R2 只 load scheduler 就
     开始 `step()` 回放，torch 2.9.1 在没有 optimizer lr 的前提下把 lr 重算成接近 warmup 底
     的值（实测 `5e-08*(1+n/13)`）。回放本身也是多余的：它步的是已经前进过的 SequentialLR。
   - 现在 R2 的三段断言：① 保存的 scheduler 标量字段与"未中断跑到同一步"的字段 diff 必须为
     空（含 `_milestones=[2000]`、`base_lrs=[1e-4]`、`sched0.total_iters=2000`、
     `sched1.T_max=38000`、`sched1.eta_min=1e-6`）；② `optimizer.pt` 里存的 lr ==
     `lr_at_local(step)`；③ 按 trainer 顺序冷恢复后，接下来 3 步各自使用的 lr == 曲线在该步
     的值。
   - 交叉验证：从 step_6 节点冷恢复算出的 step 7..12 lr 序列，与真实 GPU 第二段日志逐行吻合
     （预测 3.4985e-07 / 3.99825e-07 / 4.498e-07 / 4.99775e-07 / 5.4975e-07 / 5.99725e-07 /
     6.497e-07，日志按 3 位有效数字打印为 3.498500e-07 / 4.00e-07 / 4.50e-07 / 5.00e-07 /
     5.50e-07 / 6.00e-07 / 6.50e-07）。
9. 顺带记一条 lr 约定，免得下次又推半天：训练循环是
   `optimizer.step(); scheduler.step(); global_step += 1`，所以 `global_step == N` 时
   scheduler 的 `last_epoch == N`，而第 N+1 步用的 lr 就是曲线上的 `curve(N)`。

## 4. 值守脚本的规则

`watch_backup40k.py --loop` 是独立进程，不改正在跑的 driver（`drive40k.py --all` 已把模块加载
进内存，此刻改 driver 只影响下一次事故，不影响这一场）。它存在的理由就是这几条：

- 节点必须在两次相隔 60 s 的轮询里 (文件名, 大小) 完全一致、且 `trainer_state.json` 已落盘，
  才开始上传（trainer 逐文件写 `.tmp` 再 rename，目录整体不是原子的）；
- 每个节点起一个全新的 `backup40k.py` 子进程，torch 段错误带不走值守；
- 失败记台账、重试至多 3 次、每次退避 300 s、单次硬超时 3600 s；**任何情况下都不删、不移、
  不改本地 checkpoint**；
- 台账里 `verify_only` 的记录不算"deep 已验证"，不会让节点被跳过；
- 单实例锁 `watch.lock`，拒绝双开重复上传。

启动命令（写在脚本 docstring 里）：

```
cd /workspace/_v1_stage/fw
mkdir -p /workspace/_v1_stage/backup40k
setsid nohup env PYTHONPATH=/workspace/_v1_stage/fw/src HF_TOKEN_FILE=/root/.cache/hf_token \
    /opt/venv/bin/python -u scripts/watch_backup40k.py --loop > \
    /workspace/_v1_stage/backup40k/watch.log 2>&1 &
```

（`-u` 是必要的：这个循环要跑两天，块缓冲的日志看起来和死掉的一模一样。2026-11-10T14:01:50Z
以 pid 39283 启动，心跳 `watch_heartbeat.json` 当时为 `state=idle, nodes_seen=[]`。）

### 4.1 每段边界检查：`segment_check.py`

任务书要求"每段结束检查 global_step、scheduler、sampler、loss、校准指纹和参数完整性"。
driver 本身已经会在下一段启动前拒绝一个与计划不符的前驱节点，所以这个脚本不是闸门，而是
**记录**：每一项都从第二个来源重新推导一遍（seg 台账、节点自身、训练日志、上传用的校准
JSON、冻结的 rehearsal 契约），把两边怎么对上的写进报告。它只读，不联网，不碰 GPU，不改
checkpoint 目录，因此下一段已经在跑的时候照样可以执行。

```
cd /workspace/_v1_stage/dev            # 40k 期间只有 dev/ 允许改动，fw/ 是冻结的
PYTHONDONTWRITEBYTECODE=1 /opt/venv/bin/python segment_check.py \
    --step 5000            # 报告落在 /workspace/_v1_stage/backup40k/segment005000_check.json
```

`PYTHONDONTWRITEBYTECODE=1` 不是可选的：这个副本 import 的是 `fw/scripts/` 里的 `backup40k`
和 `drive40k`，不留这个环境变量就会往正在被主训练 import 的目录里写 `__pycache__`。

六个判据（2026-11-10 用 rehearsal 的 step_000012 节点在 dev/ 里实跑，6/6 PASS，报告
`backup40k/segment_check_dry4.json`）：

| 判据 | 断言 | 第二个来源 |
| --- | --- | --- |
| S1 global_step | 台账 `segment_end`==节点 `trainer_state.json`==日志最后一条进度行；`hit_stop` 为真；台账里那个 lr 是否上行的布尔（`lr_moved_up`，老脚本叫 `lr_climbed`）必须**存在**且等于用冻结曲线算出来的 `curve(end) > curve(start)`——段 1 含 2,000 步 warmup 所以本该为真，段 2~8 落在 cosine 下降段所以必须为假 | seg 台账 + 日志 |
| S2 scheduler | `scheduler.pt` 的 `last_epoch`==该步；`optimizer.pt` 里存的 lr、`extra.lr`、台账 lr 三者都等于同一条 40k 曲线在该步的值（相对 1e-12）；`base_lrs`（含内层两个阶段）==[1e-4]；milestone==2000 | 曲线由 trainer 自己的 `_build_scheduler` 现推 |
| S3 sampler | `resume_batch_offset`==`batch_in_epoch`；`global_step == epoch*batches_per_epoch + batch_in_epoch` 且所有节点解出同一个整数；`batch_size`==6、`num_processes`==1、seed 不变；解出的 epoch 长度还要等于语料算出来的那个 | `seg1.log` 里 trainer 自己打印的 `Train/val dataset size` |
| S4 loss | 无 NaN/inf；`force_valid_step_frac` 最小值==1.0；日志里的 lr 与曲线的偏差 ≤2e-3（打印只给 3 位有效数字，四舍五入本身就有 5e-4，真出问题差的是数量级）；段末 20 条均值 / 全段均值 ≤3；跨段时上一段尾与下一段头的比值 ≤3 | `rich` 进度行按记录重新拼接后解析 |
| S5 校准指纹 | 节点指纹==冻结契约指纹，且上传的校准 JSON 在**恰好一种** dtype 舍入下同时复现 mean/std 两个 sha256（本轮为 bfloat16） | `force_calib_v1.json` 现读现算 |
| S6 参数完整性 | format/`adapter_bytes`/`coverage`/`optimizer_member_order` 逐项等于契约；`files` 声明的四个 .pt 与五个状态文件都在 | rehearsal 契约 |

**每个 epoch 到底多少 optimizer steps——实测，不是拿帧数除。** 语料 `a3_fw_v1g` 有 42,681
**帧**，但模型吃的是 33 帧滑窗，`A3VideoDataset` 按 `sum(ep_len - num_frames + 1)` 计窗口，
100 条 episode ⇒ `42,681 - 100×32 = 39,481` 个样本。三个来源互相印证：trainer 启动即打印
`Train/val dataset size: 39481/39481`；`dev/loader_len_probe.py` 用真实的
`ResumableEpochSampler` + `DataLoader(batch_size=6)`（`drop_last` 保持默认 False）跑出来
`len(sampler)=39481`、`len(train_loader)=6581`、末批 1 个样本、完整过一遍 6,581 个 batch /
39,481 个样本。`gradient_accumulation_steps=1` 且单卡，所以 micro step == optimizer step ⇒
**每个 epoch 6,581 个 optimizer steps**。之前按 42,681÷6 得到的 7,113 是错的（把帧当成了窗口）。

八个节点应当落在（`batches_per_epoch=6581`）：

| 节点 step | 5000 | 10000 | 15000 | 20000 | 25000 | 30000 | 35000 | 40000 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| epoch | 0 | 1 | 2 | 3 | 3 | 4 | 5 | 6 |
| batch_in_epoch | 5000 | 3419 | 1838 | 257 | 5257 | 3676 | 2095 | 514 |

epoch 跨界点 6581 / 13162 / 19743 / 26324 / 32905 / 39486 分别落在段 2、3、4、6、7、8 内部
（段 1 和段 5 一次都不跨），第一个跨界在第 **6,582** 步——即节点 1 之后 1,581 步，
在段 2 里面。S3 从段 2 起就会第一次真正解出 `batches_per_epoch`；段 1 的节点满足
`global_step == batch_in_epoch`（epoch 0），单点解不出唯一整数，所以那一段以日志行与语料
推导为准。任何一段解出 6,581 以外的整数，都意味着中途换过数据。

**第一个跨界已经真实发生（2026-11-10 16:57 UTC 读 `run40k/logs/seg2.log`）。** 段 2 是从
`step_005000` 续训的，`trainer_state.json` 里 `resume_batch_offset=5000`、`batch_in_epoch=5000`、
`epoch=0`；日志里最后一条 `epoch=0` 是 `step=6580/10000`，第一条 `epoch=1` 是 `step=6600/10000`
（`log_every=20`，所以 6,581 本身不打印）。跨界就落在 6,581 与 6,582 之间，和上面由
`39,481 ÷ 6` 推出来的 6,581 对上了：续训没有重放已经吃过的 batch，也没有换过数据。
这一步是观测，不是断言——段 1 的节点解不出唯一整数，S3 真正的第一次交叉验证只有在这里才发生。


## 5. 基座权重的独立恢复（W2 的依据）

`FW_BASE_CKPT = /workspace/FasterWAM/checkpoints/fasterwam_release/libero/step_021700.pt`，
11,117,679,929 B，sha256 `01be4efca565d1890fd8a7dc1507b01c00c8cc63c16ae252287492ba9ac7550d`。

- 我们自己的归档仓 `wsadwsad6779/fasterwam-a3-cookie-lora`（1490 条目）**没有**这个文件：4 个
  >1 GB 的是 `data/frames.dat`、`step_002000_merged.pt`、`trainer/state/step_002000/model.safetensors`、
  `trainer/step_002000.pt`——注意第二个 11,117,977,659 B，与基座只差约 300 KB，是完全不同的模型，
  这就是 W2 必须同时比字节数不能只比大小的原因。
- 上游公开仓 `hustvl/FasterWAM`（private=false，lastModified 2026-08-08）里
  `libero/step_021700.pt`：size 11,117,679,929 ✓，`lfs.sha256` == 上面那个 pin ✓。
- 所以 W2 现在的写法是：对 `BASE_SOURCES = (hustvl/FasterWAM, 我们的归档仓)` 逐仓扫描，命中要
  sha256 **和** 字节数同时相等，并把 `huggingface-cli download <repo> <path>` 写进报告。节点目录
  里同时上传 `base_weights.json`（版本、路径、字节数、sha256、hub_sources、合并配方）。
- 待用户决策（不阻塞训练）：是否要把 11 GB 基座再复制一份进自有私有仓（按 §1 的实测吞吐约
  40 分钟量级），以及上游权重的再分发条款是否允许这样做。

## 6. 状态分档（按任务书 §十五）

- **已实测**：传输层、单节点 22 文件 V2 全绿、W1 逐字节回下全绿、R1–R6 在真实节点字节上跑通、
  U7 校准闭合唯一命中 bfloat16、基座在上游公开仓 sha256+size 命中、lr 冷恢复序列与真实第二段
  日志逐行一致。
- **推理/接口层面已验，但本会话未重复执行**：v3 adapter→merged 发布权重往返、6 步实机演练。
- **尚未证明**：step 5000 首个生产节点的 `--deep`（含真实下载+重新加载）要在节点落地后由值守
  脚本完成；60 回合 B0 评估未跑；任何云端冗余文件的清理都还没做，也不会在 §2 的 deep 判据全绿
  之前做。
