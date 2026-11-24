# C-20261124-01 · 高优先级：夹爪平衡采样代码部署漂移（pm1 语义回退）

发现时间：2026-11-24T01:15Z
状态：OPEN（等待 task7 修复并自证）
影响：若用当前 pod/工作区副本启动任何训练，平衡采样会**静默失效**，close_frac 指标含义反转。

## 证据

### 1) HEAD（V3 正确版本，commit e74e1ea）
- `scripts/t7_data.py`：`phase_open[ep] = np.flatnonzero(g >= 0.8)`（V3 连续标签，1=开）。
- `scripts/t7_train.py`：`close_frac = (action[:, :, 6] <= 0.6).mean()`，`open_frac = (action[:, :, 6] >= 0.8).mean()`。

### 2) 当前 pod 副本（已被回退成 pm1）
- `/workspace/t7_data.py` 与 `/workspace/_t7_data.py`（8782 B）：
  `phase_open[ep] = np.flatnonzero(g <= -0.8)`；`phase_rise` 同样用 `g <= -0.8`。
- `/workspace/t7_train.py` 与 `/workspace/_t7_train.py`（8034 B）：
  `'close_frac': (action[:, :, 6] >= 0.8).mean()`，与 `open_frac` 完全同义（指标名与含义不符）。

### 3) 本地工作区同样存在未提交的回退
`git diff` 显示 `p_items/scripts/t7_data.py` 与 `t7_train.py` 被改成上述 pm1 版本（未提交，HEAD 仍是正确版）。

### 4) 为什么能确定 V3 run 用的是正确版
- V3 训练日志每步 `close_frac` 与 `open_frac` 数值不同（例：step 70 = 0.648 vs 0.301）；若用当前 pod 版，两者必然逐位相等。
- 00:45Z 的采样探针（`SAMPLER_BALANCE_CHECK.json`）显示平衡开启后 open(≥0.8) 帧占比 4.77% → 20.84%（4.37×），说明当时 pod 的 `t7_data.py` 是 HEAD 正确版；回退发生在该探针之后。

## 要求的修复（task7）
1. 把 HEAD 的 `p_items/scripts/t7_data.py`、`t7_train.py` 逐字节部署到 pod：`/workspace/t7_data.py`、`/workspace/t7_train.py`，并同步覆盖 `_t7_*` 旧副本（或删除旧副本避免误用）。
2. 运行 `_t7_gripper_sampler_probe.py`（V3 标签）：balanced 的 `mean_open_frac_ge0.8` 必须 ≥3× unbalanced，且 ≥10/20 个 batch 的 open 帧 >10%。
3. 跑 1 个真实训练步，证明 metrics 中 `close_frac`（≤0.6 口径）与 `open_frac`（≥0.8 口径）不再相等。
4. 提交修复并附 sha256；不得提交 pm1 回退版本。

## 监督复验方式
- 读 pod 两处文件对应行 + sha256；重跑探针；抽 1 个 metrics 行核对两指标定义。
- 未通过前 G-D 维持 blocked，训练禁止启动。