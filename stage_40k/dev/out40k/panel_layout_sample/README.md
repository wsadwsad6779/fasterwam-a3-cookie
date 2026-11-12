# 面板录像布局样片（非真实回合）

生成命令（本地 CPU，`py -3.13`，无权重、无 MuJoCo）：

```
cd _v1_stage/dev/eval40k
py -3.13 panel_sample40k.py
```

走的是正式录制路径 `record40k.PanelSink.add_frame` → `panelvideo40k.compose_frame`，
所以这份样片的版式与 pod 上真实录出来的文件同一份代码；不同之处在于力、动作链与相机画面
全部是构造的。它只回答「格式对不对」，不回答「策略为什么失败」。

## 文件

| 文件 | 内容 |
|---|---|
| `LAYOUT_SAMPLE_synth_300f_1664x576_20fps.mp4` | 300 帧 / 15 s，1664×576，20 fps |
| `sample_frame_000.png` | reset 帧（`frame 0`）：两路相机与四路指尖力都在，但**没有一行动作数字** —— 策略此时还没动过，画一行 0 就是造假 |
| `sample_frame_110.png` | 左指尖按压（自动出现 `L_pad first contact >=20 N` 竖线）。动作段回放 seed 11 的记录值：全程未越 ±5，`at the +-1 rail: 1/7`（只有夹爪列按定义落在 ±1） |
| `sample_frame_182.png` | 右指尖穿透爆发。动作段回放 seed 17 chunk 26 row 31 的记录值，六行一眼分层：模型输出 `+40.00`（归一化单位，夹爪列）→ 反归一化 `+40.000`（action 单位）→ pm1 后 `+1.000` → `env.step` 指令 `-1.000`；平移/旋转六列全部落在 ±1 导轨上，`at the +-1 rail: 6/7` |
| `sample_frame_245.png` | 急停后 |
| `sample_frame_299.png` | 释放段，两路指尖力回零 |
| `LAYOUT_SAMPLE_meta.json` | 几何、帧数、编码器、自检结论，以及每段动作数字来自哪个种子的哪一行（`chain_provenance`） |

## 与历史 FasterWAM 结果视频的对齐（实测，非推断）

| 项 | 历史 `r4_c0_seed21_annotated_20fps.mp4` | 本布局 |
|---|---|---|
| 画布 | 864×576（单相机） | 1664×576（双相机＋面板） |
| 相机块 | 512×512 = 原生 256×256 的 2× | 512×512，同样 2×，像素复制 |
| 帧率 | 20.0 fps（= control_hz） | 20 fps，且断言 `pv["fps"] == meta["video_fps"]` |
| 编码 | h264 | pod：`imageio/libx264`；本地样片：`cv2/mp4v`（见下方限制） |
| 相机顺序 | 历史包只有单相机文件，无法证实双相机顺序 | front 左、left_wrist 中（按需求文字执行） |

## 已知限制（如实写）

1. 本地无 `imageio`，样片由 `cv2/mp4v` 编码：版式、同步、解码可读性在此验证，**码流与 pod 的
   libx264 一致性未验证**——该项由 pod 上的 `panel_preflight` 与 `recorder_selftest40k.py` 负责。
2. 相机画面是占位图（会动的方框与夹爪），不是渲染出来的场景。
3. 力与相机画面是构造的；**动作链不是构造的**，它回放 `ROOT_CAUSE_40K_20261112.md` §1.1 里逐层记录过的
   真实见证（爆发段 = seed 17 chunk 26 row 31，其余 = seed 11 的逐维 episode 最大值）。`44.37` 在这里
   不再出现：§1.1 已确定它是反归一化之后、0–5 维的量，早期样片把它画在归一化那一行是错的，
   `LAYOUT_SAMPLE_meta.json` 的 `chain_provenance["44.37"]` 把这件事留在产物里而不只是留在对话里。
4. 样片回答的是版式与同步；「策略为什么失败」只能由 pod 上 `step_040000` 的 seed 11 / 17 真实录制回答。
5. `2775 N` 依旧没有传感器证据，本样片与正式报告都不把它当作已确认的实际接触力。
