# CPU 40k 配置审计（任务 A）

- 身份：**RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL**。
- 审计日期：2026-11-18；仅本地 CPU 文件读取。未运行模型、训练、仿真、GPU 检测或远端访问；未读取凭据、历史报告。
- 结论：已定位并读取实际 `step_040000/run_config.yaml` 的 processor/数据配置，并读取 force 专用 YAML 及推理反归一化分支。配置链可核验，但**源码、统计文件和 40k 历史 checkpoint 的同源绑定尚未证明**。不能把当前重建 normalizer 升级为原始历史 normalizer。

## 证据文件（绝对路径）

根目录：`C:/Users/ASUS/Documents/Qoder/2026-10-31/b462d38d/_clone_default`。

1. `C:/Users/ASUS/Documents/Qoder/2026-10-31/b462d38d/_clone_default/restore_20261117/step_040000/run_config.yaml`：Glob 找到的实际文件；本次读取 1–240 行。
2. `C:/Users/ASUS/Documents/Qoder/2026-10-31/b462d38d/_clone_default/stage_40k/fw/configs/data/a3_2cam_force.yaml`。
3. `C:/Users/ASUS/Documents/Qoder/2026-10-31/b462d38d/_clone_default/stage_40k/fw/configs/task/a3_force_40k.yaml`。
4. `C:/Users/ASUS/Documents/Qoder/2026-10-31/b462d38d/_clone_default/stage_40k/fw/configs/model/fasterwam_force.yaml`。
5. `C:/Users/ASUS/Documents/Qoder/2026-10-31/b462d38d/_clone_default/stage_40k/fw/src/fasterwam/datasets/lerobot/processors/fastwam_processor.py`：定点搜索。
6. `C:/Users/ASUS/Documents/Qoder/2026-10-31/b462d38d/_clone_default/stage_40k/fw/src/fasterwam/datasets/lerobot/utils/normalizer.py`：读取 1–220 行。
7. `C:/Users/ASUS/Documents/Qoder/2026-10-31/b462d38d/_clone_default/bridge/infer_a3_worker.py`：定点搜索及读取 202–240 行；未执行。

## 实际 40k 配置

`run_config.yaml:69–129` 的 processor 是 `fasterwam.datasets.lerobot.processors.fastwam_processor.FastWAMProcessor`：

| 项目 | 实际值 |
| --- | --- |
| action / state | 单键 default；7 / 14 维 |
| 图像 | image、wrist_image；各 3×224×224 |
| num_obs_steps / num_output_cameras | 33 / 2 |
| action_output_dim / proprio_output_dim | 7 / 14 |
| delta_action_dim_mask.default | [true,true,true,true,true,true,false] |
| action_state_transforms | null |
| use_stepwise_action_norm | false |
| norm_default_mode / norm_exception_mode | min/max / null |
| action_state_merger | ConcatLeftAlign |
| train / val transforms | ToTensor → Resize(224,224) |

注意：delta mask 不意味着推理时自动把所有动作转换成绝对量；当前查阅的 processor 源码显示该 mask 参与 action padding 处理，不能据字段名扩展语义。

`run_config.yaml:25–68,130–142,149`：数据集为 `/workspace/a3_data/a3_fw_v1g`；历史 stats 引用为 `/workspace/a3_data/a3_fw_v1g/dataset_stats.json`；force arrays 引用 `/workspace/a3_data/a3_fw_force/arrays.npz`；force channels=[0,1,2,12,13,16]，history=16、horizon=32；model.proprio_dim=14。这里的 `/workspace/...` 是配置内历史路径，不代表本机已验证存在的文件。

## 不得用普通 a3_2cam.yaml 替代 force 配置

- `a3_force_40k.yaml:15–18` 明确覆盖 `/data: a3_2cam_force` 和 `/model: fasterwam_force`。
- `a3_2cam_force.yaml:21–42` 虽继承 a3_2cam，但将 dataset/stats 指向 v1g、state 设为 14、processor.proprio_output_dim=14，并接入 force arrays、channels、history、horizon。
- `fasterwam_force.yaml:77–118` 定义 force conditioner、6 通道、history=16、horizon=32，以及 calibration 文件 `/workspace/a3_data/a3_fw_force/force_calib_v1.json`。通道不是六轴 wrench：前三个为腕部三轴力，后三个为接触/夹爪标量；此处是 YAML 的说明，不是本次传感器实测。
- 实际 run_config 优先于模板：任务模板 save_every=5000、resume=null，而实际配置 save_every=40000、resume 指向 step_035000；model 模板的若干开关也被任务/实际配置覆盖。不能仅凭模板名恢复历史运行。
- 本次没有读取普通 a3_2cam.yaml，也没有把它当作 force 实际配置的替代证据。

## normalizer 与 infer gripper denorm

`fastwam_processor.py:111–118` 将上述 mode、shape_meta、stepwise 和传入 stats 交给 LinearNormalizer。

`normalizer.py:32–62,91–133`：非 stepwise action 选择 `global_*`，state 也选择 `global_*`；min/max 通常映射到 [-1,1]，反变换为 `(x-offset)/scale`。范围小于 1e-4 的维度有专门 fallback，不能直接除以零；forward clamp 至 [-5,5]，backward 本身不 clamp。只有拿到匹配的历史 stats 才能确定每维数值参数及 gripper 的真实范围。

`infer_a3_worker.py:83–92,146–155`：先调用 action/default normalizer.backward，并转到 CPU，再处理最后一维：

- 环境变量 `FASTERWAM_A3_GRIPPER_MODE=pm1`：反归一化后以 >=0 为界输出 +1，否则 -1；代码注释称 +1=close，但本次未用数据或执行验证该约定。
- 默认 `libero01`：先 `g*2-1`，再乘 -1；若 cfg.EVALUATION.binarize_gripper 为真，再 np.sign。它不能不加区分地套到本来就是 ±1 的 A3 动作上。

`infer_a3_worker.py:48–51,95–99,226–240`：默认 CKPT/STATS 指向 libero 发布资产；build_proprio 是 eef6+fingers（8 维），不是 force 的 14 维；processor 从 cfg 实例化，stats 从独立 STATS 路径加载。因此该 worker 只是可见的 denorm/gripper 逻辑证据，**不是已证明适配或实际使用于 force 40k 的推理链**。本次未读取 load_cfg 函数体，也未确认历史环境变量/实际启动参数。

## 本轮实际诊断入口的补充绑定

本轮入口不是直接执行 worker.main。`diag_reconstructed_40k.py:151–161` 从恢复的 run_config 只读加载并将 stats 映射到独立重建资产；203–208 行实例化该配置的 Processor，显式传入 `proprio_fn=E.build_proprio14` 与 `gripper_mode=pm1`。因此 worker 默认的 8 维 build_proprio 与 libero01 不是本轮实际采用的分支。

`stage_40k/dev/eval40k/policy40k.py:142–152` 实际执行 bridge.denormalize_action 后，用 `>=0` 映射 ±1，再调用 Adapter。已落盘 raw/denorm/pm1/handed 链是该入口的运行证据；这只绑定本轮重建条件诊断，不追溯证明训练时源码字节身份。严格 CPU 真实资产清单另外给出本轮远端配置及源码 SHA。

## 证据缺口与结论边界

1. **历史 stats 绑定未闭合**：Glob 发现以下候选，但未读内容、未核对维数/哈希/生成 provenance：
   - `C:/Users/ASUS/Documents/Qoder/2026-10-31/b462d38d/_clone_default/data/dataset_stats.json`
   - `C:/Users/ASUS/Documents/Qoder/2026-10-31/b462d38d/_clone_default/restore_20261117/archive_evidence/data/a3_fw/dataset_stats.json`
   这不证明它们就是 v1g 的 14 维历史 stats；定点 Glob 未返回 v1g/dataset_stats.json，亦不作为全盘不存在的证明。
2. **calibration 仅定位**：Glob 返回 `C:/Users/ASUS/Documents/Qoder/2026-10-31/b462d38d/_clone_default/restore_20261117/step_040000/force_calib_v1.json`；未读/未验证内容及与 arrays 的绑定。force calibration 和 action/state dataset stats 不可相互替代。
3. **源码—40k checkpoint 绑定未闭合**：当前 stage_40k 源码和恢复配置可解释行为，但本次未验证历史 commit、打包清单、文件哈希、checkpoint 内嵌 normalizer 或保存状态；不证明当前源码与训练时逐字一致。
4. **历史 infer 绑定未闭合**：未获取 force 40k 实际入口、环境、gripper 模式和输入维度链；不能把 bridge 默认 libero 分支认作历史实跑。
5. **未做数值/行为验收**：未加载权重、未实例化 processor、未重建或比较 stats、未做 round-trip、未执行动作。预测、代码意图与实际执行结果严格区分。

最终判定：任务 A 的配置取证已完成并落盘；原始历史 normalizer 身份验证仍未通过。后续若需闭合，应仅补齐带来源/哈希的 v1g stats、对应训练源码版本及 checkpoint/真实 infer 入口绑定；在此之前始终保留 **RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL**，不得宣称原始恢复或历史等价。
