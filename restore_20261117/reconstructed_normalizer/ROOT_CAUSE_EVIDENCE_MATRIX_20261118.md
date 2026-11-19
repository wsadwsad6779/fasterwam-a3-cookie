# FasterWAM 40k 失败机制交付

NORMALIZER_IDENTITY=RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL
original_stats_sha256=UNKNOWN

## 本轮执行范围

完成正式配置核验、既有1210条执行命令的CPU分析、真实资产CPU前置验收、8项合成启动器回归、37项真实fixed记录器合成环境回归，并在用户追加允许小规模GPU后执行一次seed12严格A/B（每侧10步，1次推理）。未训练、未启动160回合、未重跑seed11/17完整回合，未改模型/pm1/Adapter/normalizer/控制频率/场景。独立启动器默认仅CPU preflight，显式--run-pair才允许10步配对，已移除通过后自动600步回合。

## 配置绑定

正式恢复run_config的Processor：action7、state14；delta_action_dim_mask=[true,true,true,true,true,true,false]；action_state_transforms=null；global min/max、非stepwise。force配置通道[0,1,2,12,13,16]、历史16、未来32。不是以普通a3_2cam模板替代正式配置。

本轮实际入口diag_reconstructed_40k.build_policy_reconstructed加载该run_config，显式注入build_proprio14与pm1；ForcePolicy先denormalize再>=0阈值，再调用Adapter。训练预处理可见代码将opening减小标+1，Adapter映射-1闭合，语义一致。但训练arrays与历史源码/checkpoint逐字同源未闭合；不能将可见预处理代码认定为历史真实训练标签的最终证明。

## 严格验收结果

- 真实权重11126493579字节，SHA256 d118eb2172b04165990c032e291057aaa9313b1d9fd0e6cd7a9ada3637c9cc20。
- 真实run_config SHA256 4aa66c57cdfe2804ae56cf244a907be9e13fe232d4ede65ba7ef0695eb79abfd。
- 原始记录器 SHA256 6e41095da4b54c9f8261f62203647532eba8c727f7d35a4b1977682630a1a80d。
- fixed副本 SHA256 380ca4c24ca5858c22f5d6ad7bb72984320c85e0f057099c67f2b7e5e87d9641。
- 启动器 SHA256 28b53a2f895b9c69612b8c074b0117e7443de78838d96ebca5a1098688702a67。
- 原始hash_files严格规则未更改；真实资产验收与合成fixture报告严格分开。CPU前置进程未导入torch。
- 本次严格GPU A/B 56项全PASS，pair_rc=0。A/B实际动作、raw/denorm/pm1/Adapter、三相机、FORCE18、state16、velocity、eef、time、safety、MuJoCo qpos/qvel/ctrl/act/qacc/sensordata、CPU与GPU RNG一致。
- 仅证明本次seed12十步中生产录像器相对共同基础记录器和见证器的增量影响未被上述检查检出；不是所有seed、所有长度的无副作用证明，也不是任务成功。此前无记录器对基础记录器的测试范围仍按原报告，不得随本次结果扩大。
- 本次视频11帧、20fps，CPU解码成功；关键帧已保存。图像工具未提供可辨识视觉反馈，不声称完成全部文字截断视觉验收。

## 执行翻转

|seed|执行命令|flip|块内flip/机会|边界flip/机会|小/大/混合flip|
|---|---:|---:|---|---|---|
|11|600|38|32/540=5.93%|6/59=10.17%|2/12/24|
|12|10|2|2/9=22.22%|无机会|0/2/0|
|17|600|96|75/540=13.89%|21/59=35.59%|5/57/34|

小定义两端abs<=0.1，大定义两端abs>=0.3，其他混合；阈值只是描述性分类。三组raw中已有变号，不是pm1凭空制造。Seed12完整32行计划5次翻转中，仅row5/8实际执行，row16/18/19未执行。

首次pad>1N前10步以外：seed11为5/28，seed17为16/52；其中两端均>=0.3的大翻转分别3和14次。seed12两次均大翻转且全程pads未>1N。early指垫接触前也有明显翻转，不能用后期高力或偏离单独解释；但腕力初始已>10N，不能排除力条件。

边界单位机会率分别为块内1.72/2.56倍。pre阶段seed11边界1/2、块内4/26；其样本极少，不做总体显著性或重规划因果结论。完整分层与幅度在trajectory JSON/CSV。

### 同时刻计划改写补充

另做纯CPU比较：旧chunk row9为上一条实际命令，旧row10为同一未来时刻的未执行提案，新chunk row0为实际执行。59个边界中，旧row10与新row0符号不同：seed11为4处、seed17为17处。实际边界flip中，旧计划本来不翻、但新计划变为翻转的分别3/6与16/21处。这进一步定位了重规划时计划变化，但旧row10从未执行，不能当实际轨迹或证明若继续旧计划就能成功；变化可能来自新观测、噪声、缓存或条件，尚不能叫算法bug。完整逐边界见CPU_BOUNDARY_PLAN_REWRITE_20261118.json。

## 时间线（命令1基；frame0 reset，frame f为第f条执行后的观察）

|事件|Seed11|Seed17|
|---|---:|---:|
|首次raw夹爪flip|11|3|
|首次pad abs>1N|40|64|
|首次pad abs>10N|40|129|
|首次姿态相对reset>5度|47|78|
|首次姿态>15度|60|144|
|首次pad>50N|未观察到|143|
|首次XYZ命令rail|332|151|
|首次腕力0:3范数>100N|未观察到|153|
|首次pad>100N|未观察到|157|
|首次pad>300N|未观察到|159|
|首次姿态>30度|98|159|

rail为归一化执行命令abs>=1-1e-6，非实测XYZ越界，也不等于已测得preclip越界；夹爪天然±1不计异常。Seed11全回合只有1步XYZ rail、pads峰19.10N；Seed17有248步XYZ rail、pads峰338.87N。两条失败不可强行归为同一高力机制。腕力用force0:3范数，四pads为12:16，不是把18维混合单位整体求范数。

末端离开饼干区域和饼干散乱未有绑定空间ROI/物体位姿日志，本轮无法给出精确控制步。交接作者的Seed11约5秒撤离、12秒后远离，以及Seed17约20秒后散乱仅作为外部视觉估计线索，未独立确认，不混入精确数值时间线。

## 根因排序与缺口

|候选|当前证据|判断与下一证据|
|---|---|---|
|生成动作块时间不一致|同次推理中正负大跳，先于指垫接触；跨seed存在|优先级高，但只是最早可见异常层；需同源示范32步窗口统计，区分监督与生成|
|重规划不连续|边界机会率升高且有大跳|优先级高；应比较旧chunk未执行row10与新chunk row0的同一未来时刻计划，不把row10算已执行|
|力输入影响早期动作|reset腕部已有负载；后期seed17高力与漂移共现|仍未证明；需固定观测、相同噪声、只变P1/P2条件的最小单chunk敏感性对照，零力不代表无力模型，不能直接评成功率|
|训练标签噪声|尚无40k训练arrays同源绑定|不能宣布根因，历史1892翻转不可直接移植|
|整体夹爪符号接反|可见标签到Adapter到实测方向一致|当前不支持，历史版本同源缺口保留|
|pm1作为直接起因|raw已变号；small-only翻转占比较少|不能解释主要大跳；阈值可放大小幅变号，但不在本轮修改|
|生产记录器改变10步执行|本次严格56项全相等|本次未检出；不得外推所有完整回合|
|动作rail是统一首因|Seed11明显姿态变化早于332步唯一rail|不支持统一解释；Seed17高力与rail在143–159步附近共现，不能定因果|

未进行额外力消融：本轮严格配对已补齐必要仪器证据；目前仍需将同源训练窗口和同时间戳新旧计划比较闭合，贸然多跑完整回合信息增益低。保留冻结接口，不加P5纠偏器，不宣称已定位全部根因。

## 交付文件

- cpu40k_config_audit.md / .json：正式Processor与当前推理入口核验。
- forensics_download/cpu40k_trajectory_analysis.json：输入SHA、定义、逐事件状态和全量汇总。
- forensics_download/cpu40k_trajectory_flips.csv：136条执行flip。
- forensics_download/cpu40k_trajectory_chunks.csv：121条执行chunk。
- forensics_download/cpu40k_trajectory_report_zh.md：分析说明。
- forensics_download/STRICT_ARTIFACT_PREFLIGHT_20261118.json：实际启动器真实资产CPU验收。
- CPU_LAUNCHER_REGRESSION.json：8项启动器合成回归。
- CPU_REAL_CAPTURE_REGRESSION.json：37项合成环境真实记录器回归。
- forensics_download/STRICT_PRODUCTION_PAIR_SEED12_20261118.json：56项严格配对结果。
- forensics_download/strict20261118_seed12_composed_panel.mp4：本次最短录像。

没有删除文件、推送代码或改变外部权限。
