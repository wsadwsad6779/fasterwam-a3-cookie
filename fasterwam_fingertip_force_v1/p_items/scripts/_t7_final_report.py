import json
import os
import time

P = '/workspace/fwv1/p_items'
EV = '/workspace/fwv1/eval_fingertip_v1'
TS = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|']
    for r in rows:
        out.append('| ' + ' | '.join('' if x is None else str(x) for x in r) + ' |')
    return '\n'.join(out)


def seed_rows(d, base=0):
    rows = []
    for r in (d or {}).get('results', []):
        if 'error' in r:
            rows.append([r.get('episode'), 'ERROR', '', '', '', '', ''])
            continue
        fs = r.get('force_stats') or {}
        rows.append([r.get('episode'), r.get('final_in_target'), int(bool(r.get('exact_fill'))),
                     r.get('grasps'), round(fs.get('carry_force_frac_gt_0p5') or 0, 4),
                     round(fs.get('force_max') or 0, 2),
                     round((r.get('action_stats') or {}).get('open_cmd_frac') or 0, 4)])
    return rows


f0 = load(EV + '/cmp_f0_3k/SUMMARY_step_3000.json')
f2 = load(EV + '/cmp_f2_3k/SUMMARY_step_3000.json')
selfchk = load(P + '/F0_GRIP_SELFCHECK_3K.json')
rc = load(P + '/R_COMPARE_15K.json')
tf = load(P + '/TEACHER_FORCED_15K.json')
frozen = load(P + '/FROZEN_ENV_3K_CHECK.json')
f0_hold = load(P + '/F0_STAGE1_HOLD.json')
abl = {}
for v in ('forcezero', 'skipfilm'):
    for s in (7504, 7508):
        abl[(v, s)] = load(EV + '/diag15k_%s_s%d/SUMMARY_step_diag15k_%s.json' % (v, s, v))


def means(d):
    rs = (d or {}).get('results', [])
    if not rs:
        return None
    n = len(rs)
    return {'n': n,
            'mean_in_target': sum((r.get('final_in_target') or 0) for r in rs) / n,
            'exact_fill_rate': sum(1 for r in rs if r.get('exact_fill')) / n,
            'mean_grasps': sum((r.get('grasps') or 0) for r in rs) / n,
            'mean_carry': sum(((r.get('force_stats') or {}).get('carry_force_frac_gt_0p5') or 0) for r in rs) / n}


m0 = means(f0)
m2 = means(f2)
lines = []
lines.append('# Task7 双轨阶段报告（F0 基线与 F2 根因）')
lines.append('')
lines.append('- 生成时间：`%s`' % TS)
lines.append('- 报告范围：F0 step3000、F2 step3000 配对、r=10/16/24/32 诊断、teacher-forced、ForceZero/SkipFiLM')
lines.append('- 官方口径：strict `cookies_in_target` / `exact_fill`；relaxed 计数仅作诊断（用户规则，见下）')
lines.append('')

lines.append('## 0. 结论摘要')
if m0:
    lines.append('- **F0 3k**：10-seed 平均入盒 %.2f/10，exact_fill %.0f%%，grasp %.2f。' % (
        m0['mean_in_target'], 100 * m0['exact_fill_rate'], m0['mean_grasps']))
if m2:
    lines.append('- **F2 3k**：10-seed 平均入盒 %.2f/10，exact_fill %.0f%%，grasp %.2f。' % (
        m2['mean_in_target'], 100 * m2['exact_fill_rate'], m2['mean_grasps']))
if selfchk:
    lines.append('- **F0 夹爪自检**：`%s`（中位 %s，<0.30 %s，open %s，n=%s）。' % (
        selfchk.get('verdict'), (selfchk.get('pooled') or {}).get('median'),
        (selfchk.get('pooled') or {}).get('lt_030'), (selfchk.get('pooled') or {}).get('open_ge_080'),
        (selfchk.get('pooled') or {}).get('n_steps')))
if rc:
    runs = [r for r in rc.get('runs', []) if 'missing' not in r]
    lines.append('- **r 扫描**：' + '；'.join(
        'r%d 用户规则 %.2f / strict %.2f / exact %.0f%%' % (
            r['replan'], r.get('mean_in_target_user') or 0, r.get('mean_in_target_strict') or 0,
            100 * (r.get('exact_fill_rate') or 0)) for r in runs))
    lines.append('- **r 结论**：r32 的优势目前只在个别 seed（7504）出现；不是已证明的 r 因果，正式成绩仍按 r10 + exact_fill。')
if tf:
    lines.append('- **teacher-forced**：见第 6 节（专家观测下的夹爪分布，用于区分输出/监督问题与状态偏移）。')
lines.append('')

lines.append('## 1. F0 step3000（10 seeds 7100-7109，官方 eval）')
if f0:
    lines.append(table(['seed', 'in_target(strict)', 'exact_fill', 'grasps', 'carry>0.5', 'force_max(N)', 'open_cmd_frac'],
                       seed_rows(f0)))
    lines.append('')
    lines.append('- checkpoint：`%s` sha `%s`' % (f0.get('checkpoint'), f0.get('checkpoint_sha256')))
    lines.append('- scene all_match：`%s`；eval 协议：600 步 / 5 去噪 / replan 10' % (
        (f0.get('scene_manifest') or {}).get('all_match')))
else:
    lines.append('- 缺失：`%s`' % (EV + '/cmp_f0_3k/SUMMARY_step_3000.json'))
lines.append('')

lines.append('## 2. F2 step3000（配对，同一 seeds/场景/脚本）')
if f2:
    lines.append(table(['seed', 'in_target(strict)', 'exact_fill', 'grasps', 'carry>0.5', 'force_max(N)', 'open_cmd_frac'],
                       seed_rows(f2)))
    lines.append('')
    lines.append('- checkpoint：`%s` sha `%s`' % (f2.get('checkpoint'), f2.get('checkpoint_sha256')))
else:
    lines.append('- 缺失：`%s`' % (EV + '/cmp_f2_3k/SUMMARY_step_3000.json'))
lines.append('')

lines.append('## 2A. Base anchor（官方 step_021700，同一 10 seeds 7100-7109）')
base = load(P + '/BASE_ANCHOR_V31.json')
if base:
    rows = []
    for name, key in (('base', 'base'), ('F0 3k', 'f0_3k'), ('F2 3k', 'f2_3k')):
        m = (base.get(key) or {}).get('means') or {}
        rows.append([name, m.get('mean_in_target'), m.get('exact_fill_rate'),
                     m.get('mean_grasps'), m.get('mean_carry'), m.get('open_cmd_mean')])
    lines.append(table(['model', 'in_target', 'exact', 'grasps', 'carry>0.5', 'open_cmd'], rows))
    lines.append('- 读法：base anchor 是“未微调官方基座”在同 10 场景的下限；F0/F2 3k 高于它才有训练增益。')
else:
    lines.append('- 缺失或未完成：`p_items/BASE_ANCHOR_V31.json`（base anchor 在 post_r 之后串行跑）。')
lines.append('')

lines.append('## 3. F0 vs F2 同 seed 配对')
if f0 and f2:
    m2map = {r.get('episode'): r for r in f2.get('results', [])}
    rows = []
    for r in f0.get('results', []):
        o = m2map.get(r.get('episode'))
        if not o:
            continue
        rows.append([r.get('episode'), r.get('final_in_target'), o.get('final_in_target'),
                     (r.get('final_in_target') or 0) - (o.get('final_in_target') or 0),
                     int(bool(r.get('exact_fill'))), int(bool(o.get('exact_fill'))),
                     r.get('grasps'), o.get('grasps')])
    lines.append(table(['seed', 'F0 in_target', 'F2 in_target', 'delta', 'F0 exact', 'F2 exact', 'F0 grasps', 'F2 grasps'], rows))
else:
    lines.append('- 需要两份 3k summary 同时存在。')
lines.append('')

lines.append('## 4. r 扫描（F2 step_015000，seeds 7500-7509，诊断）')
if rc:
    rows = []
    for r in [x for x in rc.get('runs', []) if 'missing' not in x]:
        rows.append([r['replan'], r.get('label'), r.get('mean_in_target_user'), r.get('mean_in_target_strict'),
                     round(100 * (r.get('mean_in_target_user') or 0) / 10, 1),
                     round(100 * (r.get('exact_fill_rate') or 0), 1), round(r.get('mean_grasps') or 0, 2),
                     round(r.get('mean_exec_lt030') or 0, 3), round(r.get('mean_elapsed_s') or 0, 1),
                     r.get('mean_replan_calls')])
    lines.append(table(['r', 'label', '用户规则入盒', 'strict 入盒', 'placement %', 'exact %', 'grasps',
                        'exec<0.30', '耗时 s', '模型调用'], rows))
    o = rc.get('official_r10_strict')
    if o:
        lines.append('')
        lines.append('- 官方 strict r10 anchor（15k 里程碑）：平均入盒 %s，exact %.0f%%。' % (
            o.get('mean_in_target'), 100 * (o.get('exact_fill_rate') or 0)))
    r10 = [x for x in rc.get('runs', []) if x.get('replan') == 10 and x.get('official_strict_match')]
    if r10:
        mm = r10[0]['official_strict_match']
        lines.append('- r10 strict 复算与里程碑一致：`%s`（逐 seed in_target/exact_fill/grasps）。' % mm.get('all'))
    lines.append('')
    lines.append('### 每 seed 原始表')
    rows = []
    for r in [x for x in rc.get('runs', []) if 'missing' not in x]:
        for row in r.get('rows', []):
            rows.append([r['replan'], row.get('seed'), row.get('in_target_user'), row.get('in_target_strict'),
                         int(bool(row.get('exact_fill'))), row.get('grasps'),
                         round(row.get('exec_lt030') or 0, 3), row.get('elapsed_s'),
                         row.get('strict_reimpl_mismatch_steps')])
    lines.append(table(['r', 'seed', '用户入盒', 'strict 入盒', 'exact', 'grasps', 'exec<0.30', '耗时 s', 'strict自检失配步'], rows))
    lines.append('')
    lines.append('- counting rule：%s' % ((rc.get('counting_rule') or {}).get('primary')))
    lines.append('- 说明：r10 网格是 relaxed-r10 诊断，不顶替官方 strict r10；r>32 与 r32 等价，未作为新点。')
else:
    lines.append('- 缺失：`%s`' % (P + '/R_COMPARE_15K.json'))
lines.append('')

lines.append('## 5. F0 夹爪自检与冻结环境')
if selfchk:
    rows = []
    for p in selfchk.get('training_probes', []):
        rows.append([p.get('step'), p.get('median'), p.get('lt_0.30'), p.get('open_ge_0.8')])
    lines.append('- 训练探针：')
    lines.append(table(['step', 'median', '<0.30', 'open>=0.8'], rows))
    lines.append('- 判定：`%s`；规则：%s' % (selfchk.get('verdict'), selfchk.get('rule')))
if f0_hold:
    lines.append('- **F0 3k 自检（仅记录，不阻断）**：%s；续训按用户 2026-11-25T04:03:50Z 指令无条件进行（见第 8 节）。' % f0_hold.get('reason'))
if frozen:
    lines.append('- 冻结环境 3k 前后一致：`%s`（sim/eval/freeze/scene manifest）。' % frozen.get('all_frozen_match'))
lines.append('')

lines.append('## 6. Teacher-forced 夹爪分布（F2 step_015000）')
if tf:
    lines.append('- 覆盖：n=%s，variants=%s，phases=%s；未覆盖 APPROACH/RESET/RELEASE/DESCEND（同一 val 批次）。' % (
        tf.get('n_samples'), list((tf.get('summary') or {}).keys()),
        list(((tf.get('summary') or {}).get('normal') or {}).keys())))
    rows = []
    for variant, per in (tf.get('summary') or {}).items():
        for ph, v in per.items():
            rows.append([variant, ph, v.get('n'), v.get('label0_lt_0.30'), v.get('pred0_lt_0.30'),
                         v.get('pred9_lt_0.30'), v.get('pred31_lt_0.30'), v.get('pred0_median')])
    lines.append(table(['variant', 'phase', 'n', 'label<0.30', 'pred0<0.30', 'pred9<0.30', 'pred31<0.30', 'pred0 median'], rows))
    lines.append('')
    lines.append('- 读法：若专家观测下 pred0<0.30 仍低，说明是输出/监督分布问题；若 pred0 明显升高，则更像闭环状态偏移。')
else:
    lines.append('- 缺失：`%s`' % (P + '/TEACHER_FORCED_15K.json'))
lines.append('')

lines.append('## 6A. 第 7 维 action loss / 梯度审计（F2 15k）')
dim7 = load(P + '/DIM7_LOSS_GRAD_AUDIT.json')
if dim7:
    pdim = dim7.get('per_dim_loss') or []
    lines.append(table(['dim', 'MSE'], [[i, None if v is None else round(v, 6)] for i, v in enumerate(pdim)]))
    lines.append('- dim6 统计：`%s`' % json.dumps(dim7.get('dim6_stats')))
    lines.append('- action expert 梯度总范数：`%s`（%s 个参数有梯度）' % (
        dim7.get('grad_total_norm_action_expert'), dim7.get('n_params_with_grad')))
    lines.append('- 说明：单 val batch 前向/反向，无 optimizer step、不改权重。')
else:
    lines.append('- 缺失：`p_items/DIM7_LOSS_GRAD_AUDIT.json`（base anchor 前串行跑）。')
lines.append('')

lines.append('## 6B. Teacher-forced 3k vs 15k（同 val 观测，换 checkpoint）')
tf3 = load(P + '/TEACHER_FORCED_3K.json')
if tf3:
    rows = []
    for variant, per in (tf3.get('summary') or {}).items():
        for ph, v in per.items():
            rows.append([variant, ph, v.get('n'), v.get('label0_lt_0.30'), v.get('pred0_lt_0.30'),
                         v.get('pred9_lt_0.30'), v.get('pred31_lt_0.30'), v.get('pred0_median')])
    lines.append(table(['variant', 'phase', 'n', 'label<0.30', 'pred0<0.30', 'pred9<0.30', 'pred31<0.30', 'pred0 median'], rows))
    lines.append('- 读法：同一批专家观测下，3k 与 15k 的 pred 分布差异反映 checkpoint/训练阶段，不是观测差异。')
else:
    lines.append('- 缺失：`p_items/TEACHER_FORCED_3K.json`（base 链在 r sweep 之后跑）。')
lines.append('')

lines.append('## 7. ForceZero / SkipFiLM 消融（F2 step_015000，seeds 7504/7508）')
rows = []
for (v, s), d in abl.items():
    mm = means(d)
    if mm:
        rows.append([v, s, mm['mean_in_target'], round(100 * mm['exact_fill_rate'], 1),
                     round(mm['mean_grasps'], 2), round(mm['mean_carry'], 4)])
lines.append(table(['variant', 'seed', 'in_target', 'exact %', 'grasps', 'carry>0.5'], rows) if rows else '- 缺失（未运行或未完成）。')
lines.append('')

lines.append('## 8. 预算、队列与下一步')
lines.append('- F2：暂停在 15800/21000，剩 5200；无自动续训。')
lines.append('- F0：step3000 完成；用户 2026-11-25T03:45:38Z 批准从 3000 续训到 15000，每 3000 保存 checkpoint 并做 10-seed 评估；15000 为硬上限，15k→21k 仍为 BLOCKED_USER。')
lines.append('- **F0 3k 自检（仅记录，不阻断）**：selfcheck 结论会写进报告与 `F0_SELFCHECK_GATE.json`，但续训按用户 2026-11-25T04:03:50Z 指令**无条件**执行 6k/9k/12k/15k，避免 GPU 空转。')
lines.append('- 续训条件：同数据/标签/7D/8D/采样器/loss/优化器/lr/batch/clip/seed，无 force 模块；LR 余弦总跨度仍为 21000（与 F2 对齐）。')
lines.append('- GPU 时间与顺序：GPU_TIME_STATE 的 remaining_hours 只能从 Radeon 控制台读（agent 读不到）；按最后一次控制台读数的线性估计安排——先 POSTR 诊断（r sweep + teacher + 消融 + dim7 + base，约 1.5–2h）再 F0 12k（约 10.3h）；若控制台显示剩余 <12h，则改为 F0 先走、纯评估后置。')
lines.append('- 里程碑 seeds：6k=7200-7209、9k=7300-7309、12k=7400-7409、15k=7500-7509（与 F2 同组，配对比较）。')
lines.append('- 队列顺序：F0 3k → F2 3k → r 扫描 → teacher-forced → ForceZero/SkipFiLM → 本报告 → F0 6k/9k/12k/15k。')
lines.append('')

lines.append('## 8A. 夹爪专用设计 A/B/C 规格草案（BLOCKED_USER，未实施）')
lines.append('- **A 状态头 + 连续开度**：在 action expert 的 1024 维特征上加一个夹爪状态头（3 态 open/close/release + 连续开度），头约 1–2M 参数；从 15800 起联合微调头 + action expert 2–4k 步；目标是显式学双峰而不是回归均值；验证：15k/15800 权重、10-seed r10、V3.1 冻结场景；风险：头可能仍在模式内取均值，需要分类/分布损失配合。')
lines.append('- **B 分布式损失**：把第 7 维的 MSE 换成分布式目标（two-hot/HL-Gauss 开度分箱，或 mixture-density 头），其余 6 维保持 MSE；从 15800 起 3–6k 步；需要新的标签分箱与 loss 实现；风险：会扰动其他维，需分维权重与消融。')
lines.append('- **C Flow Matching 采样修复**：先做零训练推理修复（denoise 5→10/20、采样器/guidance、只对第 7 维做约束），不改权重；若有效再做 1–2k 步短微调；风险：推理变慢、可能仍不破均值；这是最便宜的验证项。')
lines.append('- 共同约束：V3 连续标签、同一 8D/7D、同一 V3.1 薄垫与 r10 官方口径、exact_fill 仍按 strict；A/B/C 均需用户规格 + 预算批准后才实现；当前只写规格，不动 F2 checkpoint / STOP_TRAINING。')
lines.append('')

lines.append('## 9. 产物')
for p in ['p_items/R0_PROTOCOL_FINGERPRINT.json', 'p_items/R0_RUNTIME_FACTS.json',
          'p_items/R_COMPARE_15K.json', 'p_items/R_SWEEP_RAW_RESULTS.csv',
          'p_items/R_SWEEP_SUMMARY.json', 'p_items/TEACHER_FORCED_15K.json',
          'p_items/F0_GRIP_SELFCHECK_3K.json', 'p_items/FROZEN_ENV_3K_CHECK.json',
          'p_items/USER_INSTRUCTION_20261125_RELAXED_INBOX.json',
          'p_items/USER_INSTRUCTION_20261125_F0_CONT_15K.json']:
    lines.append('- `%s`' % p)

md = '\n'.join(lines) + '\n'
open(P + '/T7_REPORT_20261125.md', 'w', encoding='utf-8').write(md)
json.dump({'ts': TS, 'f0_3k': m0, 'f2_3k': m2,
           'base_anchor': base,
           'dim7_audit': dim7,
           'f0_selfcheck_verdict': (selfchk or {}).get('verdict'),
           'r_sweep': [{k: v for k, v in r.items() if k != 'rows'} for r in (rc or {}).get('runs', [])],
           'teacher_samples': (tf or {}).get('n_samples'),
           'frozen_env_match': (frozen or {}).get('all_frozen_match'),
           'ablations': {('%s_%d' % k): means(v) for k, v in abl.items()},
           'report_md': P + '/T7_REPORT_20261125.md'},
          open(P + '/T7_REPORT_20261125.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('WROTE', P + '/T7_REPORT_20261125.md')