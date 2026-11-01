'''Rebuild the backup inventory as v2 from the byte-verified pod TSV.

v1 covered 963 files because it enumerated only the directories we thought of. A
full `find` reconciliation added 2,925 more, so v2 is the whole pod minus scratch.
Input is `_r6_stage/dl/manifest_rows.tsv`, downloaded in 8 slices and checked against
the pod-side sha256 62afef81caed177f1746af7c5eec42dc1c30dbbc59caabd23a9084b94d7d782e
before this script runs.
'''
import hashlib
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = HERE / 'dl' / 'manifest_rows.tsv'
OUTMD = HERE.parent / 'BACKUP_MANIFEST_v2.md'
OUTJSON = HERE.parent / 'backup_manifest_v2.json'
EXPECT_SHA = '62afef81caed177f1746af7c5eec42dc1c30dbbc59caabd23a9084b94d7d782e'

GROUP_CAT = {
    'model_base_merged': '最终模型',
    'trainer_state_full': '训练恢复状态',
    'training_state_final': '训练恢复状态',
    'light_ckpts_series': '训练恢复状态',
    'light_ckpts_probe': '临时测试产物（诊断小跑 checkpoint）',
    'temp_test_artifacts': '临时测试产物',
    'pod_archives': '临时测试产物',
    'eval_results': '科研数据（评估结果）',
    'bridge_eval_outputs': '科研数据（评估结果）',
    'new_demos_corpus': '科研数据（新轨迹语料）',
    'dagger_corpus': '科研数据（新轨迹语料）',
    'dagger_raw_frames': '科研数据（新轨迹语料，未转换原始帧）',
    'dataset_metadata': '科研数据（数据集与派生数组）',
    'raw_demos_100': '科研数据（原始 100 条专家示范）',
    'repro_env': '复现材料（环境、commit 与工作区 diff）',
    'training_config_log': '复现材料（配置与日志）',
    'run_logs_root': '复现材料（配置与日志）',
    'live_code': '复现材料（代码）',
    'code_bridge': '复现材料（代码）',
    'code_sim': '复现材料（代码）',
    'code_cookie_repo': '复现材料（代码）',
    'code_upstream_fasterwam': '复现材料（上游代码）',
    'stage_artifacts': '复现材料（审计脚本与中间产物）',
    'other': '复现材料（审计脚本与中间产物）',
    'upstream_base_wan': '上游基础模型（可从外部源重建）',
    'upstream_base_fasterwam': '上游基础模型（可从外部源重建）',
}
ORDER = ['最终模型', '训练恢复状态', '科研数据（原始 100 条专家示范）',
         '科研数据（新轨迹语料）', '科研数据（新轨迹语料，未转换原始帧）',
         '科研数据（数据集与派生数组）', '科研数据（评估结果）',
         '复现材料（代码）', '复现材料（上游代码）', '复现材料（配置与日志）',
         '复现材料（环境、commit 与工作区 diff）', '复现材料（审计脚本与中间产物）',
         '上游基础模型（可从外部源重建）', '临时测试产物',
         '临时测试产物（诊断小跑 checkpoint）']

# What already sits outside the pod, and at what strength the claim is checked.
# 'verified' rows were re-downloaded/re-read and compared byte for byte this round.
EXTERNAL = [
    ('a3_train_runs/a3_lora_r1/step_002000_merged.pt',
     'HuggingFace model wsadwsad6779/fasterwam-a3-cookie-lora: step_002000_merged.pt',
     '4091b02a23b3e35fb63b8bac38ea8e9c6d44439c052ed5aa5db151435a175601', 'verified'),
    ('a3_data/a3_fw/frames.dat',
     'HuggingFace model wsadwsad6779/fasterwam-a3-cookie-lora: data/frames.dat',
     '5fc73ae59729cef7', 'size_only'),
    ('data/a3_front_close_left_100',
     'HuggingFace dataset Eter0109/a3-front-close-left-100 @8fdd8a7caf8275eba8be8a83b53a99176a084062',
     None, 'size_only'),
    ('FasterWAM/src', 'GitHub hustvl/FasterWAM @83667817df0d4f823f39d90700e61ea2f432ac45',
     None, 'upstream'),
    ('FasterWAM/checkpoints/fasterwam_release',
     'HuggingFace/Wan-AI upstream release step_021700.pt (re-downloadable)',
     '11117679929', 'size_only'),
    ('FasterWAM/checkpoints/Wan-AI',
     'ModelScope Wan-AI/Wan2.2-TI2V-5B (re-downloadable)', None, 'upstream'),
    ('a3_dual_arm_sim/', 'GitHub Eter0109/a3_dual_arm_sim @08ebe8c', None, 'upstream'),
    ('fasterwam-a3-cookie/',
     'GitHub wsadwsad6779/fasterwam-a3-cookie branch backup-2026-11-01-r6-audit '
     '@781f55af3404a77a04da92f4608ea5074bd77470 (diff patch verified)',
     None, 'verified'),
    ('_r6_stage/',
     'GitHub wsadwsad6779/fasterwam-a3-cookie pod_code/bundle_code_r6_stage.txt '
     '@781f55af (bundle sha 933c228ca5e0ad2b)', None, 'verified'),
]

HIST = ('r4/r5 训练末态的 proprio_encoder 权重（2 张量 / 36,864 元素 / 147,456 B 张量字节）'
        '在 pod 上没有任何副本，本次备份不能恢复它；备份覆盖的是 LoRA adapter、优化器/调度器/RNG '
        '状态与 r3 基座，因此历史评估组合可复现，训练末态模型不可复现。')


def rel_of(p):
    return p.replace('/workspace/', '')


def external_of(p):
    r = rel_of(p)
    hits = [(s, sha, kind) for key, s, sha, kind in EXTERNAL
            if r == key or r.startswith(key)]
    return hits[0] if hits else (None, None, None)


def load():
    raw = ROWS.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    assert got == EXPECT_SHA, 'input TSV sha mismatch: %s' % got
    rows = []
    for line in raw.decode('utf-8').splitlines():
        if not line.strip():
            continue
        f = line.split('\t')
        rows.append({'path': f[0], 'bytes': int(f[1]), 'mtime_utc_pod': f[2],
                     'suffix': f[3], 'sha256': f[4],
                     'referenced_by_live_config': f[5] == 'True',
                     'groups': f[6].split('|'), 'hash_seconds': f[7]})
    return rows


def main():
    rows = load()
    bysha = defaultdict(list)
    for r in rows:
        if r['sha256']:
            bysha[r['sha256']].append(r['path'])
    for r in rows:
        store, csha, kind = external_of(r['path'])
        r['claimed_external_copy'] = store
        r['external_copy_check'] = kind
        r['pod_local_copies'] = [q for q in bysha.get(r['sha256'], [])
                                 if q != r['path']] if r['sha256'] else []
        r['category'] = GROUP_CAT[r['groups'][0]]
        r['other_verified_copy'] = (
            'pod 内同 sha 副本 %d 份' % len(r['pod_local_copies'])
            if r['pod_local_copies'] else
            ('外部已逐字节核验' if kind == 'verified' else
             ('外部存在但只核过大小' if kind == 'size_only' else
              ('上游可重建' if kind == 'upstream' else '无'))))
        r['backup_status'] = (
            '仅存在于 pod，必须外存' if (not r['claimed_external_copy']
                                          and not r['pod_local_copies'])
            else '已有副本（见 other_verified_copy）')

    total = sum(r['bytes'] for r in rows)
    hashed = sum(r['bytes'] for r in rows if r['sha256'])
    by_cat = defaultdict(lambda: {'files': 0, 'bytes': 0, 'sha_ok': 0, 'pod_only_bytes': 0})
    for r in rows:
        c = by_cat[r['category']]
        c['files'] += 1
        c['bytes'] += r['bytes']
        c['sha_ok'] += 1 if r['sha256'] else 0
        if r['backup_status'] == '仅存在于 pod，必须外存':
            c['pod_only_bytes'] += r['bytes']

    # directory rollup: first three path segments
    dirs = defaultdict(lambda: {'files': 0, 'bytes': 0, 'pod_only': 0, 'cat': set()})
    for r in rows:
        seg = rel_of(r['path']).split('/')
        key = '/'.join(seg[:3]) if len(seg) > 2 else '/'.join(seg[:2])
        d = dirs[key]
        d['files'] += 1
        d['bytes'] += r['bytes']
        d['cat'].add(r['category'])
        if r['backup_status'] == '仅存在于 pod，必须外存':
            d['pod_only'] += r['bytes']

    pod_only_big = [r for r in rows if r['backup_status'] == '仅存在于 pod，必须外存'
                    and r['bytes'] >= 50 * 2**20]
    doc = {
        'written_by': 'build_backup_manifest_v2.py (local)',
        'input': {'file': 'dl/manifest_rows.tsv', 'sha256': EXPECT_SHA,
                  'pod_source': '_r6_stage/out/manifest_rows.tsv',
                  'generated_utc_pod': '2026-11-01T03:51:39Z'},
        'pod': 'rgapi-one-click-24316-57b6b870 (/workspace)',
        'totals': {
            'files': len(rows), 'bytes': total, 'gb': round(total / 2**30, 3),
            'hashed_bytes': hashed,
            'files_without_sha': sum(1 for r in rows if not r['sha256']),
            'duplicate_groups': sum(1 for s, p in bysha.items() if len(p) > 1),
            'pod_only_files': sum(1 for r in rows
                                  if r['backup_status'] == '仅存在于 pod，必须外存'),
            'pod_only_bytes': sum(r['bytes'] for r in rows
                                  if r['backup_status'] == '仅存在于 pod，必须外存'),
        },
        'by_category': {k: v for k, v in sorted(
            by_cat.items(), key=lambda kv: ORDER.index(kv[0])
            if kv[0] in ORDER else 99)},
        'by_directory': {k: {'files': v['files'], 'bytes': v['bytes'],
                             'pod_only_bytes': v['pod_only'],
                             'categories': sorted(v['cat'])}
                         for k, v in sorted(dirs.items(), key=lambda kv: -kv[1]['bytes'])},
        'duplicate_groups': {s: p for s, p in bysha.items() if len(p) > 1},
        'historical_limitation': HIST,
        'entries': rows,
    }
    OUTJSON.write_text(json.dumps(doc, indent=1), encoding='utf-8')

    L = ['# BACKUP_MANIFEST v2（pod 全量对账版）', '',
         '- 生成：2026-11-01，pod `rgapi-one-click-24316-57b6b870`，根目录 `/workspace`。',
         '- v1（`BACKUP_MANIFEST.md`，963 个 / 47.88 GiB）只枚举了当时想到的目录。'
         '本版用一次全量 `find`（4,933 个文件 / 92.08 GiB）与清单做差集，'
         '补入 2,925 个此前未记录的文件，跳过 1,046 个 shell 碎片/缓存/字节码。',
         '- 数据链路：`backup_scan.py` → `scan_extra.py` → `scan_checkpoints.py`（枚举）→ '
         '`backup_hash.py`（流式 sha256）→ `manifest_rows.py`（合并成 TSV）→ 本脚本。'
         '合并表回传 8 个分片后本地 sha256 = `%s`，与 pod 端记录一致，逐字节相同。' % EXPECT_SHA[:16],
         '',
         '## 1 汇总', '',
         '| 指标 | 数值 |', '|---|---|',
         '| 文件数 | %d |' % len(rows),
         '| 总字节 | %s B（%.2f GiB） |' % (format(total, ','), total / 2**30),
         '| 已算出 sha256 | %d 个 / %s B（%.1f%%） |' % (
             doc['totals']['files'] - doc['totals']['files_without_sha'],
             format(hashed, ','), 100.0 * hashed / total),
         '| 无 sha256 | %d 个 |' % doc['totals']['files_without_sha'],
         '| pod 内同 sha 重复组 | %d 组 |' % doc['totals']['duplicate_groups'],
         '| 仅存在于 pod、尚无外部副本 | %d 个 / %s B（%.2f GiB） |' % (
             doc['totals']['pod_only_files'], format(doc['totals']['pod_only_bytes'], ','),
             doc['totals']['pod_only_bytes'] / 2**30),
         '| 被现役配置/脚本按绝对路径引用 | %d 个 |' % sum(
             1 for r in rows if r['referenced_by_live_config']),
         '',
         '每个条目带 8 个字段：绝对路径、字节数、pod 端 mtime(UTC)、完整 sha256、'
         '分组、类别、是否被现役引用、副本状态（pod 内副本 / 外部副本 / 校验强度）。'
         '逐条目原始记录在 `backup_manifest_v2.json` 的 `entries`（%d 条）与 `manifest_rows.tsv`。' % len(rows),
         '',
         '## 2 类别覆盖', '',
         '| 类别 | 文件数 | 字节 | 有 sha256 | 仅 pod 字节 |', '|---|---|---|---|---|']
    for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1]['bytes']):
        L.append('| %s | %d | %s | %d | %s |' % (
            k, v['files'], format(v['bytes'], ','), v['sha_ok'],
            format(v['pod_only_bytes'], ',')))
    L += ['', '> **历史限制（必须随备份一起引用）**：' + HIST, '',
          '## 3 目录级汇总（前 3 层路径）', '',
          '| 目录 | 文件数 | 字节 | 其中仅 pod | 类别 |', '|---|---|---|---|---|']
    for k, v in sorted(dirs.items(), key=lambda kv: -kv[1]['bytes'])[:60]:
        L.append('| `%s` | %d | %s | %s | %s |' % (
            k, v['files'], format(v['bytes'], ','), format(v['pod_only'], ','),
            '；'.join(sorted(v['cat']))))
    L += ['', '## 4 尚无外部副本的大文件（≥50 MiB，必须先传到 pod 外才算备份完成）', '',
          '| 路径 | 字节 | mtime(UTC) | SHA256（pod 端） | 类别 |', '|---|---|---|---|---|']
    for r in sorted(pod_only_big, key=lambda x: -x['bytes']):
        L.append('| `%s` | %s | %s | `%s…` | %s |' % (
            r['path'], format(r['bytes'], ','), r['mtime_utc_pod'], r['sha256'][:24],
            r['category']))
    L += ['', '共 %d 个，%s B（%.2f GiB）。' % (
        len(pod_only_big), format(sum(r['bytes'] for r in pod_only_big), ','),
        sum(r['bytes'] for r in pod_only_big) / 2**30), '',
        '## 5 与 v1 的差异', '',
        '- v1 的 963 条全部保留；新增 2,925 条。',
        '- 新增中最大的一块是上游基础模型 `FasterWAM/checkpoints/`（25.32 GB，8 个文件）——'
        '它不属于科研成果，可外部重建，不需要占用外存预算，但已入清单并给出 sha256 以便重建后比对。',
        '- 新发现 `a3_train_runs/a3_lora_r3/checkpoints/weights/step_003000.pt`（11,179,613,340 B）'
        '是 r3 的完整训练状态（模型+优化器），v1 漏掉了它；这是 r4/r5 的训练基座，'
        '目前只有 pod 一份，属于最高优先级外存对象。',
        '- `a3_train_runs/*/checkpoints/state` 全部为空目录（0 个文件）：'
        'pod 上根本不存在独立的 sampler/dataloader 状态文件，任何"可断点续训"的说法都不包含采样器状态。',
        '- r4/r5 的 `checkpoints/light` 每个 run 有 8 个中间 step（40 个文件），'
        'v1 只记录了最后一步；诊断小跑 chk3_A/A2/B/B2/B3、chk4_N/S、r4_crashed_step500 '
        '此前完全未入清单。',
        '', '## 6 pod 内部同 sha256 重复组', '',
        '同内容多副本意味着"pod 内复制"不等于"pod 外备份"。共 %d 组，'
        '完整列表见 `backup_manifest_v2.json` 的 `duplicate_groups`。前 25 组：'
        % doc['totals']['duplicate_groups'], '']
    for s, p in sorted(doc['duplicate_groups'].items(), key=lambda kv: -len(kv[1]))[:25]:
        L.append('- `%s`（%d 份）: %s' % (
            s[:16], len(p), '，'.join('`%s`' % q.replace('/workspace/', '') for q in p[:6])))
    OUTMD.write_text('\n'.join(L), encoding='utf-8')
    print('files %d bytes %s gb %.2f sha_ok %d dup %d pod_only %d/%s B' % (
        len(rows), format(total, ','), total / 2**30,
        sum(1 for r in rows if r['sha256']), doc['totals']['duplicate_groups'],
        doc['totals']['pod_only_files'], format(doc['totals']['pod_only_bytes'], ',')))
    for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1]['bytes']):
        print('%-32s files=%-5d bytes=%-15s pod_only=%-15s' % (
            k, v['files'], format(v['bytes'], ','), format(v['pod_only_bytes'], ',')))
    print('md %d json %d' % (OUTMD.stat().st_size, OUTJSON.stat().st_size))


main()
