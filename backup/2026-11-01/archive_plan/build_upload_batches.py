import json, collections, re, os

GB = 1024 ** 3
REPO = 'wsadwsad6779/fasterwam-a3-cookie-lora'
W = '/workspace/'
d = json.load(open('backup_status_abc.json', encoding='utf-8'))
E = [{'path': e['path'], 'bytes': int(e['bytes']), 'abc': e['abc'], 'sha256': e['sha256'],
      'where': e['where'], 'dup': str(e['pod_internal_dup']) == 'True'} for e in d['entries']]

CODE_EXT = ('.py', '.sh', '.yaml', '.yml', '.toml', '.md', '.txt', '.json', '.csv', '.lock', '.cfg', '.ini', '.b64')
BIGCODE = ('.npz', '.dat', '.pt', '.pkl', '.gz', '.png', '.mp4', '.npy', '.safetensors', '.bin', '.h5')

RUN_RE = re.compile(r'^/workspace/a3_train_runs/(a3_lora_r\d|a3_lora_r4_crashed)/(.*)$')
LIGHT_RE = re.compile(r'^/workspace/a3_train_runs/(a3_lora_r\d(?:_crashed)?)/checkpoints/light/(step_\d+)/([^/]+)$')
CHKSMOKE_RE = re.compile(r'^/workspace/a3_train_runs/chk[34]_[A-Za-z0-9]+/')
STAGE_RE = re.compile(r'^/workspace/(_r\d_stage|_r6_stage|_selftest|dagger_logs)/(.*)$')


def rel(p, pre):
    return p[len(pre):]


def classify(e):
    """returns (batch, action, path_in_repo, why)"""
    p, tier = e['path'], e['abc']

    # ---- licence-gated: never copied into a public repo ----
    if p.startswith(W + 'data/a3_front_close_left_100'):
        return ('L', 'do_not_upload_license', '',
                '上游 Eter0109/a3-front-close-left-100 无 license 字段；只引用 revision 8fdd8a7caf… 并做回读核验')
    if p.startswith(W + 'FasterWAM/checkpoints/') or p.startswith(W + 'fasterwam_release'):
        return ('L', 'do_not_upload_upstream', '',
                'FasterWAM / Wan-AI 上游权重，未获再分发许可；只记来源与校验值')

    # ---- already byte-identical outside the pod (A/A0) or claimed-only (B) ----
    if tier == 'A0':
        return ('V', 'verify_only', '', '0 字节占位符，哈希相等不构成备份，无需上传')
    if tier == 'A' and p.endswith('.mp4'):
        return ('P0', 'upload', 'videos/' + p[len(W):], '评估视频，需进 HF 才算持久')
    if tier == 'A':
        return ('V', 'verify_readback', '', 'pod 外已有逐字节相同副本，复验即可')

    # ---- irreplaceable weights ----
    m = RUN_RE.match(p)
    if m and p.endswith('step_003000_merged.pt') and 'a3_lora_r3' in p:
        return ('P0', 'upload', 'weights/r3/step_003000_merged.pt', 'r4/r5 评估基座，pod 唯一副本')
    if p == W + 'a3_train_runs/a3_lora_r3/checkpoints/weights/step_003000.pt':
        return ('P0', 'upload', 'weights/r3/full_state/step_003000.pt', 'r3 末态全量权重，pod 唯一副本')
    if p.endswith('step_003000_merged.pt') and 'a3_lora_r2' in p:
        return ('P1', 'upload', 'weights/r2/step_003000_merged.pt', 'r3 的基座，历史权重')
    if p.endswith('step_002000_merged.pt') and 'a3_lora_r1' in p:
        return ('P1', 'upload', 'weights/r1/step_002000_merged.pt', 'r1 历史权重')

    lm = LIGHT_RE.match(p)
    if lm:
        run, step, fn = lm.group(1), lm.group(2), lm.group(3)
        if run == 'a3_lora_r4' and step == 'step_004000':
            return ('P0', 'upload', 'weights/r4/light/step_004000/' + fn, 'r4 末态轻量训练状态')
        if run == 'a3_lora_r5' and step == 'step_008000':
            return ('P0', 'upload', 'weights/r5/light/step_008000/' + fn, 'r5 末态轻量训练状态')
        if run in ('a3_lora_r4', 'a3_lora_r5'):
            return ('P1', 'upload', 'weights/%s/light/%s/%s' % (run[-2:], step, fn), '中间 step，用于恢复与曲线复现')
        if run == 'a3_lora_r4_crashed':
            return ('P1', 'upload', 'weights/r4_crashed/%s/%s' % (step, fn), '崩溃现场留档')
        return ('P2', 'not_upload_smoke', '', '%s %s 只跑了 <=10 step 的冒烟权重，无研究价值；代码可复现，故不上传但仍是 pod 唯一副本，不得删除' % (run, step))

    if CHKSMOKE_RE.match(p):
        if p.endswith(CODE_EXT):
            return ('GIT', 'git_only', '', '冒烟测试脚本/配置，进 GitHub')
        return ('P2', 'not_upload_smoke', '', 'chk3/chk4 冒烟运行产物，不上传但不得删除')

    if p.startswith(W + 'a3_train_runs/'):
        rest = rel(p, W + 'a3_train_runs/')
        if rest.endswith(('.yaml', '.json', '.txt', '.log', '.md', '.csv')) or '/weights/' in rest:
            return ('P0', 'upload', 'runs/' + rest, '训练配置、日志与恢复信息')
        if rest.endswith('.pt'):
            return ('P1', 'upload', 'runs/' + rest, '训练产物权重')
        return ('P2', 'not_upload_other', '', 'a3_train_runs 内杂项')

    # ---- corpora we produced ----
    if p.startswith(W + 'a3_data/a3_fw_dagger/'):
        return ('P0', 'upload', 'data/a3_fw_dagger/' + rel(p, W + 'a3_data/a3_fw_dagger/'), '自建低力/纠正语料')
    if p.startswith(W + 'a3_data/a3_fw_dagger_near/'):
        return ('P0', 'upload', 'data/a3_fw_dagger_near/' + rel(p, W + 'a3_data/a3_fw_dagger_near/'), '自建近距纠正语料')
    if p.startswith(W + 'a3_data/a3_fw_force/'):
        return ('P0', 'upload', 'data/a3_fw_force/' + rel(p, W + 'a3_data/a3_fw_force/'), '力统计与力数组')
    if p.startswith(W + 'a3_data/a3_fw/'):
        fn = rel(p, W + 'a3_data/a3_fw/')
        if fn == 'frames.dat':
            return ('V', 'verify_readback', 'data/frames.dat', 'HF 已有同字节数副本但未逐字节核验；必须回读算 sha256')
        return ('P0', 'upload', 'data/a3_fw/' + fn, '归一化统计与索引')
    if p.startswith(W + 'a3_data/text_embeds_cache/'):
        return ('P2', 'not_upload_cache', '', '文本嵌入缓存，可由上游模型重新生成')
    if p.startswith(W + 'a3_data/'):
        return ('P0', 'upload', 'data/' + rel(p, W + 'a3_data/'), 'a3_data 其余自建件')

    # ---- evaluation records ----
    if p.startswith(W + 'a3_eval/'):
        rest = rel(p, W + 'a3_eval/')
        if rest.endswith('.mp4'):
            return ('P0', 'upload', 'videos/' + rest, '评估视频')
        if rest.endswith('.pt'):
            return ('P1', 'upload', 'tests/' + rest, 'light_state 修复验证产物')
        return ('P0', 'upload', 'eval/a3_eval/' + rest, '评估记录本体')
    if p.startswith(W + 'fasterwam_a3_bridge/bridge/outputs/'):
        rest = rel(p, W + 'fasterwam_a3_bridge/bridge/outputs/')
        if rest.endswith('.mp4'):
            return ('P0', 'upload', 'videos/bridge/' + rest, '评估视频')
        if rest.endswith('.dat'):
            return ('P1', 'upload', 'replay/bridge/' + rest, '物理回放帧，需 GPU 才能重生成')
        return ('P0', 'upload', 'eval/bridge/' + rest, '评估记录本体')

    # ---- code / docs / stage artifacts ----
    sm = STAGE_RE.match(p)
    if sm:
        rest = rel(p, W)
        if rest.endswith(CODE_EXT):
            return ('GIT', 'git_only', '', '阶段脚本与日志，进 GitHub 的 pod_snapshot')
        if rest.endswith('.npz') or rest.endswith('.csv'):
            return ('P1', 'upload', 'research/' + rest, '研究数据（专家力分布、掩膜、周期表）')
        return ('P2', 'not_upload_scratch', '', '阶段目录内的临时/中间文件')
    if p.startswith(W + 'FasterWAM/') or p.startswith(W + 'a3_dual_arm_sim/') or p.startswith(W + 'fasterwam_a3_bridge/'):
        rest = rel(p, W)
        if rest.endswith(CODE_EXT):
            return ('GIT', 'git_only', '', '项目代码，进 GitHub')
        if rest.endswith(BIGCODE):
            return ('P2', 'not_upload_upstream_or_big', '', '上游仓库附带资源，不复制进公开仓库')
        return ('GIT', 'git_only', '', '项目内其余文本文件，进 GitHub')

    # ---- pod root ----
    base = p[len(W):]
    if '/' not in base:
        if base.endswith(('.md', '.json', '.txt', '.log', '.py', '.sh', '.yaml', '.csv')):
            return ('GIT', 'git_only', '', 'pod 根目录报告/脚本，进 GitHub')
        if base.endswith(('.npz', '.pkl')):
            return ('P1', 'upload', 'research/' + base, 'pod 根目录研究数据')
        return ('P2', 'not_upload_archive', '', 'pod 根目录打包件（a3.tar.gz 等），内容与仓库重复')
    return ('P2', 'not_upload_other', '', '未分类：默认为不上传，保留在 pod 且不得删除')


rows = []
for e in sorted(E, key=lambda x: x['path']):
    b, a, ir, why = classify(e)
    rows.append({'batch': b, 'action': a, 'pod_path': e['path'], 'bytes': e['bytes'], 'tier': e['abc'],
                 'sha256': e['sha256'], 'path_in_repo': ir, 'note': why})

json.dump({'repo': REPO, 'source_scan': 'backup_status_abc.json', 'n_files': len(rows), 'rows': rows},
          open('upload_batches.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)

agg = collections.defaultdict(lambda: [0, 0])
for r in rows:
    k = (r['batch'], r['action'], r['tier'])
    agg[k][0] += 1
    agg[k][1] += r['bytes']
lines = []
for k in sorted(agg):
    lines.append('%-4s %-26s tier=%-3s n=%-5d %9.3f GiB' % (k[0], k[1], k[2], agg[k][0], agg[k][1] / GB))
txt = '\n'.join(lines)
print(txt)

up = sum(r['bytes'] for r in rows if r['action'] == 'upload')
p0 = sum(r['bytes'] for r in rows if r['action'] == 'upload' and r['batch'] == 'P0')
p1 = sum(r['bytes'] for r in rows if r['action'] == 'upload' and r['batch'] == 'P1')
git = sum(r['bytes'] for r in rows if r['action'] == 'git_only')
nu = sum(r['bytes'] for r in rows if r['action'].startswith('not_upload') or r['action'] == 'do_not_upload_license')
print('upload total %.3f GiB (P0 %.3f + P1 %.3f) | git %.3f GiB | not-uploaded %.3f GiB' % (
    up / GB, p0 / GB, p1 / GB, git / GB, nu / GB))
open(os.path.join('_r6_stage', 'out', 'batch_plan_summary.txt'), 'w', encoding='utf-8').write(
    txt + '\nupload %.3f GiB P0 %.3f P1 %.3f git %.3f notupload %.3f\n' % (up / GB, p0 / GB, p1 / GB, git / GB, nu / GB))
