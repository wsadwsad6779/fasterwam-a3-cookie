#!/bin/bash
# Pre-flight for step 6, read-only except for files this script creates in _probe_1112/.
# The claim it exists to test is the one RUN_BATCH_STEP6.sh depends on: that the 40k node costs no
# merge because its publish record is on disk. batch_eval.py reuses a merged artifact only when
# BOTH the file and $OUT/_work/publish_record_stepNNNNNN.json exist, so with a fresh --out the first
# node would re-merge 11 GiB over the bytes the live smoke scored. Everything below is written to
# find out, before 40 hours of rollouts, which of the two is true.
export PYTHONDONTWRITEBYTECODE=1
ST=/workspace/_v1_stage/dev/_probe_1112
D=/workspace/_v1_stage/dev/eval40k
B=/workspace/_v1_stage/dev/out40k
PY=/opt/venv/bin/python
cd "$ST" || exit 9

echo "== A. volume and the merged slot =="
{ date -u +%FT%TZ; df -h /workspace; echo; ls -l $D/merged/ 2>&1; echo; ls -l $D/merged_hold/ 2>&1; } > pf_disk.out 2>&1

echo "== B. where the publish records live, and what the step-40000 one says =="
{ find $B -maxdepth 3 -name 'publish_record_step*.json' -printf '%s\t%p\n' 2>&1
  echo "--- also under dev/eval40k ---"
  find $D -maxdepth 3 -name 'publish_record_step*.json' -printf '%s\t%p\n' 2>&1; } > pf_records.out 2>&1
$PY - <<'PYX' > pf_records.json 2>&1
import json, subprocess
from pathlib import Path
out = []
for p in sorted(Path('/workspace/_v1_stage/dev/out40k').glob('*/_work/publish_record_step*.json')) + \
          sorted(Path('/workspace/_v1_stage/dev/eval40k').glob('*/publish_record_step*.json')):
    try:
        d = json.loads(p.read_text())
    except Exception as exc:
        out.append({'path': str(p), 'unreadable': str(exc)[:200]})
        continue
    merged = d.get('output') or d.get('merged') or d.get('path')
    rec = {'path': str(p), 'published_step': d.get('published_step'),
           'published_force_calibration': str(d.get('published_force_calibration'))[:120],
           'bytes': d.get('bytes') or d.get('size') or d.get('merged_bytes'),
           'output': merged, 'keys': sorted(d)[:14]}
    if merged and Path(str(merged)).is_file():
        rec['output_exists'] = True
        rec['output_size'] = Path(str(merged)).stat().st_size
    out.append(rec)
print(json.dumps({'records': out, 'count': len(out)}, indent=2, sort_keys=True, default=str))
PYX

echo "== C. the identity of the artifact the smoke scored =="
$PY - <<'PYX' > pf_merged_sha.out 2>&1
import hashlib, json
from pathlib import Path
p = Path('/workspace/_v1_stage/dev/eval40k/merged/step_040000_merged.pt')
out = {'path': str(p), 'exists': p.is_file()}
if p.is_file():
    out['bytes'] = p.stat().st_size
    with p.open('rb') as fh:
        out['first_mib_sha16'] = hashlib.sha256(fh.read(1 << 20)).hexdigest()[:16]
    out['expect'] = {'bytes': 11126493579, 'first_mib_sha16': 'f9fe3c9407c89169'}
    out['matches_smoke_identity'] = (out['bytes'] == out['expect']['bytes']
                                     and out['first_mib_sha16'] == out['expect']['first_mib_sha16'])
print(json.dumps(out, indent=2, sort_keys=True))
PYX
cat pf_merged_sha.out

echo "== D. is the box otherwise idle, and what the GPU reports =="
{ command -v nvidia-smi; ls -l /usr/bin/nvidia-smi /usr/local/nvidia/bin/nvidia-smi 2>&1; } > pf_smi.out 2>&1
{ ps -eo pid,ppid,etime,rss,cmd --sort=-rss | head -20; } > pf_ps.out 2>&1
$PY - <<'PYX' > pf_gpu.out 2>&1
import json
out = {}
try:
    import torch
    out['torch'] = torch.__version__
    out['cuda_available'] = bool(torch.cuda.is_available())
    if out['cuda_available']:
        free, total = torch.cuda.mem_get_info()
        out['device'] = torch.cuda.get_device_name(0)
        out['free_gib'] = round(free / 2**30, 2)
        out['total_gib'] = round(total / 2**30, 2)
except Exception as exc:
    out['error'] = str(exc)[:300]
print(json.dumps(out, indent=2, sort_keys=True))
PYX

echo "== E. the protocol the batch would freeze against (fresh dir, nothing existing touched) =="
$PY $D/batch_eval.py --dry-run --out $ST/pf_dry > pf_batchdry.out 2>&1
echo "  batch_eval --dry-run rc=$?"
$PY - <<'PYX' > pf_plan_summary.json 2>&1
import json
from pathlib import Path
p = Path('/workspace/_v1_stage/dev/_probe_1112/pf_dry/plan.json')
out = {'plan_exists': p.is_file()}
if p.is_file():
    d = json.loads(p.read_text())
    out['generated'] = d.get('generated')
    out['seeds'] = d.get('seeds')
    out['n_seeds'] = len(d.get('seeds') or [])
    out['nodes'] = [{'step': n.get('step'), 'status': n.get('status'),
                     'merged': bool(n.get('merged_artifact'))} for n in d.get('nodes', [])]
    out['top_level_keys'] = sorted(d)[:24]
    for k in ('blockers', 'backend', 'protocol', 'scene', 'force_calibration', 'calibration'):
        if k in d:
            out[k] = d[k]
print(json.dumps(out, indent=2, sort_keys=True, default=str))
PYX
cat pf_plan_summary.json
echo "== done =="
ls -l pf_*.out pf_*.json r9_dryrun.json 2>/dev/null
