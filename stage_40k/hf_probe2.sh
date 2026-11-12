#!/usr/bin/env bash
# Connectivity + layout probe for the HF backup path. Read-only, no GPU, no writes
# outside /workspace/_v1_stage/backup40k.
set -u
echo "=== proxy env (credentials masked) ==="
env | grep -iE 'http_proxy|https_proxy|no_proxy|all_proxy' || echo "(none)"

echo "=== dns ==="
getent ahostsv4 huggingface.co | head -3
getent ahostsv6 huggingface.co | head -3 || echo "(no v6)"

echo "=== curl -4 huggingface.co ==="
curl -4 -m 20 -sS -o /dev/null -w '%{http_code} %{time_total}s\n' https://huggingface.co/api/models || echo "curl -4 failed rc=$?"

echo "=== curl -6 huggingface.co ==="
curl -6 -m 12 -sS -o /dev/null -w '%{http_code} %{time_total}s\n' https://huggingface.co/api/models || echo "curl -6 failed rc=$?"

echo "=== python urllib, default family ==="
/opt/venv/bin/python - <<'PY'
import urllib.request, socket
try:
    print('status', urllib.request.urlopen('https://huggingface.co/api/models', timeout=20).status)
except Exception as exc:
    print(type(exc).__name__, str(exc)[:160])
PY

echo "=== python urllib, forced IPv4 ==="
/opt/venv/bin/python - <<'PY'
import urllib.request, socket
socket.setdefault_family(socket.AF_INET)
try:
    print('status', urllib.request.urlopen('https://huggingface.co/api/models', timeout=20).status)
except Exception as exc:
    print(type(exc).__name__, str(exc)[:160])
PY

echo "=== whoami through huggingface_hub, real endpoint ==="
/opt/venv/bin/python - <<'PY'
import os
from huggingface_hub import HfApi
tok = open('/root/.cache/hf_token').read().strip()
for ep in ('https://huggingface.co', 'https://hf-mirror.com'):
    try:
        info = HfApi(endpoint=ep, token=tok).whoami()
        keys = {k: info[k] for k in ('name', 'type', 'fullname', 'plan') if k in info}
        print(ep, 'OK', keys, 'scopes=', info.get('scope') or info.get('scopes'))
    except Exception as exc:
        print(ep, 'FAIL', type(exc).__name__, str(exc)[:160])
PY

echo "=== rehearsal node layout ==="
ls -l /workspace/_v1_stage/rehearsal65/run/checkpoints/light/step_000012/

echo "=== ram (bytes) ==="
free -b | head -2

echo "=== run40k tree ==="
ls -1 /workspace/_v1_stage/run40k
find /workspace/_v1_stage/run40k -maxdepth 4 -name 'overrides.yaml' -o -maxdepth 4 -name '*.yaml' | head -5

echo "=== training heartbeat ==="
tail -c 6000 /workspace/_v1_stage/run40k/logs/seg1.log | grep -oE 'step=[0-9]+/5000 loss=[0-9.]+' | tail -2
date -u
