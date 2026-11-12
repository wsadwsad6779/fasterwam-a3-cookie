#!/bin/bash
# Which network path can actually carry an upload off this instance?
# Three things have to be separated: the API host (control plane, works via the
# mirror), the S3 host the LFS PUT goes to (data plane), and whether the proxy is
# a requirement or merely an environment default. Prints no credential.
set -u
T=/workspace/_v1_stage
log() { echo "=== $*"; }

log "proxy env (values other than the proxy host are masked)"
env | grep -iE '^(http|https|no|all)_proxy=' | sed -E 's#(://)[^@]*@#\1<cred>@#'

log "curl, proxy honoured (what the shell does by default)"
for h in https://huggingface.co/api/models https://hf-mirror.com/api/models \
         https://hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com/ \
         https://cdn-lfs.hf-mirror.com/ https://s3.amazonaws.com/ ; do
  printf '%-70s ' "$h"
  curl -sS -o /dev/null -m 25 -w 'http=%{http_code} t=%{time_total}s\n' "$h" 2>&1 | head -1
done

log "curl, --noproxy '*' (is the proxy load-bearing or an artefact?)"
for h in https://huggingface.co/api/models https://hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com/ ; do
  printf '%-70s ' "$h"
  curl -sS -o /dev/null -m 25 --noproxy '*' -w 'http=%{http_code} t=%{time_total}s\n' "$h" 2>&1 | head -1
done

log "python, default env (proxy from environment)"
python3 - <<'PY'
import urllib.request, socket
for u in ("https://huggingface.co/api/models", "https://hf-mirror.com/api/models"):
    try:
        t0 = socket.getdefaulttimeout()
        with urllib.request.urlopen(u, timeout=25) as r:
            print("OK   ", u, r.status, len(r.read()))
    except Exception as e:
        print("FAIL ", u, type(e).__name__, str(e)[:120])
PY

log "python, proxy env unset in-process"
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
python3 - <<'PY'
import urllib.request
for u in ("https://huggingface.co/api/models",
          "https://hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com/",
          "https://huggingface.co/api/whoami-v2"):
    try:
        with urllib.request.urlopen(u, timeout=25) as r:
            print("OK   ", u, r.status, len(r.read()))
    except Exception as e:
        print("FAIL ", u, type(e).__name__, str(e)[:120])
PY

log "python, huggingface_hub whoami with proxy unset (mirror + direct)"
for ep in https://hf-mirror.com https://huggingface.co; do
  env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
    HF_ENDPOINT=$ep HF_HUB_DISABLE_XET=1 python3 - <<'PY' 2>&1 | tail -3
import os
from huggingface_hub import HfApi
tok = open("/root/.cache/hf_token").read().strip()
try:
    w = HfApi(endpoint=os.environ["HF_ENDPOINT"]).whoami(token=tok)
    print("whoami OK", os.environ["HF_ENDPOINT"], w.get("name"))
except Exception as e:
    print("whoami FAIL", os.environ["HF_ENDPOINT"], type(e).__name__, str(e)[:160])
PY
done

log "training still alive"
tail -2 $T/run40k/driver.log
grep -c . $T/run40k/logs/*.jsonl 2>/dev/null | tail -3
df -B1 --output=avail /workspace | tail -1
echo DONE_PROBE3
