"""Which upload path does this instance actually have? Determined by uploading, not reading docs.

Three independent variables decide whether an eight-node backup is possible from here, and
the earlier probe conflated them:

  * the endpoint serving the API (the mirror answers the control plane but rewrites upload
    URLs; huggingface.co is reachable from the shell but refused by the proxy from Python);
  * the upload mode the server picks for the extension — a small .json travels inside a
    commit request (same host as the API), an LFS object travels to a separate S3 host, and
    only the second one needs a route to `*.s3-accelerate.amazonaws.com`;
  * whether the egress proxy is load-bearing or just an environment default.

Each cell is one real write, timed, with the full masked exception chain, and each remote
path is read back afterwards so a "success" that stored nothing cannot be reported. Uploads
go to `transport_probe/`, never into the namespace the checkpoint nodes will use.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

TOKEN_FILE = "/root/.cache/hf_token"
REPO = "wsadwsad6779/fasterwam-a3-force-40k"
OUT = Path("/workspace/_v1_stage/backup40k/write_probe2.json")
TOKEN = open(TOKEN_FILE, encoding="utf-8").read().strip()


def mask(text):
    return str(text).replace(TOKEN, "<HF_TOKEN>")


def chain(exc):
    seen, cur, out = set(), exc, []
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        out.append("%s: %s" % (type(cur).__name__, mask(cur)[:200]))
        cur = cur.__cause__ or cur.__context__
    return out


def blob(nbytes, seed):
    """Deterministic pseudo-random payload that will not collapse under compression."""
    h = hashlib.sha256(("%s|%s" % (seed, nbytes)).encode()).digest()
    out = bytearray()
    while len(out) < nbytes:
        h = hashlib.sha256(h).digest()
        out += h
    return bytes(out[:nbytes])


def no_proxy_env():
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(k, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def run_cell(endpoint, sizes, proxy_unset, rows):
    label = "%s | proxy=%s" % (endpoint, "unset" if proxy_unset else "default")
    if proxy_unset:
        no_proxy_env()
    for var in ("HF_ENDPOINT",):
        os.environ[var] = endpoint
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    for mod in [m for m in list(sys.modules) if m.startswith("huggingface_hub")]:
        del sys.modules[mod]
    try:
        from huggingface_hub import HfApi
        from huggingface_hub import constants as hc
        hc.HF_ENDPOINT = endpoint
    except Exception as exc:  # noqa: BLE001
        rows.append({"cell": label, "fatal": chain(exc)})
        return
    api = HfApi(endpoint=endpoint, token=TOKEN)
    cell = {"cell": label, "endpoint_seen_by_lib": getattr(hc, "HF_ENDPOINT", None), "writes": []}
    for name, nbytes in sizes:
        path = Path("/tmp") / name
        path.write_bytes(blob(nbytes, name))
        rel = "transport_probe/%s/%s" % ("noproxy" if proxy_unset else "proxy", name)
        rec = {"file": name, "bytes": nbytes, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "path_in_repo": rel}
        t0 = time.time()
        try:
            api.upload_file(path_or_fileobj=str(path), path_in_repo=rel, repo_id=REPO,
                            repo_type="model", commit_message="probe %s %d" % (name, nbytes))
            rec["ok"] = True
        except Exception as exc:  # noqa: BLE001
            rec["ok"] = False
            rec["chain"] = chain(exc)
            rec["tb_tail"] = mask(traceback.format_exc().strip().splitlines()[-1])[:200]
        rec["seconds"] = round(time.time() - t0, 2)
        if rec.get("ok"):
            rec["mbps"] = round(nbytes / 1e6 / max(rec["seconds"], 1e-3), 2)
        cell["writes"].append(rec)
        path.unlink(missing_ok=True)
        print(json.dumps({k: rec[k] for k in ("file", "bytes", "ok", "seconds")}), flush=True)
    try:
        listed = [e.path for e in api.list_repo_tree(repo_id=REPO, repo_type="model",
                                                     path_in_repo="transport_probe",
                                                     recursive=True)
                  if getattr(e, "size", None)]
        cell["remote_listed"] = sorted(listed)
    except Exception as exc:  # noqa: BLE001
        cell["remote_listed"] = {"error": chain(exc)}
    rows.append(cell)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sizes = [("meta_64k.json", 65536), ("blob_8m.bin", 8 * 1024 * 1024),
             ("blob_70m.bin", 70 * 1024 * 1024)]
    rows = []
    for endpoint, proxy_unset in (("https://hf-mirror.com", False),
                                  ("https://hf-mirror.com", True),
                                  ("https://huggingface.co", True)):
        try:
            run_cell(endpoint, sizes, proxy_unset, rows)
        except Exception as exc:  # noqa: BLE001
            rows.append({"cell": endpoint, "fatal": chain(exc)})
        print("CELL DONE " + endpoint + " proxy_unset=%s" % proxy_unset, flush=True)
    doc = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "dns": {h: sorted({i[4][0] for i in socket.getaddrinfo(h, 443)})
                   for h in ("huggingface.co", "hf-mirror.com")},
           "cells": rows}
    text = json.dumps(doc, indent=2, sort_keys=True, default=str)
    assert TOKEN not in text
    OUT.write_text(text)
    print(text[-4000:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
