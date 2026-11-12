"""Which endpoint can this instance actually WRITE to, and at what size?

The 40k backup needs an endpoint that accepts a 141 MB `optimizer.pt`. Two candidates,
and the container environment is misleading: `HF_ENDPOINT` is exported as hf-mirror.com,
`curl -4 https://huggingface.co` succeeds in 3 s through the egress proxy, and Python's
`huggingface_hub` on that same endpoint dies with a ProxyError. Reading the env vars does
not settle it, so this probe does the only thing that settles it: real uploads, small
first, then large, with the full exception chain printed for each.

Everything that leaves this script is masked: the credential is read from a file and
every string is passed through `mask()` so a presigned URL or an Authorization header
that some library decided to echo cannot put a token in a log file.

    /opt/venv/bin/python hf_write_probe.py --token-file /root/.cache/hf_token \
        --out backup40k/write_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import time
import traceback
from pathlib import Path

TOKEN = ""


def mask(text):
    return str(text).replace(TOKEN, "<HF_TOKEN>") if TOKEN else str(text)


def chain(exc):
    """Every link of the exception chain, masked. The outermost message is usually a lie."""
    out = []
    seen = exc
    while seen is not None:
        out.append("%s: %s" % (type(seen).__name__, mask(str(seen))[:400]))
        seen = seen.__cause__ or seen.__context__
        if len(out) > 6:
            break
    return out


def probe_endpoint(endpoint, repo, token, scratch):
    os.environ["HF_ENDPOINT"] = endpoint
    for mod in [m for m in list(sys_modules) if m.startswith("huggingface_hub")]:
        del sys_modules[mod]
    import huggingface_hub  # noqa: F401  (re-imported so constants.py picks the new endpoint)
    from huggingface_hub import HfApi

    api = HfApi(endpoint=endpoint, token=token)
    res = {"endpoint": endpoint, "hf_hub_version": huggingface_hub.__version__}
    try:
        api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        res["create_repo"] = "ok"
    except Exception as exc:  # noqa: BLE001
        res["create_repo"] = chain(exc)
        return res
    try:
        info = api.repo_info(repo, repo_type="model")
        res["repo"] = {"id": info.id, "private": info.private}
    except Exception as exc:  # noqa: BLE001
        res["repo"] = chain(exc)

    for size in (64 << 10, 8 << 20, 70 << 20):
        path = Path(scratch) / ("probe_%d.bin" % size)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(bytes(random.getrandbits(8) for _ in range(65536)) * (size // 65536))
        t0 = time.time()
        try:
            api.upload_file(path_or_file=str(path), repo_id=repo, repo_type="model",
                            path="writeprobe/%s_%d.bin" % (endpoint.split("//")[1].split(".")[0], size),
                            commit_message="write probe %d B" % size)
            res["upload_%d" % size] = {"ok": True, "seconds": round(time.time() - t0, 1),
                                       "bytes_per_second": int(size / max(time.time() - t0, 0.1))}
        except Exception as exc:  # noqa: BLE001
            res["upload_%d" % size] = {"ok": False, "seconds": round(time.time() - t0, 1),
                                       "chain": chain(exc)}
        finally:
            path.unlink(missing_ok=True)
    try:
        rows = []
        for e in api.list_repo_tree(repo_id=repo, repo_type="model",
                                    path_in_repo="writeprobe", recursive=True):
            get = e.get if isinstance(e, dict) else (lambda k: getattr(e, k, None))
            lfs = get("lfs") or {}
            lget = lfs.get if isinstance(lfs, dict) else (lambda k: getattr(lfs, k, None))
            rows.append({"path": get("path"), "size": get("size"),
                         "oid": get("oid") or get("sha") or get("blob_id"),
                         "lfs_oid": lget("oid"), "lfs_size": lget("size")})
        res["remote_writeprobe"] = rows
    except Exception as exc:  # noqa: BLE001
        res["remote_writeprobe"] = chain(exc)
    return res


def main():
    global TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="wsadwsad6779/fasterwam-a3-force-40k")
    ap.add_argument("--token-file", default="/root/.cache/hf_token")
    ap.add_argument("--out", default="backup40k/write_probe.json")
    ap.add_argument("--endpoints", default="https://hf-mirror.com,https://huggingface.co")
    a = ap.parse_args()
    TOKEN = Path(a.token_file).read_text().strip()
    import sys
    global sys_modules
    sys_modules = sys.modules

    out = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "dns": {}, "results": []}
    for host in ("huggingface.co", "hf-mirror.com",
                 os.environ.get("https_proxy", "").split("//")[-1].split(":")[0] or "none"):
        try:
            out["dns"][host] = sorted({r[4][0] for r in socket.getaddrinfo(host, None)})[:4]
        except Exception as exc:  # noqa: BLE001
            out["dns"][host] = chain(exc)
    scratch = "/tmp/hfwriteprobe"
    for ep in a.endpoints.split(","):
        try:
            out["results"].append(probe_endpoint(ep.strip(), a.repo, TOKEN, scratch))
        except Exception as exc:  # noqa: BLE001
            out["results"].append({"endpoint": ep, "fatal": chain(exc),
                                   "trace": mask(traceback.format_exc())[-800:]})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(mask(json.dumps(out, indent=2, default=str)))
    print(mask(json.dumps(out, indent=2, default=str)))
    return 0


sys_modules = {}

if __name__ == "__main__":
    raise SystemExit(main())
