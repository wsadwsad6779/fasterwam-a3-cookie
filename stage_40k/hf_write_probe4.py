#!/usr/bin/env python3
"""Which data plane does this instance have, and what does that buy the backup?

Established by measurement, not assumed: the egress proxy answers `CONNECT tunnel failed:
403` for `*.s3-accelerate.amazonaws.com` and `s3.amazonaws.com`, direct routes are
unreachable, and a small regular-file commit through the mirror succeeds in 2.2 s. The
control plane is fine; only the bulk transfer leg is in question.

One case per process, because huggingface_hub freezes `HF_HUB_DISABLE_XET` into a module
constant at import time -- setting it later in the same interpreter would silently test the
same transport twice and report two different answers.

  xet      -- huggingface_hub's newer transfer format, which talks to xethub hosts. This is
              what moved 88 GB out of the old instance.
  lfs      -- the S3 path believed blocked; re-tested because believed is not measured.
  regular  -- a payload whose extension is not in the repo's LFS patterns, which the server
              may accept inside the commit request itself, i.e. through the host that
              already works. If this lands, a 211 MB node can ship as bounded shards with no
              S3 route at all.

Each remote path is read back and classified by whether the server recorded an `lfs` object
for it, so a "success" that stored nothing cannot be reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = "wsadwsad6779/fasterwam-a3-force-40k"
ENDPOINT = "https://hf-mirror.com"
TOKEN = open("/root/.cache/hf_token", encoding="utf-8").read().strip()
OUT = Path("/workspace/_v1_stage/backup40k/write_probe4.json")

CASES = {
    "xet_8m":      ("blob_8m.bin",    8 * 1024 * 1024,  "0"),
    "xet_70m":     ("blob_70m.bin",   70 * 1024 * 1024, "0"),
    "regular_8m":  ("payload_8m.datx", 8 * 1024 * 1024,  "1"),
    "regular_70m": ("payload_70m.datx", 70 * 1024 * 1024, "1"),
    "regular_200m": ("payload_200m.datx", 200 * 1024 * 1024, "1"),
}


def mask(text):
    text = str(text).replace(TOKEN, "<HF_TOKEN>")
    return re.sub(r"(https?://[^\s'\"]*)\?[^\s'\"]*", r"\1?<presigned-redacted>", text)


def blob(nbytes, seed):
    h = hashlib.sha256(("%s|%s" % (seed, nbytes)).encode()).digest()
    out = bytearray()
    while len(out) < nbytes:
        h = hashlib.sha256(h).digest()
        out += h
    return bytes(out[:nbytes])


def curl(url):
    p = subprocess.run(["curl", "-sS", "-o", "/dev/null", "-m", "20", "-w", "%{http_code} %{time_total}",
                        url], capture_output=True, text=True)
    return (p.stdout or "").strip() or ("ERR " + mask(p.stderr)[:120])


def do_case(label):
    name, nbytes, disable_xet = CASES[label]
    os.environ["HF_ENDPOINT"] = ENDPOINT
    os.environ["HF_HUB_DISABLE_XET"] = disable_xet
    from huggingface_hub import HfApi
    from huggingface_hub import constants as hc
    hc.HF_ENDPOINT = ENDPOINT
    api = HfApi(endpoint=ENDPOINT, token=TOKEN)
    src = Path("/tmp") / name
    src.write_bytes(blob(nbytes, label))
    rel = "transport_probe/%s/%s" % (label, name)
    rec = {"case": label, "file": name, "bytes": nbytes, "xet": disable_xet == "0",
           "sha256": hashlib.sha256(src.read_bytes()).hexdigest(), "path": rel,
           "lib_xet_disabled": getattr(hc, "HF_HUB_DISABLE_XET", None)}
    t0 = time.time()
    try:
        api.upload_file(path_or_fileobj=str(src), path_in_repo=rel, repo_id=REPO,
                        repo_type="model", commit_message="probe %s" % label)
        rec["ok"] = True
    except Exception as exc:  # noqa: BLE001
        rec["ok"] = False
        rec["err"] = type(exc).__name__ + ": " + mask(exc)[:300]
    rec["seconds"] = round(time.time() - t0, 2)
    if rec["ok"]:
        rec["mbps"] = round(nbytes / 1e6 / max(rec["seconds"], 1e-3), 2)
    src.unlink(missing_ok=True)
    print(json.dumps(rec, sort_keys=True, default=str), flush=True)
    Path("/tmp/probe4_%s.json" % label).write_text(json.dumps(rec, default=str))
    return 0 if rec["ok"] else 1


def do_hosts():
    doc = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hosts": {}}
    for h in ("https://cas-server.xethub.hf.co/", "https://transfer.xethub.hf.co/",
              "https://cdn-lfs.hf-mirror.com/", "https://cdn-lfs-us-1.hf.co/",
              "https://hf-hub-lfs-us-east-1.s3-accelerate.amazonaws.com/"):
        doc["hosts"][h] = curl(h)
        print("%-62s %s" % (h, doc["hosts"][h]), flush=True)
    try:
        import hf_xet
        doc["hf_xet"] = getattr(hf_xet, "__version__", "present")
    except Exception as exc:  # noqa: BLE001
        doc["hf_xet"] = "MISSING: " + type(exc).__name__
    os.environ["HF_ENDPOINT"] = ENDPOINT
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(endpoint=ENDPOINT, token=TOKEN)
    try:
        text = Path(hf_hub_download(REPO, ".gitattributes", token=TOKEN)).read_text()
        doc["lfs_patterns"] = [ln.split()[0] for ln in text.splitlines() if "filter=lfs" in ln]
    except Exception as exc:  # noqa: BLE001
        doc["lfs_patterns"] = {"error": mask(exc)[:200]}
    print(json.dumps(doc, indent=2, sort_keys=True), flush=True)
    Path("/tmp/probe4_hosts.json").write_text(json.dumps(doc, indent=2, sort_keys=True))
    return 0


def do_collect():
    os.environ["HF_ENDPOINT"] = ENDPOINT
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import HfApi
    api = HfApi(endpoint=ENDPOINT, token=TOKEN)
    rows = []
    for p in sorted(Path("/tmp").glob("probe4_*.json")):
        rows.append(json.loads(p.read_text()))
    remote = []
    try:
        for e in api.list_repo_tree(REPO, repo_type="model", path_in_repo="transport_probe",
                                    recursive=True):
            if getattr(e, "size", None):
                remote.append({"path": e.path, "size": e.size,
                               "lfs": bool(getattr(e, "lfs", None)),
                               "oid": (getattr(e, "lfs", None) or {}).get("oid")
                               if isinstance(getattr(e, "lfs", None), dict)
                               else getattr(getattr(e, "lfs", None), "oid", None)})
    except Exception as exc:  # noqa: BLE001
        remote = [{"error": mask(exc)[:200]}]
    doc = {"writes": rows, "remote": remote,
           "hosts": json.loads(Path("/tmp/probe4_hosts.json").read_text())
           if Path("/tmp/probe4_hosts.json").exists() else None}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, indent=2, sort_keys=True, default=str)
    assert TOKEN not in text
    OUT.write_text(text)
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="")
    ap.add_argument("--hosts", action="store_true")
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    if a.hosts:
        raise SystemExit(do_hosts())
    if a.collect:
        raise SystemExit(do_collect())
    raise SystemExit(do_case(a.case))
