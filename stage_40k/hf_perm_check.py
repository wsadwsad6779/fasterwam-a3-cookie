"""Read-only check of the Hugging Face write path before the 40k run uploads to it.

Prints token *names, lengths and file modes*, never a token, and never reads one from a
command line. Three things have to be true before an eight-checkpoint upload is planned,
and none of them is visible from the training code:

  1. a credential exists and resolves to an account;
  2. that account can write the target repo, and the repo is not gated or disabled (a
     repo that silently rejects large files costs a whole segment);
  3. the weights we would upload have a license that permits redistribution.

The license half reads the upstream model cards on disk rather than the model card's
marketing text, and reports exactly which file said what.

A7/A8 are leak guards: the credential has to live outside the trees that get synced to
GitHub or HF, and it must not be readable by the world.
"""
from __future__ import annotations

import argparse
import json
import os
import stat
import urllib.error
import urllib.request

TOKEN_VARS = ("HF_TOKEN", "HF_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN", "ACCESS_TOKEN")
BASE = "/workspace/FasterWAM"
# The stage tree is what `__sync` pushes around and what the archive scripts tar up, so
# a credential must never be parked inside it.
FORBIDDEN_DIRS = ("/workspace/_v1_stage", "/workspace/FasterWAM", "/workspace/fasterwam_a3_bridge")


def _get(url, token=None, timeout=30):
    headers = {"User-Agent": "fastwam-40k-backup"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode()), dict(resp.headers)


def _resolve_token(token_file):
    """Return (token, evidence). The token itself is only ever returned, never printed."""
    present = {v: len(os.environ.get(v) or "") for v in TOKEN_VARS}
    token = next((os.environ[v] for v in TOKEN_VARS if os.environ.get(v)), None)
    source = "env" if token else None
    if token_file:
        if not os.path.exists(token_file):
            return None, {"error": "token file missing", "path": token_file, "env_vars": present}
        with open(token_file, encoding="utf-8") as fh:
            disk_token = fh.read().strip()
        mode = stat.S_IMODE(os.stat(token_file).st_mode)
        token = disk_token or token
        source = "file"
        present = dict(present)
        present["token_file"] = {"path": os.path.abspath(token_file),
                                 "length": len(disk_token), "mode": oct(mode)}
    return token, {"source": source, "vars": present}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="wsadwsad6779/fasterwam-a3-cookie-lora")
    ap.add_argument("--token-file", default=os.environ.get("HF_TOKEN_FILE", ""))
    ap.add_argument("--endpoint", default="", help="explicit endpoint; empty means env HF_ENDPOINT")
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    out = {"checks": [], "repo": a.repo}

    def check(name, ok, evidence):
        out["checks"].append({"name": name, "ok": bool(ok), "evidence": evidence})
        print(("PASS " if ok else "FAIL ") + name + " :: "
              + json.dumps(evidence, sort_keys=True, default=str)[:600])

    endpoint = (a.endpoint or os.environ.get("HF_ENDPOINT", "https://huggingface.co")).rstrip("/")
    out["endpoint"] = endpoint
    token, ev = _resolve_token(a.token_file)
    check("A1 a credential is present (env var or --token-file)", bool(token), ev)
    check("A7 the credential is not parked inside a synced tree",
          bool(token) and not any(os.path.abspath(a.token_file or "").startswith(d)
                                  for d in FORBIDDEN_DIRS),
          {"token_file": os.path.abspath(a.token_file) if a.token_file else None,
           "forbidden_dirs": FORBIDDEN_DIRS})
    check("A8 the credential file is owner-readable only",
          bool(a.token_file) and stat.S_IMODE(os.stat(a.token_file).st_mode) in (0o600, 0o400),
          {"mode": oct(stat.S_IMODE(os.stat(a.token_file).st_mode)) if a.token_file else "n/a (env)"})

    who = None
    if token:
        try:
            who, _ = _get(endpoint + "/api/whoami-v2", token)
        except Exception as exc:  # noqa: BLE001
            who = {"error": type(exc).__name__, "msg": str(exc)[:200]}
    check("A2 the credential resolves to an account",
          bool(who) and "name" in (who or {}),
          {"whoami": {k: who.get(k) for k in ("name", "fullname", "type", "plan", "error", "msg")}
           if who else None})

    info = None
    try:
        info, _ = _get(endpoint + "/api/models/" + a.repo, token)
    except urllib.error.HTTPError as exc:
        info = {"http_error": exc.code, "body": exc.read()[:200].decode(errors="replace")}
    except Exception as exc:  # noqa: BLE001
        info = {"error": type(exc).__name__, "msg": str(exc)[:200]}
    writable = bool(info) and info.get("write") is True and not info.get("disabled")
    check("A3 target repo exists, is writable, not disabled",
          writable,
          {"id": info.get("id"), "private": info.get("private"), "gated": info.get("gated"),
           "disabled": info.get("disabled"), "write": info.get("write"),
           "files_on_repo": len(info.get("siblings", []) or []),
           "sample": [s.get("rfilename") for s in (info.get("siblings") or [])[:6]],
           "raw_if_error": {k: info[k] for k in ("http_error", "body", "error", "msg") if k in info}})

    sizes = [s.get("size") for s in (info.get("siblings") or []) if s.get("size") is not None]
    check("A4 repo file sizes visible through the API (upload limits are not)",
          True, {"largest_file_bytes": max(sizes) if sizes else None,
                 "total_bytes_reported": sum(sizes) if sizes else 0,
                 "note": "LFS is enabled per-extension by the repo; a 11 GB torch file needs "
                         "the repo to accept it, which only an actual upload proves."})

    reach = {}
    for url in ("https://huggingface.co/api/models", "https://hf-mirror.com/api/models"):
        try:
            _, headers = _get(url, None, 20)
            reach[url] = "ok"
        except Exception as exc:  # noqa: BLE001
            reach[url] = type(exc).__name__
    check("A9 both endpoints reachable (mirror hides real write errors)",
          all(v == "ok" for v in reach.values()) and len(reach) == 2,
          {"reach": reach, "using": endpoint})

    lic = {}
    for rel in ("LICENSE", "README.md", "checkpoints/fasterwam_release/README.md"):
        p = os.path.join(BASE, rel)
        if os.path.exists(p):
            text = open(p, encoding="utf-8", errors="replace").read()
            hits = [ln.strip() for ln in text.splitlines()
                    if any(w in ln.lower() for w in ("license", "licens", "apache", "mit", "cc",
                                                     "non-commercial", "research only", "redistribut"))]
            lic[rel] = {"bytes": len(text), "lines": hits[:8]}
    check("A5 upstream license statements found on disk", bool(lic),
          {"files": list(lic), "excerpts": lic,
           "action_needed": "if no explicit license is stated for the base weights, the "
                            "published merged checkpoint inherits that ambiguity; record it "
                            "on the repo card instead of asserting a license"})

    wan = os.path.join(BASE, "checkpoints/Wan-AI")
    check("A6 which upstream trees are present to be derived from",
          os.path.isdir(wan),
          {"wan_dir": wan, "entries": sorted(os.listdir(wan))[:6] if os.path.isdir(wan) else None})

    with open(a.report, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True, default=str)
    print(json.dumps({"passed": sum(1 for c in out["checks"] if c["ok"]),
                      "failed": [c["name"] for c in out["checks"] if not c["ok"]]}, indent=2))
    return 0 if all(c["ok"] for c in out["checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
