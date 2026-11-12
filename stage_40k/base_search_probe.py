"""Where did the 11 GB base checkpoint come from, and is a verified copy of it anywhere remote?

W2 in backup40k searches one repo (ARCHIVE_REPO) for an LFS sha256 equal to the pinned base
hash, and reported found=false after scanning 1490 entries. Two ways that can be true while
the backup is still fine, and one way it is a real hole:

  * the base lives in a *different* repo (it came with the upstream FasterWAM release, not
    with this project's archive), so the search looked in the wrong place;
  * the Hub no longer reports a plain sha256 for Xet-stored bulk files, so a byte-identical
    copy is invisible to a sha256 comparison;
  * or there really is no copy off this instance, and "restore the eight nodes on a fresh
    box" silently needs an 11 GB file that dies with the rental.

This prints the provenance markers on disk and, for both repos, every large entry with the
hash fields the API actually returns. Read-only: no upload, no delete, and the token never
appears in the output.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/_v1_stage/fw/src")
sys.path.insert(0, "/workspace/_v1_stage/fw/scripts")

import backup40k as bk  # noqa: E402

BASE = Path("/workspace/FasterWAM/checkpoints/fasterwam_release/libero/step_021700.pt")
TOKEN = Path(os.environ.get("HF_TOKEN_FILE", bk.TOKEN_FILE_DEFAULT)).read_text().strip()
REPOS = [bk.ARCHIVE_REPO, bk.DEFAULT_REPO]
WANT_SHA = "01be4efca565d1890fd8a7dc1507b01c00c8cc63c16ae252287492ba9ac7550d"

print("== 1) the file on disk")
if BASE.exists():
    print(json.dumps({"path": str(BASE), "bytes": BASE.stat().st_size,
                      "mtime_utc": __import__("time").strftime(
                          "%Y-%m-%dT%H:%MZ", __import__("time").gmtime(BASE.stat().st_mtime))}))
else:
    print("   MISSING -- the pinned base file is not on this instance")

print("\n== 2) provenance markers next to it (git-lfs pointer, hf snapshot metadata)")
for cmd in ("ls -la /workspace/FasterWAM/checkpoints/fasterwam_release/libero/ | head",
            "find /workspace/FasterWAM/checkpoints -maxdepth 3 -name '.gitattributes' -o "
            "-maxdepth 3 -name '*.metadata' | head",
            "ls /workspace/FasterWAM/checkpoints/fasterwam_release/ | head -20"):
    print("   $ %s" % cmd)
    os.system(cmd)
meta = Path("/root/.cache/huggingface")
if meta.is_dir():
    out = []
    for p in meta.rglob("*.metadata"):
        try:
            txt = p.read_text(errors="ignore")[:200]
        except Exception:  # noqa: BLE001
            continue
        if "step_021700" in str(p) or "fasterwam" in txt.lower():
            out.append({"file": str(p), "head": txt.splitlines()[:4]})
    print("   hf cache metadata mentions: %s" % json.dumps(out[:6], indent=1)[:900])

print("\n== 3) git remote of the FasterWAM checkout (where the weights were pulled from)")
for cmd in ("cd /workspace/FasterWAM && git remote -v 2>/dev/null | head -4",
            "cd /workspace/FasterWAM && git log --oneline -3 2>/dev/null"):
    print("   $ %s" % cmd)
    os.system(cmd)

print("\n== 4) every large entry in the two repos, with the hash fields the API gives")
HfApi, _ = bk._hub(os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
api = HfApi(endpoint=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"), token=TOKEN)
for repo in REPOS:
    try:
        entries = [bk._entry_info(e) for e in api.list_repo_tree(repo_id=repo, repo_type="model",
                                                                 recursive=True, expand=True)]
    except Exception as exc:  # noqa: BLE001
        print("   %s: %s %s" % (repo, type(exc).__name__, bk.mask(str(exc))[:150]))
        continue
    big = [e for e in entries if (e["size"] or 0) > 1_000_000_000]
    named = [e for e in entries if "step_021700" in str(e["path"])]
    print("   %s entries=%d  >1GB=%d  name-match=%d" % (repo, len(entries), len(big), len(named)))
    for e in (named or big)[:8]:
        print("      %s" % json.dumps(e, sort_keys=True))
    hit = [e for e in entries if str(e.get("lfs_oid") or "").lower() == WANT_SHA]
    print("      sha256 hit: %s" % bool(hit))

print("\n== 5) is the base file anywhere on the Hub under a name we control?")
try:
    models = api.list_models(search="fasterwam", limit=20)
    for m in models:
        print("   %s  downloads=%s lastModified=%s"
              % (m.id, getattr(m, "downloads", None), getattr(m, "lastModified", None)))
except Exception as exc:  # noqa: BLE001
    print("   search failed: %s" % type(exc).__name__)
