"""Is the 11 GB base checkpoint the run resumes from restorable from anywhere but this box?

W2's sha256 search came back empty for a concrete reason: the base file was never put in
the archive repo (its four >1 GB entries are frames.dat, step_002000_merged.pt,
model.safetensors and trainer/step_002000.pt -- none of them step_021700.pt). The question
is therefore whether the *upstream* release already carries the exact bytes. If it does,
"independently restorable" is satisfied by a pinned repo + revision + SHA256 and no upload
is needed; if it does not, eight nodes cannot be resumed on a fresh instance until 11 GB
moves somewhere durable.

Read-only, no upload, token never printed.
"""
import json
import os
import sys

sys.path.insert(0, "/workspace/_v1_stage/fw/src")
sys.path.insert(0, "/workspace/_v1_stage/fw/scripts")

import backup40k as bk  # noqa: E402

TOKEN = open(os.environ.get("HF_TOKEN_FILE", bk.TOKEN_FILE_DEFAULT)).read().strip()
WANT_SHA = "01be4efca565d1890fd8a7dc1507b01c00c8cc63c16ae252287492ba9ac7550d"
WANT_BYTES = 11117679929
CANDIDATES = ["hustvl/FasterWAM", "Eter0109/FasterWAM", "Wan-AI/Wan2.2-TI2V-5B"]

HfApi, hf_hub_download = bk._hub(os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
api = HfApi(endpoint=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"), token=TOKEN)

for repo in CANDIDATES:
    print("\n== %s" % repo)
    try:
        info = api.repo_info(repo, repo_type="model")
        print("   exists: private=%s lastModified=%s" % (info.get("private")
                                                         if isinstance(info, dict)
                                                         else info.private,
                                                         info.get("lastModified")
                                                         if isinstance(info, dict)
                                                         else info.lastModified))
    except Exception as exc:  # noqa: BLE001
        print("   repo_info: %s %s" % (type(exc).__name__, bk.mask(str(exc))[:160]))
        continue
    try:
        entries = [bk._entry_info(e) for e in api.list_repo_tree(repo_id=repo, repo_type="model",
                                                                 recursive=True, expand=True)]
    except Exception as exc:  # noqa: BLE001
        print("   tree: %s %s" % (type(exc).__name__, bk.mask(str(exc))[:160]))
        continue
    pts = [e for e in entries if str(e["path"] or "").endswith(".pt")]
    print("   entries=%d  .pt files=%d" % (len(entries), len(pts)))
    for e in pts[:15]:
        print("      %s" % json.dumps(e, sort_keys=True))
    same_name = [e for e in entries if "step_021700" in str(e["path"])]
    same_sha = [e for e in entries if str(e.get("lfs_oid") or "").lower() == WANT_SHA]
    same_size = [e for e in entries if e.get("lfs_size") == WANT_BYTES
                 or e.get("size") == WANT_BYTES]
    print("   name-match=%d sha256-match=%d size-match=%d"
          % (len(same_name), len(same_sha), len(same_size)))
    for e in (same_name or same_size)[:3]:
        print("      CANDIDATE %s" % json.dumps(e, sort_keys=True))
