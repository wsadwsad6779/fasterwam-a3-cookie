"""What does list_repo_tree actually return on this instance, and why did V2 see nothing?

The remote really holds the node (verified with curl: 25 entries under the prefix on both
huggingface.co and the mirror), and the re-download half of the drill came back byte-exact,
so the only broken link is the hash-comparison call. Print the object type and the fields of
the first entries for each argument combination instead of guessing which one the client
needs, with the payload shrunk to names and sizes so no URL or credential can appear.
"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_XET"] = "0"
import inspect
from huggingface_hub import HfApi, __version__

REPO = "wsadwsad6779/fasterwam-a3-force-40k"
TOKEN = open("/root/.cache/hf_token", encoding="utf-8").read().strip()
api = HfApi(endpoint="https://hf-mirror.com", token=TOKEN)

print("huggingface_hub", __version__)
sig = inspect.signature(api.list_repo_tree)
print("list_repo_tree params:", list(sig.parameters))

combos = [
    {"path_in_repo": "v1_force_rehearsal"},
    {"path_in_repo": "v1_force_rehearsal", "recursive": True},
    {"path_in_repo": "v1_force_rehearsal", "recursive": True, "expand": True},
    {"path_in_repo": "v1_force_rehearsal/README.md", "recursive": True, "expand": True},
    {"path_in_repo": "v1_force_rehearsal/nodes/step_000012", "recursive": True, "expand": True},
]
for kw in combos:
    try:
        rows = list(api.list_repo_tree(repo_id=REPO, repo_type="model", **kw))
        print("\n== %s -> %d entries" % (kw, len(rows)))
        for r in rows[:4]:
            d = r if isinstance(r, dict) else None
            get = (lambda k: d.get(k)) if d else (lambda k: getattr(r, k, None))
            print("   %s path=%r type=%r size=%r lfs=%r oid=%r" % (
                type(r).__name__, get("path"), get("type"), get("size"),
                str(get("lfs"))[:40], str(get("oid"))[:20]))
    except Exception as exc:  # noqa: BLE001
        print("\n== %s -> ERROR %s %s" % (kw, type(exc).__name__, str(exc)[:180]))
