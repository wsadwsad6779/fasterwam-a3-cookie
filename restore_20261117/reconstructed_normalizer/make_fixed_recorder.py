"""Phase1: build an INDEPENDENT corrected copy of the execution-chain recorder.

The archived original deploy/capture/execution_chain_capture.py is read ONLY and kept
byte-identical. This writes reconstructed_normalizer/execution_chain_capture_fixed.py with
exactly one functional change: Capture.chunk's raw shape assertion (1,n,7) -> (n,7), matching
the REAL model output pred["action"]=(n,7). denorm stays (1,n,7) (the real bridge return after
its internal unsqueeze). No batch dim is faked onto raw; FasterWAM/Bridge/pm1/Adapter/normalizer/
training config are NOT touched.

It also records before/after SHA256, the exact diff, and a downstream raw-shape audit into
recorder_fix_manifest.json. NORMALIZER_IDENTITY=RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE = Path("/workspace/eval_restore_20261117")
ORIG = BASE / "deploy" / "capture" / "execution_chain_capture.py"
FIXED = BASE / "reconstructed_normalizer" / "execution_chain_capture_fixed.py"
MANIFEST = BASE / "reconstructed_normalizer" / "recorder_fix_manifest.json"

OLD = b'"raw": checked(raw, (1, n, 7), "raw"),'
NEW = b'"raw": checked(raw, (n, 7), "raw"),'

HEADER = (
    "# INDEPENDENT DIAGNOSTIC COPY - RECONSTRUCTED_NORMALIZER 40k diagnosis (Req2 / Phase1).\n"
    "# Original: deploy/capture/execution_chain_capture.py is kept BYTE-IDENTICAL, not modified.\n"
    "# ONLY functional change: Capture.chunk raw shape assertion (1,n,7) -> (n,7) to match the\n"
    "#   REAL model output pred['action']=(n,7). denorm stays (1,n,7) (real bridge return after\n"
    "#   its internal unsqueeze). No batch dim is faked onto raw. See recorder_fix_manifest.json.\n"
    "# NORMALIZER_IDENTITY=RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL\n"
)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def main():
    orig_bytes = ORIG.read_bytes()
    orig_sha = sha256_bytes(orig_bytes)
    count = orig_bytes.count(OLD)
    if count != 1:
        raise SystemExit(f"expected exactly 1 raw-shape line, found {count}; aborting (no write)")
    fixed_body = orig_bytes.replace(OLD, NEW)
    fixed_bytes = HEADER.encode("utf-8") + fixed_body
    FIXED.write_bytes(fixed_bytes)
    fixed_sha = sha256_bytes(fixed_bytes)

    # Downstream raw-shape audit: confirm nothing else assumes a leading batch dim on raw.
    text = fixed_body.decode("utf-8")
    audit = {
        "raw_only_stored_not_batch_indexed": (
            'raw is stored as arrays["chunk_{id}_raw"] and never indexed with a batch axis; '
            'selected()/executed()/save() do not read raw at all'
        ),
        "raw_occurrences_in_fixed": text.count("raw"),
        "denorm_assertion_unchanged": '"denorm": checked(denorm, (1, n, 7), "denorm"),' in text,
        "rewritten_assertion": '"rewritten": checked(rewritten, (n, 7), "rewritten"),' in text,
        "adapter_assertion": '"adapter": checked(adapter, (n, 14), "adapter"),' in text,
        "copy_isolation": (
            "checked() calls copy_cpu() = np.array(value, copy=True) for raw/denorm/rewritten/"
            "adapter; _Bridge/_Adapter proxies also copy_cpu their inputs/outputs, so the "
            "in-place pm1 rewrite action[..., -1]=... (a view of the bridge result) cannot "
            "pollute the already-copied denorm record"
        ),
        "evaluation_capture_raw_use": (
            "CapturedPolicy.chunk unpacks raw,denorm=bridge_proxy.calls[0] and forwards to "
            "capture.chunk; it compares returned actions to adapter outputs only, no raw rank "
            "assumption"
        ),
    }
    if not audit["denorm_assertion_unchanged"]:
        raise SystemExit("denorm assertion was altered; aborting")

    manifest = {
        "NORMALIZER_IDENTITY": "RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL",
        "original_stats_sha256": "UNKNOWN",
        "phase": "Phase1 recorder raw-shape contract fix",
        "original_path": str(ORIG),
        "original_sha256": orig_sha,
        "original_bytes": len(orig_bytes),
        "original_untouched": True,
        "fixed_path": str(FIXED),
        "fixed_sha256": fixed_sha,
        "fixed_bytes": len(fixed_bytes),
        "replacement_count": count,
        "diff": {"old_line": OLD.decode(), "new_line": NEW.decode(),
                 "location": "Capture.chunk values dict, raw entry"},
        "denorm_shape_kept": "(1, n, 7)",
        "raw_shape_corrected": "(n, 7)",
        "header_added": HEADER,
        "downstream_audit": audit,
        "forbidden_changes_avoided": [
            "no artificial batch dim added to raw",
            "FasterWAM / Bridge / pm1 / Adapter / normalizer / training config NOT modified",
            "archived execution_chain_capture.py NOT modified",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    # Re-verify the archived original is still byte-identical after the write.
    if sha256_bytes(ORIG.read_bytes()) != orig_sha:
        raise SystemExit("original changed unexpectedly; aborting")
    print(json.dumps({"original_sha256": orig_sha, "fixed_sha256": fixed_sha,
                      "replacement_count": count, "fixed_bytes": len(fixed_bytes),
                      "manifest": str(MANIFEST)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())