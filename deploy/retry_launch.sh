#!/usr/bin/env bash
# Bring the pod up, every INTERVAL seconds, until one attempt succeeds.
#
# The operator opens the paid instance; this loop only waits for it and then
# runs the steps that must happen on the GPU box. Nothing here starts, stops or
# resizes an instance, and no secret is written to disk: HF_TOKEN is forwarded
# through the environment only when it is already set in the calling shell.
#
#   POD_SSH="ssh -p 45339 root@1.2.3.4" bash deploy/retry_launch.sh
#   INTERVAL=60 MAX_ATTEMPTS=30 POD_SSH=... bash deploy/retry_launch.sh
#
# One attempt = reachable -> repo at a3-cookie-r27 -> bootstrap --smoke ->
# 2-step launch probe that actually loads the LoRA checkpoint. Any failure just
# sleeps and tries again; success exits 0.
set -uo pipefail

INTERVAL="${INTERVAL:-60}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"          # 0 = retry until success
POD_SSH="${POD_SSH:-}"                     # e.g. "ssh -p 45339 root@host"
REMOTE_DIR="${REMOTE_DIR:-/workspace/fasterwam-a3-cookie}"
REMOTE_WORKSPACE="${REMOTE_WORKSPACE:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/wsadwsad6779/fasterwam-a3-cookie.git}"
BRANCH="${BRANCH:-a3-cookie-r27}"
LOG="${LOG:-outputs/launch_retry.log}"

if [ -z "$POD_SSH" ]; then
    echo "POD_SSH is required, e.g. POD_SSH='ssh -p 45339 root@1.2.3.4' $0" >&2
    exit 2
fi

mkdir -p "$(dirname "$LOG")"

attempt=0
while :; do
    attempt=$((attempt + 1))
    echo "[$(date -Is)] attempt $attempt" | tee -a "$LOG"

    if $POD_SSH -o BatchMode=yes -o ConnectTimeout=15 true >>"$LOG" 2>&1; then
        echo "[$(date -Is)] ssh reachable; deploying" | tee -a "$LOG"
        if $POD_SSH -o BatchMode=yes -o ConnectTimeout=30 \
            "REMOTE_DIR='$REMOTE_DIR' REMOTE_WORKSPACE='$REMOTE_WORKSPACE' REPO_URL='$REPO_URL' BRANCH='$BRANCH' bash -s" \
            >>"$LOG" 2>&1 <<'REMOTE'
set -euo pipefail

# 1. Code: pinned branch, fast-forward only, so operator edits on the pod are
#    never discarded.
if [ ! -d "$REMOTE_DIR/.git" ]; then
    git clone --branch "$BRANCH" "$REPO_URL" "$REMOTE_DIR"
else
    git -C "$REMOTE_DIR" fetch origin "$BRANCH"
    git -C "$REMOTE_DIR" checkout -q "$BRANCH"
    git -C "$REMOTE_DIR" pull --ff-only origin "$BRANCH"
fi

# 2. Dependencies + weights + activate script. Idempotent: existing files are
#    verified, not overwritten.
bash "$REMOTE_DIR/deploy/bootstrap_pod.sh" --smoke

# 3. Launch probe: load the LoRA checkpoint and step the simulator twice. This
#    is the first thing that proves the model is actually up on this box.
source "$REMOTE_WORKSPACE/activate_fasterwam_a3.sh"
cd "$REMOTE_WORKSPACE/fasterwam_a3_bridge/bridge"
python3 _next/b0_probe.py --out "$REMOTE_WORKSPACE/a3_results/b0"
python3 _next/run_matrix.py --model lora --controls C0 --seeds 11 \
    --max-steps 2 --out "$REMOTE_WORKSPACE/a3_results/launch_probe"
REMOTE
        then
            echo "[$(date -Is)] SUCCESS: pod is up, B0 sign check passed, LoRA stepped" | tee -a "$LOG"
            exit 0
        fi
        echo "[$(date -Is)] attempt $attempt failed; retrying in ${INTERVAL}s" | tee -a "$LOG"
    else
        echo "[$(date -Is)] ssh not reachable yet; retrying in ${INTERVAL}s" | tee -a "$LOG"
    fi

    if [ "$MAX_ATTEMPTS" -gt 0 ] && [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
        echo "[$(date -Is)] gave up after $attempt attempts" | tee -a "$LOG"
        exit 1
    fi
    sleep "$INTERVAL"
done
