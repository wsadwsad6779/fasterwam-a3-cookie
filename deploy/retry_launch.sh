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
#   E0=0 POD_SSH=... bash deploy/retry_launch.sh      # readiness only
#
# One attempt = reachable -> repo at a3-cookie-r27 -> bootstrap --smoke ->
# B0 sign check on the checkpoint -> 2-step LoRA probe -> start the E0 C0
# baseline (one seed, 600 steps, video) with nohup. Any failure just sleeps and
# tries again; success exits 0 once E0 is running.
set -uo pipefail

INTERVAL="${INTERVAL:-60}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"          # 0 = retry until success
POD_SSH="${POD_SSH:-}"                     # e.g. "ssh -p 45339 root@host"
REMOTE_DIR="${REMOTE_DIR:-/workspace/fasterwam-a3-cookie}"
REMOTE_WORKSPACE="${REMOTE_WORKSPACE:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/wsadwsad6779/fasterwam-a3-cookie.git}"
BRANCH="${BRANCH:-a3-cookie-r27}"
LOG="${LOG:-outputs/launch_retry.log}"

# The endpoint lives in deploy/pod.env (gitignored) and is re-read before every
# attempt, so the loop can run before the instance exists and picks up a new
# port without a restart.
POD_ENV="${POD_ENV:-deploy/pod.env}"
POD_SSH_ENV="${POD_SSH:-}"

load_endpoint() {
    if [ -n "$POD_SSH_ENV" ]; then
        POD_SSH="$POD_SSH_ENV"
    elif [ -f "$POD_ENV" ]; then
        set -a
        # shellcheck disable=SC1090
        . "$POD_ENV"
        set +a
        POD_SSH="${POD_SSH%$'\r'}"
    fi
}
load_endpoint

if [ -z "$POD_SSH" ]; then
    echo "POD_SSH is required, e.g. POD_SSH='ssh -p 45339 root@1.2.3.4' $0" >&2
    echo "or copy deploy/pod.env.example to deploy/pod.env and fill it in" >&2
    exit 2
fi

mkdir -p "$(dirname "$LOG")"

attempt=0
while :; do
    load_endpoint
    attempt=$((attempt + 1))
    echo "[$(date -Is)] attempt $attempt" | tee -a "$LOG"

    if $POD_SSH -o BatchMode=yes -o ConnectTimeout=15 true >>"$LOG" 2>&1; then
        echo "[$(date -Is)] ssh reachable; deploying" | tee -a "$LOG"
        if $POD_SSH -o BatchMode=yes -o ConnectTimeout=30 \
            "REMOTE_DIR='$REMOTE_DIR' REMOTE_WORKSPACE='$REMOTE_WORKSPACE' REPO_URL='$REPO_URL' BRANCH='$BRANCH' E0='${E0:-1}' E0_STEPS='${E0_STEPS:-600}' bash -s" \
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

# B0 with the checkpoint: the sign check has to see what the weights actually
# emit, not only the adapter table.
python3 _next/b0_probe.py --model lora --out "$REMOTE_WORKSPACE/a3_results/b0"
python3 _next/run_matrix.py --model lora --controls C0 --seeds 11 \
    --max-steps 2 --out "$REMOTE_WORKSPACE/a3_results/launch_probe"

# First real job: one complete C0 trajectory with video. Kept even at 0/10 --
# it is the baseline the release ladder is judged against. Backgrounded so a
# dropped SSH session cannot kill it, and skipped if one is already running.
if [ "${E0:-1}" = "1" ]; then
    if pgrep -f "run_matrix.py --model lora --controls C0" >/dev/null 2>&1; then
        echo "E0 C0 already running; not starting a second one"
    else
        mkdir -p "$REMOTE_WORKSPACE/a3_results/E0_C0"
        nohup setsid python3 _next/run_matrix.py --model lora --controls C0 \
            --seeds 11 --max-steps "${E0_STEPS:-600}" --video \
            --out "$REMOTE_WORKSPACE/a3_results/E0_C0" \
            > "$REMOTE_WORKSPACE/a3_results/E0_C0/run.log" 2>&1 &
        echo "E0 C0 started: $REMOTE_WORKSPACE/a3_results/E0_C0/run.log"
    fi
fi
REMOTE
        then
            echo "[$(date -Is)] SUCCESS: pod up, B0 sign check passed, LoRA stepped, E0 C0 started" | tee -a "$LOG"
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
