#!/usr/bin/env bash
# Wait until deploy/pod.env names a real endpoint, then hand over to the
# 60-second retry loop. Started by the agent so the operator never has to
# launch a script by hand.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
LOG="${LOG:-outputs/autostart.log}"
mkdir -p "$(dirname "$LOG")"

ready() {
    [ -f deploy/pod.env ] || return 1
    # POD_SSH may come from the environment instead of the file.
    local v="${POD_SSH:-$(sed -n 's/^[[:space:]]*POD_SSH=["'\'']\{0,1\}\([^"'\'']*\)["'\'']\{0,1\}[[:space:]]*$/\1/p' deploy/pod.env | head -1)}"
    [ -n "$v" ] || return 1
    case "$v" in *1.2.3.4*) return 1 ;; esac   # pod.env.example placeholder
    return 0
}

echo "[$(date -Is)] autostart waiting for deploy/pod.env" >>"$LOG"
while :; do
    if ready; then
        echo "[$(date -Is)] endpoint found; starting retry_launch.sh" >>"$LOG"
        INTERVAL="${INTERVAL:-60}" exec bash deploy/retry_launch.sh
    fi
    sleep 60
done
