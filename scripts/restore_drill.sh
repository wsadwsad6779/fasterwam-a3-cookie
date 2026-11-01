#!/usr/bin/env bash
# End-to-end recovery drill: start from an empty directory with no local
# ledgers, restore everything the archive holds, then hash it all.
#
#   ST=/root/drill_stage WS=/root/drill bash scripts/restore_drill.sh [extra args]
#
# ST is a copy of this repository (only scripts/ and the ledgers are read from
# it). WS is the pretend /workspace of a new instance. Both must not exist yet:
# this script never deletes anything, so it refuses instead of cleaning up.
# The archive side is only read; every fetched file is verified against the
# ledger sha256, and an assembled >2 GiB file is verified as a whole.
set -u
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ST="${ST:-/root/drill_stage}"
WS="${WS:-/root/drill}"
LOG="${LOG:-$HERE/drill.log}"
MIRROR="${MIRROR:---mirror}"
JOBS="${JOBS:-6}"

if [ -e "$WS" ] || [ -e "$ST" ]; then
  echo "REFUSE: $WS or $ST already exists (nothing was deleted)" >&2
  exit 9
fi
mkdir -p "$ST/scripts" "$WS" || exit 9
cp "$HERE/scripts/restore_new_instance.sh" "$ST/scripts/" || exit 9

run() {
  local name="$1"; shift
  printf '=== %s start %s\n' "$name" "$(date -u +%FT%TZ)" >> "$LOG"
  ( cd "$ST" && WS="$WS" JOBS="$JOBS" bash scripts/restore_new_instance.sh $MIRROR "$@" ) >> "$LOG" 2>&1
  printf '%s rc=%d %s\n' "$name" "$?" "$(date -u +%FT%TZ)" >> "$LOG"
}

# code and deps are deliberately not drilled: the upstream repositories and the
# torch base image are not part of the archive, and installing packages would
# mutate the machine running the drill.
run weights --weights
run rest --rest
run data --data
run parts --parts
run verify --verify
