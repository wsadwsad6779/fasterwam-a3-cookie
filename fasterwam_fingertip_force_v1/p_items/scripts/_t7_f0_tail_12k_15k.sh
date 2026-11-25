#!/bin/bash
# HELD. Per user rule 2026-11-25: the 12k continuation must NOT start.
# This script is intentionally a no-op hold; it does not train anything.
set -u
echo "TAIL_HOLD $(date -u +%FT%TZ) 12k NOT started per user 2026-11-25; F0 stays at 9k"
exit 0
