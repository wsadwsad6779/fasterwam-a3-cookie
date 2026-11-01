#!/usr/bin/env bash
set -euo pipefail
source /workspace/activate_fasterwam_a3.sh
C=$1
CKPT=$2
OUT=$3
if [ -z "$OUT" ]; then OUT=r2; fi
if [ ! -f "$CKPT" ]; then echo "CKPT_MISSING $CKPT" >&2; exit 2; fi
SZ=$(stat -c %s "$CKPT")
if [ "$SZ" -lt 1000000000 ]; then echo "CKPT_TOO_SMALL $SZ" >&2; exit 3; fi
echo "STAGE_START control=$C ckpt=$CKPT size=$SZ out=outputs/$OUT/$C"
cd /workspace/fasterwam_a3_bridge/bridge
python3 _next/run_matrix.py --model lora --controls "$C" --seeds 11 21 22 --video --out "outputs/$OUT/$C" --ckpt "$CKPT"
python3 _next/analyze_runs.py "outputs/$OUT/$C"
grep -q "loaded_ckpt=$CKPT" "outputs/$OUT/$C/manifest_lora.json" || echo CKPT_IDENTITY_WARN
echo "R2_$C""_DONE"
