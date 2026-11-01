#!/usr/bin/env bash
set -u
source /workspace/activate_fasterwam_a3.sh
cd /workspace/fasterwam_a3_bridge/bridge/_next || exit 1
{
  date -u +%FT%TZ
  echo == deps
  python3 -c "import PIL, imageio, numpy; print(chr(39)+chr(39), PIL.__version__, imageio.__version__, numpy.__version__)"
  echo == fonts
  ls /usr/share/fonts/truetype/dejavu | head -5
  echo == replay mask seed21
  python3 -u /workspace/_r4_stage/tko_mask.py --npz /workspace/a3_eval/r4_c0_seed21_video/lora_C0_21.npz --seed 21 --out /workspace/_r4_stage/tko_mask_seed21.npz --movers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
  echo mask21_rc=$?
  date -u +%FT%TZ
} > /workspace/_r4_stage/tko_mask21.log 2>&1
echo __M21_DONE__ >> /workspace/_r4_stage/tko_mask21.log