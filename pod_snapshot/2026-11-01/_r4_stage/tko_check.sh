#!/usr/bin/env bash
set -u
S=/workspace/_r4_stage
V=/workspace/a3_eval/r4_c0_seed22_video
F=/workspace/a3_eval/r4_c0_frozen
{
  date -u +%FT%TZ
  echo "== run_matrix.py sha (must stay 01ef77cb)"
  sha256sum /workspace/fasterwam_a3_bridge/bridge/_next/run_matrix.py
  echo "== fonts"
  ls /usr/share/fonts/truetype/dejavu/ 2>&1 | head -5
  echo "== npz sha old vs new"
  sha256sum $F/lora_C0_22.npz $V/lora_C0_22.npz
  echo "== verify_video.py"
  python3 -u $S/verify_video.py
  echo "== mp4 probe"
  python3 -u -c "import imageio,os;p=chr(47).join([chr(0),''])" 2>/dev/null
  python3 -u - <<'PY'
import imageio, os
p = '/workspace/a3_eval/r4_c0_seed22_video/lora_C0_22_front.mp4'
r = imageio.get_reader(p)
print("mp4_nframes", r.count_frames())
print("mp4_meta", r.get_meta_data())
print("mp4_bytes", os.path.getsize(p))
r.close()
PY
  echo "== entry_analysis.py"
  python3 -u $S/entry_analysis.py
  echo "== df"
  df -h /workspace | tail -1
  date -u +%FT%TZ
} > $S/tko_check.log 2>&1
echo __TKO_CHECK_DONE__ >> $S/tko_check.log
