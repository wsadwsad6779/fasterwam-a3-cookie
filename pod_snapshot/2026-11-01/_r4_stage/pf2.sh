#!/bin/bash
set +e
O=/workspace/_r4_stage/pf2.txt
echo === py deps === > $O
/opt/venv/bin/python3 -m pip list > $O.deps 2>/dev/null
grep -Ei "imageio|opencv|av$|moviepy|pillow|numpy" $O.deps >> $O
echo === import check === >> $O
/opt/venv/bin/python3 -c "import imageio;print(1,imageio.__version__)" >> $O 2>&1
/opt/venv/bin/python3 -c "import imageio_ffmpeg;print(2,imageio_ffmpeg.get_ffmpeg_exe())" >> $O 2>&1
/opt/venv/bin/python3 -c "import cv2;print(3,cv2.__version__)" >> $O 2>&1
/opt/venv/bin/python3 -c "import av;print(4,av.__version__)" >> $O 2>&1
/opt/venv/bin/python3 -c "import PIL;print(5,PIL.__version__)" >> $O 2>&1
echo === which python in eval env === >> $O
bash -lc "source /workspace/activate_fasterwam_a3.sh; which python3; python3 -c "import imageio,imageio_ffmpeg;print(imageio.__version__,imageio_ffmpeg.get_ffmpeg_exe())"" >> $O 2>&1
echo === cookie truth log === >> $O
cat /workspace/fasterwam_a3_bridge/bridge/_next/cookie_truth_log.py >> $O 2>&1
echo __PFDONE__ >> $O