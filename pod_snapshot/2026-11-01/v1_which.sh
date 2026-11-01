md5sum /workspace/FasterWAM/src/fasterwam/light_state.py /workspace/fasterwam-a3-cookie/finetune/light_state.py /workspace/FasterWAM/src/fasterwam/datasets/lerobot/a3_video_dataset.py /workspace/fasterwam-a3-cookie/finetune/a3_video_dataset.py
echo SITEPKG
ls -d /opt/venv/lib/python3*/site-packages/fasterwam* 2>/dev/null; python3 -c 'import fasterwam,sys;print(fasterwam.__file__)'
echo RUNR5
sed -n '1,25p' /workspace/_r4_stage/run_r5.sh
echo DONE