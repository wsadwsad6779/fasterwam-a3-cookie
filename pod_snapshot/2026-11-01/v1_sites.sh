sed -n "20,50p" /workspace/FasterWAM/src/fasterwam/light_state.py
echo BUILD_PROPRIO
sed -n "88,118p" /workspace/fasterwam_a3_bridge/bridge/infer_a3_worker.py
echo DATASET
grep -nE "proprio|state|frames.dat|dataset_stats" /workspace/FasterWAM/src/fasterwam/datasets/lerobot/a3_video_dataset.py | head -25
echo STATS
python3 -c 'import json;d=json.load(open("/workspace/a3_data/a3_fw/dataset_stats.json"));print(sorted(d.keys()));print({k:(sorted(v.keys()) if isinstance(v,dict) else type(v).__name__) for k,v in d.items()})'
echo DONE