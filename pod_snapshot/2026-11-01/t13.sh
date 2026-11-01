echo ==SUMMARY==
python3 -c "import json;d=json.load(open('/workspace/data/a3_front_close_left_100/collection_summary.json'));d.pop('config',None);print(json.dumps(d,indent=1)[:2500])"
echo ==BATCH_EXPERT_DONE_LOGIC==
sed -n '435,470p' /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/controllers/batch_expert.py
echo ==BENCH_TOP==
sed -n '35,70p' /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/workflows/benchmark.py
echo DONE