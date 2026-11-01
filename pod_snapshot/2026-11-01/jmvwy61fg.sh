date -u +%H:%M:%S
ls /workspace/a3_eval/demos_correct_c7/*.npz | wc -l
ls -d /proc/478449 2>/dev/null || echo DRIVER_GONE
python3 -c "import json;d=json.load(open('/workspace/a3_eval/demos_correct_c7/summary.json'));rows=d["episodes"] if isinstance(d,dict) and "episodes" in d else d;print(len(rows));[print(r.get("seed"),r.get("anomaly_observed"),[x.get("anomaly_kind") for x in (r.get("retry_log") or [])][:1],r.get("success")) for r in rows]" 2>&1 | tail -10
df -BG /workspace | tail -1