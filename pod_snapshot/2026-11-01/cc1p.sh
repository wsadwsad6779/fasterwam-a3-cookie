date
grep -c . /workspace/cc1.out
python3 -c "import json;rows=json.load(open('/workspace/a3_eval/demos_correct_c1/summary.json'));print('rows',len(rows));print([[r['seed'],r['variant'],r['steps'],r['stop'][:16],r['anomaly_observed'],r['recovered_and_valid'],r['accept'],[q['anomaly_kind'][:6] for q in r['retry_log']],r['pad_max_n'],r['frames_over_2n'],r['in_target_final']] for r in rows])"
echo __END__
