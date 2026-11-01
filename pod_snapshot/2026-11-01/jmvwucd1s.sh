python3 -c "
import json,re
rows=[]
for line in open('/workspace/adj1.out'):
    line=line.strip()
    if line.startswith('{') and 'strict_accept' in line:
        rows.append(json.loads(line))
print('%-6s %-7s %-6s %-8s %-9s %-8s'%('seed','variant','strict','abnorm','recpeak','recover2'))
for r in rows:
    rec=r['segments'][-1]['segments']
    sp=r['recovery_spikes']
    print('%-6d %-7d %-6s %-8s %-9s %-8d  spikes=%s'%(r['seed'],r['variant'],r['strict_accept'],r['abnormal_observed'],rec['peak'],rec['over_2p0'],[ (s['frame'],s['frames_after_cut']) for s in sp]))
"; echo '--- wc2 ---'; date; wc -c /workspace/wc2.out; tail -c 700 /workspace/wc2.out