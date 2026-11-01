import json, numpy as np, sys
for p in ['/workspace/a3_eval/r4_b0_c0_paired/lora_B0_22.npz',
          '/workspace/a3_eval/r4_b0_c0_paired/lora_C0_22.npz']:
    d=np.load(p, allow_pickle=False)
    print('==',p)
    for k in sorted(d.files):
        a=d[k]
        print('  %-34s %-8s %s' % (k, str(a.dtype), a.shape))
print('== summary')
print(open('/workspace/a3_eval/r4_b0_c0_paired/summary_lora.json').read())
print('== manifest')
print(open('/workspace/a3_eval/r4_b0_c0_paired/manifest_lora.json').read())
