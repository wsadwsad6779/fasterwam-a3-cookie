import json
for sd in (11,21,22):
    g=json.load(open('/workspace/a3_eval/r3_c0_diag_600/gripper_%d.json'%sd))
    print('=== SEED',sd,'steps',g['steps'],'chunks',g['chunks'])
    for k in ('executed','predicted_chunks','threshold_distance','context'):
        print(k, json.dumps(g[k], sort_keys=True)[:700])
