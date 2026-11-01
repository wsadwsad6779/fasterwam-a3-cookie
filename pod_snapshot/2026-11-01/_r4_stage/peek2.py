import json
N='/workspace/fasterwam_a3_bridge/bridge/_next'
g=json.load(open('/workspace/a3_eval/r3_c0_diag_600/gripper_11.json'))
print('TOP', {k:(type(v).__name__) for k,v in g.items()})
print('RAW_STATS', json.dumps(g.get('raw_stats'))[:900])
print('CHUNKS_TYPE', type(g['chunks']).__name__, (list(g['chunks'].keys())[:8] if isinstance(g['chunks'],dict) else len(g['chunks'])))
c=g['chunks'];
if isinstance(c,dict):
    k0=list(c.keys())[0]; print('CHUNK0KEY', k0, json.dumps(c[k0])[:600])
elif isinstance(c,list) and c: print('CHUNK0', json.dumps(c[0])[:600])
print('CONTEXT', json.dumps(g.get('context'))[:600])
print('EXEC', json.dumps(g.get('executed'))[:600])
