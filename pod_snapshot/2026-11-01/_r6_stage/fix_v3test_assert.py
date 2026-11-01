p = '/workspace/_r6_stage/test_light_state_v3.py'
L = open(p, encoding='utf-8').read().split(chr(10))
hit = [k for k, x in enumerate(L) if 'check_v1_drops_proprio' in x]
assert len(hit) == 1, hit
i = hit[0]
assert 'proprio_encoder.bias' in L[i + 1], repr(L[i:i+2])
new = ["    R['check_v1_drops_proprio'] = (set(cov['dropped_by_v1']) == {'proprio_encoder.bias',",
       "                                                                  'proprio_encoder.weight'}",
       "                                   and {'proprio_encoder.weight'} <= set(R['v1_drops']))"]
L[i:i + 2] = new
open(p, 'w', encoding='utf-8', newline=chr(10)).write(chr(10).join(L))
print('PATCHED at line', i, 'total', len(L))