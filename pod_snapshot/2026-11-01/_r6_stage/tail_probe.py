p = '/workspace/_r6_stage/light_state_v3.py'
s = open(p, encoding='utf-8').read()
L = s.split(chr(10))
print('chars', len(s), 'items', len(L))
print('tail', repr(s[-70:]))
print('around', [repr(x) for x in L[114:120]])