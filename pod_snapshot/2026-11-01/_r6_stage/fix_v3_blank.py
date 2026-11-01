p = '/workspace/_r6_stage/light_state_v3.py'
s = open(p, encoding='utf-8').read()
a = '}' + chr(10)*2 + 'def save_adapter_v3'
b = '}' + chr(10)*3 + 'def save_adapter_v3'
assert s.count(a) == 1, (s.count(a), s.count(b))
assert s.count(b) == 0, s.count(b)
open(p, 'w', encoding='utf-8', newline=chr(10)).write(s.replace(a, b))
print('PATCHED', len(s), len(s.replace(a, b)))