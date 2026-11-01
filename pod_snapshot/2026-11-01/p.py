p = "/workspace/r2_metrics.py"
s = open(p).read()
NL = chr(10)
old = '        o["cookies"] = json.load(open(sm)).get("cookies")'
lines = []
lines.append('        sj = json.load(open(sm))')
lines.append('        row = [r for r in sj if r.get("seed") == seed]')
lines.append('        o["cookies"] = row[0].get("cookies_in_target") if row else None')
lines.append('        o["verbatim"] = row[0].get("verbatim_pct") if row else None')
new = NL.join(lines)
assert s.count(old) == 1, s.count(old)
open(p, "w").write(s.replace(old, new))
print("PATCHED")
