"""Pure-numpy rehearsal of the proprio attribution, for a machine with no torch.

`flow_test.check_proprio_attribution` is the authority; this only mirrors its arithmetic (the band
map, the clamp, the excursion widths) so a broken expectation is caught here in a second instead of
on the pod after a model load.
"""
import numpy as np

lo = np.array([0.05462999641895294, 0.09031549096107483, 0.7638311982154846,
               -1.2248071432113647, -2.4337832927703857, -1.2333282232284546,
               0.012251690961420536, -0.04023492708802223,
               -0.7338846325874329, -0.5887312889099121, -0.18089430034160614,
               -0.6814043521881104, -0.6854666471481323, -1.3969910144805908])
hi = np.array([0.13350403308868408, 0.3028571903705597, 0.919999361038208,
               2.480116605758667, 1.233620047569275, 2.443288803100586,
               0.04023492708802223, -0.012251690961420536,
               0.7338846325874329, 0.5887312889099121, 0.18089430034160614,
               0.6814043521881104, 0.6854666471481323, 1.3969910144805908])
mid = (lo + hi) / 2.0
rows = [mid.copy(), mid.copy(), mid.copy(), mid.copy()]
rows[1][0] = lo[0] - 2.5 * (hi[0] - lo[0])
rows[2][2] = hi[2] + 2.5 * (hi[2] - lo[2])
rows[3][9] = lo[9] - 3.0 * (hi[9] - lo[9])
mapped = lambda r: 2.0 * (r - lo) / (hi - lo) - 1.0          # noqa: E731
fed = [np.clip(mapped(r), -5.0, 5.0) for r in rows]

col_min = np.min(fed, axis=0)
col_max = np.max(fed, axis=0)
raw_min = np.min(rows, axis=0)
raw_max = np.max(rows, axis=0)
exc = []
for i in range(14):
    if col_min[i] >= -1.02 and col_max[i] <= 1.02:
        continue
    w = hi[i] - lo[i]
    below = (lo[i] - raw_min[i]) / w
    above = (raw_max[i] - hi[i]) / w
    ok = True
    if col_min[i] < -1.02:
        ok = ok and raw_min[i] < lo[i] and (col_min[i] > -4.99 or below >= 2.0)
    if col_max[i] > 1.02:
        ok = ok and raw_max[i] > hi[i] and (col_max[i] < 4.99 or above >= 2.0)
    exc.append((i, float(col_min[i]), float(col_max[i]), round(below, 3), round(above, 3), bool(ok)))

print("exceedances (col, norm_min, norm_max, widths_below_lo, widths_above_hi, explained):")
for e in exc:
    print("  %s" % (e,))
clamped = sorted({int(i) for i in range(14)
                  if np.any(np.abs(np.abs(np.array([f[i] for f in fed])) - 5.0) < 1e-6)})
print("clamped columns:", clamped)
forged = np.clip(mapped(mid), -5, 5).copy()
forged[4] = -5.0
# mirrored from smoke40k.proprio_band_rollup, so this rehearses the real predicate
raw_min_forged = mid.copy()
ok_forged = True
if forged[4] < -1.02:
    ok_forged = ok_forged and raw_min_forged[4] < lo[4]
    if forged[4] <= -4.99:
        ok_forged = ok_forged and ((lo[4] - raw_min_forged[4]) / (hi[4] - lo[4])) >= 2.0
print("forged clamp on col4 -> raw mid-band, explained?", bool(ok_forged), "(must be False)")
print("fed[1][0] == -5.0:", fed[1][0] == -5.0, " fed[2][2] == +5.0:", fed[2][2] == 5.0)
assert {e[0] for e in exc} == {0, 2, 9} and all(e[5] for e in exc)
assert clamped == [0, 2, 9] and not ok_forged
print("numpy rehearsal consistent with the flow_test expectations")
