import numpy as np
E = np.load("/workspace/_r4_stage/expert_force.npz")
A = np.load("/workspace/a3_data/a3_fw/arrays.npz", mmap_mode="r")
FO = E["force"].astype(np.float64); EP = E["episode_index"]
op = np.asarray(A["opening"], float); pr = np.asarray(A["proprio"], float)
jc = np.asarray(A["joint_cmd"], float); ac = np.asarray(A["action"], float)
pad = np.maximum(FO[:, 12], FO[:, 13])
padR = np.maximum(FO[:, 14], FO[:, 15])
w = np.flatnonzero(EP == 0)
print("ep0 frames", w.size)
print("  t  opening    pad    padR   actL  jointcmd  g6      z     xy_disp")
p0 = pr[w[0], 0:2]
for k in range(0, w.size, 10):
    i = w[k]
    print("%4d %7.3f %6.3f %6.3f %6.3f %8.3f %6.2f %7.4f %7.1f" % (
        k, op[i], pad[i], padR[i], FO[i, 16], jc[i], ac[i, 6], pr[i, 2],
        np.linalg.norm(pr[i, 0:2] - p0) * 1000))
print("R finger actuator(17) p50/p99/max %.3f %.3f %.3f" % tuple(np.percentile(FO[w, 17], [50, 99]) ) + " %.3f" % FO[w, 17].max())
print("opening uniq sample", np.unique(np.round(op[w], 3))[:12], np.unique(np.round(op[w], 3))[-6:])
