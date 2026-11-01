import numpy as np, glob
for f in sorted(glob.glob('/workspace/fasterwam_a3_bridge/bridge/outputs/C3/*.npz')):
    z = np.load(f)
    e = z['eef_xyz']; op = z['opening']
    print(f.split('/')[-1], e.shape, 'start', np.round(e[0],3), 'end', np.round(e[-1],3))
    print('  min', np.round(e.min(0),3), 'max', np.round(e.max(0),3), 'opmin', round(float(op.min()),3), 'opend', round(float(op[-1]),3))
    for i in list(range(0, len(e), 100)):
        print('  ', i, np.round(e[i],3), round(float(op[i]),3))
d = np.load('/workspace/expert_seed11.npz')
e = d['eef']; op = d['opening']
print('expert', e.shape, 'start', np.round(e[0],3), 'end', np.round(e[-1],3))
for i in list(range(0, len(e), 60)):
    print('  E', i, np.round(e[i],3), round(float(op[i]),3))
