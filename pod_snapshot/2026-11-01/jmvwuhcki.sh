cd /workspace/fasterwam_a3_bridge/bridge/_next && export PYTHONPATH=/workspace/a3_dual_arm_sim/src && python3 -c "
import sys,numpy as np
sys.path.insert(0,'/workspace/_r5_stage')
import replay_forces as R
z=np.load('/workspace/a3_eval/replay_bundle_dist/expert_bundle_7031.npz')
xyz,eef,z0=z['cookie_xyz'],z['eef_xyz'],z['cookie_xyz0']
print('keys',sorted(z.files))
rise=xyz[:,:,2]-z0[:,2]
d=np.linalg.norm(xyz[:,:,:2]-eef[:,None,:2],axis=2)
g=((d<0.025)&(rise>0.02))
f=np.flatnonzero(g[100])
print('lifted_at_100_indices',f.tolist())
print('their_x',np.round(z0[f,0],4).tolist())
print('their_y',np.round(z0[f,1],4).tolist())
cols=sorted(set(z0[:,0].tolist()))
print('source_columns_x',cols)
print('variant_to_col', {v: cols[v%2] for v in (0,1)})
for v in (0,1):
    x=cols[v%2]; order=np.argsort(z0[:,1][np.isclose(z0[:,0],x)])
    print('variant',v,'col_x',x,'row0_index',order[0],'row0_x',round(float(z0[order[0],0]),4))
"