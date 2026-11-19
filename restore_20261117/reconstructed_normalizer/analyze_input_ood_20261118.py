"""Read-only local CPU NumPy audit; no model, network, GPU or asset modifications."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent.parent
P = ROOT / 'forensics_download'
OUT = ROOT / 'INPUT_OOD_ANALYSIS_20261118.json'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def bounds(a):
    return {'min':a.min(axis=0).tolist(), 'max':a.max(axis=0).tolist(), 'q01':np.quantile(a,.01,axis=0).tolist(), 'q99':np.quantile(a,.99,axis=0).tolist()}
def compare(x, ref):
    x=np.asarray(x).reshape(-1,ref.shape[1]); lo=ref.min(0); hi=ref.max(0)
    outside=(x<lo-1e-6)|(x>hi+1e-6)
    return {'outside_dims0':np.flatnonzero(outside.any(0)).tolist(), 'outside_count_per_dim':outside.sum(0).tolist(), 'min':x.min(0).tolist(),'max':x.max(0).tolist(), 'max_excess_in_training_ranges':(np.maximum(np.maximum(lo-x,x-hi),0)/np.maximum(hi-lo,1e-8)).max(0).tolist()}
calpath=ROOT.parent/'step_040000/force_calib_v1.json'
cal=json.loads(calpath.read_text()); channels=cal['channels']; mean=np.array(cal['mean']); std=np.array(cal['std']); tare=np.array(cal['zero_load_ref_median'])
up=np.load(P/'training_source_a3_fw_arrays.npz',allow_pickle=False)
fo=np.load(P/'training_source_a3_fw_force_arrays.npz',allow_pickle=False)
assert sha(P/'training_source_a3_fw_force_arrays.npz')==cal['source_sha256']['force_arrays']
common={k:bool(np.array_equal(up[k],fo[k])) for k in up.files}; assert all(common.values())
ftrain=fo['force18'][:,channels]
p8=up['proprio']; p14=np.concatenate([p8,(ftrain.astype(np.float64)-tare).astype(np.float32)],axis=1)
stats=json.loads((ROOT/'dataset_stats_14d_RECONSTRUCTED.json').read_text())
s=stats['state']['default']; lo=np.array(s['global_min'],dtype=np.float32); hi=np.array(s['global_max'],dtype=np.float32)
scale=np.float32(2)/(hi-lo); offset=np.float32(-1)-scale*lo
ntrain=np.clip(p14*scale+offset,-5,5)
am=stats['action']['default']; al=np.array(am['global_min']); ah=np.array(am['global_max'])
def raw_proprio(z):
    pose=z['observation.eef_pose'].astype(np.float64); q=pose[3:7]; q=q/np.linalg.norm(q); w=np.clip(q[0],-1,1); den=np.sqrt(max(0,1-w*w))
    rot=np.zeros(3,dtype=np.float32) if den<1e-8 else (q[1:]*(2*np.arccos(w)/den)).astype(np.float32)
    opening=np.clip(float(z['observation.state'][7]),0,1)
    base=np.concatenate([pose[:3],rot]).astype(np.float32)
    base=np.concatenate([base,np.array([.04*opening,-.04*opening],np.float32)])
    f=z['observation.force'][channels]
    return np.concatenate([base.astype(np.float64),f.astype(np.float64)-tare]).astype(np.float32)
def bf16(a):
    u=np.ascontiguousarray(a,dtype=np.float32).view(np.uint32)
    return ((u+np.uint32(32767)+((u>>16)&1))&np.uint32(4294901760)).view(np.float32)
summary=json.loads((P/'mechanism_summary38.json').read_text()); records=[]
for job in summary['results']:
    path=P/('mechanism_'+job['name']+'.npz'); z=np.load(path,allow_pickle=False)
    assert sha(path)==job['npz_sha256']
    rawp=raw_proprio(z); pre=rawp*scale+offset; norm=np.clip(pre,-5,5); saved=z['normalized_proprio'].reshape(14)
    assert np.max(np.abs(norm-saved))<2e-5,(job['name'],norm,saved)
    raw=z['raw32']; hist=z['force_history']; mask=z['force_history_mask']; valid=hist[mask]; fut=z['force_future_newton']; after=z['trace_after'][:,0]; before=z['trace_before'][:,0]
    assert np.array_equal(hist[-1],z['observation.force'][channels])
    assert np.array_equal(mask,np.arange(16)>=16-int(mask.sum()))
    assert np.all(hist[~mask]==0)
    errs=[float(np.abs(bf16(before[i]+bf16(z['trace_velocity'][i,0]*z['trace_delta'][i]))-after[i]).max()) for i in range(10)]
    assert max(errs)==0 and np.array_equal(after[-1],raw)
    stepmax=np.abs(after).max(1)
    r={'name':job['name'],'seed':job['seed'],'frame':job['frame'],'noise':job['noise'], 'sha256':sha(path),
       'schema':{k:{'shape':list(z[k].shape),'dtype':str(z[k].dtype)} for k in z.files},
       'raw_proprio14':rawp.tolist(),'normalized_proprio':saved.tolist(),'normalizer_unclipped':pre.tolist(),'reconstruction_max_error':float(np.abs(norm-saved).max()),
       'normalized_abs_gt1_dims0':np.flatnonzero(np.abs(saved)>1+1e-6).tolist(),'clip_boundary_dims0':np.flatnonzero(np.abs(saved)>=5-1e-6).tolist(),
       'raw_proprio_vs_source':compare(rawp,p14),'normalized_vs_source':compare(saved,ntrain),
       'raw_force6':z['observation.force'][channels].tolist(),'current_force_vs_source':compare(z['observation.force'][channels],ftrain),
       'history_mask':mask.tolist(),'history_valid_newton':valid.tolist(),'history_valid_vs_source':compare(valid,ftrain),
       'history_valid_zscore_min':((valid-mean)/std).min(0).tolist(),'history_valid_zscore_max':((valid-mean)/std).max(0).tolist(),
       'gripper_range32':[float(raw[:,6].min()),float(raw[:,6].max())],
       'raw_action_absmax32_per_dim':np.abs(raw).max(0).tolist(),'raw_action_absmax_first10_per_dim':np.abs(raw[:10]).max(0).tolist(),
       'raw_action_gt1_count_per_dim':(np.abs(raw)>1).sum(0).tolist(),
       'denorm_motion6_range':[(((raw*(ah-al)/2)+(ah+al)/2)[:,:6]).min(0).tolist(),(((raw*(ah-al)/2)+(ah+al)/2)[:,:6]).max(0).tolist()],
       'future_force_vs_source':compare(fut,ftrain),'future_force_zscore_absmax_per_dim':np.abs((fut-mean)/std).max(0).tolist(),
       'trace_initial_absmax_per_dim':np.abs(before[0]).max(0).tolist(),'trace_after_absmax_per_step_per_dim':stepmax.tolist(),
       'trace_velocity_absmax_per_step_per_dim':np.abs(z['trace_velocity'][:,0]).max(1).tolist(),
       'trace_first_after_gt5_step1_per_dim':[next((i+1 for i in range(10) if stepmax[i,j]>5),None) for j in range(7)],
       'trace_first_after_gt10_step1_per_dim':[next((i+1 for i in range(10) if stepmax[i,j]>10),None) for j in range(7)],
       'euler_bf16_max_error':max(errs),'history_padding_zero':True,'history_current_exact':True}
    records.append(r)
assets=[P/'training_source_a3_fw_arrays.npz',P/'training_source_a3_fw_force_arrays.npz',calpath,ROOT/'dataset_stats_14d_RECONSTRUCTED.json',ROOT.parent/'step_040000/run_config.yaml',ROOT/'diag_reconstructed_40k.py',BASE/'bridge/infer_a3_worker.py',BASE/'bridge/observation_adapter.py',BASE/'bridge/gripper_proprio_adapter.py',BASE/'stage_40k/dev/eval40k/eval40k.py',BASE/'stage_40k/dev/eval40k/policy40k.py',BASE/'stage_40k/dev/eval40k/force_stream.py',BASE/'stage_40k/fw/src/fasterwam/datasets/lerobot/utils/normalizer.py',BASE/'stage_40k/fw/src/fasterwam/datasets/lerobot/a3_video_dataset.py',BASE/'stage_40k/fw/src/fasterwam/models/wan22/force_condition.py',P/'mechanism_summary38.json']
groups=[]
for key in sorted(set((r['seed'],r['frame']) for r in records)):
    rr=[r for r in records if (r['seed'],r['frame'])==key]; a=rr[0]
    groups.append({'seed':key[0],'frame':key[1],'n':len(rr),'normalized_absmax':max(np.abs(a['normalized_proprio'])),'input_ood_dims0':a['raw_proprio_vs_source']['outside_dims0'],'clip_dims0':a['clip_boundary_dims0'],'history_ood_dims0':a['history_valid_vs_source']['outside_dims0'],'g_absmax_range':[min(r['raw_action_absmax32_per_dim'][6] for r in rr),max(r['raw_action_absmax32_per_dim'][6] for r in rr)],'motion6_max_range':[min(max(r['raw_action_absmax32_per_dim'][:6]) for r in rr),max(max(r['raw_action_absmax32_per_dim'][:6]) for r in rr)],'future_zmax_range':[min(max(r['future_force_zscore_absmax_per_dim']) for r in rr),max(max(r['future_force_zscore_absmax_per_dim']) for r in rr)]})
report={'NORMALIZER_IDENTITY':'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL','original_stats_sha256':'UNKNOWN',
 'scope':'CPU NumPy only, local files; no inference/control, network, GPU, labels research or original-asset writes',
 'assets_sha256':{str(p):sha(p) for p in assets},'source_force_sha_matches_calibration':True,'upstream_common_fields_exact':common,
 'reference_limit':'42681-row a3_fw upstream 8D + calibration-matched actual force18 gathered channels; constructed 14D reference is not recovered a3_fw_v1g nor proof of original statistics identity. Marginal range only; in-range does not establish joint/temporal ID.',
 'normalizer_contract':{'mode':'min/max','formula':'clip(raw*2/(max-min)+(-1-2*min/(max-min)), -5, 5)','clip_not_training_range':True,'force_conditioner':'(raw_newton-mean)/std * scale; no amplitude clamp; only valid history used in comparisons','indices0':['x','y','z','axis_angle_x','axis_angle_y','axis_angle_z','finger1','finger2','tare_fx','tare_fy','tare_fz','tare_pad_inner','tare_pad_outer','tare_gripper']},
 'training_reference':{'proprio8':bounds(p8),'raw_force6':bounds(ftrain),'constructed_proprio14':bounds(p14),'normalized_constructed_proprio14':bounds(ntrain),'force_zscore':bounds((ftrain-mean)/std),'normalizer_minmax_vs_actual_source_bounds':{'stats_min':lo.tolist(),'stats_max':hi.tolist(),'actual_min':p14.min(0).tolist(),'actual_max':p14.max(0).tolist()}},
 'states':groups,'records':records,
 'findings':{'association':'All nine seed17 severe-state traces (frames143/159/207, 3 noises each) have all six force-proprio coordinates8:14 saturated at +5 and all six valid-history force channels outside actual source ranges. History remains unclamped and reaches about2175.68 calibration std at frame207. This is association, not causal attribution.',
 'normalizer_vs_source':'Force slots in reconstructed min/max are symmetric +/- calibration std, NOT actual source extrema. Actual-source in-range force can therefore normalize beyond1, and clip presence is not itself sufficient proof of training-source OOD. Use raw/source comparison separately.',
 'specific_motion_ood':'Base coordinate2 z: 0.761320/0.744069/0.733227 vs source[0.763831,0.919999]; coordinate4 axis-angle-y:1.239154/1.279420/1.527308 vs max1.233620. Frame143 also coordinate5 below min. Rotation coordinates are representation-dependent, not automatically physical orientation novelty.',
 'trace':'Protocol gripper after-step maxima: frame143 2.391->12.312, frame159 4.25->32.5, frame207 6.375->101.5; initial Gaussian gripper maxima1.898/2.516/2.516. Amplification begins early, accumulates through10 exact BF16 Euler steps; no action scheduler arithmetic discrepancy found.',
 'co_amplification':'Not gripper-only: frame207 protocol final action per-dim absmax [9,34.25,32,31.125,22.875,4.53125,101.5]; all7 exceed1, not all exceed5. Future force all6 channels leave source range; protocol gripper-force channel peaks43.002/107.680/176.902 N at frames143/159/207 vs training max2.952N. Future-force per-step timing unavailable.',
 'counterexamples':'Seed11f98 saturates coord10 and has history OOD yet final gripper absmax0.871..1.078; seed11f332 multiple saturations but gripper1.469..1.766. Thus OOD/saturation not sufficient and scalar max normalized proprio cannot rank severity. Seed11/12/17f0 no source-marginal input OOD yet gripper slightly>1; >1 alone not evidence of severe explosion. Seed11f47 is intermediate: gripper3.953..4.781, motion2.359..2.703, future force12.984..16.480std.'},
 'limitations':['Trace records action Euler states/velocities only; no per-step future-force intermediate trace, so future-force amplification timing cannot be established.','Selected correlated closed-loop states and repeated noise are not independent randomized interventions.','Raw action >1 alone is not explosion; initial Gaussian noise naturally exceeds 1. Report final per-dim values, >5/>10 onset and controls.'],
 'counterevidence_conditions':['An OOD-but-normal-output state refutes sufficiency; a severe output with no marginal input OOD refutes necessity for that marginal criterion.','Proprio clamp saturation shared by mild/severe states cannot alone explain severity; unchanged input across noise with output variation shows other determinants.','If matched-noise replacing only offending input coordinates does not reproducibly reduce severe amplitudes, the specific-coordinate causal hypothesis weakens; nonspecific reductions from unrelated replacements weaken selectivity.'],
 'next_minimal_controlled_intervention':{'offline_only':True,'baseline':'seed17 frame143 protocol; cache identical image/text, model eval, solver, precision and separate action/video/force RNG tensors; first replay baseline','single_factor':'clip only valid force_history[:,2] (raw wrist fz) to actual source bounds [-1.3804129362,0.4544959664] N. Keep normalized proprio (including already saturated coord10), other history channels, mask and all cached noise byte-identical; retain unaltered input. This isolates one unclamped history channel rather than conflating two force pathways.','controls':'paired same-noise baseline vs wrist-fz-history intervention; same intervention at seed17 frame63 as near-ID negative control. If negative, separately test other channels or all-history bounding; do not combine with proprio changes in the first pair.','metrics':'7 action per-dim max at every step and final first10/all32; gripper max; future-force per-channel range/zscore. Repeat matched N202/N303 only after first pair.','interpretation':'Tests sensitivity to this coordinate under reconstructed normalization, not causal proof of original normalizer error or real-world control safety.'}}
OUT.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'states':groups,'records':len(records),'output':str(OUT)},indent=2))
