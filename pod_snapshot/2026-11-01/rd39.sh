cd /workspace
python3 - <<PY
import glob,numpy as np,pyarrow.parquet as pq,json
fs=sorted(glob.glob("/workspace/data/a3_front_close_left_100/data/*/*.parquet"))
F=[];EP=[];TS=[];CT=[]
for p in fs:
  t=pq.read_table(p,columns=["episode_index","timestamp","observation.force","frame_index"]).to_pylist()
  for r in t:
    F.append(r["observation.force"]); EP.append(r["episode_index"]); TS.append(r["timestamp"]); CT.append(r["frame_index"])
F=np.asarray(F,dtype=np.float64); EP=np.asarray(EP); TS=np.asarray(TS); CT=np.asarray(CT)
print("rows",F.shape,"eps",len(set(EP.tolist())),"null",int(np.isnan(F).sum()))
print("ts monoton per ep",all(bool(np.all(np.diff(TS[EP==e])>0)) for e in sorted(set(EP.tolist()))[:20]))
print("ts range",TS[0],TS[-1],"dt med",float(np.median(np.diff(TS[EP==EP[0]]))))
names=["L_fx","L_fy","L_fz","L_tx","L_ty","L_tz","R_fx","R_fy","R_fz","R_tx","R_ty","R_tz","L_in","L_out","R_in","R_out","L_act","R_act"]
for i,nm in enumerate(names):
  v=F[:,i]
  print(nm,"p50",round(float(np.percentile(v,50)),3),"p95",round(float(np.percentile(v,95)),3),"max",round(float(v.max()),3),"min",round(float(v.min()),3))
pad=np.maximum(F[:,12],F[:,13])
padr=np.maximum(F[:,14],F[:,15])
print("Lpadmax p50",round(float(np.percentile(pad,50)),3),"p95",round(float(np.percentile(pad,95)),3),"max",round(float(pad.max()),3))
print("Rpadmax p50",round(float(np.percentile(padr,50)),3),"p95",round(float(np.percentile(padr,95)),3),"max",round(float(padr.max()),3))
for th in (2.0,25.0,40.0):
  print("frames over",th,"L",int((pad>th).sum()),"R",int((padr>th).sum()))
mm=[]
for e in sorted(set(EP.tolist())):
  sub=pad[EP==e]
  mm.append((e,int((CT[EP==e]).max()+1),round(float(np.median(sub)),3),round(float(sub.max()),3)))
print("per-episode med of ep pad max",round(float(np.median([m[3] for m in mm])),3),"min",round(float(min(m[3] for m in mm)),3),"max",round(float(max(m[3] for m in mm)),3))
print("ep lens first 6",mm[:6])
json.dump({"episodes":mm},open("/workspace/_r5_stage/corpus_force_eps.json","w"))
PY
