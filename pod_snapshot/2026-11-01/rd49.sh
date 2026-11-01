cd /workspace
pgrep -af replay_forces.py | head -2
python3 - <<PY
import torch
d=torch.load("/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_004000/optimizer.pt",map_location="cpu",weights_only=False)
st=d["state"]
ids=d["param_groups"][0]["params"]
print("param ids",len(ids),"state entries",len(st))
big=[]
for i in ids:
  e=st.get(i)
  if not e: continue
  sh=tuple(e["exp_avg"].shape)
  if sh[0]<200000: big.append((i,sh,int(e["exp_avg"].numel()+e["exp_avg_sq"].numel())/2))
nz=[x for x in big if x[1][0]!=0]
for x in big[:6]: print("small param",x)
tot=0
for i in ids:
  e=st.get(i)
  if e: tot+=int(e["exp_avg"].numel())
print("optimizer total numel",tot)
PY
