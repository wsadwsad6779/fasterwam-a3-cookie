import numpy as np
E = np.load("/workspace/expert_seed11.npz", allow_pickle=True)
R = np.load("/workspace/fasterwam_a3_bridge/bridge/outputs/C5/lora_C5_11.npz", allow_pickle=True)
def rle(seq):
    out=[]; s=0
    for i in range(1,len(seq)):
        if seq[i]!=seq[i-1]:
            out.append((str(seq[s]), s, i-s)); s=i
    out.append((str(seq[s]), s, len(seq)-s)); return out
print("EXPERT_RLE", rle(list(E["phases"]))[:14])
print("ROLL_RLE", rle(list(R["phase"]))[:14])
pi = R["pad_force_inner"]; po = R["pad_force_outer"]
print("PAD_SHAPE", np.shape(pi), np.shape(po))
pad = np.maximum(np.abs(np.asarray(pi).ravel()), np.abs(np.asarray(po).ravel()))
print("PAD max/med/n", round(float(pad.max()),1), round(float(np.median(pad)),1), pad.size)
rp=R["eef_xyz"]; ep=E["eef"][:, :3]
print("roll_start", np.round(rp[0],3), "roll_end", np.round(rp[-1],3))
print("exp_start ", np.round(ep[0],3), "exp_end ", np.round(ep[-1],3))
