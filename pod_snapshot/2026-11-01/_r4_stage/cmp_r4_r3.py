import json,sys,numpy as np
sys.path.insert(0,"/workspace/fasterwam_a3_bridge/bridge/_next")
from analyze_gripper import sign,flips
C="/workspace/a3_eval/r4_c0_frozen/cmp/"
P3="/workspace/a3_eval/r3_c0_diag_600/"
P4="/workspace/a3_eval/r4_c0_frozen/"
def L(p): return json.load(open(p))
def by(x): return dict((int(r["seed"]),r) for r in x)
def grip(p):
    m=np.load(p,allow_pickle=False)
    pre=m["pre_14d"][:,6];post=m["post_14d"][:,6];ci=m["chunk_idx"];op=m["opening"];fo=m["pad_force_inner"];ch=np.asarray(m["changed"])
    e=sign(pre);b=np.zeros(len(pre),bool);b[1:]=ci[1:]!=ci[:-1];f=e[1:]!=e[:-1];ct=fo>5.0;hd=ct&(op<0.6)
    return dict(pre_flips=int(flips(pre)),post_flips=int(flips(post)),seam=int(np.sum(f&b[1:])),within=int(np.sum(f&~b[1:])),while_hold=int(np.sum(f&hd[1:])),safety_steps=int(ch.sum()),safety_pct=round(100*float(ch.mean()),2),holding=int(hd.sum()),contact=int(ct.sum()))
CFG={"r3":(P3+"rep0/lora_C0_%d.npz",C+"r3_lift_recheck.json",C+"r3_force_recheck.json",P3+"summary.json",P3+"gripper_%d.json"),"r4":(P4+"lora_C0_%d.npz",C+"r4_lift.json",C+"r4_force.json",P4+"summary_lora.json","")}
K=[("in","cookies_in_target","s"),("succ","success","s"),("lift_mm","max_cookie_lift_mm","l"),("follow","stable_follow_steps","l"),("transport_mm","transport_xy_mm","l"),("flip_within","within","g"),("flip_seam","seam","g"),("flip_while_hold","while_hold","g"),("safety_steps","safety_steps","g"),("safety_pct","safety_pct","g"),("verbatim_pct","verbatim_pct","s"),("pad_max_n","pad_force_max_n","f"),("pad_p50","pad_force_p50_n","f"),("pad_p90","pad_force_p90_n","f"),("pad_p99","pad_force_p99_n","f"),("frac_over5n","frac_over_5n","f"),("frac_over10n","frac_over_10n","f"),("frac_above_soft","frac_steps_above_soft","f"),("hold_frac","hold_frac","f"),("prem_release","premature_release_events","f"),("open_cmd_while_hold","open_cmd_events_while_holding","f"),("ever_lifted","cookies_ever_lifted","f"),("wrist_max_n","wrist_force_max_n","f"),("wrapper_steps","wrapper_rewritten_steps","f"),("wrapper_bitwise","wrapper_stage_bitwise_identical_to_model_action","f"),("grasp_frac","grasp_frames_frac","f"),("dur_s","duration_s","s"),("estop","estop_steps","s"),("binfloor","binfloor_steps","s"),("scene_ok","scene_ok","s"),("oracle_reads","oracle_cookie_reads","s"),("steps","steps","s"),("replans","replans","s")]
OUT={}
for run in ("r3","r4"):
    cf=CFG[run]
    lf=by(L(cf[1]));fc=by(L(cf[2]));sm=by(L(cf[3]))
    for sd in (11,21,22): OUT[run+str(sd)]={"s":sm[sd],"l":lf[sd],"f":fc[sd],"g":grip(cf[0]%sd)}
print("field,r3_11,r4_11,r3_21,r4_21,r3_22,r4_22")
for name,key,blk in K:
    print(name+","+",".join(str(OUT[run+str(sd)][blk].get(key,"NA")) for sd in (11,21,22) for run in ("r3","r4")))
print("r3_archived_vs_recomputed")
for sd in (11,21,22):
    a=L(CFG["r3"][4]%sd);mm=OUT["r3"+str(sd)]["g"]
    chk={"seam":(a["executed"]["at_chunk_boundary"],mm["seam"]),"within":(a["executed"]["within_chunk"],mm["within"]),"pre":(a["executed"]["flips_pre_14d"],mm["pre_flips"]),"post":(a["executed"]["flips_post_safety"],mm["post_flips"]),"safety_steps":(a["executed"]["safety_rewritten_steps"],mm["safety_steps"]),"safety_pct":(a["executed"]["safety_pct"],mm["safety_pct"]),"hold":(a["context"]["flips_while_holding"],mm["while_hold"]),"holding":(a["context"]["holding_steps"],mm["holding"]),"contact":(a["context"]["in_contact_steps"],mm["contact"])}
    print(sd,json.dumps(dict((k,v) for k,v in chk.items() if v[0]!=v[1])) or "MATCH_ALL")
