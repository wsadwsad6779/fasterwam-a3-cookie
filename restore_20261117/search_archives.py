import os,json,hashlib,shutil,pathlib
os.environ['HF_HUB_DISABLE_XET']='0'
from huggingface_hub import HfApi,hf_hub_download
r=pathlib.Path('/workspace/eval_restore_20261117')
out=r/'archive_evidence'
out.mkdir(exist_ok=True)
api=HfApi()
report={'repositories':[],'downloads':[],'errors':[]}
for repo,rev in [('wsadwsad6779/fasterwam-a3-cookie-lora','f2aaf55c6ce1b164de97c5b6ec2cd75e942d0de0'),('wsadwsad6779/fasterwam-a3-force-40k','c25f73640670b1be1592feab05cc8e84f88300b9')]:
 try:
  files=api.list_repo_files(repo,revision=rev)
  candidates=[p for p in files if any(s in p.lower() for s in ('stats','normal','v1g','embed','corpus','calib','tare','manifest','vae','tokenizer'))]
  report['repositories'].append({'repo':repo,'revision':rev,'file_count':len(files),'candidates':candidates})
  if repo.endswith('cookie-lora'):
   for p in ['data/a3_fw/arrays.npz','data/a3_fw/dataset_stats.json','data/a3_fw_force/arrays.npz','data/a3_fw_force/force_stats.json']:
    if p not in files: continue
    src=pathlib.Path(hf_hub_download(repo,p,revision=rev))
    dst=out/p
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists(): raise FileExistsError(str(dst))
    shutil.copyfile(src,dst)
    h=hashlib.sha256(dst.read_bytes()).hexdigest()
    report['downloads'].append({'path':str(dst),'repo_path':p,'bytes':dst.stat().st_size,'sha256':h})
 except Exception as e:
  report['errors'].append({'repo':repo,'type':type(e).__name__})
(r/'archive_search.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'archive_search_done':True,'downloads':report['downloads'],'errors':report['errors']}))
