import os,json,hashlib,pathlib,shutil
os.environ['HF_HUB_DISABLE_XET']='0'
from huggingface_hub import hf_hub_download
r=pathlib.Path('/workspace/eval_restore_20261117')
items=[('Wan-AI/Wan2.2-TI2V-5B','921dbaf3f1674a56f47e83fb80a34bac8a8f203e','Wan2.2_VAE.pth','20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36',2818839170),('Wan-AI/Wan2.2-TI2V-5B','921dbaf3f1674a56f47e83fb80a34bac8a8f203e','models_t5_umt5-xxl-enc-bf16.pth','7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d',11361920418)]
for name,sha,size in [('special_tokens_map.json','7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d',6623),('spiece.model','e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458',4548313),('tokenizer.json','6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b',16837417),('tokenizer_config.json','ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05',61728)]:
 items.append(('Wan-AI/Wan2.1-T2V-1.3B','37ec512624d61f7aa208f7ea8140a131f93afc9a','google/umt5-xxl/'+name,sha,size))
if shutil.disk_usage(r).free < sum(x[4] for x in items)+2_000_000_000: raise RuntimeError('disk headroom insufficient')
results=[]
for repo,rev,name,sha,size in items:
 src=pathlib.Path(hf_hub_download(repo,name,revision=rev))
 h=hashlib.sha256()
 with src.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
 actual=h.hexdigest()
 if actual!=sha or src.stat().st_size!=size: raise RuntimeError('asset identity mismatch '+name)
 dest=r/'auxiliary'/repo/name
 dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists() or dest.is_symlink(): raise FileExistsError(str(dest))
 dest.symlink_to(src)
 results.append({'path':str(dest),'cache':str(src),'sha256':actual,'bytes':size,'repo':repo,'revision':rev})
 (r/'auxiliary_inventory.json').write_text(json.dumps(results,indent=2))
 print(json.dumps({'verified':name,'bytes':size}),flush=True)
print('AUXILIARY_COMPLETE',flush=True)
