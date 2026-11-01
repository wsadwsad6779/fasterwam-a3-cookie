import io
p='/workspace/FasterWAM/src/fasterwam/trainer.py'
s=open(p).read()
old='            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),'
new='            "none" if self.accelerator.state.deepspeed_plugin is None else self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),'
assert s.count(old)==1, s.count(old)
open(p,'w').write(s.replace(old,new))
print('PATCH_OK')
