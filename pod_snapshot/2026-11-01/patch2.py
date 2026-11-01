p='/workspace/FasterWAM/src/fasterwam/trainer.py'
t=open(p).read()
a=('        trainable_params = list(self.model.dit.parameters())\n'
   '        proprio_encoder = getattr(self.model, "proprio_encoder", None)\n'
   '        if proprio_encoder is not None:\n'
   '            trainable_params.extend(list(proprio_encoder.parameters()))\n')
assert t.count(a)==1, t.count(a)
b=('        trainable_params = [p for p in self.model.dit.parameters() if p.requires_grad]\n'
   '        proprio_encoder = getattr(self.model, "proprio_encoder", None)\n'
   '        if proprio_encoder is not None:\n'
   '            trainable_params.extend([p for p in proprio_encoder.parameters() if p.requires_grad])\n')
open(p,'w').write(t.replace(a,b))
q='/workspace/FasterWAM/configs/task/a3_fasterwam.yaml'
s=open(q).read()
if 'redirect_common_files' not in s:
    s=s.replace('model:\n  mot_checkpoint_mixed_attn: false\n',
                'model:\n  redirect_common_files: false\n  mot_checkpoint_mixed_attn: false\n')
    open(q,'w').write(s)
print('PATCH2_OK')
