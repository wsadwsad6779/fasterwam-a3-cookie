p='/workspace/FasterWAM/src/fasterwam/runtime.py'
s=open(p).read()
a1='    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)\n    train_ds, val_ds = build_datasets(cfg.data)\n'
assert s.count(a1)==1, s.count(a1)
b1=('    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)\n'
    '    if cfg.get("lora"):\n'
    '        from fasterwam.lora import apply_lora\n'
    '        n_layers, n_trainable = apply_lora(model, cfg.lora)\n'
    '        logging.info("Applied LoRA to %s layers (trainable params = %s)", n_layers, n_trainable)\n'
    '    train_ds, val_ds = build_datasets(cfg.data)\n')
open(p,'w').write(s.replace(a1,b1))

p2='/workspace/FasterWAM/src/fasterwam/trainer.py'
t=open(p2).read()
a2='    @staticmethod\n    def _apply_dit_only_train_mode(model):\n        model.eval()\n'
assert t.count(a2)==1, t.count(a2)
t=t.replace(a2,'    def _apply_dit_only_train_mode(self, model):\n        model.eval()\n')
a3='        model.dit.requires_grad_(True)\n'
assert t.count(a3)==1, t.count(a3)
b3=('        model.dit.requires_grad_(True)\n'
    '        if self.cfg.get("lora"):\n'
    '            for _name, _p in model.named_parameters():\n'
    '                if "lora_" not in _name:\n'
    '                    _p.requires_grad_(False)\n'
    '            logger.info("LoRA mode: only lora_ params trainable.")\n')
t=t.replace(a3,b3)
open(p2,'w').write(t)
print('PATCHED')
