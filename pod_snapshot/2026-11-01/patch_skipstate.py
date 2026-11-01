p='/workspace/FasterWAM/src/fasterwam/trainer.py'
s=open(p).read()
old='''        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)'''
new='''        state_path = os.path.join(self.state_dir, step_tag)
        if os.environ.get("FW_SKIP_STATE") == "1":
            return {"weights_path": ckpt_path, "state_path": None}
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)'''
if s.count(old)==0:
    print('ALREADY_OR_MISMATCH'); raise SystemExit
open(p,'w').write(s.replace(old,new,1))
print('PATCH_OK')
