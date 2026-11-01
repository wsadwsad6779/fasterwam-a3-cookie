set -u
source /workspace/activate_fasterwam_a3.sh
echo ==self.cfg / lr_scheduler_type / max_steps
grep -n -e "self.cfg" -e "lr_scheduler_type" -e "self.max_steps" -e "self.seed" /workspace/FasterWAM/src/fasterwam/trainer.py | head -30
echo ==AcceleratedScheduler
python3 -c "import accelerate,inspect;from accelerate.scheduler import AcceleratedScheduler as A;src=inspect.getsource(A);print(src[:1200])"
echo ==ver
python3 -c "import accelerate,torch;print(accelerate.__version__, torch.__version__)"