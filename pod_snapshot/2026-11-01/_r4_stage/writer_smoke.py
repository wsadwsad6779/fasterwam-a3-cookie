import os, time, numpy as np, imageio
t = time.time()
frames = [np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8) for _ in range(30)]
p = "/tmp/_writer_smoke.mp4"
imageio.mimsave(p, frames, fps=20)
print("writer_ok", os.path.getsize(p), round(time.time() - t, 2), imageio.__version__)