import json
import os
import traceback

import imageio

P = '/workspace/fwv1/data/hf_videos/videos/observation.images.front/chunk-000/file-000.mp4'
OUT = '/workspace/fwv1/p_items/VIDEO_DECODE_TEST.json'
res = {'path': P, 'exists': os.path.exists(P)}
try:
    rd = imageio.get_reader(P)
    res['count_frames_reported'] = int(rd.count_frames())
    f = rd.get_data(0)
    res['frame_shape'] = list(f.shape)
    res['frame_dtype'] = str(f.dtype)
    res['frame_mean'] = float(f.mean())
    f2 = rd.get_data(4)
    res['frame4_mean'] = float(f2.mean())
    rd.close()
    res['ok'] = bool(res['frame_shape'][0] > 0 and res['frame_shape'][1] > 0)
except Exception:
    res['error'] = traceback.format_exc()[:800]
    res['ok'] = False
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps({k: res.get(k) for k in ['ok', 'frame_shape', 'count_frames_reported', 'error']}))