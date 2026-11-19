import json
import numpy as np

base = '/workspace/eval_restore_20261117/reconstructed_normalizer'
man = json.load(open(base + '/teacher_diverse_20261119/manifest.json'))
center = np.array(man['episodes'][0]['batch_center'])[:2]
for seed in (29, 30):
    z = np.load(base + '/c_class_eval_v3_step8000/episode_C_seed%d.npz' % seed)
    m = json.load(open(base + '/c_class_eval_v3_step8000/episode_C_seed%d.json' % seed))
    a = z['action']; eef = z['eef']; op = z['opening']
    d = np.linalg.norm(eef[:, :2] - center, axis=1)
    print('seed%d action.shape=%s' % (seed, a.shape))
    print('  A3 action[:,6] uniq=%s frac==-1(close) %.3f' % (np.unique(a[:, 6]), (a[:, 6] == -1).mean()))
    print('  opening min %.4f max %.4f frac<0.005 %.3f' % (op.min(), op.max(), (op < 0.005).mean()))
    print('  eef xy dist to batch_center: min %.4f median %.4f final %.4f m'
          % (d.min(), np.median(d), d[-1]))
    print('  eef z min %.3f max %.3f final %.3f | path %.1f mm | grasps %s close_ev %s cookies %s src %s'
          % (eef[:, 2].min(), eef[:, 2].max(), eef[-1, 2], m['eef_path_mm'],
             m['grasps_confirmed'], m['gripper_close_events'], m['cookies_final'], m['source_final']))
poses = np.load(base + '/teacher_diverse_20261119/poses.npz')
p0 = poses['poses_ep00']
print('teacher poses_ep00 shape', p0.shape, 'cols:', p0.shape[1] if p0.ndim == 2 else '')
