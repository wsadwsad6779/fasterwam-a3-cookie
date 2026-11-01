import numpy as np
from _next.run_teacher_forcing import replay_expert

for seed in (11, 21, 22):
    r = replay_expert(seed, 600, 20, render=False)
    np.savez_compressed(
        f'/workspace/expert_seed{seed}.npz', eef=r['eef'], opening=r['opening'],
        phases=np.asarray(r['phases']), step_ids=r['step_ids'], act14=r['act14'],
        cookies=r['cookies_in_target'], failed=r['failed'],
        failure_phase=r['failure_phase'])
    print('dump', seed, 'cookies', r['cookies_in_target'], 'steps', len(r['act14']), flush=True)
print('EXPERT_DUMP_DONE', flush=True)
