import json
import sys
from pathlib import Path

K = ('control', 'seed', 'success', 'cookies_in_target', 'max_pad_force', 'max_wrist_force',
     'verbatim_pct', 'steps', 'replans', 'estop_steps', 'binfloor_steps',
     'safety_would_have_changed_pct', 'safety_would_have_hard', 'safety_would_have_binfloor',
     'final_phase', 'duration_s')

for d in sys.argv[1:]:
    rows = json.loads(Path(d, 'summary_lora.json').read_text(encoding='utf-8'))
    print('DIR', d, 'n', len(rows))
    for row in rows:
        print('  ' + ' '.join('%s=%s' % (k, row.get(k)) for k in K))
