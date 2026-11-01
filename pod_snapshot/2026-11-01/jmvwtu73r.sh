cd /workspace/a3_eval/replay_bundle_dist && ls | head -5 && python3 -c "
import json
rows=json.load(open('summary.json'))
print('rows',len(rows))
for r in rows[:6]:
    print(r['seed'],r['mode'],r['steps'],r.get('stop'),'padmax',r.get('pad_max_n'),'over2n',r.get('frames_over_2n'))
    for c in r.get('clamps',[]):
        print('   clamp mc',c.get('max_cookies_at_grip'),'clos',c.get('closure_peak_pad_n'),'carry',c.get('carry_peak_pad_n'),'lift',c.get('carried_lift_mm'),'up',c.get('placed_upright'),'acc',c.get('accept'))
"