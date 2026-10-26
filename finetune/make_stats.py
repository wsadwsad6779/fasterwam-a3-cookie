import json
import numpy as np
OUT = "/workspace/a3_data/a3_fw"
arr = np.load(OUT + "/arrays.npz")
act, prop = arr["action"], arr["proprio"]

def block(x):
    return {"global_min": x.min(0).tolist(), "global_max": x.max(0).tolist(),
            "global_mean": x.mean(0).tolist(), "global_std": x.std(0).tolist(),
            "global_q01": np.percentile(x, 1, axis=0).tolist(),
            "global_q99": np.percentile(x, 99, axis=0).tolist()}

stats = {"action": {"default": block(act)}, "state": {"default": block(prop)},
         "num_episodes": int(len(arr["ep_len"])), "num_transition": int(act.shape[0])}
json.dump(stats, open(OUT + "/dataset_stats.json", "w"), indent=2)
for k in ("action", "state"):
    b = stats[k]["default"]
    print(k, "min", np.round(b["global_min"], 4).tolist())
    print(k, "max", np.round(b["global_max"], 4).tolist())
