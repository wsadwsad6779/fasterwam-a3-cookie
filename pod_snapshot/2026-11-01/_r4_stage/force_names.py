import json, numpy as np
info = json.load(open("/workspace/data/a3_front_close_left_100/meta/info.json"))
f = info["features"]
for k in ("observation.state", "observation.velocity", "observation.eef_pose", "observation.force", "action"):
    print(k, f[k]["shape"], f[k]["names"])
print("robot_type", info.get("robot_type"), "data_path", info.get("data_path"))
