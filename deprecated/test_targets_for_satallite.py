import json

out = []
for i in range(0, 101):
    d = {"im_id": i, "inst_count": 1, "obj_id": 1, "scene_id": 0, }
    out.append(d)
with open("test_targets.json", "w") as f:
    f.write(json.dumps(out).replace("}, ", "},\n"))
