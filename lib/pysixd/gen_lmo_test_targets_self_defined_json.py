import os
import os.path as osp
import json
from bop_toolkit_lib.inout import load_json

cur_dir = osp.dirname(osp.abspath(__file__))

data_root = "/home/dataset/pbr/tless/test_primesense"


def main():
    test_targets = []  # {"im_id": , "inst_count": , "obj_id": , "scene_id": }
    test_scenes = [int(item) for item in os.listdir(data_root)]
    test_scenes.sort()
    for scene_id in test_scenes:
        print("scene_id", scene_id)
        BOP_gt_file = osp.join(data_root, f"{scene_id:06d}/scene_gt.json")
        assert osp.exists(BOP_gt_file), BOP_gt_file
        gt_dict = load_json(BOP_gt_file, keys_to_int= True)
        all_ids = gt_dict.keys()
        print("this scene contains", len(all_ids), "images")
        for idx in all_ids:
            annos = gt_dict[idx]
            obj_ids = [anno["obj_id"] for anno in annos]
            num_inst_dict = {}
            # stat num instances for each obj
            for obj_id in obj_ids:
                if obj_id not in num_inst_dict:
                    num_inst_dict[obj_id] = 1
                else:
                    num_inst_dict[obj_id] += 1
            for obj_id in num_inst_dict:
                target = {"im_id": idx, "inst_count": num_inst_dict[obj_id], "obj_id": obj_id, "scene_id": scene_id}
                test_targets.append(target)
    res_file = osp.join(cur_dir, "tless_test_targets_all.json")
    print(res_file)
    print(len(test_targets))  # 50904
    with open(res_file, "w") as f:
        f.write("[\n" + ",\n".join(json.dumps(item) for item in test_targets) + "]\n")
    print("done")


if __name__ == "__main__":
    main()
