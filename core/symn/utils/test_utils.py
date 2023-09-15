import logging
import os.path as osp
import mmcv

logger = logging.getLogger(__name__)


def _to_str(item):
    if isinstance(item, (list, tuple)):
        return " ".join(["{}".format(e) for e in item])
    else:
        return "{}".format(item)

def to_list(array):
    return array.flatten().tolist()

def save_results(cfg, results_all, output_dir):
    save_root = output_dir  # eval_path
    split_type_str = f"-{cfg.VAL.SPLIT_TYPE}" if cfg.VAL.SPLIT_TYPE != "" else ""
    mmcv.mkdir_or_exist(save_root)
    header = "scene_id,im_id,obj_id,score,R,t,time"
    keys = header.split(",")
    result_names = []
    for name, result_list in results_all.items():
        method_name = f"{cfg.EXP_ID.replace('_', '-')}-{name}"
        result_name = f"{method_name}_{cfg.DATASETS.NAME}-test{split_type_str}.csv"
        res_path = osp.join(save_root, result_name)
        result_names.append(result_name)
        with open(res_path, "w") as f:
            f.write(header + "\n")
            for line_i, result in enumerate(result_list):
                items = []
                for res_k in keys:
                    items.append(_to_str(result[res_k]))
                f.write(",".join(items) + "\n")
        logger.info("wrote results to: {}".format(res_path))
