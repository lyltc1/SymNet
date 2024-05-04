"""
usages:
python core/symn/run_evaluate.py --eval_folder output/SymNet_ycbv_obj15_xxx
"""

import os
import sys

sys.path.insert(0, os.getcwd())
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../../"))  # add project directory to sys.path

import argparse
from tqdm import tqdm
import numpy as np

import torch
from mmengine import Config
from bop_toolkit_lib import inout

from core.symn.MetaInfo import MetaInfo
from core.symn.datasets.BOPDataset_utils import build_BOP_test_dataset, batch_data_test
from core.symn.models.SymNetLightning import build_model
from lib.utils.time_utils import add_timing_to_list
from core.symn.utils.visualize_utils import visualize_v2
from core.symn.utils.renderer import ObjCoordRenderer
from core.symn.utils.obj import load_objs

from utils.pose_optimization import pose_pnp


def write_cvs(evaluation_result_path, file_name_prefix, predictions):
    if not os.path.exists(evaluation_result_path):
        os.makedirs(evaluation_result_path)
    for obj_id, predict_list in predictions.items():
        filename = file_name_prefix + "-test"
        filename = os.path.join(evaluation_result_path, filename + ".csv")
        with open(filename, "w") as f:
            f.write("scene_id,im_id,obj_id,score,R,t,time\n")
            for data in predict_list:
                scene_id = data["scene_id"]
                img_id = data["im_id"]
                r = data["R"]
                t = data["t"]
                time = data["time"]
                score = data["score"]
                r11 = r[0][0]
                r12 = r[0][1]
                r13 = r[0][2]

                r21 = r[1][0]
                r22 = r[1][1]
                r23 = r[1][2]

                r31 = r[2][0]
                r32 = r[2][1]
                r33 = r[2][2]

                f.write(str(scene_id))
                f.write(",")
                f.write(str(img_id))
                f.write(",")
                f.write(str(obj_id))
                f.write(",")
                f.write(str(score))  # score
                f.write(",")
                # R
                f.write(str(r11))
                f.write(" ")
                f.write(str(r12))
                f.write(" ")
                f.write(str(r13))
                f.write(" ")
                f.write(str(r21))
                f.write(" ")
                f.write(str(r22))
                f.write(" ")
                f.write(str(r23))
                f.write(" ")
                f.write(str(r31))
                f.write(" ")
                f.write(str(r32))
                f.write(" ")
                f.write(str(r33))
                f.write(",")
                # t
                f.write(str(t[0]))
                f.write(" ")
                f.write(str(t[1]))
                f.write(" ")
                f.write(str(t[2]))
                f.write(",")
                # time
                f.write(f"{str(time)}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_folder", metavar="FILE", help="path to eval folder")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ckpt", default="all", help="ckpt name to be evaluated")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--detection", required=False)

    args = parser.parse_args()
    # parse --eval_folder, generate args.config_file
    assert args.eval_folder
    for file in os.listdir(args.eval_folder):
        if os.path.splitext(file)[1] == ".py":
            args.config_file = os.path.join(args.eval_folder, file)

    cfg = Config.fromfile(args.config_file)
    # parse --ckpt, generate args.ckpt, do not use ckpt_for_last now
    ckpt = list()
    for file in os.listdir(args.eval_folder):
        if os.path.splitext(file)[1] == ".ckpt":
            ckpt.append(file)
    if args.ckpt == "all":
        args.ckpt = ckpt
    elif args.ckpt == "best":
        args.ckpt = [ck for ck in ckpt if ck !="last.ckpt"]
        assert len(args.ckpt) == 1, "not able to determine which ckpt to evaluate"
    else:
        assert args.ckpt in ckpt
        if isinstance(args.ckpt, str):
            args.ckpt = [
                args.ckpt,
            ]
    # parse --debug
    cfg.DEBUG = args.debug
    # parse device
    device = torch.device(args.device)

    # parse --detection
    if "gdrnppdet" in args.detection:
        cfg.DATASETS.TEST_DETECTION_TYPE = "type2"
        cfg.DATASETS.TEST_DETECTION_PATH = args.detection
    elif "zebrapose" in args.detection:
        cfg.DATASETS.TEST_DETECTION_TYPE = "type1"
        cfg.DATASETS.TEST_DETECTION_PATH = args.detection

    # get info used in calculate metric
    obj_ids = cfg.DATASETS.OBJ_IDS
    dataset_name = cfg.DATASETS.NAME
    meta_info = MetaInfo(dataset_name)
    models_info = inout.load_json(meta_info.models_info_path, keys_to_int=True)
    sym_obj_id = cfg.DATASETS.SYM_OBJS_ID
    if sym_obj_id == "bop":
        sym_obj_id = [
            k
            for k, v in models_info.items()
            if "symmetries_discrete" in v or "symmetries_continuous" in v
        ]
    objs = load_objs(meta_info, obj_ids)
    if cfg.DATASETS.CODE_TYPE == 'SymCode':
        from utils.class_id_encoder_decoder import load_decoders
        decoders = load_decoders(
            meta_info.models_GT_color_folder,
            bit=cfg.MODEL.GEOMETRY_NET.get("CODE_BIT", 16),
            obj_ids=obj_ids,
        )
    elif cfg.DATASETS.CODE_TYPE == 'ZebraCode':
        from utils.class_id_encoder_decoder import load_decoders_zebracode
        decoders = load_decoders_zebracode(meta_info.zebrapose_code_folder, obj_ids=obj_ids)

    renderer = ObjCoordRenderer(objs, [k for k in objs.keys()], cfg.DATASETS.RES_CROP)
    # CODE_BIT is a new parameter that needs a default value of 16, ensuring compatibility with older configuration
    cfg.MODEL.GEOMETRY_NET.CODE_BIT = cfg.MODEL.GEOMETRY_NET.get("CODE_BIT", 16)
    # set output_dir and resume
    cfg.OUTPUT_ROOT = os.path.abspath(args.eval_folder)

    for ck in args.ckpt:
        cfg.OUTPUT_DIR = os.path.join(
            cfg.OUTPUT_ROOT, os.path.splitext(ck)[0] + "ablation_pnp"
        )
        if not os.path.exists(cfg.OUTPUT_DIR):
            os.mkdir(cfg.OUTPUT_DIR)
        cvs_path = cfg.OUTPUT_DIR
        cfg.VIS_DIR = os.path.join(cvs_path, "visualize")
        if not os.path.exists(cfg.VIS_DIR):
            os.mkdir(cfg.VIS_DIR)
        cfg.RESUME = os.path.join(cfg.OUTPUT_ROOT, ck)

        assert cfg.MODEL.NAME == "SymNet"
        model = build_model(cfg)
        model.load_state_dict(torch.load(cfg.RESUME)["state_dict"])
        model.eval().to(device).freeze()

        # load data
        data_test = build_BOP_test_dataset(cfg, cfg.DATASETS.TEST, debug=cfg.DEBUG)
        loader_test = torch.utils.data.DataLoader(
            data_test,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            collate_fn=batch_data_test,
        )
        predictions = dict()
        time_forward, time_optimize = [], []
        for idx, batch in enumerate(tqdm(loader_test)):
            with add_timing_to_list(time_forward):
                out_dict = model.infer(
                    batch["rgb_crop"].to(device),
                    obj_idx=batch["obj_idx"].to(device),
                    K=batch["K_crop"].to(device),
                    AABB=batch["AABB_crop"].to(device),
                )
            out_rots = out_dict["rot"].detach().cpu().numpy()  # [b,3,3]
            out_transes = out_dict["trans"].detach().cpu().numpy()  # [b,3]

            for i in range(len(out_rots)):
                scene_id = batch["scene_id"][i]
                im_id = batch["img_id"][i]
                score = batch["det_score"][i] if "det_score" in batch.keys() else 1.0
                time = batch["det_time"][i] if "det_time" in batch.keys() else 1000.0

                obj_id = batch["obj_id"][i]
                gt_R = batch["cam_R_obj"][i]
                gt_t = batch["cam_t_obj"][i]
                # get pose
                est_R = out_rots[i]
                est_t = out_transes[i]

                K_d_2 = np.copy(batch["K_crop"][0].cpu().numpy())
                K_d_2[:2, :] = K_d_2[:2, :] / 2
                with add_timing_to_list(time_optimize):
                    est_R, est_t, success = pose_pnp(
                        K_d_2,
                        out_dict["visib_mask_prob"][0, 0],
                        out_dict["binary_code_prob"][0],
                        decoders[obj_id],
                    )
                if success is False:
                    continue
                if obj_id not in predictions:
                    predictions[obj_id] = list()
                result = {
                    "score": score,
                    "R": est_R,
                    "t": est_t,
                    "gt_R": gt_R,
                    "gt_t": gt_t,
                    "scene_id": scene_id,
                    "im_id": im_id,
                    "time": time,
                }
                if args.visualize:
                    visualize_v2(batch, cfg.VIS_DIR, out_dict, renderer=renderer)
                predictions[obj_id].append(result)
        time_forward = np.array(time_forward)
        print("time_forward", np.mean(time_forward))
        for d in predictions[obj_id]:
            d["time"] = d["time"] + np.mean(time_forward)
        write_cvs(cvs_path, cfg.MODEL.NAME + "_" + cfg.DATASETS.NAME, predictions)


if __name__ == "__main__":
    main()