"""
usages:
python core/symn/run_evaluate_with_human_help.py --eval_folder output/SymNet_tless_obj04_xxx --debug
"""
import os
import sys

sys.path.insert(0, os.getcwd())
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../../"))  # add project directory to sys.path

import argparse
import importlib
from tqdm import tqdm
import numpy as np
import torch
from mmcv import Config
import cv2

from bop_toolkit_lib import inout
from bop_toolkit_lib.pose_error import add, adi

from core.symn.MetaInfo import MetaInfo
from core.symn.datasets.BOPDataset_utils import build_BOP_test_dataset, batch_data_test
from core.symn.models.SymNetLightning import build_model
from lib.utils.utils import iprint
from lib.utils.time_utils import get_time_str
from core.symn.utils.visualize_utils import visualize_v2
from core.symn.utils.renderer import ObjCoordRenderer
from core.symn.utils.obj import load_objs
from core.symn.datasets.GDRN_aux import AugRgbAux, ReplaceBgAux
from core.symn.datasets.std_auxs import RgbLoader, MaskLoader, RandomRotatedMaskCrop, NormalizeAux
from core.symn.datasets.symn_aux import GTLoader, GT2CodeAux, PosePresentationAux, LoadSymInfoExtentsAux, LoadPointsAux


def get_aux(cfg, aug_bg=False, aug_rgb=False, detection=True, crop_scale=[1.5, 1.5]):
    auxs = [RgbLoader()]
    if detection:
        crop_aux = RandomRotatedMaskCrop(cfg.DATASETS.RES_CROP, use_bbox_est=True, offset_scale=0.,
                                         crop_scale=crop_scale,
                                         crop_keys=('rgb', 'mask_visib'),
                                         crop_keys_crop_res_divide2=('mask', 'mask_visib', 'GT'),
                                         rgb_interpolation=cv2.INTER_LINEAR)
    else:
        crop_aux = RandomRotatedMaskCrop(cfg.DATASETS.RES_CROP, mask_key='mask_visib',
                                         crop_keys=('rgb', 'mask_visib'),
                                         crop_keys_crop_res_divide2=('mask', 'mask_visib', 'GT'),
                                         rgb_interpolation=cv2.INTER_LINEAR)
    # train_aux
    auxs.extend([MaskLoader(),
                 GTLoader(),
                 crop_aux.definition_aux,
                 crop_aux.apply_aux,
                 ])
    if aug_bg is True:
        auxs.extend([ReplaceBgAux(cfg.DATASETS.BG_AUG_PROB, cfg.DATASETS.BG_AUG_TYPE), ])
    if aug_rgb is True:
        auxs.extend([AugRgbAux(cfg.DATASETS.COLOR_AUG_PROB), ])

    auxs.extend([crop_aux.apply_aux_d2,
                 GT2CodeAux(),
                 PosePresentationAux(cfg.DATASETS.RES_CROP,
                                     R_type=cfg.MODEL.PNP_NET.R_type, t_type=cfg.MODEL.PNP_NET.t_type),
                 LoadPointsAux(num_points=cfg.MODEL.PNP_NET.PM_NUM_POINTS),
                 LoadSymInfoExtentsAux(),
                 ])
    auxs.extend([NormalizeAux(), ])
    return auxs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_folder", metavar="FILE", help="path to eval folder")
    parser.add_argument("--use_last_ckpt", action="store_true", help="else use best ckpt")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--device', default='cuda:0')

    args = parser.parse_args()
    # parse --eval_folder, generate args.config_file
    assert args.eval_folder
    for file in os.listdir(args.eval_folder):
        if os.path.splitext(file)[1] == '.py':
            args.config_file = os.path.join(args.eval_folder, file)
    cfg = Config.fromfile(args.config_file)
    print(f"use config file: {args.config_file}")

    # parse --use_last_ckpt, generate args.ckpt
    args.ckpt = os.path.join(args.eval_folder, 'last.ckpt')
    if not args.use_last_ckpt:
        for file in os.listdir(args.eval_folder):
            if os.path.splitext(file)[1] == '.ckpt' and os.path.splitext(file)[0].startswith("epoch"):
                args.ckpt = os.path.join(args.eval_folder, file)
    cfg.RESUME = args.ckpt
    print(f"use checkpoint: {args.ckpt}")

    # parse --debug
    cfg.DEBUG = args.debug
    # parse device
    device = torch.device(args.device)
    # set output_dir
    if cfg.OUTPUT_DIR.lower() == "auto":
        out_str = cfg.MODEL.NAME + "_" + cfg.DATASETS.NAME
        for obj_id in cfg.DATASETS.OBJ_IDS:
            out_str += f"_obj{obj_id}"
        out_str += '_' + get_time_str()
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_ROOT, out_str)
        if not os.path.exists(cfg.OUTPUT_DIR):
            os.mkdir(cfg.OUTPUT_DIR)
    # get info used in calculate metric
    obj_ids = cfg.DATASETS.OBJ_IDS
    dataset_name = cfg.DATASETS.NAME
    meta_info = MetaInfo(dataset_name)
    models_3d = {obj_id: inout.load_ply(meta_info.model_tpath.format(obj_id=obj_id)) for obj_id in obj_ids}
    models_info = inout.load_json(meta_info.models_info_path, keys_to_int=True)
    diameters = {obj_id: models_info[obj_id]['diameter'] for obj_id in obj_ids}
    sym_obj_id = cfg.DATASETS.SYM_OBJS_ID
    if sym_obj_id == "bop":
        sym_obj_id = [k for k, v in models_info.items() if 'symmetries_discrete' in v or 'symmetries_continuous' in v]
    cfg.DATASETS.SYM_OBJS_ID = sym_obj_id
    objs = load_objs(meta_info, obj_ids)
    renderer = ObjCoordRenderer(objs, [k for k in objs.keys()], cfg.DATASETS.RES_CROP)

    # load model
    assert cfg.MODEL.NAME == "SymNet"
    model = build_model(cfg)
    model.load_state_dict(torch.load(cfg.RESUME)['state_dict'])
    model.eval().to(device).freeze()

    # load data
    use_detection = True if cfg.DATASETS.TEST_DETECTION_PATH else False
    auxs_test = get_aux(cfg, gt=True, detection=use_detection, aug_bg=False, aug_rgb=False, debug=debug)
    # ---- get detections ----
    detections = None
    if use_detection:
        print(f"use detections from {cfg.DATASETS.TEST_DETECTION_PATH}")
        # get detections use imported function
        get_det_fun_name = "get_detection_results_" + cfg.DATASETS.TEST_DETECTION_TYPE
        get_det_module = importlib.import_module(".get_detection_results", "datasets")
        get_det_fun = getattr(get_det_module, get_det_fun_name)
        detections = get_det_fun(os.path.join(meta_info.detections_folder, cfg.DATASETS.TEST_DETECTION_PATH))

    data = build_BOP_test_dataset(cfg, cfg.DATASETS.TEST, debug=True)

    print()
    print('With an opencv window active:')
    print("press 'a', 'd' and 'x'(random) to get a new input image,")
    while True:
        print('------------ input -------------')



    for idx, batch in enumerate(tqdm(loader_test)):
        out_dict = model.infer(
            batch["rgb_crop"].to(device),
            obj_idx=batch["obj_idx"].to(device),
            K=batch["K_crop"].to(device),
            AABB=batch["AABB_crop"].to(device),
        )
        out_rots = out_dict["rot"].detach().cpu().numpy()  # [b,3,3]
        out_transes = out_dict["trans"].detach().cpu().numpy()  # [b,3]

        for i in range(len(out_rots)):
            scene_id = batch['scene_id'][i]
            im_id = batch['img_id'][i]
            score = batch["det_score"][i] if "det_score" in batch.keys() else 1.0
            time = batch["det_time"][i] if "det_time" in batch.keys() else 1000.0

            obj_id = batch["obj_id"][i]
            gt_R = batch["cam_R_obj"][i]
            gt_t = batch["cam_t_obj"][i]
            # get pose
            est_R = out_rots[i]
            est_t = out_transes[i]

            if obj_id not in predictions:
                predictions[obj_id] = list()
            result = {"score": score, "R": est_R, "t": est_t, "gt_R": gt_R, "gt_t": gt_t,
                      "scene_id": scene_id, "im_id": im_id, "time": time + 100.}
            visualize_v2(batch, cfg.VIS_DIR, out_dict, renderer=renderer)
            predictions[obj_id].append(result)



if __name__ == "__main__":
    main()
