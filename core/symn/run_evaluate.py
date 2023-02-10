"""
usages:
python core/symn/run_evaluate.py --eval_folder output/SymNet_ycbv_obj15_xxx --debug
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
from mmcv import Config

from bop_toolkit_lib import inout
from bop_toolkit_lib.pose_error import add, adi

from core.symn.MetaInfo import MetaInfo
from core.symn.datasets.BOPDataset_utils import build_BOP_test_dataset, batch_data_test
from core.symn.models.SymNetLightning import build_model
from lib.utils.time_utils import get_time_str
from core.symn.utils.visualize_utils import visualize_v2
from core.symn.utils.renderer import ObjCoordRenderer
from core.symn.utils.obj import load_objs


# function to calculate ycbv metric
def compute_auc_posecnn(errors):
    errors = errors.copy()
    d = np.sort(errors)
    d[d > 0.1] = np.inf
    accuracy = np.cumsum(np.ones(d.shape[0])) / d.shape[0]
    ids = np.isfinite(d)
    d = d[ids]
    accuracy = accuracy[ids]
    if len(ids) == 0 or ids.sum() == 0:
        return np.nan
    rec = d
    prec = accuracy
    mrec = np.concatenate(([0], rec, [0.1]))
    mpre = np.concatenate(([0], prec, [prec[-1]]))
    for i in np.arange(1, len(mpre)):
        mpre[i] = max(mpre[i], mpre[i - 1])
    i = np.arange(1, len(mpre))
    ids = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = ((mrec[ids] - mrec[ids - 1]) * mpre[ids]).sum() * 10
    return ap


def write_cvs(evaluation_result_path, file_name_prefix, predictions):
    if not os.path.exists(evaluation_result_path):
        os.makedirs(evaluation_result_path)
    for obj_id, predict_list in predictions.items():
        filename = file_name_prefix + '-test'
        filename = os.path.join(evaluation_result_path, filename + '.csv')
        with open(filename, "w") as f:
            f.write("scene_id,im_id,obj_id,score,R,t,time\n")
            for data in predict_list:
                scene_id = data['scene_id']
                img_id = data['im_id']
                r = data['R']
                t = data['t']
                time = data['time']
                score = data['score']
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
        os.system("python /home/lyltc/git/GDR-Net/bop_toolkit/scripts/eval_bop19_pose.py " + f"--result_filenames {os.path.abspath(filename)} " + f"--results_path {os.path.abspath(evaluation_result_path)} " +f"--eval_path {os.path.abspath(evaluation_result_path)}")
        os.system("python /home/lyltc/git/GDR-Net/bop_toolkit/scripts/vis_est_poses.py " + f"--result_filenames {os.path.abspath(filename)} " + f"--output_path {os.path.abspath(evaluation_result_path)}")

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
    # parse --use_last_ckpt, generate args.ckpt
    args.ckpt = os.path.join(args.eval_folder, 'last.ckpt')
    if not args.use_last_ckpt:
        for file in os.listdir(args.eval_folder):
            if os.path.splitext(file)[1] == '.ckpt' and os.path.splitext(file)[0].startswith("epoch"):
                args.ckpt = os.path.join(args.eval_folder, file)
    cfg.RESUME = args.ckpt
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
        cfg.VIS_DIR = os.path.join(cfg.OUTPUT_DIR, "visualize")
        if not os.path.exists(cfg.OUTPUT_DIR):
            os.mkdir(cfg.OUTPUT_DIR)
        if not os.path.exists(cfg.VIS_DIR):
            os.mkdir(cfg.VIS_DIR)
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
    objs = load_objs(meta_info, obj_ids)
    renderer = ObjCoordRenderer(objs, [k for k in objs.keys()], cfg.DATASETS.RES_CROP)

    # load model
    assert cfg.MODEL.NAME == "SymNet"
    model = build_model(cfg)
    model.load_state_dict(torch.load(cfg.RESUME)['state_dict'])
    model.eval().to(device).freeze()

    # load data
    data_test = build_BOP_test_dataset(cfg, cfg.DATASETS.TEST, debug=cfg.DEBUG)
    loader_test = torch.utils.data.DataLoader(data_test,
                                              batch_size=1,
                                              num_workers=4,
                                              pin_memory=True,
                                              collate_fn=batch_data_test,
                                              )
    predictions = dict()
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

    # calculate matric
    for obj_id, predict_list in predictions.items():
        adx = adi if obj_id in sym_obj_id else add
        diameter = diameters[obj_id]
        ADX_passed = np.zeros(len(predict_list))
        ADX_passed_5 = np.zeros(len(predict_list))
        ADX_passed_2 = np.zeros(len(predict_list))
        ADX_error = np.zeros(len(predict_list))
        ADX_passed_posecnn_10 = np.zeros(len(predict_list))
        for i, d in enumerate(predict_list):
            adx_error = adx(d['R'], d['t'], d['gt_R'], d['gt_t'], pts=models_3d[obj_id]["pts"])
            if np.isnan(adx_error):
                adx_error = 10000
            ADX_error[i] = adx_error
            if adx_error < diameter * 0.1:
                ADX_passed[i] = 1
            if adx_error < diameter * 0.05:
                ADX_passed_5[i] = 1
            if adx_error < diameter * 0.02:
                ADX_passed_2[i] = 1
            if adx_error < 100:
                ADX_passed_posecnn_10[i] = 1
        ADX_passed = np.mean(ADX_passed)
        ADX_passed_5 = np.mean(ADX_passed_5)
        ADX_passed_2 = np.mean(ADX_passed_2)
        ADX_passed_posecnn_10 = np.mean(ADX_passed_posecnn_10)
        AUC_ADX_error_posecnn = compute_auc_posecnn(ADX_error / 1000.0)
        adx_str = 'adi' if obj_id in sym_obj_id else "add"
        print(f"obj{obj_id}-{adx_str}_10d : {ADX_passed}")
        print(f"obj{obj_id}-{adx_str}_5d : {ADX_passed_5}")
        print(f"obj{obj_id}-{adx_str}_2d : {ADX_passed_2}")
        print(f"obj{obj_id}-AUC_posecnn_{adx_str} : {AUC_ADX_error_posecnn}")
        print(f"obj{obj_id}-_posecnn_{adx_str} : {ADX_passed_posecnn_10}")
        print(f"diameter: {diameter}")
        result_path = os.path.join(cfg.OUTPUT_DIR, "result.txt")
        print('save ADD results to', result_path)
        f = open(result_path, "w")
        f.write(f"obj{obj_id}-{adx_str}_10d : {ADX_passed}")
        f.write(f"obj{obj_id}-{adx_str}_5d : {ADX_passed_5}")
        f.write(f"obj{obj_id}-{adx_str}_2d : {ADX_passed_2}")
        f.write(f"obj{obj_id}-AUC_posecnn_{adx_str} : {AUC_ADX_error_posecnn}")
        f.write(f"obj{obj_id}-_posecnn_{adx_str} : {ADX_passed_posecnn_10}")
        f.write(f"diameter: {diameter}")
        f.write("\n")
    cvs_path = os.path.join(cfg.OUTPUT_DIR, 'result_bop/')
    write_cvs(cvs_path, cfg.MODEL.NAME + '_' + cfg.DATASETS.NAME, predictions)


if __name__ == "__main__":
    main()
