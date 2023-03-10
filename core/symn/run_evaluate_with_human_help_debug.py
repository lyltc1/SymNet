"""
this scripts aims to help find better strategy to do evaluation.
For example,
1. mask origin rgb with predict mask
2. use different crop with predict mask
3. different augmentation/rotation of input
4. different optimization method
5. others
usages:
python core/symn/run_evaluate_with_human_help.py --eval_folder output/SymNet_tless_obj04_xxx
"""
import os
import sys

sys.path.insert(0, os.getcwd())
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../../"))  # add project directory to sys.path

import argparse
import numpy as np
import cv2
import torch
from mmcv import Config

from bop_toolkit_lib import inout
from bop_toolkit_lib import misc
from bop_toolkit_lib import pose_error

from lib.utils.time_utils import get_time_str

from core.symn.MetaInfo import MetaInfo
from core.symn.datasets.BOPDataset_utils import build_BOP_test_dataset, build_BOP_train_dataset
from core.symn.models.SymNetLightning import build_model
from core.symn.utils.renderer import ObjCoordRenderer
from core.symn.utils.obj import load_objs
from core.symn.utils.visualize_utils import show_rgb, show_mask_contour, show_mask_code, show_pose,\
                                            preprogress_mask, preprogress_rgb
from core.symn.datasets.std_auxs import RandomRotatedMaskCrop, NormalizeAux
from core.symn.datasets.symn_aux import GT2CodeAux


def preprogress_Rt(R, t, K=None):
    """ use to make R t K easy to draw and evaluate """
    if isinstance(R, torch.Tensor):
        R = R.detach().cpu().numpy()
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    if len(t.shape) == 1:  # convert [3,] to  [3, 1]
        t = t[:, np.newaxis]
    if K is not None:
        if isinstance(K, torch.Tensor):
            K = K.detach().cpu().numpy()
        return R, t, K
    else:
        return R, t


class CropSelector:
    """
    select crop region by human
    """

    def __init__(self, is_square=True):
        self.region = []
        self.image = None
        self.start_pos = None
        self.end_pos = None
        self.is_drawing = None
        self.is_square = is_square

    def draw_crop(self, img_bgr, window_name):
        self.image = img_bgr.copy()
        cv2.setMouseCallback(window_name, self.mouse_handler)
        while cv2.waitKey(50) & 0xFF != ord('q'):
            if self.start_pos is not None and self.end_pos is not None:
                x, y = self.start_pos
                cv2.putText(self.image, f'{x},{y}', (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
                cv2.arrowedLine(self.image, self.start_pos, self.end_pos, (128, 128, 0))
                cv2.rectangle(self.image, self.start_pos, self.end_pos, (128, 128, 0), 1)
                x, y = self.end_pos
                cv2.putText(self.image, f'{x},{y}', (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
                cv2.imshow(window_name, self.image)
            self.image = img_bgr.copy()
            cv2.waitKey(30)
        cv2.setMouseCallback(window_name, self.empty_handler)
        return self.region

    def empty_handler(self, event, x, y, flags, params):
        pass

    def mouse_handler(self, event, x, y, flags, params):
        self.mouse_callback(event, x, y, flags, params)

    def mouse_callback(self, event, x, y, flags, params):
        if event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:
                if self.is_square:
                    y = x - self.start_pos[0] + self.start_pos[1]
                self.end_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.start_pos = (x, y)
        if event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            if self.start_pos is not None and self.end_pos is not None:
                self.region = [*self.start_pos, *self.end_pos]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_folder", metavar="FILE", help="path to eval folder")
    parser.add_argument("--use_last_ckpt", action="store_true", help="else use best ckpt")
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    # parse --eval_folder, generate args.config_file
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
    models_sym = {}
    for obj_id in obj_ids:
        models_sym[obj_id] = misc.get_symmetry_transformations(models_info[obj_id], 0.01)
    sym_obj_id = cfg.DATASETS.SYM_OBJS_ID
    if sym_obj_id == "bop":
        sym_obj_id = [k for k, v in models_info.items() if 'symmetries_discrete' in v or 'symmetries_continuous' in v]
    cfg.DATASETS.SYM_OBJS_ID = sym_obj_id

    objs = load_objs(meta_info, obj_ids)
    renderer = ObjCoordRenderer(objs, [k for k in objs.keys()], cfg.DATASETS.RES_CROP)

    # load model
    model = build_model(cfg)
    model.load_state_dict(torch.load(cfg.RESUME)['state_dict'])
    model.eval().to(device).freeze()

    # load data
    data = build_BOP_test_dataset(cfg, cfg.DATASETS.TEST, debug=True)
    # data = build_BOP_test_dataset(cfg, cfg.DATASETS.TEST, debug=cfg.DEBUG)

    # initialize opencv windows
    print()
    print('With an opencv window active:')
    print("press 'a', 'd' and 'x'(random) to get a new input image,")
    print("press 'q' to quit.")
    print("press 'c' to recrop")
    print("press 'n' to input scene id and image id")
    data_i = 0
    current_data_i = -1
    while True:
        print('------------ input -------------')
        # current_data_i != data_i, we need to generate new inst from dataset, else use old inst
        if current_data_i != data_i:
            inst = data[data_i]
            current_data_i = data_i
        obj_id = inst['obj_id']
        print(f"i: {data_i}, obj_id: {obj_id}")
        rgb = np.copy(inst['rgb'])
        cv2.rectangle(rgb, (int(inst['bbox_est'][0]), int(inst['bbox_est'][1])),
                      (int(inst['bbox_est'][2]), int(inst['bbox_est'][3])), (0, 0, 255), 1)
        cv2.putText(rgb, 'bbox_est', (int(inst['bbox_est'][0]), int(inst['bbox_est'][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
        cv2.rectangle(rgb, (int(inst['AABB_crop'][0]), int(inst['AABB_crop'][1])),
                      (int(inst['AABB_crop'][2]), int(inst['AABB_crop'][3])), (255, 0, 0), 1)
        cv2.putText(rgb, 'crop', (int(inst['AABB_crop'][0]), int(inst['AABB_crop'][1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
        cv2.imshow('rgb', rgb[..., ::-1])
        rgb_crop = np.copy(inst['rgb_crop'])
        show_rgb('rgb_crop', rgb_crop)
        out_dict = model.infer(
            torch.stack([torch.tensor(inst["rgb_crop"], dtype=torch.float32)], dim=0).to(device),
            obj_idx=torch.tensor(inst["obj_idx"], dtype=torch.long).to(device),
            K=torch.stack([torch.tensor(inst["K_crop"], dtype=torch.float32)], dim=0).to(device),
            AABB=torch.stack([torch.tensor(inst["AABB_crop"], dtype=torch.long)], dim=0).to(device),
        )
        mask_crop = inst["mask_crop"]
        mask_visib_crop = inst["mask_visib_crop"]
        code_crop = inst["code_crop"]
        show_mask_code('gt visib amodal code', mask_visib_crop, mask_crop, code_crop)

        inst['visib_mask_prob'] = visib_mask_prob = out_dict["visib_mask_prob"]
        amodal_mask_prob = out_dict["amodal_mask_prob"]

        code_prob = out_dict["binary_code_prob"]
        show_mask_code('est visib amodal code', visib_mask_prob, amodal_mask_prob, code_prob)
        cv2.namedWindow('gt est mask contour', cv2.WINDOW_NORMAL)
        show_mask_contour('gt est mask contour', rgb_crop, [mask_crop, amodal_mask_prob])
        cv2.namedWindow('gt est visib mask contour', cv2.WINDOW_NORMAL)
        show_mask_contour('gt est visib mask contour', rgb_crop, [mask_visib_crop, visib_mask_prob])

        K = inst['K_crop']
        gt_R = inst['cam_R_obj']
        gt_t = inst['cam_t_obj']
        est_R = out_dict['rot'][0]
        est_t = out_dict['trans'][0]
        gt_R, gt_t, K = preprogress_Rt(gt_R, gt_t, K)
        est_R, est_t = preprogress_Rt(est_R, est_t)
        show_pose("gt pose", K, gt_R, gt_t, rgb_crop, renderer, obj_id)
        show_pose("est pose", K, est_R, est_t, rgb_crop, renderer, obj_id)

        # caculate error
        error_mssd = pose_error.mssd(est_R, est_t, gt_R, gt_t, models_3d[obj_id]["pts"], models_sym[obj_id])
        error_mspd = pose_error.mspd(est_R, est_t, gt_R, gt_t, K, models_3d[obj_id]["pts"], models_sym[obj_id])
        error_add = pose_error.add(est_R, est_t, gt_R, gt_t, models_3d[obj_id]["pts"])
        d = diameters[obj_id]
        print(f"mssd: {error_mssd}, threshold: [{d * 0.05},{d * 0.1},...,{d * 0.5}]")
        print(f"mspd: {error_mspd}, threshold: [5, 10, ..., 45, 50]")
        if obj_id in sym_obj_id:
            error_adi = pose_error.adi(est_R, est_t, gt_R, gt_t, models_3d[obj_id]["pts"])
            print(f"adi: {error_adi}, threshold:[{d * 0.02},{d * 0.05},{d * 0.1}]")
        else:
            error_add = pose_error.add(est_R, est_t, gt_R, gt_t, models_3d[obj_id]["pts"])
            print(f"add: {error_add}, threshold:[{d * 0.02},{d * 0.05},{d * 0.1}]")

        while True:
            print()
            key = cv2.waitKey()
            if key == ord('q'):
                quit()
            elif key == ord('a'):
                data_i = (data_i + 1) % len(data)
                break
            elif key == ord('d'):
                data_i = (data_i - 1) % len(data)
                break
            elif key == ord('x'):
                data_i = np.random.randint(len(data))
                break
            elif key == ord('c'):
                print('mode: crop by human')
                print('drag a crop region on windows_name=\'rgb\', press q to quit, you can drag many times')
                cs = CropSelector()
                crop = cs.draw_crop(rgb[..., ::-1], 'rgb')
                inst['bbox_est'] = crop
                print(f"crop change from {inst['AABB_crop']} to {crop}")
                crop_aux = RandomRotatedMaskCrop(cfg.DATASETS.RES_CROP, use_bbox_est=True,
                                                 crop_scale=1, offset_scale=0, rgb_interpolation=cv2.INTER_LINEAR,
                                                 crop_keys=('rgb', 'mask_visib'),
                                                 crop_keys_crop_res_divide2=('mask', 'mask_visib', 'GT'), )
                auxs = [crop_aux, crop_aux.definition_aux,
                        crop_aux.apply_aux, crop_aux.apply_aux_d2,
                        GT2CodeAux(), NormalizeAux()]
                for aux in auxs:
                    inst = aux(inst, None)
                break
            elif key == ord('v'):
                print('mode:crop and padding with zero on rgb_crop')
                Ms = np.concatenate((inst['M_crop'], [[0, 0, 1]]))
                bbox_est = inst['bbox_est']
                bbox_est = np.array(((bbox_est[0], bbox_est[1], 1),(bbox_est[2], bbox_est[3], 1)))
                bbox_est_in_crop = Ms @ bbox_est.T
                left, top, right, down = int(bbox_est_in_crop[0, 0]), int(bbox_est_in_crop[1, 0]), \
                                         int(bbox_est_in_crop[0, 1]), int(bbox_est_in_crop[1, 1])
                tmp_mask = np.zeros_like(inst['rgb_crop'])
                tmp_mask[:, top:down+1, left:right+1] = 1
                inst['rgb_crop'] *= tmp_mask
                break

            elif key == ord('g'):
                print('mode: choose region and set zero to test occlusion influence')
                print('drag a crop region on windows_name=\'rgb_crop\', press q to quit, you can drag many times')
                cs = CropSelector(is_square=False)
                left, top, right, down = cs.draw_crop(preprogress_rgb(rgb_crop)[..., ::-1], 'rgb_crop')
                tmp_mask = np.ones_like(inst['rgb_crop'])
                tmp_mask[:, top:down + 1, left:right + 1] = 0
                inst['rgb_crop'] *= tmp_mask
                break

            elif key == ord('b'):
                print('mode: use the predict visib mask to mask rgb')
                visib_mask_prob = cv2.resize(preprogress_mask(inst['visib_mask_prob']), (256, 256))
                inst['rgb_crop'] *= (visib_mask_prob != 0)
                break

            elif key == ord('n'):
                print('mode: find particular image')
                flag = False
                scene = int(input('input the scene'))
                image = int(input('input the image id'))
                for i, inst in enumerate(data):
                    if inst['scene_id'] == scene and inst['img_id'] == image:
                        flag = True
                        data_i = i
                        break
                if flag:
                    print(f"find scene {scene} image {image}")
                    print(f"current data_i is {data_i}")
                else:
                    print(f"can not find scene {scene} image {image}")
                    print(f"current data_i is {data_i}")
                break

            elif key == ord('r'):
                print('mode: rotaion anticlockwise')
                inst['rgb_crop'] = np.flip(inst['rgb_crop'], axis=-1).swapaxes(-1, -2).copy()
                break