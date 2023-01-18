""" usage: python run_datasets.py --config-file configs/symn/tless/symn_tless_obj04.py
    :param train: True for train dataset, False for test dataset
    :param debug: True for not using KeyFilterOutAux
    :param no_detection: used when train==False, if True, no detection result used
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
from mmcv import Config
import cv2
from core.utils.default_args_setup import my_default_argument_parser
from lib.utils.utils import iprint
from core.symn.datasets.BOPDataset_utils import build_BOP_train_dataset, build_BOP_test_dataset
from core.symn.datasets.std_auxs import denormalize

parser = my_default_argument_parser()
parser.add_argument("--train", default=True, help="True for train dataset, False for test dataset")
parser.add_argument("--debug", default=False, help="True for not using KeyFilterOutAux")
parser.add_argument("--no_detection", default=False, help="used when train==False, if True, no detection result used")
args = parser.parse_args()
cfg = Config.fromfile(args.config_file)

# debug setting
debug = args.debug
train = args.train
no_detection = False
if debug:
    iprint("debug mode: on")
else:
    iprint("debug mode: off")
if train:
    iprint("visualize train_bop_instance_dataset")
else:
    iprint("visualize test_bop_instance_dataset")
if no_detection is False:
    iprint("use detection file")
    assert cfg.DATASETS.TEST_DETECTION_PATH
else:
    iprint("no detection file")
    cfg.DATASETS.TEST_DETECTION_PATH = None
args.num_gpus = 1
args.num_machines = 1
cfg.TRAIN.NUM_WORKERS = 0
cfg.TRAIN.PRINT_FREQ = 1
cfg.TRAIN.IMS_PER_BATCH = 2

# ---- main function build datasets
if train:
    iprint(f"build_BOP_train_dataset {cfg.DATASETS.TRAIN}, debug mode: {debug}")
    data = build_BOP_train_dataset(cfg, cfg.DATASETS.TRAIN, debug)
else:
    iprint(f"build_BOP_test_dataset {cfg.DATASETS.TEST}, debug mode: {debug}")
    data = build_BOP_test_dataset(cfg, cfg.DATASETS.TEST, debug)

obj_ids = cfg.DATASETS.OBJ_IDS
res_crop = cfg.DATASETS.RES_CROP
window_names = ['rgb_crop', 'mask_crop', 'mask_visib_crop', 'obj_coord']
for j, name in enumerate(window_names):
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, res_crop, res_crop)
    cv2.moveWindow(name, 1 + 250 * j, 1 + 250 * 0)
for j in range(16):
    row, col = j // 6, j % 6
    cv2.namedWindow(str(j), cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(str(j), res_crop, res_crop)
    cv2.moveWindow(str(j), 1 + 250 * col, 280 + 250 * row)

print()
print('With an opencv window active:')
print("press 'a'(add), 'd'(decrease) and 'x'(random) to get a new input image,")
print("press 'q' to quit.")
data_i = 0

while True:
    print()
    print('------------ new input -------------')
    inst = data[data_i]
    obj_idx = inst['obj_idx']
    print(f'i: {data_i}, obj_id: {obj_ids[obj_idx]}')
    if 'iou' in inst:
        print('iou:', inst['iou'], 'det_score', inst['det_score'])
    rgb_crop = inst['rgb_crop']
    rgb_crop = denormalize(rgb_crop)
    mask_crop = inst['mask_crop']
    mask_visib_crop = inst['mask_visib_crop']
    code_crop = inst['code_crop']
    if not code_crop.any():
        data_i = np.random.randint(len(data))
        continue

    cv2.imshow('rgb_crop', rgb_crop[..., ::-1])
    cv2.imshow('mask_crop', mask_crop)
    cv2.imshow('mask_visib_crop', mask_visib_crop)

    if 'obj_coord' in inst.keys():
        obj_coord = inst['obj_coord']
        render_mask = obj_coord[..., 3] == 1.
        obj_coord[..., :3][render_mask] = obj_coord[..., :3][render_mask] * 0.5 + 0.5
        cv2.imshow('obj_coord', obj_coord[..., 2::-1])

    for j in range(16):
        cv2.imshow(str(j), code_crop[j, ...])
    while True:
        print()
        key = cv2.waitKey()
        if key == ord('q'):
            cv2.destroyAllWindows()
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
