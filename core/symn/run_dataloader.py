""" usage: python run_dataloader.py --config-file /home/lyltc/git/GDR-Net/configs/symn/ycbv/symn_net_config.py

"""
import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
from mmcv import Config
from lib.utils.utils import iprint
from core.utils.default_args_setup import my_default_argument_parser
from core.symn.datasets.BOPDataset_utils import build_train_dataloader, build_test_dataloader

parser = my_default_argument_parser()
parser.add_argument("--debug", default=False, help="True for not using KeyFilterOutAux")
parser.add_argument("--no_detection", default=True, help="used when train==False, if True, no detection result used")
args = parser.parse_args()
cfg = Config.fromfile(args.config_file)

# debug setting
debug = args.debug
no_detection = False
if debug:
    iprint("debug mode: on")
else:
    iprint("debug mode: off")
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
cfg.TRAIN.IMS_PER_BATCH = 4

# ---- train_dataloader
iprint(f"build_BOP_train_dataset {cfg.DATASETS.TRAIN}, debug mode: {debug}")
train_dataloader = build_train_dataloader(cfg, cfg.DATASETS.TRAIN, debug)

train_dataloader_iter = iter(train_dataloader)
iprint(f"length of train dataset {len(train_dataloader.dataset)}")
train_inst = next(train_dataloader_iter)
print("----------------print all the key in train_inst----------------")
for key, value in train_inst[0].items():
    if isinstance(value, np.ndarray):
        print(f"{key}:{type(value)}-{value.shape}")
    else:
        print(f"{key}:{type(value)}")

iprint(f"build_BOP_test_dataset {cfg.DATASETS.TEST}, debug mode: {debug}")
test_dataloader = build_test_dataloader(cfg, cfg.DATASETS.TEST, debug)
test_dataloader_iter = iter(test_dataloader)
iprint(f"length of test dataset {len(test_dataloader.dataset)}")
test_inst = next(test_dataloader_iter)
print("----------------print all the key in test_inst----------------")
for key, value in test_inst[0].items():
    if isinstance(value, np.ndarray):
        print(f"{key}:{type(value)}-{value.shape}")
    else:
        print(f"{key}:{type(value)}")
