""" usage:
    python core/symn/run_train.py --config-file configs/symn/tless/symn_tless_config.py --obj_id 4
    python core/symn/run_train.py --config-file configs/symn/tless/symn_tless_config.py --obj_id 4 --debug True
"""

import os
import sys

sys.path.insert(0, os.getcwd())
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, "../../"))  # add project directory to sys.path

import argparse
import numpy as np
import os.path as osp
import torch
from mmcv import Config
import cv2
from bop_toolkit_lib import inout
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from lib.utils.time_utils import get_time_str
from core.symn.datasets.BOPDataset_utils import build_BOP_train_dataset, batch_data_train
from core.symn.models.SymNetLightning import build_model
from core.symn.MetaInfo import MetaInfo


def worker_init_fn(*_):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    np.random.seed(None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--ckpt", default=None, help="the checkpoint to resume")
    parser.add_argument("--debug", type=bool, default=False, help="use one gpu and small batch size")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument("--obj_id", type=int, nargs='+', default=[], required=True, help="the obj id to train")
    parser.add_argument('--small_dataset', action="store_true", help="quick debug with small datasets")
    args = parser.parse_args()
    # parse --config-file
    cfg = Config.fromfile(args.config_file)
    # parse --ckpt
    if args.ckpt is not None:
        cfg.RESUME = args.ckpt
    # parse --debug
    cfg.DEBUG = args.debug
    if args.debug is True:
        args.gpus = [0]
        cfg.TRAIN.NUM_WORKERS = 4
        cfg.TRAIN.PRINT_FREQ = 1
        cfg.TRAIN.BATCH_SIZE = 2
    # parse --obj_id
    if isinstance(args.obj_id, list):
        cfg.DATASETS.OBJ_IDS = args.obj_id
    else:
        cfg.DATASETS.OBJ_IDS = [args.obj_id, ]
    cfg.DATASETS.NUM_CLASSES = len(cfg.DATASETS.OBJ_IDS)

    # parse allo_sym
    if cfg.MODEL.PNP_NET.R_type == 'R_allo' and cfg.MODEL.PNP_NET.R_ALLO_SYM_LW > 0.0:
        obj_ids = cfg.DATASETS.OBJ_IDS
        assert len(obj_ids) == 1, "sym_axis only support for per object training now"
        obj_id = obj_ids[0]
        meta_info = MetaInfo(cfg.DATASETS.NAME)
        models_info = inout.load_json(meta_info.models_info_path, keys_to_int=True)
        if 'symmetries_continuous' in models_info[obj_id]:
            axis = models_info[obj_id]['symmetries_continuous'][0]['axis']
            sym_axis = axis.index(1)
        else:
            sym_axis = -1
        cfg.DATASETS.sym_axis = sym_axis  # dict of obj_idx : sym_axis, -1 for None, 0, 1, 2 for continuous

    # model and optimizer
    assert cfg.MODEL.NAME == "SymNet"
    model = build_model(cfg)

    # datasets
    data_train = build_BOP_train_dataset(cfg, cfg.DATASETS.TRAIN, args.debug)
    data_valid = build_BOP_train_dataset(cfg, cfg.DATASETS.TEST, args.debug)
    if args.small_dataset:
        data_train, _ = torch.utils.data.random_split(
            data_train, (128, len(data_train) - 128),
            generator=torch.Generator().manual_seed(0),
        )
        data_valid, _ = torch.utils.data.random_split(
            data_valid, (128, len(data_valid) - 128),
            generator=torch.Generator().manual_seed(0),
        )
    loader_args = dict(
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        persistent_workers=True,
        worker_init_fn=worker_init_fn, pin_memory=True,
        collate_fn=batch_data_train,
    )
    loader_train = torch.utils.data.DataLoader(data_train, drop_last=True, shuffle=True, **loader_args)
    loader_valid = torch.utils.data.DataLoader(data_valid, shuffle=False, **loader_args)

    # set output_dir
    if cfg.OUTPUT_DIR.lower() == "auto":
        out_str = cfg.MODEL.NAME + "_" + cfg.DATASETS.NAME
        for obj_id in cfg.DATASETS.OBJ_IDS:
            out_str += f"_obj{obj_id}"
        out_str += '_' + get_time_str()
        cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_ROOT, out_str)
        cfg.VIS_DIR = osp.join(cfg.OUTPUT_DIR, "visualize")
        if not os.path.exists(cfg.OUTPUT_DIR):
            os.mkdir(cfg.OUTPUT_DIR)
        if not os.path.exists(cfg.VIS_DIR):
            os.mkdir(cfg.VIS_DIR)

    # log config
    path = osp.join(cfg.OUTPUT_DIR, osp.basename(args.config_file))
    cfg.dump(path)

    if cfg.MODEL.PNP_NET.get("NOT_FREEZE_EPOCH", 0) > 0:
        strategy = None
    else:
        strategy = "ddp_find_unused_parameters_false"

    trainer = pl.Trainer(
        accelerator="gpu",
        strategy=strategy,
        max_epochs=300,
        devices=args.gpus,
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            # if want to save top k, set save_top_k = 1, and every_n_epochs = 1, save_last=True;
            # if want to save every n epoch, set save_top_k=-1, and every_n_epochs.
            pl.callbacks.ModelCheckpoint(dirpath=cfg.OUTPUT_DIR, save_top_k=1,
                                         save_last=True, monitor='valid/eval_loss', 
                                         every_n_epochs=1),
            TQDMProgressBar(refresh_rate=20),
        ],
        logger=[
            TensorBoardLogger(save_dir=cfg.OUTPUT_DIR),
        ],
        val_check_interval=1.0,
    )
    trainer.fit(model, loader_train, loader_valid, ckpt_path=cfg.RESUME)


if __name__ == '__main__':
    main()
