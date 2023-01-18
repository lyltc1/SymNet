""" python core/symn/run_train.py --config-file configs/symn/tless/symn_tless_obj04_pl.py
    python core/symn/run_train.py --config-file configs/symn/tless/symn_tless_obj04_pl.py --debug True
    python core/symn/run_train.py --config-file configs/symn/tless/symn_tless_obj04_pl.py --gpus 0 1 2 3 4 5
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
import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from lib.utils.time_utils import get_time_str
from core.symn.datasets.BOPDataset_utils import build_BOP_train_dataset, batch_data_train
from core.zebrapose.models.BinaryCodeNet import DeepLabV3


def worker_init_fn(*_):
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    np.random.seed(None)


class Net(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = DeepLabV3(34, 17, True, 1)
        self.resnet = self.model.resnet
        self.aspp = self.model.aspp

    def training_step(self, batch, batch_nb):
        return 0

    def configure_optimizers(self):
        self.automatic_optimization = False
        opt_cfg = self.cfg.SOLVER.OPTIMIZER_CFG
        if opt_cfg == "" or opt_cfg is None:
            raise RuntimeError("please provide cfg.SOLVER.OPTIMIZER_CFG to build optimizer")
        if opt_cfg.type == "Ranger":
            from lib.torch_utils.solver.ranger import Ranger
            opt = Ranger([
                dict(params=self.resnet.parameters(), lr=opt_cfg.lr, ),
                dict(params=self.aspp.parameters(), lr=opt_cfg.lr, ),],
                weight_decay=opt_cfg.weight_decay,
            )
        elif opt_cfg.type == "Adam":
            opt = torch.optim.Adam([
                dict(params=self.model.parameters(), lr=opt_cfg.lr),])
        else:
            raise NotImplementedError

        sche_cfg = self.cfg.SOLVER.LR_SCHEDULER_CFG
        if sche_cfg == "" or sche_cfg is None:
            raise RuntimeError("please provide cfg.SOLVER.LR_SCHEDULER_CFG to build scheduler")
        if sche_cfg.type == "LambdaLR":
            sched = dict(
                scheduler=torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1., i / sche_cfg.warm)),
                interval='step'
            )
        else:
            raise NotImplementedError
        return [opt], [sched]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--ckpt", default=None, help="the checkpoint to resume")
    parser.add_argument("--debug", type=bool, default=False, help="use one gpu and small batch size")
    parser.add_argument('--gpus', type=int, nargs='+', default=[0])
    parser.add_argument('--n-valid', type=int, default=100)
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
        cfg.TRAIN.NUM_WORKERS = 2
        cfg.TRAIN.PRINT_FREQ = 1
        cfg.TRAIN.BATCH_SIZE = 2
    # model and optimizer
    model = Net(cfg)

    # datasets
    dataset = build_BOP_train_dataset(cfg, cfg.DATASETS.TRAIN, args.debug)
    n_valid = args.n_valid
    data_train, data_valid = torch.utils.data.random_split(
        dataset, (len(dataset) - n_valid, n_valid),
        generator=torch.Generator().manual_seed(0),
    )
    loader_args = dict(
        batch_size=cfg.TRAIN.BATCH_SIZE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        persistent_workers=True, shuffle=True,
        worker_init_fn=worker_init_fn, pin_memory=True,
        collate_fn=batch_data_train,
    )
    loader_train = torch.utils.data.DataLoader(data_train, drop_last=True, **loader_args)
    loader_valid = torch.utils.data.DataLoader(data_valid, **loader_args)

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

    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_false",
        devices=args.gpus,
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.ModelCheckpoint(dirpath=cfg.OUTPUT_DIR, save_top_k=3,
                                         save_last=True, monitor='valid/total_loss'),
            TQDMProgressBar(refresh_rate=20),
        ],
        logger=[
            TensorBoardLogger(save_dir=cfg.OUTPUT_DIR),
        ],
        val_check_interval=0.5,
    )
    trainer.fit(model, loader_train, loader_valid)


if __name__ == '__main__':
    main()
