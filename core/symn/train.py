""" python core/symn/train.py --config-file configs/symn/tless/symn_tless_obj04.py --eval-only --debug
    python core/symn/train.py --config-file configs/symn/tless/symn_tless_obj04.py --resume --debug
    CUDA_VISIBLE_DEVECES=0 python core/symn/train.py --config-file configs/symn/tless/symn_tless_obj04.py --num-gpus 1
    CUDA_VISIBLE_DEVECES=0,1,2,3,4,5 python core/symn/train.py --config-file output/SymNet_tless_obj4_20221125_210524/symn_tless_obj04_5050.py --num-gpus 6
    CUDA_VISIBLE_DEVECES=0,1,2,3,4,5 python core/symn/train.py --config-file configs/symn/tless/symn_tless_obj04_no_d2d.py --num-gpus 6
    --debug batch-size=2 num_gpu=1
    --eval-only
    --resume continue traing from cfg.MODEL.WEIGHTS
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import resource
import logging
from loguru import logger as loguru_logger
import os.path as osp
from setproctitle import setproctitle
import torch
from mmcv import Config
import cv2
from pytorch_lightning import seed_everything
from core.utils.default_args_setup import my_default_argument_parser, my_default_setup
from core.utils.my_setup import setup_for_distributed
from core.utils.my_checkpoint import MyCheckpointer
from lib.utils.utils import iprint
from lib.utils.time_utils import get_time_str
from core.symn.engine.engine import SYMN_Lite
from core.symn.models import SymNet

logger = logging.getLogger("train")
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../../"))  # add project directory to sys.path


def setup(args):
    """ Create configs and perform basic setups, do not need to change
        1. set output_dir to cfg.OUTPUT_DIR and vis_dir to cfg.VIS_DIR
        2. set process title
        3. set optimizer args to cfg.SOLVER
        4. set exp_id to cfg.EXP_ID
        5. set args.resume to cfg.RESUME
    """
    cfg = Config.fromfile(args.config_file)
    if args.opts is not None:
        cfg.merge_from_dict(args.opts)
    ############## pre-process some cfg options ######################
    # ---- output_dir is output/[model_name]_[dataset_name]_[obj_id]_time
    if cfg.OUTPUT_DIR.lower() == "auto":
        out_str = cfg.MODEL.NAME + "_" + cfg.DATASETS.NAME
        for obj_id in cfg.DATASETS.OBJ_IDS:
            out_str += f"_obj{obj_id}"
        out_str += '_' + get_time_str()
        cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_ROOT, out_str)
        iprint(f"OUTPUT_DIR was automatically set to: {cfg.OUTPUT_DIR}")
    cfg.VIS_DIR = osp.join(cfg.OUTPUT_DIR, "visualize")
    if cfg.get("EXP_NAME", "") == "":
        setproctitle("{}.{}".format(osp.splitext(osp.basename(args.config_file))[0], get_time_str()))
    else:
        setproctitle("{}.{}".format(cfg.EXP_NAME, get_time_str()))

    if cfg.SOLVER.AMP.ENABLED:
        if torch.cuda.get_device_capability() <= (6, 1):
            iprint("Disable AMP for older GPUs")
            cfg.SOLVER.AMP.ENABLED = False

    # NOTE: pop some unwanted configs in detectron2
    cfg.SOLVER.pop("STEPS", None)
    cfg.SOLVER.pop("MAX_ITER", None)
    # NOTE: get optimizer from string cfg dict
    if cfg.SOLVER.OPTIMIZER_CFG != "":
        if isinstance(cfg.SOLVER.OPTIMIZER_CFG, str):
            optim_cfg = eval(cfg.SOLVER.OPTIMIZER_CFG)
            cfg.SOLVER.OPTIMIZER_CFG = optim_cfg
        else:
            optim_cfg = cfg.SOLVER.OPTIMIZER_CFG
        iprint("optimizer_cfg:", optim_cfg)
        cfg.SOLVER.OPTIMIZER_NAME = optim_cfg["type"]
        cfg.SOLVER.BASE_LR = optim_cfg["lr"]
        cfg.SOLVER.MOMENTUM = optim_cfg.get("momentum", 0.9)
        cfg.SOLVER.WEIGHT_DECAY = optim_cfg.get("weight_decay", 1e-4)
    if cfg.get("DEBUG", False) or args.debug is True:
        iprint("DEBUG")
        args.num_gpus = 1
        args.num_machines = 1
        cfg.TRAIN.NUM_WORKERS = 0
        cfg.TRAIN.PRINT_FREQ = 1
        cfg.TRAIN.BATCH_SIZE = 2
    exp_id = "{}".format(osp.splitext(osp.basename(args.config_file))[0])
    if args.eval_only:
        if cfg.TEST.USE_PNP:
            # NOTE: need to keep _test at last
            exp_id += "{}_test".format(cfg.TEST.PNP_TYPE.upper())
        else:
            exp_id += "_test"
    cfg.EXP_ID = exp_id
    cfg.RESUME = args.resume
    ####################################
    return cfg


class Lite(SYMN_Lite):
    def set_my_env(self, args, cfg):
        # ---- set log and log env ----
        my_default_setup(cfg, args)  # will set os.environ["PYTHONHASHSEED"]
        # ---- random seed init ----
        seed_everything(int(os.environ["PYTHONHASHSEED"]), workers=True)
        setup_for_distributed(is_master=self.is_global_zero)

    def run(self, args, cfg):
        # ---- set log and random seed init -----
        self.set_my_env(args, cfg)
        # ---- build model and optimizer ----
        logger.info(f"Used module name: {cfg.MODEL.NAME}")
        assert cfg.MODEL.NAME == "SymNet"
        model, optimizer = SymNet.build_model_optimizer(cfg)
        logger.info("Model:\n{}".format(model))
        # ---- distributed setup for model and opti by lite ----
        model, optimizer = self.setup(model, optimizer)
        # ---- print the model info ----
        if True:
            params = sum(p.numel() for p in model.parameters()) / 1e6
            logger.info("{}M params".format(params))

        if args.eval_only:
            MyCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            return self.do_test(cfg, model)

        self.do_train(cfg, model, optimizer, resume=args.resume)
        torch.multiprocessing.set_sharing_strategy("file_system")
        return self.do_test(cfg, model)


@loguru_logger.catch
def main(args):
    cfg = setup(args)

    logger.info(f"start to train with {args.num_machines} nodes and {args.num_gpus} GPUs")
    if args.num_gpus > 1 and args.strategy is None:
        args.strategy = "ddp"
    Lite(
        accelerator="gpu",
        strategy=args.strategy,
        devices=args.num_gpus,
        num_nodes=args.num_machines,
        precision=16 if cfg.SOLVER.AMP.ENABLED else 32,
    ).run(args, cfg)


if __name__ == "__main__":
    hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[1]  # max files number can be opened
    soft_limit = min(500000, hard_limit)
    iprint("soft limit: ", soft_limit, "hard limit: ", hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

    parser = my_default_argument_parser()
    parser.add_argument(
        "--strategy",
        default=None,
        type=str,
        help="the strategy for parallel training: dp | ddp | ddp_spawn | deepspeed | ddp_sharded",
    )
    parser.add_argument("--debug", action="store_true", default=False, help="use one gpu and small batch size")
    args = parser.parse_args()
    iprint("Command Line Args: {}".format(args))

    if args.eval_only:
        torch.multiprocessing.set_sharing_strategy("file_system")

    main(args)
