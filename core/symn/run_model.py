""" usage: python run_model.py --config-file configs/symn/tless/symn_tless_obj04.py"""
import sys
import os
sys.path.insert(0, os.getcwd())

from mmcv import Config
from core.utils.default_args_setup import my_default_argument_parser
from lib.utils.utils import iprint
from core.symn.models import SymNet


if __name__ == "__main__":
    parser = my_default_argument_parser()
    args = parser.parse_args()
    cfg = Config.fromfile(args.config_file)
    iprint(f"Used model name: {cfg.MODEL.NAME}")
    assert cfg.MODEL.NAME == "SymNet"
    model, optimizer = SymNet.build_model_optimizer(cfg)
    device = "cpu"
    iprint("Model:\n{}".format(model))
