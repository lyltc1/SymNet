import torch
from mmcv import Config
from core.utils.default_args_setup import my_default_argument_parser
from lib.utils.utils import iprint
from core.gdrn_modeling.models import GDRN  # noqa
from torchsummary import summary

if __name__ == "__main__":
    parser = my_default_argument_parser()
    args = parser.parse_args()
    cfg = Config.fromfile(args.config_file)
    optim_cfg = cfg.SOLVER.OPTIMIZER_CFG
    iprint("optimizer_cfg:", optim_cfg)
    cfg.SOLVER.BASE_LR = optim_cfg["lr"]

    iprint(f"Used GDRN module name: {cfg.MODEL.CDPN.NAME}")
    model, optimizer = eval(cfg.MODEL.CDPN.NAME).build_model_optimizer(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary(model.to(device), (3, 256, 256))
    # iprint("Model:\n{}".format(model))
