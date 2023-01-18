import numpy as np
import cv2
from mmcv import Config
from core.utils.default_args_setup import my_default_argument_parser
from lib.utils.utils import iprint
from core.gdrn_modeling.dataset_factory import register_datasets_in_cfg
from core.gdrn_modeling.data_loader import build_gdrn_train_loader
from core.gdrn_modeling.engine_utils import batch_data
from lib.vis_utils.image import grid_show


def normalize_to_01(img):
    if img.max() != img.min():
        return (img - img.min()) / (img.max() - img.min())
    else:
        return img


def get_emb_show(bbox_emb):
    show_emb = bbox_emb.copy()
    show_emb = normalize_to_01(show_emb)
    return show_emb


if __name__ == "__main__":
    parser = my_default_argument_parser()
    args = parser.parse_args()
    cfg = Config.fromfile(args.config_file)
    if cfg.get("DEBUG", False):
        iprint("DEBUG")
        args.num_gpus = 1
        args.num_machines = 1
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.TRAIN.PRINT_FREQ = 1
        cfg.SOLVER.IMS_PER_BATCH = 2

    # register datasets
    register_datasets_in_cfg(cfg)
    # build dataloader
    data_loader = build_gdrn_train_loader(cfg, cfg.DATASETS.TRAIN)
    data_loader_iter = iter(data_loader)
    dataset_len = len(data_loader.dataset)
    data = next(data_loader_iter)
    roi_img = np.uint8(data[0]['roi_img'].detach().cpu().numpy().transpose((1, 2, 0)) * 255.)
    roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
    # roi_xyz = data[0]['roi_xyz'].detach().cpu().numpy().transpose((1, 2, 0))
    # roi_xyz = np.uint8(get_emb_show(roi_xyz) * 255.)
    roi_mask_trunc = np.uint8(data[0]['roi_mask_trunc'].detach().cpu().numpy() * 255.)
    roi_mask_visib = np.uint8(data[0]['roi_mask_visib'].detach().cpu().numpy() * 255.)
    # roi_mask_obj = np.uint8(data[0]['roi_mask_obj'].detach().cpu().numpy() * 255.)
    # roi_region = np.uint8(data[0]['roi_region'].detach().cpu().numpy() * 3)
    show_ims = [
        roi_img,
        # roi_xyz,
        roi_mask_trunc,
        roi_mask_visib,
        # roi_mask_obj,
        # roi_region,
    ]
    show_titles = [
        "roi_img",
        # "roi_xyz",
        "roi_mask_trunc",
        "roi_mask_visib",
        # "roi_mask_obj",
        # "roi_region"
    ]
    grid_show(show_ims, show_titles, row=2, col=3)
    batch = batch_data(cfg, data)
    print(1)
