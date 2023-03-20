import importlib
from os.path import join

import cv2
import torch
from torch import set_num_threads
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.sampler import BatchSampler

from core.symn.MetaInfo import MetaInfo
from core.utils.dataset_utils import trivial_batch_collator
from core.utils.my_distributed_sampler import TrainingSampler, InferenceSampler
from lib.utils.utils import iprint
from .BOPDataset import BopTrainDataset, BopTestDataset
from .GDRN_aux import AugRgbAux, ReplaceBgAux
from .std_auxs import RgbLoader, MaskLoader, RandomRotatedMaskCrop, NormalizeAux, KeyFilterOutAux
from .symn_aux import GTLoader, GT2CodeAux, OccludeAux,\
                      PosePresentationAux, LoadSymInfoExtentsAux, LoadPointsAux


class EmptyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 0

    def __getitem__(self, item):
        return None


def balanced_dataset_concat(a, b):
    # makes an approximately 50/50 concat
    # by adding copies of the smallest dataset
    if len(a) < len(b):
        a, b = b, a
    assert len(a) >= len(b)
    data = a
    for i in range(round(len(a) / len(b))):
        data += b
    return data


def get_aux(cfg, gt, aug_bg=False, aug_rgb=False, aug_occ=False, detection=False, debug=False):
    auxs = [RgbLoader(), ]
    if gt is True:
        if detection:
            crop_aux = RandomRotatedMaskCrop(cfg.DATASETS.RES_CROP, use_bbox_est=True, offset_scale=0.,
                                             crop_keys=('rgb', 'mask_visib'),
                                             crop_scale=[1.3, 1.3],
                                             crop_keys_crop_res_divide2=('mask', 'mask_visib', 'GT'),
                                             rgb_interpolation=cv2.INTER_LINEAR)
        else:
            crop_aux = RandomRotatedMaskCrop(cfg.DATASETS.RES_CROP, mask_key='mask_visib',
                                             crop_scale=[1.2, 1.5],
                                             crop_keys=('rgb', 'mask_visib'),
                                             crop_keys_crop_res_divide2=('mask', 'mask_visib', 'GT'),
                                             rgb_interpolation=cv2.INTER_LINEAR)
        # train_aux
        auxs.extend([MaskLoader(), GTLoader(), ])
        if aug_occ:
            auxs.append(OccludeAux(prob=cfg.DATASETS.OCCLUDE_AUG_PROB))
        auxs.extend([crop_aux.definition_aux, crop_aux.apply_aux, ])
        if aug_bg is True:
            auxs.extend([ReplaceBgAux(cfg.DATASETS.BG_AUG_PROB, cfg.DATASETS.BG_AUG_TYPE), ])
        if aug_rgb is True:
            auxs.extend([AugRgbAux(cfg.DATASETS.COLOR_AUG_PROB), ])

        auxs.extend([crop_aux.apply_aux_d2,
                     GT2CodeAux(),
                     PosePresentationAux(cfg.DATASETS.RES_CROP,
                                         R_type=cfg.MODEL.PNP_NET.R_type, t_type=cfg.MODEL.PNP_NET.t_type),
                     LoadPointsAux(num_points=cfg.MODEL.PNP_NET.PM_NUM_POINTS),
                     LoadSymInfoExtentsAux(),
                     ])
    else:
        auxs.extend([RandomRotatedMaskCrop(cfg.DATASETS.RES_CROP, use_bbox_est=True,
                                           crop_keys=('rgb',),
                                           crop_keys_crop_res_divide2=tuple(),
                                           rgb_interpolation=cv2.INTER_LINEAR),
                     ])
    auxs.extend([NormalizeAux(), ])
    if debug is False:
        auxs.extend([KeyFilterOutAux({'rgb', 'mask', 'mask_visib',
                                      'GT', 'GT_crop', 'K', 'bbox_obj', 'bbox_visib'})])
    return auxs


def build_BOP_train_dataset(cfg, dataset_type, debug=False):
    """ dataset_type is cfg.DATASETS.TRAIN or cfg.DATASETS.TRAIN2 """
    meta_info = MetaInfo(cfg.DATASETS.NAME)
    obj_ids = cfg.DATASETS.OBJ_IDS
    if isinstance(dataset_type, str):
        dataset_type = [dataset_type]
    assert len(dataset_type), dataset_type
    dataset = []
    for name in dataset_type:
        if name == "train_real" or name == "train_primesense":
            # we use both ReplaceBgAux and AugRgbAux in real dataset
            auxs_real = get_aux(cfg, gt=True, aug_occ=True, aug_bg=True, aug_rgb=True, debug=debug)
            real_dataset = BopTrainDataset(meta_info, name, obj_ids=obj_ids, auxs=auxs_real)
            dataset.append(real_dataset)
        elif name == "train_pbr":
            # we do not replace background in BOP pbr dataset
            auxs_pbr = get_aux(cfg, gt=True, aug_occ=True, aug_bg=False, aug_rgb=True, debug=debug)
            pbr_dataset = BopTrainDataset(meta_info, name, obj_ids=obj_ids, auxs=auxs_pbr)
            dataset.append(pbr_dataset)
        elif name == "test" or name == "test_primesense":
            # use only for debug
            auxs = get_aux(cfg, gt=True, aug_occ=False, aug_bg=False, aug_rgb=False, debug=debug)
            test_dataset = BopTrainDataset(meta_info, name, obj_ids=obj_ids, auxs=auxs)
            dataset.append(test_dataset)
        else:
            raise NotImplementedError
    if len(dataset) >= 3:
        dataset = ConcatDataset(dataset)
    elif len(dataset) == 2:
        dataset = balanced_dataset_concat(dataset[0], dataset[1])
    elif len(dataset) == 1:
        dataset = dataset[0]
    return dataset


def build_BOP_test_dataset(cfg, dataset_type, debug=False):
    """ build dataset in cfg.DATASETS.TEST """
    meta_info = MetaInfo(cfg.DATASETS.NAME)
    obj_ids = cfg.DATASETS.OBJ_IDS
    if isinstance(dataset_type, str):
        dataset_type = [dataset_type]
    assert len(dataset_type), dataset_type
    if len(dataset_type):
        dataset = []
    else:
        dataset = None
    for folder_name in dataset_type:
        if folder_name == "test" or "test_primesense":
            use_detection = True if cfg.DATASETS.TEST_DETECTION_PATH else False
            auxs_test = get_aux(cfg, gt=True, detection=use_detection, aug_bg=False, aug_rgb=False, debug=debug)
            # ---- get detections ----
            detections = None
            if use_detection:
                iprint(f"use detections from {cfg.DATASETS.TEST_DETECTION_PATH}")
                # get detections use imported function
                get_det_fun_name = "get_detection_results_" + cfg.DATASETS.TEST_DETECTION_TYPE
                get_det_module = importlib.import_module(".get_detection_results", "datasets")
                get_det_fun = getattr(get_det_module, get_det_fun_name)
                detections = get_det_fun(join(meta_info.detections_folder, cfg.DATASETS.TEST_DETECTION_PATH))

            # ---- get keyframe ----
            keyframe = cfg.DATASETS.TEST_KEYFRAME
            # ---- build dataset ----
            test_dataset = BopTestDataset(meta_info, folder_name, obj_ids=obj_ids,
                                          auxs=auxs_test, detections=detections, keyframe=keyframe,
                                          )
            dataset.append(test_dataset)
        else:
            raise NotImplementedError
    if len(dataset) > 1:
        dataset = ConcatDataset(dataset)
    elif len(dataset) == 1:
        dataset = dataset[0]
    return dataset


def build_train_dataloader(cfg, dataset_type, debug=False):
    # ---- Build dataset ----
    dataset = build_BOP_train_dataset(cfg, dataset_type, debug)
    # ---- Build a batched dataloader for training ----
    batch_size = cfg.TRAIN.BATCH_SIZE
    num_workers = cfg.TRAIN.NUM_WORKERS
    if num_workers > 0:
        set_num_threads(num_workers)
    sampler = TrainingSampler(len(dataset))
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=True)
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler, pin_memory=True,
                             num_workers=num_workers, collate_fn=trivial_batch_collator)
    return data_loader


def build_test_dataloader(cfg, dataset_type, debug=False):
    # ---- Build dataset ----
    dataset = build_BOP_test_dataset(cfg, dataset_type, debug=debug)
    # ---- Build a batched dataloader for testing ----
    num_workers = cfg.TRAIN.NUM_WORKERS
    if num_workers > 0:
        set_num_threads(num_workers)
    sampler = InferenceSampler(len(dataset))
    batch_sampler = BatchSampler(sampler, 1, drop_last=False)
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler,
                             num_workers=num_workers, collate_fn=trivial_batch_collator)
    return data_loader


def batch_data_train(data):
    batch = {}
    for key in ["obj_id", "obj_idx"]:
        batch[key] = torch.tensor([d[key] for d in data], dtype=torch.long)
    for key in ["rgb_crop", "mask_visib_crop", "mask_crop", "code_crop", "K_crop", "extent",
                "cam_R_obj", "cam_t_obj", "points"]:
        batch[key] = torch.stack([torch.tensor(d[key], dtype=torch.float32) for d in data], dim=0)
    for key in ["AABB_crop"]:
        batch[key] = torch.stack([torch.tensor(d[key], dtype=torch.long) for d in data], dim=0)
    for key in ['scene_id', 'img_id', 'pose_idx', 'sym_info']:
        batch[key] = [d[key] for d in data]

    for key in ["allo_rot6d", "allo_rot", "SITE"]:
        if key in data[0]:
            batch[key] = torch.stack([torch.tensor(d[key], dtype=torch.float32) for d in data], dim=0)
        else:
            batch[key] = None
    return batch


def batch_data_test(data):
    """ data is a list of dict, for each dict contain keys like
        scene_id, img_id, K, obj_id, ...
    """
    batch = {}
    for key in data[0].keys():
        if key in ["obj_idx"]:
            batch[key] = torch.tensor([d[key] for d in data], dtype=torch.long)
        elif key in ["AABB_crop"]:
            batch[key] = torch.stack([torch.tensor(d[key], dtype=torch.long) for d in data], dim=0)
        elif key in ["rgb_crop", "K_crop"]:
            batch[key] = torch.stack([torch.tensor(d[key], dtype=torch.float32) for d in data], dim=0)
        elif key in ['scene_id', 'img_id', 'pose_idx', 'obj_id',
                     "mask_visib_crop", "mask_crop", "code_crop", "extent",
                     "cam_R_obj", "cam_t_obj", "allo_rot6d", "allo_rot", "SITE",
                     "points", "sym_info", 'det_score', 'det_time', 'M_crop',
                     "bbox_est"]:
            batch[key] = [d[key] for d in data]
    return batch
