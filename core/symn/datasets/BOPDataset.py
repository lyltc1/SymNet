""" bop train dataset
"""

from os.path import join, exists
from typing import Sequence
import glob
import warnings
from lib.utils.utils import iprint

import numpy as np
from tqdm import tqdm
import torch.utils.data
from bop_toolkit_lib.inout import load_json
from core.symn.MetaInfo import MetaInfo


def get_target_list(target_path):
    # get the test list for the bop test json
    targets = load_json(target_path)
    target_list = []
    for i in range(len(targets)):
        tgt = targets[i]
        im_id = tgt["im_id"]
        inst_count = tgt["inst_count"]
        obj_id = tgt["obj_id"]
        scene_id = tgt["scene_id"]

        target_list.append([scene_id, im_id, obj_id, inst_count])
    return target_list


def compute_iou(rec_1, rec_2):
    """
    rec_1:left up(rec_1[0],rec_1[1])    right down：(rec_1[2],rec_1[3])
    rec_2:left up(rec_2[0],rec_2[1])    right down：(rec_2[2],rec_2[3])

    （rec_1）
    1--------1
    1   1----1------1
    1---1----1      1
        1           1
        1-----------1 （rec_2）
    """
    s_rec1 = (rec_1[2] - rec_1[0]) * (rec_1[3] - rec_1[1])  # area 1
    s_rec2 = (rec_2[2] - rec_2[0]) * (rec_2[3] - rec_2[1])  # area 2
    sum_s = s_rec1 + s_rec2
    left = max(rec_1[0], rec_2[0])
    right = min(rec_1[2], rec_2[2])
    bottom = max(rec_1[1], rec_2[1])
    top = min(rec_1[3], rec_2[3])
    if left >= right or top <= bottom:
        return 0
    else:
        inter = (right - left) * (top - bottom)
        iou = (inter / (sum_s - inter)) * 1.0
        return iou


class BopTrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        meta_info: MetaInfo,
        folder_name,
        obj_ids: Sequence[int],
        scene_ids=None,
        code_type="SymCode",
        min_visib_fract=0.1,
        min_px_count_visib=1024,
        auxs: Sequence["BopDatasetAux"] = tuple(),
        show_progressbar=True,
    ):
        self.meta_info = meta_info
        bop_dataset_folder = meta_info.bop_dataset_folder
        self.data_folder = join(bop_dataset_folder, folder_name)
        if code_type == "SymCode":
            self.GT_folder = join(meta_info.symnet_code_folder, folder_name + "_GT")
        elif code_type == "ZebraCode":
            self.GT_folder = join(meta_info.zebrapose_code_folder, folder_name+ "_GT")
        self.img_folder_name = "rgb"
        self.depth_folder_name = "depth"
        self.img_ext = "png"
        self.depth_ext = "png"
        if folder_name == "train_pbr":
            self.img_ext = "jpg"
        elif meta_info.name == "itodd":
            self.img_folder_name = "gray"
            self.img_ext = "tif"
            self.depth_ext = "tif"
        elif meta_info.name == "kill" or meta_info.name == "tacsat":
            self.img_ext = "jpg"
        self.auxs = auxs

        obj_idxs = {obj_id: idx for idx, obj_id in enumerate(obj_ids)}
        self.instances = []
        if scene_ids is None:
            scene_ids = sorted(
                [int(p.split("/")[-1]) for p in glob.glob(self.data_folder + "/*")]
            )
        for scene_id in (
            tqdm(scene_ids, "loading crop info") if show_progressbar else scene_ids
        ):
            scene_folder = join(self.data_folder, f"{scene_id:06d}")
            scene_gt = load_json(join(scene_folder, "scene_gt.json"))
            scene_gt_info = load_json(join(scene_folder, "scene_gt_info.json"))
            scene_camera = load_json(join(scene_folder, "scene_camera.json"))

            for img_id, poses in scene_gt.items():
                img_info = scene_gt_info[img_id]
                K = np.array(scene_camera[img_id]["cam_K"]).reshape((3, 3)).copy()
                if folder_name == "train_pbr":
                    warnings.warn(
                        "Altering camera matrix, since PBR camera matrix doesnt seem to be correct"
                    )
                    K[:2, 2] -= 0.5
                for pose_idx, pose in enumerate(poses):
                    obj_id = pose["obj_id"]
                    if obj_ids is not None and obj_id not in obj_ids:
                        continue
                    pose_info = img_info[pose_idx]
                    if pose_info["visib_fract"] < min_visib_fract:
                        continue
                    if pose_info["px_count_visib"] < min_px_count_visib:
                        continue
                    x1, y1, w, h = pose_info["bbox_visib"]
                    x2, y2 = x1 + w, y1 + h
                    bbox_visib = [x1, x2, y1, y2]
                    x1, y1, w, h = pose_info["bbox_obj"]
                    x2, y2 = x1 + w, y1 + h
                    bbox_obj = [x1, x2, y1, y2]

                    cam_R_obj = np.array(pose["cam_R_m2c"]).reshape(3, 3)
                    cam_t_obj = np.array(pose["cam_t_m2c"]).reshape(3, 1)

                    self.instances.append(
                        dict(
                            scene_id=scene_id,
                            img_id=int(img_id),
                            K=K,
                            obj_id=obj_id,
                            pose_idx=pose_idx,
                            bbox_visib=bbox_visib,
                            bbox_obj=bbox_obj,
                            cam_R_obj=cam_R_obj,
                            cam_t_obj=cam_t_obj,
                            obj_idx=obj_idxs[obj_id],
                        )
                    )

        for aux in self.auxs:
            aux.init(self)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        instance = self.instances[i].copy()
        for aux in self.auxs:
            instance = aux(instance, self)
        return instance

    def get_occlude_inst(self, i):
        # used in OccludeAux
        instance = self.instances[i].copy()
        return instance


class BopTestDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        meta_info: MetaInfo,
        folder_name,
        obj_ids: Sequence[int],
        code_type="SymCode",
        auxs: Sequence["BopDatasetAux"] = tuple(),
        detections=None,
        keyframe=None,
        min_det_score=0.01,
    ):
        self.meta_info = meta_info
        bop_dataset_folder = meta_info.bop_dataset_folder
        self.data_folder = join(bop_dataset_folder, folder_name)
        if code_type == "SymCode":
            self.GT_folder = join(meta_info.symnet_code_folder, folder_name + "_GT")
        elif code_type == "ZebraCode":
            self.GT_folder = join(meta_info.zebrapose_code_folder, folder_name+ "_GT")
        self.img_folder_name = "rgb"
        self.depth_folder_name = "depth"
        self.img_ext = "png"
        self.depth_ext = "png"
        if meta_info.name == "itodd":
            self.img_folder_name = "gray"
            self.img_ext = "tif"
            self.depth_ext = "tif"
        elif meta_info.name == "kill" or meta_info.name == "tacsat":
            self.img_ext = "jpg"

        self.auxs = auxs
        obj_idxs = {obj_id: idx for idx, obj_id in enumerate(obj_ids)}

        self.instances = []

        if keyframe is not None and detections is not None:
            # for each item in keyframe, find corresponding detection and gt (if avaliable)
            targets_list = get_target_list(join(bop_dataset_folder, keyframe))
            current_scene_id = -1
            has_gt = False
            for scene_id, im_id, obj_id, _ in targets_list:
                if obj_id not in obj_ids:
                    continue
                # ---- build found_detections ----
                found_detections = []
                detection_list_scene_img = detections[str(scene_id) + "/" + str(im_id)]
                for detection in detection_list_scene_img:
                    if (
                        detection["obj_id"] == obj_id
                        and detection["det_score"] > min_det_score
                    ):
                        found_detections.append(detection)
                if len(found_detections) == 0:
                    iprint(
                        f"not find detection for scene_id {scene_id} img_id {im_id} obj_id {obj_id}"
                    )
                    continue
                # ---- get ground truth if available ----
                if current_scene_id != scene_id:
                    current_scene_id = scene_id
                    scene_folder = join(self.data_folder, f"{scene_id:06d}")
                    scene_camera = load_json(
                        join(scene_folder, "scene_camera.json"), keys_to_int=True
                    )
                    if exists(join(scene_folder, "scene_gt_info.json")):
                        has_gt = True
                        scene_gt_info = load_json(
                            join(scene_folder, "scene_gt_info.json"), keys_to_int=True
                        )
                        scene_gt = load_json(
                            join(scene_folder, "scene_gt.json"), keys_to_int=True
                        )
                    else:
                        has_gt = False
                K = np.array(scene_camera[im_id]["cam_K"]).reshape((3, 3)).copy()
                if not has_gt:
                    for found_detection in found_detections:
                        det_score = found_detection["det_score"]
                        det_time = found_detection["det_time"]
                        bbox_est = found_detection["bbox_est"]
                        self.instances.append(
                            dict(
                                scene_id=scene_id,
                                img_id=im_id,
                                K=K,
                                obj_id=obj_id,
                                bbox_est=bbox_est,
                                obj_idx=obj_idxs[obj_id],
                                det_score=det_score,
                                det_time=det_time,
                            )
                        )
                else:  # has_gt, # for every found_detection, find the corresponding gt
                    scene_gt_this_img = scene_gt[im_id]
                    scene_gt_info_this_img = scene_gt_info[im_id]
                    for found_detection in found_detections:
                        iou_between_detection_and_gt = -1
                        gt_for_found_detection = None
                        info_gt_for_found_detection = None
                        gt_index = -1
                        bbox_est = found_detection["bbox_est"]
                        for i, (gt_info_this_img_i, gt_this_img_i) in enumerate(
                            zip(scene_gt_info_this_img, scene_gt_this_img)
                        ):
                            if gt_this_img_i["obj_id"] == obj_id:
                                x1, y1, w, h = gt_info_this_img_i["bbox_visib"]
                                iou = compute_iou(bbox_est, [x1, y1, x1 + w, y1 + h])
                                if iou > iou_between_detection_and_gt:
                                    iou_between_detection_and_gt = iou
                                    gt_for_found_detection = gt_this_img_i
                                    info_gt_for_found_detection = gt_info_this_img_i
                                    gt_index = i
                        cam_R_obj = np.array(
                            gt_for_found_detection["cam_R_m2c"]
                        ).reshape(3, 3)
                        cam_t_obj = np.array(
                            gt_for_found_detection["cam_t_m2c"]
                        ).reshape(3, 1)
                        x1, y1, w, h = info_gt_for_found_detection["bbox_visib"]
                        bbox_visib = [x1, y1, x1 + w, y1 + h]
                        x1, y1, w, h = info_gt_for_found_detection["bbox_obj"]
                        bbox_obj = [x1, y1, x1 + w, y1 + h]
                        assert iou_between_detection_and_gt >= -0.5
                        det_score = found_detection["det_score"]
                        det_time = found_detection["det_time"]
                        self.instances.append(
                            dict(
                                scene_id=scene_id,
                                img_id=im_id,
                                K=K,
                                obj_id=obj_id,
                                pose_idx=gt_index,
                                bbox_visib=bbox_visib,
                                bbox_obj=bbox_obj,
                                cam_R_obj=cam_R_obj,
                                cam_t_obj=cam_t_obj,
                                obj_idx=obj_idxs[obj_id],
                                det_score=det_score,
                                det_time=det_time,
                                bbox_est=bbox_est,
                                iou=iou_between_detection_and_gt,
                            )
                        )
        elif keyframe is not None and detections is None:
            # for each item in keyframe, must have gt
            targets_list = get_target_list(join(bop_dataset_folder, keyframe))
            current_scene_id = -1
            for scene_id, im_id, obj_id, _ in targets_list:
                if obj_id not in obj_ids:
                    continue
                if current_scene_id != scene_id:
                    current_scene_id = scene_id
                    scene_folder = join(self.data_folder, f"{scene_id:06d}")
                    scene_camera = load_json(
                        join(scene_folder, "scene_camera.json"), keys_to_int=True
                    )
                    scene_gt_info = load_json(
                        join(scene_folder, "scene_gt_info.json"), keys_to_int=True
                    )
                    scene_gt = load_json(
                        join(scene_folder, "scene_gt.json"), keys_to_int=True
                    )
                K = np.array(scene_camera[im_id]["cam_K"]).reshape((3, 3)).copy()
                scene_gt_this_img = scene_gt[im_id]
                scene_gt_info_this_img = scene_gt_info[im_id]
                for i, (gt_info_this_img_i, gt_this_img_i) in enumerate(
                    zip(scene_gt_info_this_img, scene_gt_this_img)
                ):
                    if gt_this_img_i["obj_id"] != obj_id:
                        continue
                    if gt_info_this_img_i["visib_fract"] < 0.05:
                        continue
                    cam_R_obj = np.array(gt_this_img_i["cam_R_m2c"]).reshape(3, 3)
                    cam_t_obj = np.array(gt_this_img_i["cam_t_m2c"]).reshape(3, 1)
                    x1, y1, w, h = gt_info_this_img_i["bbox_visib"]
                    bbox_visib = [x1, y1, x1 + w, y1 + h]
                    x1, y1, w, h = gt_info_this_img_i["bbox_obj"]
                    bbox_obj = [x1, y1, x1 + w, y1 + h]
                    self.instances.append(
                        dict(
                            scene_id=scene_id,
                            img_id=im_id,
                            K=K,
                            obj_id=obj_id,
                            pose_idx=i,
                            bbox_visib=bbox_visib,
                            bbox_obj=bbox_obj,
                            cam_R_obj=cam_R_obj,
                            cam_t_obj=cam_t_obj,
                            obj_idx=obj_idxs[obj_id],
                        )
                    )
        else:
            raise NotImplementedError("should have keyframe")

        for aux in self.auxs:
            aux.init(self)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        instance = self.instances[i].copy()
        for aux in self.auxs:
            instance = aux(instance, self)
        return instance


class BopDatasetAux:
    def init(self, dataset: BopTrainDataset):
        pass

    def __call__(self, data: dict, dataset: BopTrainDataset) -> dict:
        pass


if __name__ == "__main__":
    from core.symn.MetaInfo import MetaInfo

    meta_info = MetaInfo("tless")
    for folder_name in ["train_pbr", "train_primesense"]:
        data = BopTrainDataset(meta_info, folder_name, obj_ids=range(1, 31))
        print(len(data))
