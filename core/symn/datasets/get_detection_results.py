import json
from pathlib import Path
import numpy as np


def get_detection_results_type3(detection_folder: Path):
    """ surfemb detections stored in folder """
    bboxes = np.load(str(detection_folder / 'bboxes.npy'))  # xyxy
    obj_ids = np.load(str(detection_folder / 'obj_ids.npy'))
    scene_ids = np.load(str(detection_folder / 'scene_ids.npy'))
    view_ids = np.load(str(detection_folder / 'view_ids.npy'))
    scores = np.load(str(detection_folder / 'scores.npy'))
    times = np.load(str(detection_folder / 'times.npy'))
    instances = dict()
    for i in range(len(bboxes)):
        scene_id = scene_ids[i]
        img_id = view_ids[i]
        scene_im_id = str(scene_id) + '/' + str(img_id)
        score = scores[i]
        time = times[i]
        obj_id = obj_ids[i]
        bbox_est = bboxes[i].tolist()
        if scene_im_id not in instances.keys():
            instances[scene_im_id] = [dict(
                bj_id=obj_id, bbox_est=bbox_est, det_score=score, det_time=time,)]
        else:
            instances[scene_im_id].append(dict(
                bj_id=obj_id, bbox_est=bbox_est, det_score=score, det_time=time,))
    return instances

def get_detection_results_type1(detection_file: Path):
    """ Load detections from zebrapose_fcos or gdrnpp_yolox,
        The key of detections is 'scene_id/img_id', each value for key is a list, which
        contains several dict with keys ['bbox_est','obj_id','det_score','det_time']
    """
    with open(detection_file) as jsonFile:
        detections = json.load(jsonFile)
        jsonFile.close()
    instances = dict()
    for scene_im_id, dets in detections.items():
        for det in dets:
            bbox_est = det["bbox_est"]  # xywh
            bbox_est[2] = bbox_est[0] + bbox_est[2]
            bbox_est[3] = bbox_est[1] + bbox_est[3]  # xyxy
            det["bbox_est"] = bbox_est
            det["det_score"] = det["score"]
            det["det_time"] = det["time"]
            del det["score"]
            del det["time"]
            if scene_im_id not in instances.keys():
                instances[scene_im_id] = [det]
            else:
                instances[scene_im_id].append(det)
    return instances

def get_detection_results_type2(detection_file):
    """ Load detections from cosypose mask R-CNN,
        which is a large list contains dicts, each dict with keys
        ['scene_id', 'image_id', 'det_score', 'bbox', 'segmentation', 'det_time'
        we do not use segmentation now.
    """
    with open(detection_file) as jsonFile:
        detections = json.load(jsonFile)
        jsonFile.close()
    instances = dict()
    for det in detections:
        scene_id = det['scene_id']
        img_id = det['image_id']
        scene_im_id = str(scene_id) + '/' + str(img_id)
        obj_id = det['category_id']
        bbox_est = det['bbox']  # xywh
        bbox_est[2] = bbox_est[0] + bbox_est[2]
        bbox_est[3] = bbox_est[1] + bbox_est[3]  # xyxy
        score = det.get("score", 1.0)
        time = det.get("time", -1)
        if scene_im_id not in instances.keys():
            instances[scene_im_id] = [dict(obj_id=obj_id, bbox_est=bbox_est, det_score=score, det_time=time,)]
        else:
            instances[scene_im_id].append(dict(obj_id=obj_id, bbox_est=bbox_est, det_score=score, det_time=time,))
    return instances

if __name__ == "__main__":
    if 1:  # test get_detection_results_surfemb
        instances = get_detection_results_type3(Path('datasets/detections/surfemb_detections/ycbv'))
        print(len(instances))
    if 1:  # test_type1
        instances = get_detection_results_type1(Path('datasets/detections/zebrapose_detections/ycbv'
            '/fcos_V57eSE_MSx1333_ColorAugAAEWeaker_8e_ycbv_real_pbr_8e_test_keyframe.json'))
        print(len(instances))
    if 1:
        instances = get_detection_results_type2(Path('datasets/detections/cosypose_maskrcnn_synt+real'
            '/challenge2022-642947_ycbv-test.json'))
        print(len(instances))
