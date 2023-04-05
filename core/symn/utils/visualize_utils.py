import os.path
from typing import Union
from os.path import join
import numpy as np
import cv2
import mmcv
import torch


def preprogress_rgb(rgb: Union[np.ndarray, torch.Tensor]):
    mu, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    if isinstance(rgb, torch.Tensor):  # convert to numpy
        assert rgb.ndim == 3
        rgb = rgb.detach().cpu().numpy()
    assert len(rgb.shape) == 3
    if rgb.shape[0] == 3:  # convert to [h, w, 3]
        rgb = rgb.transpose(1, 2, 0)
    if rgb.min() < 0:  # denormalize
        rgb = (rgb * std + mu) * 255.
    elif rgb.max() < 1.0001:
        rgb = rgb * 255.
    rgb = rgb.astype(np.uint8)
    return rgb

def show_rgb(windows_name: str, rgb: Union[np.ndarray, torch.Tensor]):
    cv2.imshow(windows_name, preprogress_rgb(rgb)[..., ::-1])

def preprogress_mask(mask: Union[np.ndarray, torch.Tensor]):
    if isinstance(mask, torch.Tensor):  # convert to numpy
        mask = mask.detach().cpu().numpy()
    while len(mask.shape) > 2:  # convert [1, h, w] to [h, w]
        mask = mask[0]
    if mask.max() < 1.0001:
        mask = mask * 255.
    mask = mask.astype(np.uint8)
    return mask

def show_mask(windows_name: str, mask: Union[np.ndarray, torch.Tensor]):
    cv2.imshow(windows_name, preprogress_mask(mask))

def show_mask_contour(windows_name: str, rgb, mask_list, threshold=128):
    color = ((0, 255, 0), (0, 0, 255),)  # used in mouse_cb_not_show_contour
    rgb = preprogress_rgb(rgb)
    rgb = cv2.resize(rgb, (rgb.shape[0] >> 1, rgb.shape[1] >> 1), interpolation=cv2.INTER_LINEAR)
    for m, c in zip(mask_list, color):
        m = preprogress_mask(m)
        mask = np.zeros(m.shape, dtype=np.uint8)
        mask[m > threshold] = 1
        mask_contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, mask_contour, -1, c, 1)
    cv2.imshow(windows_name, rgb[..., ::-1])

def preprogress_code(code: Union[np.ndarray, torch.Tensor], last=17):
    """ preprogress code to an image to be show, last is from [0, last) need be show """
    if isinstance(code, torch.Tensor):  # convert to numpy
        code = code.detach().cpu().numpy()
    while len(code.shape) > 3:  # convert [1, 16, h, w] to [16, h, w]
        code = code[0]
    code = (code * 255).astype(np.uint8)
    if code.shape[0] == code.shape[1]:
        code = code.transpose((2, 0, 1))
    last = min(len(code), last)
    code = np.column_stack([code[i] for i in range(last)])
    return code

def show_code(windows_name: str, code: np.ndarray):
    cv2.imshow(windows_name, code)

def show_mask_code(window_name: str, mask1, mask2, code, last_code=5):
    mask1 = preprogress_mask(mask1)
    mask2 = preprogress_mask(mask2)
    code = preprogress_code(code, last_code)
    stack = np.column_stack([mask1, mask2, code])
    cv2.imshow(window_name, stack)

def preprogress_pose(obj_id, renderer, K, R, t, rgb):
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    if isinstance(R, torch.Tensor):
        R = R.detach().cpu().numpy()
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    if len(t.shape) == 1:  # convert [3,] to  [3, 1]
        t = t[:, np.newaxis]
    rgb = preprogress_rgb(rgb)
    render = renderer.render(obj_id, K, R, t)
    render_mask = render[..., 3] == 1.
    pose_img = cv2.resize(rgb, (128, 128))
    pose_img[render_mask] = pose_img[render_mask] * 0.5 + render[..., :3][render_mask] * 0.25 * 255 + 0.25 * 255
    return pose_img

def show_pose(window_name: str, K, R, t, rgb, renderer, obj_id):
    pose_img = preprogress_pose(obj_id, renderer, K, R, t, rgb)
    cv2.imshow(window_name, pose_img[..., ::-1])

def visualize_v2(inputs, out_dir, out_dict=None, renderer=None, sub_file=None):
    """ out_dir is cfg.VIS_DIR, if iteration is not None, make subdir
    """
    if sub_file is not None:
        out_dir = join(out_dir, str(sub_file))
    mmcv.mkdir_or_exist(out_dir)

    scene_id = inputs['scene_id'][0]
    img_id = inputs['img_id'][0]
    obj_id = inputs['obj_id'][0]
    # ----visualize rgb
    i_id = 0
    save_path = join(out_dir, str(scene_id)+'_'+str(img_id)+'_'+str(i_id)+'result.txt')
    while os.path.exists(save_path):
        i_id += 1
        save_path = join(out_dir, str(scene_id)+'_'+str(img_id)+'_'+str(i_id)+'result.txt')
    rgb_crop = preprogress_rgb(inputs["rgb_crop"][0])
    # ----visualize gt mask visib
    gt_mask_amodal_crop = preprogress_mask(inputs["mask_crop"][0])
    gt_mask_visib_crop = preprogress_mask(inputs["mask_visib_crop"][0])
    # ----visualize gt code
    gt_code_crop = preprogress_code(inputs["code_crop"][0])
    concat_img_channel1 = np.column_stack([gt_mask_amodal_crop, gt_mask_visib_crop, gt_code_crop])
    with open(save_path, "w") as f:
        for item_name in ['cam_R_obj', 'cam_t_obj', 'allo_rot6d', 'SITE', 'allo_rot']:
            if item_name in inputs:
                item = inputs[item_name][0]
                f.write(f"gt_{item_name}\n{item}\n\n")
            else:
                f.write(f"can not get {item_name}")
    if renderer is not None:
        K = inputs['K_crop'][0]
        gt_R = inputs['cam_R_obj'][0]
        gt_t = inputs['cam_t_obj'][0]
        pose_img = preprogress_pose(obj_id, renderer, K, gt_R, gt_t, rgb_crop)
        concat_img_channel3 = np.column_stack([rgb_crop, pose_img])
    if out_dict is not None:
        visib_mask_prob = preprogress_mask(out_dict["visib_mask_prob"][0, 0])
        amodal_mask_prob = preprogress_mask(out_dict["amodal_mask_prob"][0, 0])
        binary_code_prob = preprogress_code(out_dict["binary_code_prob"][0])
        concat_img_channel1_eval = np.column_stack([amodal_mask_prob, visib_mask_prob, binary_code_prob])
        concat_img_channel1_diff = np.abs(concat_img_channel1-concat_img_channel1_eval)
        concat_img_channel1 = np.row_stack([concat_img_channel1,
                                            concat_img_channel1_eval,
                                            concat_img_channel1_diff
                                            ])

        with open(save_path, "a") as f:
            for item_name in ['rot', 'trans', 'allo_rot6d', 'SITE']:
                if item_name in out_dict:
                    item = out_dict[item_name][0].detach().cpu().numpy()
                    f.write(f"eval_{item_name}\n{item}\n\n")
                else:
                    f.write(f"can not get {item_name}")
        if renderer is not None:
            eval_R = out_dict['rot'][0]
            eval_t = out_dict['trans'][0]
            pose_eval_img = preprogress_pose(obj_id, renderer, K, eval_R, eval_t, rgb_crop)
            concat_img_channel3 = np.column_stack([concat_img_channel3, pose_eval_img])
    save_path = join(out_dir, str(scene_id) + '_' + str(img_id) + '_' + str(i_id) + 'pose.jpg')
    cv2.imwrite(save_path, concat_img_channel3[..., ::-1])
    save_path = join(out_dir, str(scene_id)+'_'+str(img_id)+'_'+str(i_id)+'inter.png')
    cv2.imwrite(save_path, concat_img_channel1)

