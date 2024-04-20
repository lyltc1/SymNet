import os
import numpy as np
import cv2
from os.path import join
import os.path as osp
import hashlib
import logging
import mmengine
from imgaug.augmenters import (Sequential, Sometimes, SaltAndPepper, GaussianBlur, MotionBlur,
                               Add, Multiply, CoarseDropout, Invert, LinearContrast, pillike, AdditiveGaussianNoise)
from .BOPDataset import BopTrainDataset, BopDatasetAux


logger = logging.getLogger(__name__)


class AugRgbAux(BopDatasetAux):
    def __init__(self, augment_prob, augmentor_code=None):
        self.augment_prob = augment_prob
        if augmentor_code is None:
            self.color_augmentor = Sequential([
                                       Sometimes(0.1, SaltAndPepper(0.05)),
                                       Sometimes(0.1, MotionBlur(k=5)),
                                       Sometimes(0.4, CoarseDropout(p=0.1, size_percent=0.05)),
                                       Sometimes(0.2, GaussianBlur((0., 3.))),
                                       Sometimes(0.5, Add((-20, 20), per_channel=0.3)),
                                       Sometimes(0.1, Invert(0.2, per_channel=True)),
                                       Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),
                                       Sometimes(0.5, Multiply((0.6, 1.4))),
                                       Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),
                                       Sometimes(0.5, LinearContrast((0.5, 2.0), per_channel=0.3))
                                       # Sometimes(0.3, pillike.EnhanceSharpness(factor=(0., 50.))),
                                       # Sometimes(0.3, pillike.EnhanceContrast(factor=(0.2, 50.))),
                                       # Sometimes(0.5, pillike.EnhanceBrightness(factor=(0.1, 6.))),
                                       # Sometimes(0.3, pillike.EnhanceColor(factor=(0., 20.))),
                                       ], random_order=True)
        else:
            self.color_augmentor = eval(augmentor_code)

    def __call__(self, inst: dict, dataset: BopTrainDataset) -> dict:
        if np.random.rand() < self.augment_prob:
            if 'rgb_crop' in inst.keys():
                im = inst['rgb_crop']
                inst['rgb_crop'] = self.color_augmentor.augment_image(im)
            else:
                im = inst['rgb']
                inst['rgb'] = self.color_augmentor.augment_image(im)

        return inst

class ReplaceBgAux(BopDatasetAux):
    def __init__(self, bg_aug_prob, bg_type):
        self.bg_aug_prob = bg_aug_prob
        self.bg_type = bg_type
        self.bg_img_paths = None

    def init(self, dataset: BopTrainDataset):
        hashed_file_name = hashlib.md5(("{}_bg_imgs".format(self.bg_type)).encode("utf-8")).hexdigest()
        cache_path = osp.join(dataset.meta_info.data_folder, "cache_{}_{}.pkl".format(self.bg_type, hashed_file_name))
        mmengine.utils.mkdir_or_exist(osp.dirname(cache_path))
        if osp.exists(cache_path):
            logger.info("get bg_paths from cache file: {}".format(cache_path))
            bg_img_paths = mmengine.load(cache_path)
            logger.info("num bg imgs: {}".format(len(bg_img_paths)))
            assert len(bg_img_paths) > 0
            self.bg_img_paths = bg_img_paths
            return
        if self.bg_type == "VOC":
            VOC_root = join(dataset.meta_info.voc_folder, "VOC2012")
            JPEGImagesPath = join(VOC_root, "JPEGImages")
            bg_img_paths = [osp.join(JPEGImagesPath, fn.name)
                            for fn in os.scandir(JPEGImagesPath) if ".jpg" in fn.name]
        elif self.bg_type == "VOC_table":
            VOC_root = join(dataset.meta_infovoc_folder, "VOC2012")
            VOC_table_list_path = join(VOC_root, "ImageSets/Main/diningtable_trainval.txt")
            with open(VOC_table_list_path, "r") as f:
                VOC_bg_list = [
                    line.strip("\r\n").split()[0] for line in f.readlines() if line.strip("\r\n").split()[1] == "1"
                ]
            bg_img_paths = [osp.join(VOC_root, "JPEGImages/{}.jpg".format(bg_idx)) for bg_idx in VOC_bg_list]
        else:
            raise NotImplementedError
        num_bg_imgs = min(len(bg_img_paths), 10000)
        bg_img_paths = np.random.choice(bg_img_paths, num_bg_imgs)
        mmengine.dump(bg_img_paths, cache_path)
        self.bg_img_paths = bg_img_paths

    def __call__(self, inst: dict, dataset: BopTrainDataset) -> dict:
        if np.random.rand() < self.bg_aug_prob:
            if 'rgb_crop' in inst.keys():
                im = inst['rgb_crop']
                H, W = im.shape[:2]
                ind = np.random.randint(0, len(self.bg_img_paths) - 1)
                filename = self.bg_img_paths[ind]
                bg_img = get_bg_image(filename, H, W)
                mask_bg = ~inst['mask_visib_crop'].astype(bool)
                im[mask_bg] = bg_img[mask_bg]
                inst['rgb_crop'] = im
            else:
                im = inst['rgb']
                H, W = im.shape[:2]
                ind = np.random.randint(0, len(self.bg_img_paths) - 1)
                filename = self.bg_img_paths[ind]
                bg_img = get_bg_image(filename, H, W)
                mask_bg = ~inst['mask_visib'].astype(bool)
                im[mask_bg] = bg_img[mask_bg]
                inst['rgb'] = im
        return inst

def get_bg_image(filename, imH, imW):
    """ keep aspect ratio of bg during resize target image size
        return rgb format
    """
    target_size = min(imH, imW)
    max_size = max(imH, imW)
    real_hw_ratio = float(imH) / float(imW)
    bg_image = cv2.imread(filename, cv2.IMREAD_COLOR)
    bg_h, bg_w, _ = bg_image.shape
    bg_hw_ratio = float(bg_h) / float(bg_w)
    bg_image_resize = np.zeros((imH, imW, 3), dtype="uint8")
    if (real_hw_ratio < 1 and bg_hw_ratio < 1) or (real_hw_ratio >= 1 and bg_hw_ratio >= 1):
        if bg_h >= bg_w:
            bg_h_new = int(np.ceil(bg_w * real_hw_ratio))
            if bg_h_new < bg_h:
                bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
            else:
                bg_image_crop = bg_image
        else:
            bg_w_new = int(np.ceil(bg_h / real_hw_ratio))
            if bg_w_new < bg_w:
                bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]
            else:
                bg_image_crop = bg_image
    else:
        if bg_h >= bg_w:
            bg_h_new = int(np.ceil(bg_w * real_hw_ratio))
            bg_image_crop = bg_image[0:bg_h_new, 0:bg_w, :]
        else:  # bg_h < bg_w
            bg_w_new = int(np.ceil(bg_h / real_hw_ratio))
            # logger.info(bg_w_new)
            bg_image_crop = bg_image[0:bg_h, 0:bg_w_new, :]
    bg_image_resize_0 = resize_short_edge(bg_image_crop, target_size, max_size)
    h, w, c = bg_image_resize_0.shape
    bg_image_resize[0:h, 0:w, :] = bg_image_resize_0
    return bg_image_resize[..., ::-1]

def resize_short_edge(im, target_size, max_size, stride=0, interpolation=cv2.INTER_LINEAR):
    """Scale the shorter edge to the given size, with a limit of `max_size` on
    the longer edge. If `max_size` is reached, then downscale so that the
    longer edge does not exceed max_size. only resize input image to target
    size.

    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)
    if stride == 0:
        return im
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[: im.shape[0], : im.shape[1], :] = im
        return padded_im