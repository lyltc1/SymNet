""" the path to datasets (use absolute path because docker can not support soft links)
    and basic dataset info
"""
import os
from os.path import join
from bop_toolkit_lib.dataset_params import get_model_params


project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_folder = join(project_root, 'datasets')
# concrete folder
bop_folder = join(data_folder, 'BOP_DATASETS')
detections_folder = join(data_folder, "detections")
voc_folder = join(data_folder, "VOCdevkit")
models_GT_color_folder = join(data_folder, "models_GT_color")
symnet_code_folder = join(data_folder, "symnet_code")
zebrapose_code_folder = join(data_folder, "zebrapose_code")

class MetaInfo:
    def __init__(self, name, model_type=None):
        # dataset related info
        self.name = name
        self.data_folder = bop_folder
        self.bop_dataset_folder = join(bop_folder, name)
        model_params = get_model_params(bop_folder, name, model_type)
        self.obj_ids = model_params['obj_ids']
        self.symmetric_obj_ids = model_params['symmetric_obj_ids']
        self.model_tpath = model_params['model_tpath']
        self.models_info_path = model_params['models_info_path']
        # method related folder
        self.detections_folder = detections_folder
        self.voc_folder = voc_folder
        self.models_GT_color_folder = join(models_GT_color_folder, name)
        # self.binary_code_folder = join(binary_code_folder, name)
        self.zebrapose_code_folder = join(zebrapose_code_folder, name)
        self.symnet_code_folder = join(symnet_code_folder, name)

