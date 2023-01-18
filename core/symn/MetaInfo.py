""" the path to datasets (use absolute path because docker can not support soft links)
    and basic dataset info
"""
import os
from os.path import join
import getpass
import socket

from bop_toolkit_lib.dataset_params import get_model_params

hostname = socket.gethostname()
username = getpass.getuser()

project_root = None

if username == "root" and len(hostname) > 10:  # run in author specific docker
    # docker path
    public_dataset_path = "/home/pub_datasets/"
    user_dataset_path = "/home/linyongliang/dataset"
    # project root and datasets
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_folder = join(project_root, 'datasets')
    # concrete folder
    bop_folder = join(user_dataset_path, 'pbr')
    detections_folder = join(user_dataset_path, "detections")
    voc_folder = join(public_dataset_path, "det", "VOCdevkit")
    models_GT_color_folder = join(user_dataset_path, "symnet", "models_GT_color_v3")
    binary_code_folder = join(user_dataset_path, "symnet", "binary_code_v3")
else:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_folder = join(project_root, 'datasets')
    # concrete folder
    bop_folder = join(data_folder, 'BOP_DATASETS')
    detections_folder = join(data_folder, "detections")
    voc_folder = join(data_folder, "VOCdevkit")
    models_GT_color_folder = join(data_folder, "models_GT_color")
    binary_code_folder = join(data_folder, "binary_code")


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
        self.binary_code_folder = join(binary_code_folder, name)
