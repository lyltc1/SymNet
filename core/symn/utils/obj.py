from typing import Iterable
import numpy as np
from tqdm import tqdm
import trimesh
from bop_toolkit_lib.inout import load_json
from core.symn.MetaInfo import MetaInfo

class Obj:
    def __init__(self, obj_id, mesh: trimesh.Trimesh, diameter: float):
        self.obj_id = obj_id
        self.mesh = mesh
        self.diameter = diameter
        bounding_sphere = self.mesh.bounding_sphere.primitive
        self.offset, self.scale = bounding_sphere.center, bounding_sphere.radius
        self.mesh_norm = mesh.copy()
        self.mesh_norm.apply_translation(-self.offset)
        self.mesh_norm.apply_scale(1 / self.scale)

    def normalize(self, pts: np.ndarray):
        return (pts - self.offset) / self.scale

    def denormalize(self, pts_norm: np.ndarray):
        return pts_norm * self.scale + self.offset

def load_objs(meta_info: MetaInfo, obj_ids: Iterable[int] = None, show_progressbar=True):
    model_tpath = meta_info.model_tpath
    models_info_path = meta_info.models_info_path
    models_info = load_json(models_info_path, keys_to_int=True)
    objs = {}
    if obj_ids is None:
        obj_ids = meta_info.obj_ids
    for obj_id in tqdm(obj_ids, 'loading objects') if show_progressbar else obj_ids:
        diameter = models_info[obj_id]['diameter']
        model_path = model_tpath.format(obj_id=obj_id)
        mesh = trimesh.load_mesh(model_path)
        objs[obj_id] = Obj(obj_id, mesh, diameter)
    return objs
