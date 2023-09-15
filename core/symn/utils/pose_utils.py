import torch
from core.utils.utils import allo_to_ego_mat_torch


def pose_from_param(R_allo, SITE, K, res, AABB):
    """ From pose_param to pose
    :param rot_param [bsz, 3, 3], allo R
    :param t_param [bsz, 3, 1], relative x and y and rel z
    :param K [bsz, 3, 3], camera
    """
    # absolute coords
    c = torch.stack(  # [b, 2]
        [
            (SITE[:, 0] * res) + (res - 1) / 2,
            (SITE[:, 1] * res) + (res - 1) / 2,
        ],
        dim=1,
    )
    z = SITE[:, 2:3] * res / (AABB[:, 2:3] - AABB[:, 0:1])
    pred_trans = torch.cat(
        [z * (c[:, 0:1] - K[:, 0:1, 2]) / K[:, 0:1, 0], z * (c[:, 1:2] - K[:, 1:2, 2]) / K[:, 1:2, 1], z],
        dim=1,
    )


    # pred_trans = torch.matmul(torch.linalg.inv(K), torch.stack((c, 1))) * z
    rot_ego = allo_to_ego_mat_torch(pred_trans, R_allo)
    return rot_ego, pred_trans