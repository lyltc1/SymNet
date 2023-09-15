import logging
from functools import partial
import torch.nn as nn
from .l2_loss import L2Loss
from fvcore.nn import smooth_l1_loss

from core.utils.pose_utils import get_closest_rot_batch
from lib.pysixd.misc import transform_pts_batch

logger = logging.getLogger(__name__)

class PMLoss(nn.Module):
    """Point matching loss for rotation only """
    def __init__(self, loss_type="L1", loss_weight=1.0, norm_by_extent=False, symmetric=False):
        super().__init__()

        self.loss_weight = loss_weight
        self.norm_by_extent = norm_by_extent
        self.symmetric = symmetric

        if loss_type == "L1":
            self.loss_func = nn.L1Loss(reduction="mean")
        elif loss_type == "L2":
            self.loss_func = L2Loss(reduction="mean")
        elif loss_type == "SMOOTH_L1":
            self.loss_func = partial(smooth_l1_loss, beta=1.0, reduction="mean")
        elif loss_type == "MSE":
            self.loss_func = nn.MSELoss(reduction="mean")
        else:
            raise NotImplementedError
    def forward(self, pred_rots, gt_rots, points, extents=None, sym_infos=None):
        """
        pred_rots: [B, 3, 3]
        gt_rots: [B, 3, 3]
        points: [B, n, 3]

        extents: [B, 3]
        sym_infos: list [Kx3x3 or None],
            stores K rotations regarding symmetries, if not symmetric, None
        """

        if self.symmetric:
            assert sym_infos is not None
            gt_rots = get_closest_rot_batch(pred_rots, gt_rots, sym_infos=sym_infos)

        # [B, n, 3]
        points_est = transform_pts_batch(points, pred_rots, t=None)
        points_tgt = transform_pts_batch(points, gt_rots, t=None)

        if self.norm_by_extent:
            assert extents is not None
            weights = 1.0 / extents.max(1, keepdim=True)[0]  # [B, 1]
            weights = weights.view(-1, 1, 1)  # [B, 1, 1]
        else:
            weights = 1

        loss = self.loss_func(weights * points_est, weights * points_tgt)
        return 3 * loss * self.loss_weight
