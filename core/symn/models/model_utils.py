""" This file contains some utility function used in model """
import torch
import numpy as np
from lib.pysixd.pose_error import re, te


def compute_mean_re_te(pred_transes, pred_rots, gt_transes, gt_rots):
    """ pred_transes and gt_transes [B, 3, 1]
        pred_rots and gt_rots [B, 3, 3]

        return mean error in the batch
    """

    pred_transes = pred_transes.detach().cpu().numpy()
    pred_rots = pred_rots.detach().cpu().numpy()
    gt_transes = gt_transes.detach().cpu().numpy()
    gt_rots = gt_rots.detach().cpu().numpy()

    bs = pred_rots.shape[0]
    R_errs = np.zeros((bs,), dtype=np.float32)
    T_errs = np.zeros((bs,), dtype=np.float32)
    for i in range(bs):
        R_errs[i] = re(pred_rots[i], gt_rots[i])
        T_errs[i] = te(pred_transes[i], gt_transes[i])
    return R_errs.mean(), T_errs.mean()
