import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from detectron2.utils.events import get_event_storage
from .resnet_backbone import ResNetBackboneNetForCDPN, ResNetBackboneNetForASPP, resnet_spec
from .cdpn_geo_net import CDPNGeoNet
from .aspp_geo_net import ASPPGeoNet
from .conv_pnp_net import ConvPnPNet
from core.utils.rot_reps import ortho6d_to_mat_batch
from core.utils.solver_utils import build_optimizer_with_params
from ..utils.pose_utils import pose_from_param
from .model_utils import compute_mean_re_te
from .pm_loss import PMLoss

logger = logging.getLogger(__name__)


class SymNet(nn.Module):
    def __init__(self, cfg, backbone, geometry_net, pnp_net):
        super().__init__()
        assert cfg.MODEL.NAME == "SymNet", cfg.MODEL.NAME
        self.cfg = cfg
        self.concat = cfg.MODEL.BACKBONE.CONCAT
        self.geometry_net_name = cfg.MODEL.GEOMETRY_NET.ARCH
        self.backbone = backbone
        self.geometry_net = geometry_net
        self.pnp_net = pnp_net
        self.num_classes = cfg.DATASETS.NUM_CLASSES

    def forward(self, x, K, AABB, gt_visib_mask=None, gt_binary_code=None, gt_R=None, gt_t=None, gt_SITE=None,
                gt_allo_rot6d=None, obj_idx=0, points=None, extents=None, sym_infos=None, do_loss=False):
        """
        Args:
            x [batch_size(abb. b), 3, 256, 256] rgb_crop
            K [b, 3, 3] K_crop
            AABB [b, 4] AABB_crop
            gt_visib_mask [2, 128, 128]
            gt_binary_code [b, 16, 128, 128]
            gt_R [b, 3, 3]
            gt_t [b, 3, 1]
            gt_SITE [b, 3, 1]
            gt_allo_rot6d [b, 3, 2]
            obj_idx [b, ]
            points [b, num_points, 3]
            extents [b, 3]
            sym_infos list( b of enum(ndarray[n_sym, 3, 3], None))
        """
        cfg = self.cfg
        if self.concat:
            if "aspp" in self.geometry_net_name:
                x_f32, x_f64, x_f128 = self.backbone(x)
                visib_mask, binary_code = self.geometry_net(x_f32, x_f64, x_f128)
            elif "cdpn" in self.geometry_net_name:
                x_f8, x_f16, x_f32, x_f64 = self.backbone(x)
                visib_mask, binary_code = self.geometry_net(x_f8, x_f16, x_f32, x_f64)
        else:
            if "aspp" in self.geometry_net_name:
                x_f32 = self.backbone(x)
                visib_mask, binary_code = self.geometry_net(x_f32)
            elif "cdpn" in self.geometry_net_name:
                x_f8 = self.backbone(x)
                visib_mask, binary_code = self.geometry_net(x_f8)


        device = x.device
        bs = x.shape[0]
        res = 128
        if self.num_classes > 1:
            assert obj_idx is not None
            visib_mask = visib_mask.view(bs, self.num_classes, 1, res, res)
            visib_mask = visib_mask[torch.arange(bs).to(device), obj_idx]
            binary_code = binary_code.view(bs, self.num_classes, 16, res, res)
            binary_code = binary_code[torch.arange(bs).to(device), obj_idx]

        visib_mask_prob = torch.sigmoid(visib_mask)
        binary_code_prob = torch.sigmoid(binary_code)
        allo_rot6d, SITE = self.pnp_net(visib_mask_prob, binary_code_prob)  # rot6d_allo
        SITE[:, 2] *= 1000
        R_allo = ortho6d_to_mat_batch(allo_rot6d)  # R_allo
        pred_ego_rot, pred_trans = pose_from_param(R_allo, SITE, K, res * 2, AABB)

        if not do_loss:  # test
            out_dict = {"rot": pred_ego_rot, "trans": pred_trans}
            out_dict.update({"visib_mask_prob": visib_mask_prob, "binary_code_prob": binary_code_prob,
                             "allo_rot6d": allo_rot6d, "SITE": SITE})
            return out_dict
        else:  # train
            out_dict = {"rot": pred_ego_rot, "trans": pred_trans}
            if cfg.TRAIN.DEBUG_MODE:
                out_dict.update({"visib_mask_prob": visib_mask_prob, "binary_code_prob": binary_code_prob,
                                 "allo_rot6d": allo_rot6d, "SITE": SITE})
            assert (gt_visib_mask is not None and gt_binary_code is not None and gt_R is not None and gt_t is not None
                    and gt_SITE is not None and gt_allo_rot6d is not None)
            mean_re, mean_te = compute_mean_re_te(pred_trans, pred_ego_rot, gt_t, gt_R)
            vis_dict = {
                "vis/error_re": mean_re,
                "vis/error_te": mean_te,  # mm
                # "vis/error_tx": np.abs(pred_trans[0, 0].detach().item() - gt_t[0, 0].detach().item()),  # mm
                # "vis/error_ty": np.abs(pred_trans[0, 1].detach().item() - gt_t[0, 1].detach().item()),  # mm
                # "vis/error_tz": np.abs(pred_trans[0, 2].detach().item() - gt_t[0, 2].detach().item()),  # mm
                # "vis/pred_trans_x": pred_trans[0, 0].detach().item(),
                # "vis/pred_trans_y": pred_trans[0, 1].detach().item(),
                # "vis/pred_trans_z": pred_trans[0, 2].detach().item(),
                # "vis/gt_t_x": gt_t[0, 0].detach().item(),
                # "vis/gt_t_y": gt_t[0, 1].detach().item(),
                # "vis/gt_t_z": gt_t[0, 2].detach().item(),
            }
            loss_dict = self.symn_loss(
                visib_mask=visib_mask,
                gt_visib_mask=gt_visib_mask,
                binary_code=binary_code,
                gt_binary_code=gt_binary_code,
                SITE=SITE,
                gt_SITE=gt_SITE,
                R=pred_ego_rot,
                gt_R=gt_R,
                points=points,
                extents=extents,
                sym_infos=sym_infos,
            )
            storage = get_event_storage()
            storage.put_scalars(**vis_dict)
            return out_dict, loss_dict

    def symn_loss(self, visib_mask, gt_visib_mask, binary_code,
                  gt_binary_code, SITE, gt_SITE, R, gt_R, points, extents, sym_infos):
        geometry_net_cfg = self.cfg.MODEL.GEOMETRY_NET
        pnp_net_cfg = self.cfg.MODEL.PNP_NET
        loss_dict = {}
        # ---- visib_mask loss ----
        if not geometry_net_cfg.FREEZE:
            visib_mask_loss_type = geometry_net_cfg.VISIB_MASK_LOSS_TYPE
            if visib_mask_loss_type == "BCE":
                # do the sigmoid inside BCEWithLogitsLoss
                loss_func = nn.BCEWithLogitsLoss(reduction="mean")
            elif visib_mask_loss_type == "L1":
                visib_mask = torch.sigmoid(visib_mask)
                loss_func = nn.L1Loss(reduction="mean")
            else:
                raise NotImplementedError(f"unknown visib_mask loss type: {visib_mask_loss_type}")
            loss_dict["loss_visib_mask"] = loss_func(visib_mask[:, 0, :, :], gt_visib_mask) * geometry_net_cfg.VISIB_MASK_LW
        # ---- code loss ----
        if not geometry_net_cfg.FREEZE:
            code_loss_type = geometry_net_cfg.CODE_LOSS_TYPE
            binary_code = visib_mask.clone().detach() * binary_code
            if code_loss_type == "BCE":
                loss_func = nn.BCEWithLogitsLoss(reduction="mean")
                binary_code = binary_code.reshape(-1, binary_code.shape[2], binary_code.shape[3])
                gt_binary_code = gt_binary_code.view(-1, gt_binary_code.shape[2], gt_binary_code.shape[3])
            elif code_loss_type == "L1":
                loss_func = nn.L1Loss(reduction="mean")
                binary_code = binary_code.reshape(-1, 1, binary_code.shape[2], binary_code.shape[3])
                binary_code = torch.sigmoid(binary_code)
                gt_binary_code = gt_binary_code.view(-1, 1, gt_binary_code.shape[2], gt_binary_code.shape[3])
            else:
                raise NotImplementedError(f"unknown visib_mask loss type: {code_loss_type}")
            loss_dict["loss_code"] = loss_func(binary_code, gt_binary_code) * geometry_net_cfg.CODE_LW
        # ---- point matching loss ----
        if pnp_net_cfg.PM_LW > 0:
            loss_func = PMLoss(
                loss_type=pnp_net_cfg.PM_LOSS_TYPE,
                loss_weight=pnp_net_cfg.PM_LW,
                norm_by_extent=pnp_net_cfg.PM_NORM_BY_EXTENT,
                symmetric=pnp_net_cfg.PM_LOSS_SYM,
            )
            loss_pm_dict = loss_func(
                pred_rots=R,
                gt_rots=gt_R,
                points=points,
                extents=extents,
                sym_infos=sym_infos,
            )
            loss_dict.update(loss_pm_dict)
        # ---- SITE_xy loss ----
        if pnp_net_cfg.SITE_XY_LW > 0:
            if pnp_net_cfg.SITE_XY_LOSS_TYPE == "L1":
                loss_func = nn.L1Loss(reduction="mean")
            elif pnp_net_cfg.SITE_XY_LOSS_TYPE == "MSE":
                loss_func = nn.MSELoss(reduction="mean")
            else:
                raise NotImplementedError
            loss_dict["loss_site_xy"] = loss_func(SITE[:, :2], gt_SITE[:, :2, 0]) * pnp_net_cfg.SITE_XY_LW
        # ---- SITE_z loss ----
        if pnp_net_cfg.SITE_Z_LW > 0:
            if pnp_net_cfg.SITE_Z_LOSS_TYPE == "L1":
                loss_func = nn.L1Loss(reduction="mean")
            elif pnp_net_cfg.SITE_Z_LOSS_TYPE == "MSE":
                loss_func = nn.MSELoss(reduction="mean")
            else:
                raise NotImplementedError
            loss_dict["loss_site_z"] = loss_func(SITE[:, 2], gt_SITE[:, 2, 0]) * pnp_net_cfg.SITE_Z_LW
        return loss_dict


def build_model_optimizer(cfg):
    backbone_cfg = cfg.MODEL.BACKBONE
    geometry_net_cfg = cfg.MODEL.GEOMETRY_NET
    pnp_net_cfg = cfg.MODEL.PNP_NET
    params_lr_list = []
    # ---- build backbone ----
    # input: [bsz, 3, 256, 256]
    # output: [bsz, 512, 8, 8], [bsz, 256, 16, 16], [bsz, 128, 32, 32], [bsz, 64, 64, 64]
    if "resnet" in backbone_cfg.ARCH:
        block_type, layers, channels, name = resnet_spec[backbone_cfg.NUM_LAYERS]
        if "aspp" in geometry_net_cfg.ARCH:
            backbone_net = ResNetBackboneNetForASPP(
                block_type, layers, backbone_cfg.INPUT_CHANNEL, freeze=backbone_cfg.FREEZE, concat=backbone_cfg.CONCAT
            )
        elif "cdpn" in geometry_net_cfg.ARCH:
            backbone_net = ResNetBackboneNetForCDPN(
                block_type, layers, backbone_cfg.INPUT_CHANNEL, freeze=backbone_cfg.FREEZE, concat=backbone_cfg.CONCAT
            )
        if backbone_cfg.FREEZE:
            for param in backbone_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append(
                {
                    "params": filter(lambda p: p.requires_grad, backbone_net.parameters()),
                    "lr": float(cfg.SOLVER.BASE_LR),
                }
            )
    # ---- build geometry net ----
    # input: [bsz, 512, 8, 8], [bsz, 256, 16, 16], [bsz, 128, 32, 32], [bsz, 64, 64, 64] for geo_net_cdpn
    #        [bsz, 128, 32, 32], [bsz, 64, 64, 64], [bsz, 64, 128, 128] for geo_net_aspp
    # output: visib_mask[bsz, n_class, 128, 128], binary_code[bsz, n_class * 16, 128, 128]
    if "aspp" in geometry_net_cfg.ARCH:
        geometry_net = ASPPGeoNet(cfg, channels[-3], num_classes=cfg.DATASETS.NUM_CLASSES)
    elif "cdpn" in geometry_net_cfg.ARCH:
        geometry_net = CDPNGeoNet(cfg, channels[-1], num_classes=cfg.DATASETS.NUM_CLASSES)
    else:
        raise NotImplementedError
    if geometry_net_cfg.FREEZE:
        for param in geometry_net.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, geometry_net.parameters()),
                "lr": float(cfg.SOLVER.BASE_LR),
            }
        )
    # ---- build pnp net ----
    # input: visib_mask[bsz, n_class, 128, 128], binary_code[bsz, n_class * 16, 128, 128]
    # output:
    if "ConvPnPNet" in pnp_net_cfg.ARCH:
        inChannels = 1 + 16
        rot_dim = 6
        pnp_net = ConvPnPNet(inChannels, rot_dim)
    else:
        raise NotImplementedError
    if pnp_net_cfg.FREEZE:
        for param in pnp_net.parameters():
            with torch.no_grad():
                param.requires_grad = False
    else:
        params_lr_list.append(
            {
                "params": filter(lambda p: p.requires_grad, pnp_net.parameters()),
                "lr": float(cfg.SOLVER.BASE_LR) * pnp_net_cfg.LR_MULT,
            }
        )
    # ---- build model ----
    model = SymNet(cfg, backbone_net, geometry_net, pnp_net)
    optimizer = build_optimizer_with_params(cfg, params_lr_list)

    if cfg.MODEL.WEIGHTS == "":
        ## backbone initialization
        backbone_pretrained = cfg.MODEL.BACKBONE.get("PRETRAINED", "")
        if backbone_pretrained == "":
            logger.warning("Randomly initialize weights for backbone!")
        else:
            logger.info(f"load backbone weights from: {backbone_pretrained}")
            load_checkpoint(model.backbone, backbone_pretrained, strict=False, logger=logger)

    return model, optimizer
