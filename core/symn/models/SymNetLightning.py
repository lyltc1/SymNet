import logging
from functools import partial
import torch
import torch.nn as nn
from .resnet_backbone import ResNetBackboneNetForCDPN, ResNetBackboneNetForASPP, resnet_spec
from .cdpn_geo_net import CDPNGeoNet
from .aspp_geo_net import ASPPGeoNet
from .conv_pnp_net import ConvPnPNet
from core.utils.rot_reps import ortho6d_to_mat_batch, ortho6d_to_mat_with_axis_batch
from ..utils.pose_utils import pose_from_param
from .pm_loss import PMLoss
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class SymNet(pl.LightningModule):
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

        if cfg.MODEL.PNP_NET.R_type == 'R_allo':
            sym_axis = cfg.DATASETS.get('sym_axis', -1)
            if sym_axis == -1:
                self.ortho6d_to_mat_batch = ortho6d_to_mat_batch
                assert cfg.MODEL.PNP_NET.R_ALLO_SYM_LW == 0
            elif sym_axis in [0, 1, 2]:
                self.sym_axis = sym_axis
                self.ortho6d_to_mat_batch = partial(ortho6d_to_mat_with_axis_batch, axis=sym_axis)
            else:
                raise ArithmeticError("sym_axis must be chosen in [-1, 0, 1, 2]")
        elif cfg.MODEL.PNP_NET.R_type == 'R_allo_6d':
            self.ortho6d_to_mat_batch = ortho6d_to_mat_batch
            assert cfg.MODEL.PNP_NET.R_ALLO_SYM_LW == 0

        if self.num_classes > 1 and self.sym_axis is not None:
            raise NotImplementedError("sym_axis only use for per object training")

    def step(self, x, K, AABB, obj_idx=0, gt_visib_mask=None, gt_amodal_mask=None, gt_binary_code=None, gt_R=None,
             gt_t=None, gt_SITE=None, gt_allo_rot6d=None, gt_allo=None, points=None, extents=None, sym_infos=None,
             do_loss=False, do_output=False):
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
                visib_mask, amodal_mask, binary_code = self.geometry_net(x_f32, x_f64, x_f128)
            elif "cdpn" in self.geometry_net_name:
                x_f8, x_f16, x_f32, x_f64 = self.backbone(x)
                visib_mask, amodal_mask, binary_code = self.geometry_net(x_f8, x_f16, x_f32, x_f64)
        else:
            if "aspp" in self.geometry_net_name:
                x_f32 = self.backbone(x)
                visib_mask, amodal_mask, binary_code = self.geometry_net(x_f32)
            elif "cdpn" in self.geometry_net_name:
                x_f8 = self.backbone(x)
                visib_mask, amodal_mask, binary_code = self.geometry_net(x_f8)

        device = x.device
        bs = x.shape[0]
        res = 128
        if self.num_classes > 1:
            assert obj_idx is not None
            visib_mask = visib_mask.view(bs, self.num_classes, 1, res, res)
            visib_mask = visib_mask[torch.arange(bs).to(device), obj_idx]
            amodal_mask = amodal_mask.view(bs, self.num_classes, 1, res, res)
            amodal_mask = amodal_mask[torch.arange(bs).to(device), obj_idx]
            binary_code = binary_code.view(bs, self.num_classes, 16, res, res)
            binary_code = binary_code[torch.arange(bs).to(device), obj_idx]

        visib_mask_prob = torch.sigmoid(visib_mask)
        amodal_mask_prob = torch.sigmoid(amodal_mask)
        binary_code_prob = torch.sigmoid(binary_code)
        rot_param, SITE = self.pnp_net(visib_mask_prob, amodal_mask_prob, binary_code_prob)
        SITE[:, 2] *= 1000
        R_allo = self.ortho6d_to_mat_batch(rot_param)
        pred_ego_rot, pred_trans = pose_from_param(R_allo, SITE, K, res * 2, AABB)

        if do_output:
            out_dict = {"rot": pred_ego_rot, "trans": pred_trans}
            out_dict.update({"visib_mask_prob": visib_mask_prob,
                             "amodal_mask_prob": amodal_mask_prob,
                             "binary_code_prob": binary_code_prob,
                             "rot_param": rot_param, "SITE": SITE})
        if do_loss:  # train
            loss_dict = self.symn_loss(
                visib_mask=visib_mask,
                gt_visib_mask=gt_visib_mask,
                amodal_mask=amodal_mask,
                gt_amodal_mask=gt_amodal_mask,
                binary_code=binary_code,
                gt_binary_code=gt_binary_code,
                SITE=SITE,
                gt_SITE=gt_SITE,
                R=pred_ego_rot,
                gt_R=gt_R,
                R_allo=R_allo,
                gt_allo=gt_allo,
                points=points,
                extents=extents,
                sym_infos=sym_infos,
            )
        if do_output and do_loss:
            return out_dict, loss_dict
        elif do_loss:
            return loss_dict
        elif do_output:
            return out_dict

    def symn_loss(self, visib_mask, gt_visib_mask, amodal_mask, gt_amodal_mask, binary_code,
                  gt_binary_code, SITE, gt_SITE, R, gt_R, R_allo, gt_allo, points, extents, sym_infos):
        geometry_net_cfg = self.cfg.MODEL.GEOMETRY_NET
        pnp_net_cfg = self.cfg.MODEL.PNP_NET
        loss_dict = {}
        # ---- code loss ----
        mask_for_code = torch.zeros_like(amodal_mask)
        mask_for_code[torch.sigmoid(amodal_mask) > 0.5] = 1.0
        if not geometry_net_cfg.FREEZE:
            code_loss_type = geometry_net_cfg.CODE_LOSS_TYPE
            binary_code = mask_for_code * binary_code
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
            loss_dict["loss_visib_mask"] = loss_func(visib_mask[:, 0, :, :],
                                                     gt_visib_mask) * geometry_net_cfg.VISIB_MASK_LW
        # ---- amodal_mask loss ----
        if not geometry_net_cfg.FREEZE:
            amodal_mask_loss_type = geometry_net_cfg.AMODAL_MASK_LOSS_TYPE
            if amodal_mask_loss_type == "BCE":
                loss_func = nn.BCEWithLogitsLoss(reduction="mean")
            elif amodal_mask_loss_type == "L1":
                amodal_mask = torch.sigmoid(amodal_mask)
                loss_func = nn.L1Loss(reduction="mean")
            else:
                raise NotImplementedError(f"unknown amodal_mask loss type: {amodal_mask_loss_type}")
            loss_dict["loss_amodal_mask"] = loss_func(amodal_mask[:, 0, :, :],
                                                     gt_amodal_mask) * geometry_net_cfg.AMODAL_MASK_LW

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
        # ---- R allo sym loss ----
        if pnp_net_cfg.R_ALLO_SYM_LW > 0:
            if pnp_net_cfg.R_ALLO_SYM_LOSS_TYPE == "L1":
                loss_func = nn.L1Loss(reduction="mean")
            else:
                raise NotImplementedError
            loss_dict["loss_allo_sym"] = loss_func(R_allo[:, :, self.sym_axis],
                                                   gt_allo[:, :, self.sym_axis]) * pnp_net_cfg.R_ALLO_SYM_LW
        return loss_dict

    def configure_optimizers(self):
        self.automatic_optimization = False
        opt_cfg = self.cfg.SOLVER.OPTIMIZER_CFG
        if opt_cfg == "" or opt_cfg is None:
            raise RuntimeError("please provide cfg.SOLVER.OPTIMIZER_CFG to build optimizer")
        if opt_cfg.type == "Ranger":
            from lib.torch_utils.solver.ranger import Ranger
            opt = Ranger([
                dict(params=self.backbone.parameters(), lr=opt_cfg.lr, ),
                dict(params=self.geometry_net.parameters(), lr=opt_cfg.lr),
                dict(params=self.pnp_net.parameters(), lr=opt_cfg.lr)],
                weight_decay=opt_cfg.weight_decay,
            )
        elif opt_cfg.type == "Adam":
            opt = torch.optim.Adam([
                dict(params=self.backbone.parameters(), lr=opt_cfg.lr),
                dict(params=self.geometry_net.parameters(), lr=opt_cfg.lr),
                dict(params=self.pnp_net.parameters(), lr=opt_cfg.lr)
            ])
        else:
            raise NotImplementedError

        sche_cfg = self.cfg.SOLVER.LR_SCHEDULER_CFG
        if sche_cfg == "" or sche_cfg is None:
            raise RuntimeError("please provide cfg.SOLVER.LR_SCHEDULER_CFG to build scheduler")
        if sche_cfg.type == "LambdaLR":
            sched = dict(
                scheduler=torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1., i / sche_cfg.warm)),
                interval='step'
            )
        else:
            raise NotImplementedError
        return [opt], [sched]

    def training_step(self, batch, batch_nb):
        loss_dict = self.step(x=batch['rgb_crop'],
                              K=batch["K_crop"],
                              AABB=batch["AABB_crop"],
                              obj_idx=batch["obj_idx"],
                              gt_visib_mask=batch["mask_visib_crop"],
                              gt_amodal_mask=batch["mask_crop"],
                              gt_binary_code=batch["code_crop"],
                              gt_R=batch["cam_R_obj"],
                              gt_t=batch["cam_t_obj"],
                              gt_SITE=batch["SITE"],
                              gt_allo_rot6d=batch["allo_rot6d"],
                              gt_allo=batch["allo_rot"],
                              points=batch["points"],
                              extents=batch["extent"],
                              sym_infos=batch["sym_info"],
                              do_loss=True,
                              do_output=False)
        losses = sum(loss_dict.values())
        assert torch.isfinite(losses).all(), loss_dict
        self.manual_backward(losses)
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        self.log(f'train/total_loss', losses)
        for k, v in loss_dict.items():
            self.log(f'train/{k}', v)
        return losses

    def validation_step(self, batch, batch_nb):
        loss_dict = self.step(x=batch['rgb_crop'],
                              K=batch["K_crop"],
                              AABB=batch["AABB_crop"],
                              obj_idx=batch["obj_idx"],
                              gt_visib_mask=batch["mask_visib_crop"],
                              gt_amodal_mask=batch["mask_crop"],
                              gt_binary_code=batch["code_crop"],
                              gt_R=batch["cam_R_obj"],
                              gt_t=batch["cam_t_obj"],
                              gt_SITE=batch["SITE"],
                              gt_allo_rot6d=batch["allo_rot6d"],
                              gt_allo=batch["allo_rot"],
                              points=batch["points"],
                              extents=batch["extent"],
                              sym_infos=batch["sym_info"],
                              do_loss=True,
                              do_output=False)
        losses = sum(loss_dict.values())
        self.log(f'valid/total_loss', losses, sync_dist=True)
        for k, v in loss_dict.items():
            self.log(f'valid/{k}', v, sync_dist=True)
        return losses

    @torch.no_grad()
    def infer(self, x, K, AABB, obj_idx):
        out_dict = self.step(x, K, AABB, obj_idx=obj_idx, do_loss=False, do_output=True)
        return out_dict


def build_model(cfg):
    backbone_cfg = cfg.MODEL.BACKBONE
    geometry_net_cfg = cfg.MODEL.GEOMETRY_NET
    pnp_net_cfg = cfg.MODEL.PNP_NET
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
    # ---- build geometry net ----
    # input: [bsz, 512, 8, 8], [bsz, 256, 16, 16], [bsz, 128, 32, 32], [bsz, 64, 64, 64] for geo_net_cdpn
    #        [bsz, 512, 32, 32], [bsz, 64, 64, 64], [bsz, 64, 128, 128] for geo_net_aspp
    # output: visib_mask[bsz, n_class, 128, 128], binary_code[bsz, n_class * 16, 128, 128]
    if "aspp" in geometry_net_cfg.ARCH:
        geometry_net = ASPPGeoNet(cfg, channels[-1], num_classes=cfg.DATASETS.NUM_CLASSES)
    elif "cdpn" in geometry_net_cfg.ARCH:
        geometry_net = CDPNGeoNet(cfg, channels[-1], num_classes=cfg.DATASETS.NUM_CLASSES)
    else:
        raise NotImplementedError
    # ---- build pnp net ----
    # input: visib_mask[bsz, n_class, 128, 128], binary_code[bsz, n_class * 16, 128, 128]
    # output:
    if "ConvPnPNet" in pnp_net_cfg.ARCH:
        inChannels = 1 + 1 + 16
        rot_dim = 6
        pnp_net = ConvPnPNet(inChannels, rot_dim)
    else:
        raise NotImplementedError
    # ---- build model ----
    model = SymNet(cfg, backbone_net, geometry_net, pnp_net)
    if cfg.RESUME is None:
        from mmcv.runner import load_checkpoint
        backbone_pretrained = cfg.MODEL.BACKBONE.get("PRETRAINED", "")
        load_checkpoint(model.backbone, backbone_pretrained, strict=False)

    return model
