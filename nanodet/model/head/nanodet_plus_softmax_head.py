# nanodet/model/head/nanodet_plus_softmax_head.py
# This is the MODIFIED head for Softmax/Cross-Entropy analysis.
# Imports have been moved into methods to prevent circular dependencies.

import math
import torch
import torch.nn as nn

# Only keep essential, non-conflicting imports at the top
from ..module.conv import ConvModule, DepthwiseConvModule
from ..module.init_weights import normal_init


class NanoDetPlusSoftmaxHead(nn.Module):
    """
    Detection head for Softmax-based analysis, used alongside the main head.
    """
    def __init__(
        self,
        num_classes,
        loss,
        input_channel,
        feat_channels=96,
        stacked_convs=2,
        kernel_size=5,
        strides=[8, 16, 32],
        conv_type="DWConv",
        norm_cfg=dict(type="BN"),
        reg_max=7,
        activation="LeakyReLU",
        assigner_cfg=dict(topk=13),
        **kwargs
    ):
        super(NanoDetPlusSoftmaxHead, self).__init__()
        
        # --- IMPORTS MOVED HERE ---
        from ..loss.gfocal_loss import DistributionFocalLoss
        from ..loss.iou_loss import GIoULoss
        from .assigner.dsl_assigner import DynamicSoftLabelAssigner
        from .gfl_head import Integral

        self.num_classes = num_classes
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_size = kernel_size
        self.strides = strides
        self.reg_max = reg_max
        self.activation = activation
        self.ConvModule = ConvModule if conv_type == "Conv" else DepthwiseConvModule
        self.loss_cfg = loss
        self.norm_cfg = norm_cfg
        self.assigner = DynamicSoftLabelAssigner(**assigner_cfg)
        self.distribution_project = Integral(self.reg_max)

        self.loss_cls = nn.CrossEntropyLoss(reduction='none')
        self.cls_loss_weight = self.loss_cfg.loss_cls.loss_weight
        
        self.loss_dfl = DistributionFocalLoss(
            loss_weight=self.loss_cfg.loss_dfl.loss_weight
        )
        self.loss_bbox = GIoULoss(loss_weight=self.loss_cfg.loss_bbox.loss_weight)
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        for _ in self.strides:
            cls_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)
        self.gfl_cls = nn.ModuleList(
            [
                nn.Conv2d(
                    self.feat_channels,
                    self.num_classes + 4 * (self.reg_max + 1),
                    1,
                    padding=0,
                )
                for _ in self.strides
            ]
        )

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                )
            )
        return cls_convs

    def init_weights(self):
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        bias_cls = -4.595
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
        print("Finish initialize NanoDet-Plus SOFTMAX Head.")

    def forward(self, feats):
        outputs = []
        for feat, cls_convs, gfl_cls in zip(
            feats,
            self.cls_convs,
            self.gfl_cls,
        ):
            for conv in cls_convs:
                feat = conv(feat)
            output = gfl_cls(feat)
            outputs.append(output.flatten(start_dim=2))
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        return outputs

    def loss(self, preds, gt_meta):
        # --- IMPORTS MOVED HERE ---
        from nanodet.util import distance2bbox, multi_apply

        device = preds.device
        batch_size = preds.shape[0]
        gt_bboxes = gt_meta["gt_bboxes"]
        gt_labels = gt_meta["gt_labels"]
        gt_bboxes_ignore = gt_meta.get("gt_bboxes_ignore", None)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(batch_size)]

        input_height, input_width = gt_meta["img"].shape[2:]
        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width / stride))
            for stride in self.strides
        ]
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                batch_size,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(self.strides)
        ]
        center_priors = torch.cat(mlvl_center_priors, dim=1)

        cls_preds, reg_preds = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        decoded_bboxes = distance2bbox(
            center_priors[..., :2],
            self.distribution_project(reg_preds) * center_priors[..., 2, None],
        )

        batch_assign_res = multi_apply(
            self.target_assign_single_img,
            cls_preds.detach(),
            center_priors,
            decoded_bboxes.detach(),
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
        )

        loss, loss_states = self._get_loss_from_assign(
            cls_preds, reg_preds, decoded_bboxes, batch_assign_res
        )
        return loss, loss_states

    def _get_loss_from_assign(self, cls_preds, reg_preds, decoded_bboxes, assign):
        # --- IMPORTS MOVED HERE ---
        from .gfl_head import reduce_mean

        device = cls_preds.device
        labels, label_scores, _, bbox_targets, dist_targets, num_pos = assign
        num_total_samples = max(reduce_mean(torch.tensor(sum(num_pos)).to(device)).item(), 1.0)

        labels = torch.cat(labels, dim=0)
        label_scores = torch.cat(label_scores, dim=0)
        bbox_targets = torch.cat(bbox_targets, dim=0)
        
        cls_preds = cls_preds.reshape(-1, self.num_classes)
        reg_preds = reg_preds.reshape(-1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)

        pos_inds = torch.nonzero(
            (labels >= 0) & (labels < self.num_classes - 1), as_tuple=False
        ).squeeze(1)
        
        loss_cls = torch.tensor(0.0, device=device)
        if len(pos_inds) > 0:
            pos_cls_preds = cls_preds[pos_inds]
            pos_labels = labels[pos_inds]
            
            raw_cls_loss = self.loss_cls(pos_cls_preds, pos_labels)
            pos_iou_scores = label_scores[pos_inds]
            weighted_cls_loss = raw_cls_loss * pos_iou_scores
            loss_cls = weighted_cls_loss.sum() / num_total_samples

        if len(pos_inds) > 0:
            weight_targets = cls_preds[pos_inds].detach().softmax(dim=-1).max(dim=1)[0]
            bbox_avg_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)

            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                avg_factor=bbox_avg_factor,
            )

            dist_targets = torch.cat(dist_targets, dim=0)
            loss_dfl = self.loss_dfl(
                reg_preds[pos_inds].reshape(-1, self.reg_max + 1),
                dist_targets[pos_inds].reshape(-1),
                weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0 * bbox_avg_factor,
            )
        else:
            loss_bbox = reg_preds.sum() * 0
            loss_dfl = reg_preds.sum() * 0

        loss = (self.cls_loss_weight * loss_cls) + loss_bbox + loss_dfl
        loss_states = dict(loss_cls=loss_cls, loss_bbox=loss_bbox, loss_dfl=loss_dfl)
        return loss, loss_states

    @torch.no_grad()
    def target_assign_single_img(
        self, cls_preds, center_priors, decoded_bboxes, gt_bboxes, gt_labels, gt_bboxes_ignore=None
    ):
        # --- IMPORTS MOVED HERE ---
        from nanodet.util import bbox2distance

        device = center_priors.device
        gt_bboxes = torch.from_numpy(gt_bboxes).to(device)
        gt_labels = torch.from_numpy(gt_labels).to(device)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        if gt_bboxes_ignore is not None:
            gt_bboxes_ignore = torch.from_numpy(gt_bboxes_ignore).to(device)
            gt_bboxes_ignore = gt_bboxes_ignore.to(decoded_bboxes.dtype)

        assign_result = self.assigner.assign(
            cls_preds, center_priors, decoded_bboxes, gt_bboxes, gt_labels, gt_bboxes_ignore
        )
        pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes
        )

        num_priors = center_priors.size(0)
        bbox_targets = torch.zeros_like(center_priors)
        dist_targets = torch.zeros_like(center_priors)
        
        background_label = self.num_classes - 1
        labels = center_priors.new_full(
            (num_priors,), background_label, dtype=torch.long
        )
        
        label_weights = center_priors.new_zeros(num_priors, dtype=torch.float)
        label_scores = center_priors.new_zeros(labels.shape, dtype=torch.float)

        if len(pos_inds) > 0:
            pos_ious = assign_result.max_overlaps[pos_inds]
            bbox_targets[pos_inds, :] = pos_gt_bboxes
            dist_targets[pos_inds, :] = (
                bbox2distance(center_priors[pos_inds, :2], pos_gt_bboxes)
                / center_priors[pos_inds, None, 2]
            )
            dist_targets = dist_targets.clamp(min=0, max=self.reg_max - 0.1)
            labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
            label_scores[pos_inds] = pos_ious
            label_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        return (labels, label_scores, label_weights, bbox_targets, dist_targets, pos_inds.size(0))

    def sample(self, assign_result, gt_bboxes):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        if gt_bboxes.numel() == 0:
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds
    
    def get_single_level_center_priors(self, batch_size, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = (torch.arange(w, dtype=dtype, device=device)) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device)) * stride
        y, x = torch.meshgrid(y_range, x_range, indexing="ij")
        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0],), stride)
        proiors = torch.stack([x, y, strides, strides], dim=-1)
        return proiors.unsqueeze(0).repeat(batch_size, 1, 1)
