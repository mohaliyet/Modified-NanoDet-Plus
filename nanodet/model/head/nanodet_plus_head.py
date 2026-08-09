import math

import cv2
import numpy as np
import torch
import torch.nn as nn

from nanodet.util import bbox2distance, distance2bbox, multi_apply, overlay_bbox_cv

from ...data.transform.warp import warp_boxes
from ..loss.gfocal_loss import DistributionFocalLoss, QualityFocalLoss
from ..loss.iou_loss import GIoULoss
from ..module.conv import ConvModule, DepthwiseConvModule
from ..module.init_weights import normal_init
from ..module.nms import multiclass_nms
from .assigner.dsl_assigner import DynamicSoftLabelAssigner
from .gfl_head import Integral, reduce_mean


class NanoDetPlusHead(nn.Module):
    """Detection head used in NanoDet-Plus.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        loss (dict): Loss config.
        input_channel (int): Number of channels of the input feature.
        feat_channels (int): Number of channels of the feature.
            Default: 96.
        stacked_convs (int): Number of conv layers in the stacked convs.
            Default: 2.
        kernel_size (int): Size of the convolving kernel. Default: 5.
        strides (list[int]): Strides of input multi-level feature maps.
            Default: [8, 16, 32].
        conv_type (str): Type of the convolution.
            Default: "DWConv".
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN').
        reg_max (int): The maximal value of the discrete set. Default: 7.
        activation (str): Type of activation function. Default: "LeakyReLU".
        assigner_cfg (dict): Config dict of the assigner. Default: dict(topk=13).
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
        super(NanoDetPlusHead, self).__init__()
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

        # NMS settings — configurable via config.yaml under head.nms
        nms_kwargs = kwargs.get("nms", {})
        if hasattr(nms_kwargs, '__iter__') and not isinstance(nms_kwargs, dict):
            nms_kwargs = dict(nms_kwargs)  # handle yacs CfgNode
        self.nms_iou_threshold = float(nms_kwargs.get("iou_threshold", 0.8))
        self.nms_class_agnostic = bool(nms_kwargs.get("class_agnostic", True))
        self.nms_score_thr = float(nms_kwargs.get("score_thr", 0.001))
        self.nms_max_num = int(nms_kwargs.get("max_num", 100))
        self.nms_mismatch_filter = bool(nms_kwargs.get("mismatch_filter", False))

        self.loss_qfl = QualityFocalLoss(
            beta=self.loss_cfg.loss_qfl.beta,
            loss_weight=self.loss_cfg.loss_qfl.loss_weight,
        )
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
        # init cls head with confidence = 0.01
        bias_cls = -4.595
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
        print("Finish initialize NanoDet-Plus Head.")

    def forward(self, feats):
        if torch.onnx.is_in_onnx_export():
            return self._forward_onnx(feats)
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

    def loss(self, preds, gt_meta, aux_preds=None):
        """Compute losses.
        Args:
            preds (Tensor): Prediction output.
            gt_meta (dict): Ground truth information.
            aux_preds (tuple[Tensor], optional): Auxiliary head prediction output.

        Returns:
            loss (Tensor): Loss tensor.
            loss_states (dict): State dict of each loss.
        """
        device = preds.device
        batch_size = preds.shape[0]
        gt_bboxes = gt_meta["gt_bboxes"]
        gt_labels = gt_meta["gt_labels"]

        gt_bboxes_ignore = gt_meta["gt_bboxes_ignore"]
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(batch_size)]

        input_height, input_width = gt_meta["img"].shape[2:]
        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in self.strides
        ]
        # get grid cells of one image
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
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        decoded_bboxes = distance2bbox(center_priors[..., :2], dis_preds)

        if aux_preds is not None:
            # use auxiliary head to assign
            aux_cls_preds, aux_reg_preds = aux_preds.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
            )
            aux_dis_preds = (
                self.distribution_project(aux_reg_preds) * center_priors[..., 2, None]
            )
            aux_decoded_bboxes = distance2bbox(center_priors[..., :2], aux_dis_preds)
            batch_assign_res = multi_apply(
                self.target_assign_single_img,
                aux_cls_preds.detach(),
                center_priors,
                aux_decoded_bboxes.detach(),
                gt_bboxes,
                gt_labels,
                gt_bboxes_ignore,
            )
        else:
            # use self prediction to assign
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

        if aux_preds is not None:
            aux_loss, aux_loss_states = self._get_loss_from_assign(
                aux_cls_preds, aux_reg_preds, aux_decoded_bboxes, batch_assign_res
            )
            loss = loss + aux_loss
            for k, v in aux_loss_states.items():
                loss_states["aux_" + k] = v
        return loss, loss_states

    def _get_loss_from_assign(self, cls_preds, reg_preds, decoded_bboxes, assign):
        device = cls_preds.device
        (
            labels,
            label_scores,
            label_weights,
            bbox_targets,
            dist_targets,
            num_pos,
        ) = assign
        num_total_samples = max(
            reduce_mean(torch.tensor(sum(num_pos)).to(device)).item(), 1.0
        )

        labels = torch.cat(labels, dim=0)
        label_scores = torch.cat(label_scores, dim=0)
        label_weights = torch.cat(label_weights, dim=0)
        bbox_targets = torch.cat(bbox_targets, dim=0)
        cls_preds = cls_preds.reshape(-1, self.num_classes)
        reg_preds = reg_preds.reshape(-1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)
        loss_qfl = self.loss_qfl(
            cls_preds,
            (labels, label_scores),
            weight=label_weights,
            avg_factor=num_total_samples,
        )

        pos_inds = torch.nonzero(
            (labels >= 0) & (labels < self.num_classes), as_tuple=False
        ).squeeze(1)

        if len(pos_inds) > 0:
            weight_targets = cls_preds[pos_inds].detach().sigmoid().max(dim=1)[0]
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

        loss = loss_qfl + loss_bbox + loss_dfl
        loss_states = dict(loss_qfl=loss_qfl, loss_bbox=loss_bbox, loss_dfl=loss_dfl)
        return loss, loss_states

    @torch.no_grad()
    def target_assign_single_img(
        self,
        cls_preds,
        center_priors,
        decoded_bboxes,
        gt_bboxes,
        gt_labels,
        gt_bboxes_ignore=None,
    ):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            center_priors (Tensor): All priors of one image, a 2D-Tensor with
                shape [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
        """

        device = center_priors.device
        gt_bboxes = torch.from_numpy(gt_bboxes).to(device)
        gt_labels = torch.from_numpy(gt_labels).to(device)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)

        if gt_bboxes_ignore is not None:
            gt_bboxes_ignore = torch.from_numpy(gt_bboxes_ignore).to(device)
            gt_bboxes_ignore = gt_bboxes_ignore.to(decoded_bboxes.dtype)

        assign_result = self.assigner.assign(
            cls_preds,
            center_priors,
            decoded_bboxes,
            gt_bboxes,
            gt_labels,
            gt_bboxes_ignore,
        )
        pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes
        )

        num_priors = center_priors.size(0)
        bbox_targets = torch.zeros_like(center_priors)
        dist_targets = torch.zeros_like(center_priors)
        labels = center_priors.new_full(
            (num_priors,), self.num_classes, dtype=torch.long
        )
        label_weights = center_priors.new_zeros(num_priors, dtype=torch.float)
        label_scores = center_priors.new_zeros(labels.shape, dtype=torch.float)

        num_pos_per_img = pos_inds.size(0)
        pos_ious = assign_result.max_overlaps[pos_inds]

        if len(pos_inds) > 0:
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
        return (
            labels,
            label_scores,
            label_weights,
            bbox_targets,
            dist_targets,
            num_pos_per_img,
        )

    def sample(self, assign_result, gt_bboxes):
        """Sample positive and negative bboxes."""
        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
            .squeeze(-1)
            .unique()
        )
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

    @torch.no_grad()
    def find_source_anchors(
        self,
        all_sigmoid_scores: torch.Tensor,
        det_labels: torch.Tensor,
        winning_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Find the source anchor for each detection by closest sigmoid-score match.

        This is the single source of truth for the anchor lookup used by both
        ``post_process`` and ``get_full_scores.py``.  Given the per-anchor
        sigmoid scores and the NMS winning score for each detection, it returns
        the index of the anchor whose sigmoid score for the winning class is
        closest to the NMS score.

        Args:
            all_sigmoid_scores (Tensor): Per-anchor sigmoid scores ``(N, num_classes)``.
            det_labels         (Tensor): Winning class index per detection ``(k,)``.
            winning_scores     (Tensor): NMS winning score per detection ``(k,)``.

        Returns:
            Tensor: Best anchor index per detection ``(k,)``, dtype long.
        """
        # Gather the sigmoid score of each detection's winning class for all anchors.
        # all_sigmoid_scores[:, det_labels] → (N, k), transpose → (k, N)
        anchor_scores = all_sigmoid_scores[:, det_labels].T          # (k, N)
        # For each detection find the anchor whose score is closest to the NMS score.
        best_anchors = (
            (anchor_scores - winning_scores.unsqueeze(1)).abs().argmin(dim=1)
        )                                                             # (k,)
        return best_anchors

    def post_process(self, preds, meta, aux_preds=None, softening_factor=1.0):
        """Decode bboxes and rescale to original image size.

        Reflects the full pipeline of ``get_full_scores.py``:

        * Per-class sigmoid scores from the main head are already appended to
          every detection bbox by ``multiclass_nms`` (indices 5+).
        * When *aux_preds* is supplied the method additionally identifies each
          detection's source anchor (nearest sigmoid-score match for the
          winning class) and appends a temperature-scaled softmax score vector
          from the auxiliary head.

        Final bbox column layout:
            ``[x1, y1, x2, y2, max_score, sig_0, …, sig_C]``           (no aux)
            ``[x1, y1, x2, y2, max_score, sig_0, …, sig_C, soft_0, …, soft_C, soft_bg]``  (with aux)

        Args:
            preds (Tensor): Main head output, shape ``(B, N, num_classes + 4*(reg_max+1))``.
            meta (dict): Batch meta information.
            aux_preds (Tensor | None): Auxiliary head output, same shape as
                *preds*. When provided, temperature-scaled softmax scores are
                appended to every detection. Default: ``None``.
            softening_factor (float): Temperature divisor applied to aux logits
                before softmax. Must be >= 1.0. Default: ``1.0`` (no scaling).
        """
        cls_scores, bbox_preds = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        result_list = self.get_bboxes(cls_scores, bbox_preds, meta)

        # Pre-compute tensors needed for softmax score attachment.
        if aux_preds is not None:
            # The aux (softmax) head outputs num_classes+1 logits:
            #   [fg_0, fg_1, ..., fg_C, bg]  — background is the last column.
            # We keep ALL columns so softmax is computed over foreground + background,
            # giving a proper probability distribution that includes "no class" mass.
            # Final softmax output shape per detection: (num_classes + 1,)
            aux_cls_size = aux_preds.shape[-1] - 4 * (self.reg_max + 1)
            aux_cls_preds = aux_preds[..., :aux_cls_size]  # (B, N, num_classes+1)
            all_sigmoid_scores = cls_scores.sigmoid()  # (B, N, num_classes)

        det_results = {}
        warp_matrixes = (
            meta["warp_matrix"]
            if isinstance(meta["warp_matrix"], list)
            else meta["warp_matrix"]
        )
        img_heights = (
            meta["img_info"]["height"].cpu().numpy()
            if isinstance(meta["img_info"]["height"], torch.Tensor)
            else meta["img_info"]["height"]
        )
        img_widths = (
            meta["img_info"]["width"].cpu().numpy()
            if isinstance(meta["img_info"]["width"], torch.Tensor)
            else meta["img_info"]["width"]
        )
        img_ids = (
            meta["img_info"]["id"].cpu().numpy()
            if isinstance(meta["img_info"]["id"], torch.Tensor)
            else meta["img_info"]["id"]
        )

        for b_idx, (result, img_width, img_height, img_id, warp_matrix) in enumerate(
            zip(result_list, img_widths, img_heights, img_ids, warp_matrixes)
        ):
            det_bboxes, det_labels = result  # det_bboxes: (k, 5 + num_classes) tensor

            # Optionally append temperature-scaled softmax scores from the aux head.
            # For each detection we locate its source anchor (closest sigmoid score
            # match for the winning class) and compute softmax on its aux logits.
            # This mirrors the per-detection processing in get_full_scores.py.
            if aux_preds is not None and det_bboxes.shape[0] > 0:
                sigmoid_for_img = all_sigmoid_scores[b_idx]  # (N, num_classes)
                aux_cls_for_img = aux_cls_preds[b_idx]       # (N, num_classes+1)

                # Vectorised anchor lookup — single source of truth.
                best_anchors = self.find_source_anchors(
                    sigmoid_for_img, det_labels, det_bboxes[:, 4]
                )                                            # (k,)
                # Temperature-scaled softmax on gathered aux logits.
                scaled_logits = aux_cls_for_img[best_anchors] / softening_factor  # (k, C+1)
                softmax_tensor = torch.softmax(scaled_logits, dim=-1)             # (k, C+1)
                det_bboxes = torch.cat([det_bboxes, softmax_tensor], dim=1)

            det_bboxes = det_bboxes.detach().cpu().numpy()
            det_bboxes[:, :4] = warp_boxes(
                det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height
            )
            classes = det_labels.detach().cpu().numpy()

            det_result = {}
            for i in range(self.num_classes):
                inds = classes == i
                det_result[i] = det_bboxes[inds, :].astype(np.float32).tolist()
            det_results[img_id] = det_result

        return det_results

    def show_result(
        self, img, dets, class_names, score_thres=0.3, show=True, save_path=None
    ):
        result = overlay_bbox_cv(img, dets, class_names, score_thresh=score_thres)
        if show:
            cv2.imshow("det", result)
        return result

    # In nanodet/model/head/nanodet_plus_head.py

    def get_bboxes(self, cls_preds, reg_preds, img_metas):
        """
        Decode the outputs to bboxes.
        This is the BULLETPROOF version for the main Sigmoid head.
        """
        device = cls_preds.device
        b = cls_preds.shape[0]
        input_height, input_width = img_metas["img"].shape[2:]
        input_shape = (input_height, input_width)

        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width / stride))
            for stride in self.strides
        ]
        
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                b,
                featmap_sizes[i],
                stride,
                dtype=torch.float32,
                device=device,
            )
            for i, stride in enumerate(self.strides)
        ]
        center_priors = torch.cat(mlvl_center_priors, dim=1)
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
        
        # Ensure we are only working with the classes this head knows about.
        # This prevents any possible contamination from other parts of the config.
        num_main_head_classes = self.num_classes  # Use the configured number of classes
        
        # cls_preds has shape (batch, num_points, num_classes)
        scores = cls_preds.sigmoid()
        
        result_list = []
        for i in range(b):
            # score_i has shape (num_points, num_classes)
            score_i, bbox_i = scores[i], bboxes[i]

            # This is the key: we add a dummy background class for NMS,
            # but we do it in a way that is completely self-contained.
            # The final score tensor for NMS will have shape (num_points, num_classes + 1)
            padding = score_i.new_zeros(score_i.shape[0], 1)
            score_for_nms = torch.cat([score_i, padding], dim=1)

            results = multiclass_nms(
                bbox_i,
                score_for_nms,
                score_thr=self.nms_score_thr,
                nms_cfg=dict(
                    type="nms",
                    iou_threshold=self.nms_iou_threshold,
                    class_agnostic=self.nms_class_agnostic,
                ),
                max_num=self.nms_max_num,
            )
            
            # The `results` from NMS will have labels from 0 to num_classes-1.
            # We must filter out any accidental background detections before returning.
            final_dets, final_labels = results
            
            # Keep only the detections for our main classes
            keep_inds = final_labels < num_main_head_classes
            final_dets = final_dets[keep_inds]
            final_labels = final_labels[keep_inds]

            # Mismatch filter: discard detections where the NMS-assigned label does not
            # match the argmax of the full per-class sigmoid score vector for that anchor.
            # Mirrors get_full_scores.py: `if label != true_winning_class.item(): continue`.
            if self.nms_mismatch_filter and final_dets.shape[0] > 0 and final_dets.shape[1] > 5:
                per_class_scores = final_dets[:, 5:5 + num_main_head_classes]  # (k, C)
                true_labels = per_class_scores.argmax(dim=1)
                match_mask = true_labels == final_labels
                final_dets = final_dets[match_mask]
                final_labels = final_labels[match_mask]

            result_list.append((final_dets, final_labels))

        return result_list


    def get_single_level_center_priors(
        self, batch_size, featmap_size, stride, dtype, device
    ):
        """Generate centers of a single stage feature map.
        Args:
            batch_size (int): Number of images in one batch.
            featmap_size (tuple[int]): height and width of the feature map
            stride (int): down sample stride of the feature map
            dtype (obj:`torch.dtype`): data type of the tensors
            device (obj:`torch.device`): device of the tensors
        Return:
            priors (Tensor): center priors of a single level feature map.
        """
        h, w = featmap_size
        x_range = (torch.arange(w, dtype=dtype, device=device)) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device)) * stride
        y, x = torch.meshgrid(y_range, x_range)
        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0],), stride)
        proiors = torch.stack([x, y, strides, strides], dim=-1)
        return proiors.unsqueeze(0).repeat(batch_size, 1, 1)

    def _forward_onnx(self, feats):
        """Only used for ONNX export.

        Returns raw logits (no sigmoid) so that post_process / get_bboxes
        can apply sigmoid exactly once, matching the PyTorch inference path.
        Previously sigmoid was applied here, causing a double-sigmoid when
        post_process called cls_preds.sigmoid() — compressing all scores
        to ~0.5 and killing confidence separation between classes.
        """
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
        return torch.cat(outputs, dim=2).permute(0, 2, 1)
