import torch
from torchvision.ops import nms as tv_nms

def multiclass_nms(
    multi_bboxes, multi_scores, score_thr, nms_cfg, max_num=-1, score_factors=None
):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels \
            are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr

    # record proposal and class indices for each True in valid_mask so we can
    # later recover per-detection full-class score vectors in the same order
    inds = valid_mask.nonzero(as_tuple=False)
    if inds.numel() > 0:
        proposal_indices = inds[:, 0]
        class_indices = inds[:, 1]
    else:
        proposal_indices = inds.new_empty((0,), dtype=torch.long)
        class_indices = inds.new_empty((0,), dtype=torch.long)

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # we have to use this ugly code
    bboxes = torch.masked_select(
        bboxes, torch.stack((valid_mask, valid_mask, valid_mask, valid_mask), -1)
    ).view(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = torch.masked_select(scores, valid_mask)
    labels = class_indices

    # build per-detection full-class score vectors (kept for debugging)
    # order matches masked_select output because valid_mask.nonzero() is row-major
    if proposal_indices.numel() > 0:
        score_vectors = multi_scores[:, :-1][proposal_indices]
        if score_factors is not None:
            score_vectors = score_vectors * score_factors[proposal_indices][:, None]
    else:
        score_vectors = multi_scores.new_zeros((0, num_classes))

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5 + num_classes))  # consistent with normal path (0, 5+C)
        labels = multi_bboxes.new_zeros((0,), dtype=torch.long)

        if torch.onnx.is_in_onnx_export():
            raise RuntimeError(
                "[ONNX Error] Can not record NMS "
                "as it has not been executed this time"
            )
        return bboxes, labels

    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    # Append per-class scores to each detection so callers can access full score vectors.
    # Final dets shape: (k, 5 + num_classes)  →  [x1, y1, x2, y2, max_score, cls0, cls1, ...]
    # Index 4 is always the NMS score; indices 5+ are per-class sigmoid scores.
    if score_vectors.numel() > 0:
        kept_score_vectors = score_vectors[keep]  # (k, num_classes)
        dets = torch.cat([dets, kept_score_vectors], dim=1)
    else:
        # No detections survived — pad with zeros to keep consistent column count
        num_classes = multi_scores.size(1) - 1
        dets = torch.cat([dets, dets.new_zeros((dets.size(0), num_classes))], dim=1)

    return dets, labels[keep]


def batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.
    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.
    Arguments:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict): specify nms type and other parameters like iou_thr.
            Possible keys includes the following.
            - iou_thr (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
                number of boxes is large (e.g., 200k). To avoid OOM during
                training, the users could set `split_thr` to a small value.
                If the number of boxes is greater than the threshold, it will
                perform NMS on each group of boxes separately and sequentially.
                Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class.
    Returns:
        tuple: kept dets and indice.
    """
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop("class_agnostic", class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
    nms_cfg_.pop("type", "nms")
    split_thr = nms_cfg_.pop("split_thr", 10000)
    iou_thr = nms_cfg_.pop("iou_threshold", 0.8)  # extracted once; used by both paths
    if len(boxes_for_nms) < split_thr:
        device = boxes.device
        scores_for_nms = scores
        try:
            # Prefer to run NMS on CPU to avoid missing CUDA kernel errors in torchvision builds.
            keep_cpu = tv_nms(boxes_for_nms.cpu(), scores_for_nms.cpu(), iou_thr)
            keep = keep_cpu.to(device)
        except Exception:
            # If CPU call fails for some reason, fall back to calling on the current device (may still error).
            keep = tv_nms(boxes_for_nms, scores_for_nms, iou_thr)
        boxes = boxes[keep]
        scores = scores[keep]
    else:
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            keep = tv_nms(boxes_for_nms[mask], scores[mask], iou_threshold=iou_thr)
            total_mask[mask[keep]] = True

        keep = total_mask.nonzero(as_tuple=False).view(-1)
        keep = keep[scores[keep].argsort(descending=True)]
        boxes = boxes[keep]
        scores = scores[keep]

    return torch.cat([boxes, scores[:, None]], -1), keep
