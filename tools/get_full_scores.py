import argparse
import json
import math
import itertools
import os

import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm

from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config, distance2bbox
from nanodet.data.batch_process import stack_batch_img
from nanodet.data.transform.warp import warp_boxes

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="model config file")
    parser.add_argument("--model", required=True, help="model checkpoint (.ckpt) file")
    parser.add_argument("--output", required=True, help="output json file path")
    parser.add_argument("--score_thresh", type=float, default=0.0, help="Confidence threshold for filtering detections (default 0.0 = keep all NMS survivors)")
    parser.add_argument("--task", choices=["val", "test", "cp_cal", "cp_test"], default="val", help="Dataset split to use")
    parser.add_argument(
        "--softening_factor",
        type=float,
        default=1.0,
        help="Temperature scaling factor for Softmax. Must be >=1.0"
    )
    parser.add_argument(
        "--img",
        default=None,
        help="Path to a single image file. When provided, skips the dataset and runs inference on this image only."
    )
    return parser.parse_args()


def _prepare_single_image_batch(img_path, cfg):
    """Load and preprocess a single image into a batch dict matching the dataloader format."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    height, width = img.shape[:2]
    img_info = {
        "id": 0,
        "file_name": os.path.basename(img_path),
        "height": height,
        "width": width,
    }
    meta = dict(img_info=img_info, raw_img=img, img=img)
    pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
    meta = pipeline(None, meta, cfg.data.val.input_size)
    meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1))
    return naive_collate([meta])

def compute_iou(box1, box2):
    """IoU between two xyxy boxes. Returns 0 on invalid input."""
    if len(box1) != 4 or len(box2) != 4:
        return 0.0
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return inter / union if union > 0 else 0.0

def main(args):
    load_config(cfg, args.config)
    
    model = build_model(cfg.model)
    checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)

    # Support both .ckpt (Lightning, keys prefixed with "avg_model.") and
    # .pth (saved by save_model_state, bare keys from weight_averager.state_dict()).
    raw_sd = checkpoint.get("state_dict", checkpoint)
    avg_model_state_dict = {
        k[len("avg_model."):]: v
        for k, v in raw_sd.items()
        if k.startswith("avg_model.")
    }
    if avg_model_state_dict:
        model.load_state_dict(avg_model_state_dict)
        print("Loaded averaged model weights (from .ckpt avg_model keys).")
    else:
        # .pth: state_dict already contains bare model keys
        model.load_state_dict(raw_sd)
        print("Loaded model weights (from .pth state_dict).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    assert model.aux_head.num_classes == model.head.num_classes + 1, (
        f"aux_head.num_classes ({model.aux_head.num_classes}) must equal "
        f"head.num_classes + 1 ({model.head.num_classes + 1}). "
        "Configure the aux head with num_classes = <fg_classes> + 1 (background)."
    )

    if args.img is not None:
        # Single-image mode: bypass dataset/dataloader entirely
        batch = _prepare_single_image_batch(args.img, cfg)
        batches = tqdm([batch], desc="Processing image")
    else:
        # Dataset mode: resolve split and build dataloader
        if args.task == "test" and hasattr(cfg.data, "test") and cfg.data.test is not None:
            dataset_cfg = cfg.data.test
        elif args.task == "cp_cal" and hasattr(cfg.data, "cp_cal") and cfg.data.cp_cal is not None:
            dataset_cfg = cfg.data.cp_cal
        elif args.task == "cp_test" and hasattr(cfg.data, "cp_test") and cfg.data.cp_test is not None:
            dataset_cfg = cfg.data.cp_test
        else:
            dataset_cfg = cfg.data.val
        dataset = build_dataset(dataset_cfg, args.task)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            collate_fn=naive_collate,
        )
        batches = tqdm(dataloader, desc="Processing images")

    results = []
    threshold_discard_count = 0
    mismatch_discard_count = 0
    center_priors = None  # computed once — all images share the same input shape

    with torch.no_grad():
        for batch in batches:
            img_tensor = stack_batch_img(batch["img"], divisible=32).to(device)
            img_id = batch["img_info"]["id"][0]
            
            warp_matrix = batch["warp_matrix"][0]
            img_height = batch["img_info"]["height"][0]
            img_width = batch["img_info"]["width"][0]
            file_name = os.path.basename(batch["img_info"]["file_name"][0])
            
            if isinstance(img_height, torch.Tensor):
                img_height = img_height.item()
            if isinstance(img_width, torch.Tensor):
                img_width = img_width.item()
            if isinstance(warp_matrix, torch.Tensor):
                warp_matrix = warp_matrix.cpu().numpy()

            feat = model.backbone(img_tensor)
            fpn_feat = model.fpn(feat)
            aux_fpn_feat = model.aux_fpn(feat)
            dual_fpn_feat = tuple(
                torch.cat([f, aux_f], dim=1) 
                for f, aux_f in zip(fpn_feat, aux_fpn_feat)
            )
            
            main_head_out = model.head(fpn_feat)
            aux_head_out = model.aux_head(dual_fpn_feat)

            main_cls_scores, main_bbox_preds = main_head_out.split(
                [model.head.num_classes, 4 * (model.head.reg_max + 1)],
                dim=-1,
            )
            aux_cls_scores, aux_bbox_preds = aux_head_out.split(
                [model.aux_head.num_classes, 4 * (model.aux_head.reg_max + 1)],
                dim=-1,
            )

            final_dets, final_labels = model.head.get_bboxes(
                main_cls_scores, main_bbox_preds, {"img": img_tensor}
            )[0]

            if final_dets.shape[0] == 0:
                continue

            all_sigmoid_scores = torch.sigmoid(main_cls_scores[0])

            if center_priors is None:
                input_height, input_width = img_tensor.shape[2:]
                featmap_sizes = [
                    (
                        math.ceil(input_height / s),
                        math.ceil(input_width / s),
                    )
                    for s in model.head.strides
                ]
                mlvl_center_priors = [
                    model.head.get_single_level_center_priors(
                        1,
                        featmap_sizes[i],
                        stride,
                        torch.float32,
                        device,
                    )
                    for i, stride in enumerate(model.head.strides)
                ]
                center_priors = torch.cat(mlvl_center_priors, dim=1).squeeze(0)

            image_results = []

            for i in range(final_dets.shape[0]):
                winning_score_tensor = final_dets[i, 4]
                
                if winning_score_tensor.item() < args.score_thresh:
                    threshold_discard_count += 1
                    continue

                label = int(final_labels[i].item())

                # Delegate to the head's single source-of-truth anchor lookup.
                best_anchor_index = model.head.find_source_anchors(
                    all_sigmoid_scores,
                    final_labels[i : i + 1],
                    final_dets[i : i + 1, 4],
                )[0].item()

                sigmoid_scores_tensor = all_sigmoid_scores[best_anchor_index]

                # Sanity check: the picked anchor's true argmax should match the
                # detection label.  A mismatch means find_source_anchors picked
                # the wrong anchor (two anchors with near-identical scores).
                true_winning_class = int(sigmoid_scores_tensor.argmax().item())
                if true_winning_class != label:
                    mismatch_discard_count += 1
                    continue

                anchor_point = center_priors[best_anchor_index, :2]
                stride = center_priors[best_anchor_index, 2]

                bbox_pred_softmax = aux_bbox_preds[0, best_anchor_index]
                dis_pred_softmax = model.aux_head.distribution_project(
                    bbox_pred_softmax.unsqueeze(0)
                ).squeeze(0)
                dis_pred_softmax = dis_pred_softmax * stride
                bbox_softmax = distance2bbox(
                    anchor_point.unsqueeze(0),
                    dis_pred_softmax.unsqueeze(0),
                    max_shape=img_tensor.shape[2:],
                )[0].cpu().numpy()

                # final_dets[:, :4] already contains the decoded box from get_bboxes.
                bbox_sigmoid_original = warp_boxes(
                    final_dets[i, :4].unsqueeze(0).cpu().numpy(),
                    np.linalg.inv(warp_matrix),
                    img_width,
                    img_height,
                )[0]
                
                bbox_softmax_original = warp_boxes(
                    bbox_softmax[np.newaxis, :],
                    np.linalg.inv(warp_matrix),
                    img_width,
                    img_height
                )[0]

                sigmoid_scores = sigmoid_scores_tensor.cpu().numpy()
                
                scaled_logits = aux_cls_scores[0, best_anchor_index] / args.softening_factor
                
                softmax_scores = torch.softmax(
                    scaled_logits,
                    dim=-1,
                ).cpu().numpy()
                winning_score_float = winning_score_tensor.item()

                result_entry = {
                    "image_id": int(img_id),
                    "file_name": file_name,
                    "category_id": label,
                    "winning_score": float(winning_score_float),
                    "bbox_sigmoid": [float(c) for c in bbox_sigmoid_original],
                    "sigmoid_scores": [float(s) for s in sigmoid_scores],          # length C  (one per class)
                    "bbox_softmax": [float(c) for c in bbox_softmax_original],
                    "softmax_scores": [float(s) for s in softmax_scores],           # length C+1; last element is background class
                }
                image_results.append(result_entry)

            # === Remove near-duplicate bboxes in original image space ===
            # get_bboxes already ran NMS at IoU=0.5 in model input space.
            # This second pass at IoU=0.8 removes duplicate detections that
            # survive the coordinate inversion (warp rounding artefacts).
            # torchvision.ops.nms runs in C++ and is orders of magnitude
            # faster than the equivalent Python O(n²) loop.
            if image_results:
                boxes_t  = torch.tensor([r["bbox_sigmoid"]  for r in image_results], dtype=torch.float32)
                scores_t = torch.tensor([r["winning_score"] for r in image_results], dtype=torch.float32)
                keep_idx = torchvision.ops.nms(boxes_t, scores_t, iou_threshold=0.8)
                for idx in keep_idx.tolist():
                    results.append(image_results[idx])

    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSuccessfully saved {len(results)} detections to {args.output}")
    if threshold_discard_count > 0:
        print(f"Discarded {threshold_discard_count} detections below the {args.score_thresh} confidence threshold.")
    if mismatch_discard_count > 0:
        pct = 100 * mismatch_discard_count / max(1, mismatch_discard_count + len(results))
        level = "WARNING" if pct > 5 else "INFO"
        print(f"[{level}] Discarded {mismatch_discard_count} detections due to anchor mismatch ({pct:.1f}%)")


if __name__ == "__main__":
    args = parse_args()
    main(args)