"""
Utility functions for conformal prediction in ParaSight.
"""

import json
import os
from typing import List, Dict, Optional
from collections import defaultdict


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """Compute Intersection over Union (IoU) between two boxes.

    Boxes are [x1, y1, x2, y2] format.

    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]

    Returns:
        IoU value between 0.0 and 1.0
    """
    if len(box1) != 4 or len(box2) != 4:
        return 0.0

    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def load_pred_gt_data(pred_file: str, gt_file: str) -> Dict[str, Dict]:
    """Load and prepare prediction and ground truth data from separate JSON files.

    Args:
        pred_file: Path to predictions JSON file
        gt_file: Path to ground truth COCO JSON file

    Returns:
        Dictionary with filename keys containing detection and ground truth data
    """
    # Load predictions
    with open(pred_file, 'r') as f:
        predictions = json.load(f)

    # Load ground truth
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)

    # Create mapping from image_id to file_name
    id_to_filename = {img['id']: img['file_name'] for img in gt_data.get('images', [])}

    # Group ground truth by file_name
    gt_by_filename = defaultdict(list)
    for ann in gt_data.get('annotations', []):
        img_id = ann['image_id']
        if img_id in id_to_filename:
            filename = id_to_filename[img_id]
            gt_by_filename[filename].append(ann)

    # Group predictions by file_name
    pred_by_filename = defaultdict(list)
    for pred in predictions:
        # Assume predictions use file_name directly
        filename = pred['file_name']
        pred_by_filename[filename].append({
            'confidence': pred.get('winning_score', pred.get('score', 0)),
            'bbox': pred.get('bbox_sigmoid', pred.get('bbox', [])),
            'category_id': pred['category_id'],
            'softmax_scores': pred.get('softmax_scores', [])
        })

    # Build dataset as dict with filename as key
    dataset = {}
    for filename in gt_by_filename.keys():
        dataset[filename] = {
            'file_name': filename,
            'detections': pred_by_filename.get(filename, []),
            'num_true_objects': len(gt_by_filename[filename]),
            'true_boxes': gt_by_filename[filename]
        }

    return dataset


def find_true_class(det_bbox: List[float], true_boxes: List[Dict],
                    iou_threshold: float = 0.5) -> Optional[int]:
    """Find the true class for a detection by IoU matching against ground-truth boxes.

    Args:
        det_bbox: Detection bounding box [x1, y1, x2, y2]
        true_boxes: List of ground-truth annotations with COCO 'bbox' ([x, y, w, h])
                    and 'category_id'
        iou_threshold: Minimum IoU to consider a match

    Returns:
        category_id of the best-matching ground-truth box, or None if no match
    """
    best_iou = 0.0
    best_class = None

    for gt in true_boxes:
        gt_bbox_coco = gt.get('bbox', [])
        if len(gt_bbox_coco) != 4:
            continue
        x, y, w, h = gt_bbox_coco
        gt_bbox = [x, y, x + w, y + h]

        iou = compute_iou(det_bbox, gt_bbox)
        if iou > best_iou and iou >= iou_threshold:
            best_iou = iou
            best_class = gt.get('category_id', 0)

    return best_class


def find_true_match(det_bbox: List[float], true_boxes: List[Dict],
                    iou_threshold: float = 0.5):
    """Like find_true_class but also returns the matched GT bbox in xyxy format.

    Returns:
        (category_id, [x1, y1, x2, y2]) of the best-matching GT box, or
        (None, None) if no match exceeds iou_threshold.
    """
    best_iou  = 0.0
    best_class = None
    best_bbox  = None

    for gt in true_boxes:
        gt_bbox_coco = gt.get('bbox', [])
        if len(gt_bbox_coco) != 4:
            continue
        x, y, w, h = gt_bbox_coco
        gt_bbox = [x, y, x + w, y + h]

        iou = compute_iou(det_bbox, gt_bbox)
        if iou > best_iou and iou >= iou_threshold:
            best_iou   = iou
            best_class = gt.get('category_id', 0)
            best_bbox  = gt_bbox

    return best_class, best_bbox