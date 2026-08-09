"""
Conformal Thresholding Module

Provides conformal thresholding with CRC (Conformal Risk Control) for
controlling false negative risk in medical image detection.
"""

import numpy as np
from typing import List, Dict, Tuple

from .utils import compute_iou


class RiskCache:
    """Cache for pre-computed risk values with memory management."""

    def __init__(self, calibration_data: List[Dict], max_cache_size: int = 1000,
                 mode: str = 'instance', miss_rate: float = 0.1,
                 lambda_mode: str = 'quantile'):
        """Initialize risk cache for performance optimization.

        Args:
            calibration_data: Calibration dataset
            max_cache_size: Maximum number of cached risk values
            mode: Risk calculation mode ('instance', 'image', 'relative-miss-rate')
            miss_rate: Per-image miss rate tolerance for 'relative-miss-rate' mode
            lambda_mode: 'quantile' (default) maps lambda via sorted score quantile
                         (non-increasing CNF risk); 'direct-inverted' uses
                         threshold = 1 - lambda (non-increasing CNF risk).
        """
        self.calibration_data = calibration_data
        self.cache = {}
        self.max_cache_size = max_cache_size
        self.access_order = []  # For LRU eviction
        self.mode = mode
        self.miss_rate = miss_rate
        self.lambda_mode = lambda_mode

    def get_risks(self, lambda_val: float) -> Dict[str, float]:
        """Get cached CNF risk for a lambda value with LRU eviction.

        Args:
            lambda_val: Lambda parameter value

        Returns:
            Dictionary with computed risk values
        """
        if lambda_val not in self.cache:
            if len(self.cache) >= self.max_cache_size:
                # Remove least recently used
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]

            self.cache[lambda_val] = {
                'fn': self._compute_cnf_risk_at_lambda(lambda_val)
            }

        # Update access order
        if lambda_val in self.access_order:
            self.access_order.remove(lambda_val)
        self.access_order.append(lambda_val)

        return self.cache[lambda_val]

    def _compute_cnf_risk_at_lambda(self, lambda_param: float) -> float:
        """Compute CNF risk at lambda using proper IoU-based matching.

        Args:
            lambda_param: Lambda parameter for threshold computation

        Returns:
            False negative risk value
        """
        threshold = self._lambda_to_confidence_threshold(lambda_param)
        total_unmatched_true = 0
        total_true = 0
        total_images = 0
        images_with_violation = 0

        for sample in self.calibration_data:
            # Get detections above threshold, sorted by confidence (highest first)
            kept_detections = sorted(
                [det for det in sample.get('detections', []) if det['confidence'] >= threshold],
                key=lambda x: x['confidence'],
                reverse=True
            )
            true_boxes = sample.get('true_boxes', [])
            n_true = len(true_boxes)
            total_true += n_true
            total_images += 1

            # Match detections to true boxes using best-IoU matching
            matched_true = set()
            for det in kept_detections:
                det_bbox = det['bbox']
                best_iou = 0.0
                best_gt_idx = None

                for i, gt in enumerate(true_boxes):
                    if i in matched_true:
                        continue
                    gt_bbox_coco = gt['bbox']
                    # Convert COCO [x, y, w, h] to [x1, y1, x2, y2]
                    x, y, w, h = gt_bbox_coco
                    gt_bbox = [x, y, x + w, y + h]
                    iou = compute_iou(det_bbox, gt_bbox)
                    if iou > best_iou and iou >= 0.5:
                        best_iou = iou
                        best_gt_idx = i

                if best_gt_idx is not None:
                    matched_true.add(best_gt_idx)

            n_missed = n_true - len(matched_true)
            total_unmatched_true += n_missed

            # image-level violation tracking
            if self.mode == 'image':
                if n_missed > 0:
                    images_with_violation += 1
            elif self.mode == 'relative-miss-rate':
                per_image_miss_rate = n_missed / n_true if n_true > 0 else 0.0
                if per_image_miss_rate > self.miss_rate:
                    images_with_violation += 1

        if self.mode == 'instance':
            # Finite-sample correction for conformal guarantee
            return (total_unmatched_true + 1) / (total_true + 1) if total_true > 0 else 0.0
        else:  # 'image' or 'relative-miss-rate'
            # Finite-sample correction for conformal guarantee
            return (images_with_violation + 1) / (total_images + 1) if total_images > 0 else 0.0

    def _lambda_to_confidence_threshold(self, lambda_param: float) -> float:
        """Convert lambda to confidence threshold.

        Args:
            lambda_param: Lambda parameter value

        Returns:
            Confidence threshold
        """
        all_confidences = sorted([
            det['confidence']
            for sample in self.calibration_data
            for det in sample.get('detections', [])
        ])
        if not all_confidences:
            return 0.0

        if self.lambda_mode == 'direct-inverted':
            # threshold = 1 - lambda  (non-increasing CNF risk)
            return float(1.0 - lambda_param)

        n = len(all_confidences)
        # Flipped: lambda=0 -> highest threshold (strict, high FN),
        #          lambda=1 -> lowest threshold (loose, low FN)
        # This makes CNF risk non-increasing in lambda.
        idx = int((1.0 - lambda_param) * (n - 1))
        idx = max(0, min(idx, n - 1))
        return all_confidences[idx]



class ConformalThresholdingCRC:
    """
    Conformal thresholding using CRC to control false negative risk only.

    Focuses on CNF risk for medical safety, ignoring FP constraints.
    """

    def __init__(self, calibration_data: List[Dict], alpha_cnf: float = 0.05,
                 mode: str = 'instance', miss_rate: float = 0.1,
                 lambda_mode: str = 'quantile', verbose: bool = True):
        """Initialize conformal thresholding controller.

        Args:
            calibration_data: Calibration dataset
            alpha_cnf: False negative risk tolerance
            mode: Risk calculation mode ('instance', 'image', or 'relative-miss-rate')
            miss_rate: Per-image miss rate tolerance for 'relative-miss-rate' mode
            lambda_mode: 'quantile' (default, non-increasing CNF risk in lambda) or
                         'direct-inverted' (threshold = 1 - lambda, non-increasing
                          CNF risk, lambda=0 -> threshold=1, lambda=1 -> threshold=0)
            verbose: Whether to print progress information
        """
        self.calibration_data = calibration_data
        self.alpha_cnf = alpha_cnf
        self.mode = mode
        self.miss_rate = miss_rate
        self.lambda_mode = lambda_mode
        self.verbose = verbose

        # Initialize risk cache for performance
        self.risk_cache = RiskCache(calibration_data, mode=mode, miss_rate=miss_rate,
                                    lambda_mode=lambda_mode)

        # Pre-calculate sorted confidence scores (same as conformal_threshold.py)
        self.all_confidences = sorted([
            det['confidence']
            for sample in self.calibration_data
            for det in sample.get('detections', [])
        ])
        self.total_objects = sum(sample.get('num_true_objects', 0) for sample in self.calibration_data)
        self.n_calibration = len(self.calibration_data)

    def lambda_to_confidence_threshold(self, lambda_param: float) -> float:
        """Convert lambda to confidence threshold.

        quantile mode (default, non-increasing CNF risk):
          - lambda = 0: highest threshold (strict) -> high FN, low FP
          - lambda = 1: lowest threshold (loose)  -> low FN, high FP

        direct-inverted mode (non-increasing CNF risk):
          - threshold = 1 - lambda  (range [0, 1])
          - lambda = 0: threshold=1 -> keep none -> high FN, low FP
          - lambda = 1: threshold=0 -> keep all -> low FN, high FP

        Args:
            lambda_param: Lambda parameter value

        Returns:
            Confidence threshold
        """
        if not self.all_confidences:
            return 0.0

        if self.lambda_mode == 'direct-inverted':
            # threshold = 1 - lambda  (non-increasing CNF risk)
            return float(1.0 - lambda_param)

        n = len(self.all_confidences)
        # Flipped: lambda=0 -> highest threshold (strict, high FN),
        #          lambda=1 -> lowest threshold (loose, low FN)
        idx = int((1.0 - lambda_param) * (n - 1))
        idx = max(0, min(idx, n - 1))
        return self.all_confidences[idx]

    def compute_cnf_risk_at_lambda(self, lambda_param: float) -> float:
        """Compute empirical False Negative risk at lambda (using cache).

        Args:
            lambda_param: Lambda parameter value

        Returns:
            False negative risk
        """
        return self.risk_cache.get_risks(lambda_param)['fn']

    def optimize_threshold(self) -> Dict:
        """Optimize confidence threshold for false-negative (CNF) risk control.

        Returns:
            Dictionary containing optimization results
        """
        if not self.all_confidences:
            return {
                'confidence_threshold': 0.0,
                'lambda_opt': 0.0,
                'empirical_cnf_risk': 1.0,
                'alpha_cnf': self.alpha_cnf,
                'n_detections': 0,
                'n_true_positives': 0,
                'all_risks': []
            }

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Threshold Conformal Risk Control")
            print(f"{'='*70}")
            print(f"Risk tolerance: α_CNF = {self.alpha_cnf:.4f}")
            print(f"Mode: {self.mode}")
            print(f"Lambda mode: {self.lambda_mode}")
            print(f"Total detections: {len(self.all_confidences)}")
            print(f"Total true objects: {self.total_objects}")
            print(f"{'='*70}\n")

        # Optimize λ for CNF risk with finer grid for precision
        lambda_candidates = np.arange(0.0001, 1.0, 0.0001)
        lambda_opt, cnf_risk, risk_pairs = self.optimize_lambda_for_cnf(self.alpha_cnf, lambda_candidates, verbose=self.verbose)

        # Convert to threshold
        confidence_threshold = self.lambda_to_confidence_threshold(lambda_opt)

        # Compute detailed statistics at optimal threshold
        retained_count, matched_gt, total_gt = self._compute_threshold_stats(confidence_threshold)
        n_images = len(self.calibration_data)
        avg_retained_per_image = retained_count / n_images if n_images > 0 else 0
        retention_rate = retained_count / len(self.all_confidences) if self.all_confidences else 0

        # Validation
        validation_cnf_risk = self.compute_cnf_risk_at_lambda(lambda_opt)
        cnf_pass = "✓" if validation_cnf_risk <= self.alpha_cnf else "✗"
        missed_gt = total_gt - matched_gt
        if self.verbose:
            print(f"\n✓ Optimal λ: {lambda_opt:.4f}", end="")
            if self.lambda_mode == 'direct-inverted':
                print(f"  (confidence threshold = 1 - λ = {confidence_threshold:.4f})")
            else:
                print(f"  (quantile {1.0 - lambda_opt:.3f} of cal scores)")
            print(f"✓ Confidence threshold: {confidence_threshold:.4f}")
            print(f"✓ Retained: {retained_count} / {len(self.all_confidences)} ({retention_rate:.2%})")
            print(f"✓ Matched: {matched_gt} / {total_gt} GT instances")
            print(f"✓ Missed: {missed_gt} GT instances")
            print(f"✓ CNF risk: {validation_cnf_risk:.4f} (≤ {self.alpha_cnf}) {cnf_pass}")

        # Build all_risks from already-computed pairs — no second pass needed
        all_risks = [[lam, risk, self.lambda_to_confidence_threshold(lam)]
                     for lam, risk in risk_pairs]

        return {
            'lambda_opt': lambda_opt,
            'confidence_threshold': confidence_threshold,
            'empirical_risk': validation_cnf_risk,
            'empirical_cnf_risk': validation_cnf_risk,
            'all_risks': all_risks,
            'alpha_cnf': self.alpha_cnf,
            'n_calibration': self.n_calibration,
            'total_objects': self.total_objects,
            'mode': self.mode,
            'lambda_mode': self.lambda_mode,
            'retained_count': retained_count,
            'retention_rate': retention_rate,
            'avg_retained_per_image': avg_retained_per_image,
            'matched_gt': matched_gt,
            'total_gt': total_gt,
            'missed_gt': total_gt - matched_gt
        }

    def _compute_threshold_stats(self, threshold: float) -> Tuple[int, int, int]:
        """Compute statistics at a given threshold.
        
        Returns:
            Tuple of (retained_count, matched_gt, total_gt)
        """
        retained_count = 0
        matched_gt = 0
        total_gt = 0
        
        for sample in self.calibration_data:
            detections = sample.get('detections', [])
            true_boxes = sample.get('true_boxes', [])
            total_gt += len(true_boxes)
            
            # Count retained detections
            kept = [d for d in detections if d.get('confidence', 0) >= threshold]
            retained_count += len(kept)
            
            # Count matched GT (same matching logic as CNF risk)
            kept_sorted = sorted(kept, key=lambda x: x.get('confidence', 0), reverse=True)
            matched_indices = set()
            
            for det in kept_sorted:
                det_bbox = det.get('bbox', [])
                best_iou = 0.0
                best_idx = None
                
                for i, gt in enumerate(true_boxes):
                    if i in matched_indices:
                        continue
                    gt_bbox_coco = gt.get('bbox', [])
                    if len(gt_bbox_coco) != 4:
                        continue
                    x, y, w, h = gt_bbox_coco
                    gt_bbox = [x, y, x + w, y + h]
                    iou = compute_iou(det_bbox, gt_bbox)
                    if iou > best_iou and iou >= 0.5:
                        best_iou = iou
                        best_idx = i
                
                if best_idx is not None:
                    matched_indices.add(best_idx)
            
            matched_gt += len(matched_indices)
        
        return retained_count, matched_gt, total_gt

    def optimize_lambda_for_cnf(self, alpha_cnf: float, lambda_candidates: np.ndarray, verbose: bool = True) -> Tuple[float, float]:
        """Find the optimal λ for CNF risk control with monotonization for statistical guarantee.

        Args:
            alpha_cnf: False negative risk tolerance
            lambda_candidates: Array of lambda values to test
            verbose: Whether to print progress

        Returns:
            Tuple of (optimal_lambda, final_risk, risk_pairs) where risk_pairs is a
            list of (lambda, raw_risk) for every candidate — reuse to avoid recomputing.
        """
        if verbose:
            print(f"Optimizing λ for CNF risk control (α_CNF = {alpha_cnf}) with monotonization")

        # Compute risks for all candidates
        risk_pairs = [(lam, self.compute_cnf_risk_at_lambda(lam)) for lam in lambda_candidates]

        sorted_pairs = sorted(risk_pairs, key=lambda x: x[0])
        monotonized = []

        # quantile / direct-inverted: CNF risk is non-increasing in lambda
        # monotonize by enforcing non-increasing: take running min
        running = float('inf')
        for lambda_val, risk in sorted_pairs:
            running = min(risk, running)
            monotonized.append((lambda_val, running))
        # Find smallest lambda where risk <= alpha_cnf
        valid_lambdas = [(lam, r) for lam, r in monotonized if r <= alpha_cnf]
        if valid_lambdas:
            lambda_opt, cnf_risk = min(valid_lambdas, key=lambda x: x[0])
        else:
            lambda_opt, cnf_risk = min(monotonized, key=lambda x: x[1])

        if verbose:
            print(f"λ_opt = {lambda_opt:.4f} | Monotonized CNF risk = {cnf_risk:.4f} ≤ {alpha_cnf}")

        return lambda_opt, cnf_risk, risk_pairs

    def validate(self, test_data: List[Dict], confidence_threshold: float,
                 alpha_cnf: float) -> Dict:
        """Validate the CNF risk guarantee on held-out test data.

        Uses the same IoU-matching logic as calibration so the empirical
        risk is computed consistently.

        Args:
            test_data: Test dataset (same format as calibration_data)
            confidence_threshold: Fitted confidence threshold to validate
            alpha_cnf: CNF risk tolerance to validate against

        Returns:
            Dict with keys 'passed', 'empirical_cnf_risk', 'target_alpha'
        """
        total_fn = 0
        total_true = 0

        for sample in test_data:
            detections = sample.get('detections', [])
            true_boxes = sample.get('true_boxes', [])
            total_true += len(true_boxes)

            kept = sorted(
                [d for d in detections if d.get('confidence', 0) >= confidence_threshold],
                key=lambda x: x.get('confidence', 0),
                reverse=True
            )

            matched = set()
            for det in kept:
                det_bbox = det.get('bbox', [])
                best_iou = 0.0
                best_idx = None

                for idx, gt in enumerate(true_boxes):
                    if idx in matched:
                        continue
                    gt_bbox_coco = gt.get('bbox', [])
                    if len(gt_bbox_coco) != 4:
                        continue
                    x, y, w, h = gt_bbox_coco
                    gt_bbox = [x, y, x + w, y + h]
                    iou = compute_iou(det_bbox, gt_bbox)
                    if iou > best_iou and iou >= 0.5:
                        best_iou = iou
                        best_idx = idx

                if best_idx is not None:
                    matched.add(best_idx)

            total_fn += len(true_boxes) - len(matched)

        empirical_cnf_risk = total_fn / total_true if total_true > 0 else 0.0
        return {
            'passed': empirical_cnf_risk <= alpha_cnf + 1e-3,
            'empirical_cnf_risk': empirical_cnf_risk,
            'target_alpha': alpha_cnf,
        }