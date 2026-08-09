"""Localization conformal risk control module.

This module provides conformal risk control for bounding box localization,
ensuring statistical guarantees on coverage ratios through box expansion.
"""

from typing import Dict, List, Tuple
from .utils import compute_iou


class ConformalLocalizationCRC:
    """Localization controller using Conformal Risk Control (CRC).

    Finds the minimal multiplicative expansion factor lambda such that the
    empirical coverage risk on calibration data satisfies the given alpha
    tolerance.  Coverage is measured as the fraction of a ground-truth box
    covered by the expanded prediction box (intersection / true_area).
    """

    def __init__(self, calibration_data, confidence_threshold,
                 required_coverage=0.95, iou_threshold=0.5):
        self.calibration_data = calibration_data
        self.confidence_threshold = confidence_threshold
        self.required_coverage = required_coverage
        self.iou_threshold = iou_threshold
        self.n_calibration = len(calibration_data)

        # Precompute matches (same as LocalizationConformalController)
        self.all_matches = self._precompute_matches()
        self.total_matched_objects = len(self.all_matches)

    def _precompute_matches(self) -> List[Tuple[List, List]]:
        """Precompute all matched prediction-ground truth pairs.

        Filters detections by confidence threshold, then performs greedy
        one-to-one matching between predictions and ground-truth boxes by IoU.
        """
        all_matches = []
        for sample in self.calibration_data:
            matches = self._match_predictions_to_ground_truth(sample)
            all_matches.extend(matches)
        return all_matches

    def _precompute_matches_for_data(self, test_data: List[Dict]) -> List[Tuple[List, List]]:
        """Precompute matches for test data using same logic as calibration."""
        all_matches = []
        for sample in test_data:
            matches = self._match_predictions_to_ground_truth(sample)
            all_matches.extend(matches)
        return all_matches

    def _match_predictions_to_ground_truth(self, sample: Dict) -> List[Tuple[List, List]]:
        """Match predictions to ground truth boxes via greedy one-to-one IoU matching."""
        detections = sample.get('detections', [])
        true_boxes = sample.get('true_boxes', [])

        # Filter by confidence
        filtered_detections = [
            det for det in detections
            if det.get('confidence', 0) >= self.confidence_threshold
        ]

        if not filtered_detections or not true_boxes:
            return []

        matches = []
        used_gt_indices = set()

        # Sort predictions by confidence (highest first)
        sorted_dets = sorted(filtered_detections, key=lambda x: x.get('confidence', 0), reverse=True)

        # For each prediction, find best matching ground truth (greedy)
        for det in sorted_dets:
            pred_box = det.get('bbox', [])
            if len(pred_box) != 4:
                continue

            best_iou = -1
            best_gt_idx = -1
            best_true_box = None

            for gt_idx, true_ann in enumerate(true_boxes):
                if gt_idx in used_gt_indices:
                    continue

                # Get ground truth box - handle COCO format [x, y, w, h]
                gt_bbox = true_ann.get('bbox', [])
                if len(gt_bbox) != 4:
                    continue

                # Convert COCO [x, y, w, h] to [x1, y1, x2, y2]
                x, y, w, h = gt_bbox
                true_box = [x, y, x + w, y + h]

                # Compute IoU
                iou = compute_iou(pred_box, true_box)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    best_true_box = true_box

            # Only match if IoU exceeds threshold
            if best_iou >= self.iou_threshold and best_true_box is not None:
                matches.append((pred_box, best_true_box))
                used_gt_indices.add(best_gt_idx)

        return matches

    def _expand_box(self, box: List[float], lambda_param: float, 
                     img_width: float = None, img_height: float = None) -> List[float]:
        """Expand bounding box by lambda_param (multiplicative expansion).

        Applies multiplicative scaling: new_width = width * (1 + lambda_param),
        centred on the original box centre.  Optionally clips to image bounds.
        """
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        new_width = width * (1 + lambda_param)
        new_height = height * (1 + lambda_param)

        new_x1 = center_x - new_width / 2
        new_y1 = center_y - new_height / 2
        new_x2 = center_x + new_width / 2
        new_y2 = center_y + new_height / 2
        
        # Clip to image boundaries if provided
        if img_width is not None:
            new_x1 = max(0, new_x1)
            new_x2 = min(img_width, new_x2)
        if img_height is not None:
            new_y1 = max(0, new_y1)
            new_y2 = min(img_height, new_y2)
            
        return [new_x1, new_y1, new_x2, new_y2]

    def _box_intersection(self, box1: List[float], box2: List[float]) -> float:
        """Compute intersection area between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        return (x2 - x1) * (y2 - y1)

    def _box_area(self, box: List[float]) -> float:
        """Compute box area."""
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    def _compute_coverage_ratio(self, pred_box: List[float], true_box: List[float]) -> float:
        """Return the fraction of true_box pixels covered by pred_box (in [0, 1])."""
        intersection_area = self._box_intersection(pred_box, true_box)
        true_area = self._box_area(true_box)

        if true_area == 0:
            return 1.0  # Edge case

        return intersection_area / true_area

    def compute_localization_risk_at_lambda(self, lambda_localization: float) -> float:
        """Compute localization risk: fraction of matched objects with insufficient coverage."""
        if not self.all_matches:
            return 1.0  # No data = maximum risk

        failed_objects = 0

        for pred_box, true_box in self.all_matches:
            # Expand prediction box
            expanded_box = self._expand_box(pred_box, lambda_localization)

            # Compute coverage (fraction of ground truth covered)
            coverage = self._compute_coverage_ratio(expanded_box, true_box)

            # Count failures (objects with insufficient coverage)
            if coverage < self.required_coverage:
                failed_objects += 1

        # Finite-sample correction for conformal guarantee
        n = len(self.all_matches)
        return (failed_objects + 1) / (n + 1)

    @staticmethod
    def monotonize_localization_risks(lambda_risk_pairs: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Enforce monotonicity on (lambda, risk) pairs.

        For localization: higher lambda → larger boxes → better coverage → lower risk.
        Ensures risk is non-increasing as lambda increases.
        """
        # Sort by lambda (ascending)
        sorted_pairs = sorted(lambda_risk_pairs, key=lambda x: x[0])

        # Enforce: risk[i] <= risk[i-1] (decreasing or flat)
        monotonized = []
        min_risk_so_far = 1.0  # Start with maximum risk

        for lambda_val, risk in sorted_pairs:
            # Risk can only decrease or stay same as lambda increases
            monotonized_risk = min(risk, min_risk_so_far)
            monotonized.append((lambda_val, monotonized_risk))
            min_risk_so_far = monotonized_risk

        return monotonized

    def optimize_localization_lambda(self, alpha: float, lambda_min: float = 0.0,
                                     lambda_max: float = 2.0, tolerance: float = 0.001,
                                     verbose: bool = True) -> Tuple[float, float, List[Tuple[float, float]]]:
        """Find the minimal lambda satisfying the localization risk bound via binary search."""
        if verbose:
            print(f"\n{'='*70}")
            print(f"Localization Conformal Risk Control")
            print(f"{'='*70}")
            print(f"Confidence threshold: {self.confidence_threshold:.4f}")
            print(f"Risk tolerance (alpha): {alpha:.4f}")
            print(f"Required coverage: {self.required_coverage:.2%}")
            print(f"Matched prediction-ground truth pairs: {len(self.all_matches)}")
            print(f"{'='*70}\n")

        if not self.all_matches:
            if verbose:
                print("WARNING: No matched pairs found!")
            return lambda_max, 1.0, []

        all_evaluations = []

        # Evaluate boundaries first
        risk_at_zero = self.compute_localization_risk_at_lambda(lambda_min)
        risk_at_max = self.compute_localization_risk_at_lambda(lambda_max)
        all_evaluations.append((lambda_min, risk_at_zero))
        all_evaluations.append((lambda_max, risk_at_max))

        if verbose:
            print(f"  Risk at λ=0: {risk_at_zero:.4f}, Risk at λ={lambda_max}: {risk_at_max:.4f}")

        # If risk at λ=0 is already acceptable, return 0 (no expansion needed)
        if risk_at_zero <= alpha:
            if verbose:
                print(f"  λ=0 already satisfies α={alpha}, no expansion needed")
            return lambda_min, risk_at_zero, all_evaluations

        # Binary search between lambda_min and lambda_max
        current_min, current_max = lambda_min, lambda_max
        while (current_max - current_min) > tolerance:
            lambda_mid = (current_min + current_max) / 2.0

            # Compute risk at this lambda
            risk = self.compute_localization_risk_at_lambda(lambda_mid)
            all_evaluations.append((lambda_mid, risk))

            if risk <= alpha:
                # Risk is acceptable, try smaller expansion (save resources)
                current_max = lambda_mid
            else:
                # Risk too high, need larger expansion
                current_min = lambda_mid

        # Apply monotonization
        monotonized_risks = self.monotonize_localization_risks(all_evaluations)

        # Find smallest lambda where risk <= alpha (after monotonization)
        valid_lambdas = [(lam, r) for (lam, r) in monotonized_risks if r <= alpha]

        if valid_lambdas:
            lambda_optimal, final_risk = min(valid_lambdas, key=lambda x: x[0])
        else:
            # Fallback: use maximum expansion
            lambda_optimal = lambda_max
            final_risk = self.compute_localization_risk_at_lambda(lambda_optimal)

        # Compute exact diagnostics by recomputing failures at optimal lambda
        n_pairs = len(self.all_matches)
        n_failed = 0
        for pred_box, true_box in self.all_matches:
            expanded_box = self._expand_box(pred_box, lambda_optimal)
            coverage = self._compute_coverage_ratio(expanded_box, true_box)
            if coverage < self.required_coverage:
                n_failed += 1
        n_covered = n_pairs - n_failed
        stretch = (1 + lambda_optimal) ** 2  # area ratio

        if verbose:
            print(f"  λ_loc = {lambda_optimal:.4f}")
            print(f"  Stretch = {stretch:.4f} ({(stretch - 1) * 100:.1f}% area increase)")
            print(f"  Covered: {n_covered} / {n_pairs} ({n_covered/n_pairs:.2%})")
            print(f"  Failed: {n_failed} / {n_pairs}")
            loc_pass = "✓" if final_risk <= alpha else "✗"
            print(f"  Loc Risk: {final_risk:.4f} (≤ {alpha}) {loc_pass}")

        return lambda_optimal, final_risk, monotonized_risks

    def validate(self, test_data: List[Dict], lambda_val: float,
                 alpha_loc: float) -> Dict:
        """Validate localization coverage guarantee on held-out test data.

        Args:
            test_data: Test dataset (same format as calibration_data)
            lambda_val: Fitted localization expansion lambda
            alpha_loc: Localization risk tolerance to validate against

        Returns:
            Dict with keys 'passed', 'empirical_loc_risk', 'target_alpha', 'n_matches'
        """
        test_matches = self._precompute_matches_for_data(test_data)
        if not test_matches:
            return {
                'passed': False,
                'empirical_loc_risk': 1.0,
                'target_alpha': alpha_loc,
                'n_matches': 0,
            }

        failed = 0
        for pred_box, true_box in test_matches:
            expanded = self._expand_box(pred_box, lambda_val)
            coverage = self._compute_coverage_ratio(expanded, true_box)
            if coverage < self.required_coverage:
                failed += 1

        empirical_loc_risk = failed / len(test_matches)
        return {
            'passed': empirical_loc_risk <= alpha_loc + 1e-3,
            'empirical_loc_risk': empirical_loc_risk,
            'target_alpha': alpha_loc,
            'n_matches': len(test_matches),
        }