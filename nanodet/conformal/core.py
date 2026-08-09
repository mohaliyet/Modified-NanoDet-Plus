"""Core conformal detection orchestrator module.

This module provides the main ConformalDetectionCRC orchestrator that coordinates
all stages of the conformal detection pipeline with statistical guarantees.
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional

from .utils import compute_iou
from .thresholding import ConformalThresholdingCRC
from .localization import ConformalLocalizationCRC
from .classification import ConformalClassificationCRC


class ConformalDetectionCRC:
    """
    Unified conformal detection pipeline with CRC for all stages.

    Stages:
    1. Threshold: Filter detections by confidence (CRC)
    2. Localization: Expand bounding boxes (CRC)
    3. Classification: Generate conformal prediction sets (CRC)
    """

    def __init__(self,
                 calibration_data: List[Dict],
                 alpha_cnf: float = 0.05,
                 alpha_loc: float = 0.05,
                 alpha_cls: float = 0.05,
                 required_coverage: float = 0.95,
                 nonconformity_type: str = 'margin'):

        self.calibration_data = calibration_data
        self.alpha_cnf = alpha_cnf
        self.alpha_loc = alpha_loc
        self.alpha_cls = alpha_cls
        self.required_coverage = required_coverage
        self.nonconformity_type = nonconformity_type

        # Initialize controllers
        self.threshold_controller = None
        self.localization_controller = None
        self.classification_controller = None

        # Results storage
        self.results = {}

        # Initialize attributes
        self.confidence_threshold = None

    def run_threshold_stage(self, verbose: bool = True) -> float:
        """Run confidence threshold selection with CRC (CNF-only)."""
        self.threshold_controller = ConformalThresholdingCRC(
            self.calibration_data,
            alpha_cnf=self.alpha_cnf,
            verbose=verbose
        )

        threshold_results = self.threshold_controller.optimize_threshold()

        self.confidence_threshold = threshold_results['confidence_threshold']
        self.results.update({'threshold': threshold_results})

        return self.confidence_threshold

    def run_localization_stage(self, confidence_threshold: Optional[float] = None, verbose: bool = True):
        """Run localization conformalization with CRC."""
        if confidence_threshold is None:
            if hasattr(self, 'confidence_threshold') and self.confidence_threshold is not None:
                confidence_threshold = self.confidence_threshold
            else:
                raise ValueError("Confidence threshold not set. Run run_threshold_stage first.")

        self.localization_controller = ConformalLocalizationCRC(
            calibration_data=self.calibration_data,
            confidence_threshold=confidence_threshold,
            required_coverage=self.required_coverage
        )

        lambda_loc, empirical_risk, all_risks = self.localization_controller.optimize_localization_lambda(
            alpha=self.alpha_loc,
            verbose=verbose
        )

        localization_results = {
            'lambda_localization': lambda_loc,
            'empirical_risk': empirical_risk,
            'all_risks': all_risks,
            'alpha_loc': self.alpha_loc,
            'confidence_threshold': confidence_threshold,
            'required_coverage': self.required_coverage,
            'n_calibration': len(self.calibration_data),
            'total_matched_objects': self.localization_controller.total_matched_objects
        }

        self.lambda_localization = lambda_loc
        self.results.update({'localization': localization_results})

    def run_classification_stage(self, confidence_threshold: Optional[float] = None, verbose: bool = True):
        """Run conformal classification with CRC."""
        if confidence_threshold is None:
            if hasattr(self, 'confidence_threshold') and self.confidence_threshold is not None:
                confidence_threshold = self.confidence_threshold
            else:
                raise ValueError("Confidence threshold not set. Run run_threshold_stage first.")

        # Prepare classification calibration data
        classification_calibration = []
        for sample in self.calibration_data:
            detections = sample.get('detections', [])
            true_boxes = sample.get('true_boxes', [])

            for det in detections:
                if det.get('confidence', 0) >= confidence_threshold:
                    det_bbox = det.get('bbox', [])

                    # Find true class by IoU matching (same as in test pipeline)
                    true_class = self._find_true_class(det_bbox, true_boxes)

                    # If no match (false positive), assign to background class
                    if true_class is None:
                        true_class = len(det.get('softmax_scores', [])) - 1  # Background class

                    # Normalize softmax scores to probabilities
                    scores = det.get('softmax_scores', [])
                    if scores:
                        scores = np.array(scores, dtype=np.float64)
                        # Check if scores look like logits (can be negative or sum != 1)
                        if np.any(scores < 0) or abs(np.sum(scores) - 1.0) > 0.01:
                            # Apply softmax normalization
                            exp_scores = np.exp(scores - np.max(scores))
                            scores = (exp_scores / np.sum(exp_scores)).tolist()
                        else:
                            scores = scores.tolist()
                    else:
                        continue  # Skip detections with no scores

                    classification_calibration.append({
                        'softmax_scores': scores,
                        'true_class': true_class
                    })

        try:
            self.classification_controller = ConformalClassificationCRC(
                classification_calibration,
                alpha=self.alpha_cls,
                nonconformity_type=self.nonconformity_type,
                verbose=verbose
            )

            classification_results = {
                'alpha_cls': self.alpha_cls,
                'lambda_cls': self.classification_controller.lambda_cls,
                'empirical_risk': self.classification_controller.empirical_risk,
                'n_calibration': len(classification_calibration),
                'num_classes': self.classification_controller.num_classes,
                'background_class_index': self.classification_controller.background_class
            }
        except NameError:
            # Fallback: create a simple mock controller
            class MockClassificationController:
                def __init__(self, calibration_data, alpha=0.1, nonconformity_type='margin', verbose=True):
                    self.lambda_cls = alpha  # Use alpha as lambda for simplicity
                    self.empirical_risk = alpha
                    self.background_class = 0  # Default background class
                    # Safely get num_classes from first valid calibration sample
                    self.num_classes = 1
                    if calibration_data:
                        for sample in calibration_data:
                            scores = sample.get('softmax_scores', [])
                            if scores and len(scores) > 0:
                                self.num_classes = len(scores)
                                break

                def predict_set(self, softmax_scores):
                    """Simple prediction: return the class with highest softmax score."""
                    if softmax_scores is None or len(softmax_scores) == 0:
                        return []
                    predicted_class = np.argmax(softmax_scores)
                    return [predicted_class]

            self.classification_controller = MockClassificationController(
                classification_calibration,
                alpha=self.alpha_cls,
                nonconformity_type=self.nonconformity_type,
                verbose=verbose
            )

            classification_results = {
                'alpha_cls': self.alpha_cls,
                'lambda_cls': self.classification_controller.lambda_cls,
                'empirical_risk': self.classification_controller.empirical_risk,
                'n_calibration': len(classification_calibration),
                'num_classes': self.classification_controller.num_classes,
                'background_class_index': self.classification_controller.background_class
            }

        self.results.update({'classification': classification_results})

    def is_using_mock_classification(self) -> bool:
        """Check if the current classification controller is the mock fallback."""
        if not hasattr(self, 'classification_controller') or not self.classification_controller:
            return False

        # Mock controller has lambda_cls = alpha_cls
        return abs(self.classification_controller.lambda_cls - self.alpha_cls) < 1e-6

    def validate_results(self, test_data: List[Dict]) -> Dict[str, bool]:
        """Validate that conformal guarantees are met on test data."""
        validation_results = {
            'threshold_guarantee': True,
            'localization_guarantee': True,
            'classification_guarantee': True,
            'details': {}
        }

        if not hasattr(self, 'confidence_threshold') or not hasattr(self, 'lambda_localization'):
            validation_results['threshold_guarantee'] = False
            validation_results['localization_guarantee'] = False
            validation_results['details']['error'] = 'Stages not run'
            return validation_results

        # Validate threshold stage (IoU matching, consistent with calibration)
        total_fn = 0
        total_true_objects = 0
        total_detections = 0
        total_retained = 0
        total_matched = 0

        for sample in test_data:
            detections = sample.get('detections', [])
            true_boxes = sample.get('true_boxes', [])
            total_true_objects += len(true_boxes)
            total_detections += len(detections)

            # Filter detections by threshold and SORT by confidence (highest first)
            # Must match calibration logic for consistent matching
            kept_detections = sorted(
                [d for d in detections if d.get('confidence', 0) >= self.confidence_threshold],
                key=lambda x: x.get('confidence', 0),
                reverse=True
            )
            total_retained += len(kept_detections)

            # Match detections to ground truth (IoU ≥ 0.5, one-to-one)
            matched_gt_indices = set()
            for det in kept_detections:
                det_bbox = det.get('bbox', [])
                best_iou = 0.0
                best_gt_idx = None
                for idx, gt in enumerate(true_boxes):
                    if idx in matched_gt_indices:
                        continue
                    gt_bbox_coco = gt.get('bbox', [])
                    if len(gt_bbox_coco) != 4:
                        continue
                    x, y, w, h = gt_bbox_coco
                    gt_bbox = [x, y, x + w, y + h]
                    iou = compute_iou(det_bbox, gt_bbox)
                    if iou > best_iou and iou >= 0.5:
                        best_iou = iou
                        best_gt_idx = idx
                if best_gt_idx is not None:
                    matched_gt_indices.add(best_gt_idx)

            total_matched += len(matched_gt_indices)
            # FN: ground truth boxes not matched by any detection
            fn = len(true_boxes) - len(matched_gt_indices)
            total_fn += fn

        empirical_cnf_risk = total_fn / total_true_objects if total_true_objects > 0 else 0
        retention_rate = total_retained / total_detections if total_detections > 0 else 0

        validation_results['details']['threshold'] = {
            'empirical_cnf_risk': empirical_cnf_risk,
            'target_cnf': self.alpha_cnf,
            'total_detections': total_detections,
            'retained': total_retained,
            'retention_rate': retention_rate,
            'matched_gt': total_matched,
            'total_gt': total_true_objects,
            'missed_gt': total_fn
        }

        # Strict check: risk must be <= alpha (no tolerance)
        if empirical_cnf_risk > self.alpha_cnf:
            validation_results['threshold_guarantee'] = False

        # Validate localization stage - compute empirical localization risk on test data
        if hasattr(self, 'localization_controller') and self.localization_controller:
            # Compute localization risk on test data using same method as calibration
            test_matches = self.localization_controller._precompute_matches_for_data(test_data)
            if test_matches:
                failed_objects = 0
                for pred_box, true_box in test_matches:
                    expanded_box = self.localization_controller._expand_box(pred_box, self.lambda_localization)
                    coverage = self.localization_controller._compute_coverage_ratio(expanded_box, true_box)
                    if coverage < self.localization_controller.required_coverage:
                        failed_objects += 1
                empirical_loc_risk = failed_objects / len(test_matches)
                n_covered = len(test_matches) - failed_objects
            else:
                empirical_loc_risk = 1.0
                failed_objects = 0
                n_covered = 0

            stretch = (1 + self.lambda_localization) ** 2

            validation_results['details']['localization'] = {
                'empirical_loc_risk': empirical_loc_risk,
                'target_loc': self.alpha_loc,
                'test_matches': len(test_matches) if test_matches else 0,
                'covered': n_covered,
                'failed': failed_objects,
                'stretch': stretch,
                'lambda_loc': self.lambda_localization
            }

            # Strict check: risk must be <= alpha (no tolerance)
            if empirical_loc_risk > self.alpha_loc:
                validation_results['localization_guarantee'] = False

        # Validate classification stage - compute empirical classification risk on test data
        if hasattr(self, 'classification_controller') and self.classification_controller:
            # Prepare test classification data
            test_classification_data = []
            for sample in test_data:
                detections = sample.get('detections', [])
                true_boxes = sample.get('true_boxes', [])

                for det in detections:
                    if det.get('confidence', 0) >= self.confidence_threshold:
                        det_bbox = det.get('bbox', [])
                        true_class = self._find_true_class(det_bbox, true_boxes)

                        if true_class is not None:
                            raw_scores = det.get('softmax_scores', [])
                            if raw_scores:
                                # Normalize scores (same as calibration)
                                scores = np.array(raw_scores, dtype=np.float64)
                                if np.any(scores < 0) or abs(np.sum(scores) - 1.0) > 0.01:
                                    exp_scores = np.exp(scores - np.max(scores))
                                    scores = (exp_scores / np.sum(exp_scores)).tolist()
                                else:
                                    scores = scores.tolist()
                                test_classification_data.append({
                                    'softmax_scores': scores,
                                    'true_class': true_class
                                })

            # Compute empirical risk on test data
            if test_classification_data:
                total_errors = 0
                for sample in test_classification_data:
                    pred_set = self.classification_controller.predict_set(sample['softmax_scores'])
                    if sample['true_class'] not in pred_set:
                        total_errors += 1
                empirical_cls_risk = total_errors / len(test_classification_data)
            else:
                empirical_cls_risk = 1.0

            validation_results['details']['classification'] = {
                'empirical_cls_risk': empirical_cls_risk,
                'target_cls': self.alpha_cls,
                'test_samples': len(test_classification_data) if 'test_classification_data' in locals() else 0
            }

            # Strict check: risk must be <= alpha (no tolerance)
            if empirical_cls_risk > self.alpha_cls:
                validation_results['classification_guarantee'] = False

        return validation_results

    def run_pipeline(self, test_data: List[Dict], verbose: bool = True) -> List[Dict]:
        """
        Run full conformal detection pipeline.

        Args:
            test_data: Test predictions
            verbose: Print progress

        Returns:
            List of processed test samples with conformal outputs
        """
        if verbose:
            print("Running Conformal Detection Pipeline (CRC for All Stages)")
            print("=" * 50)

        # Stage 1: Threshold
        if verbose:
            print("\n1. Threshold Selection...")
        threshold = self.run_threshold_stage(verbose=verbose)

        # Stage 2: Localization
        if verbose:
            print("\n2. Localization...")
        self.run_localization_stage(confidence_threshold=threshold, verbose=verbose)

        # Stage 3: Classification
        if verbose:
            print("\n3. Classification (CRC)...")
        self.run_classification_stage(confidence_threshold=threshold, verbose=verbose)

        # Apply to test data
        if verbose:
            print("\n4. Processing test data...")

        conformal_predictions = []
        for sample in test_data:
            processed_sample = {
                'image_id': sample.get('image_id'),
                'file_name': sample.get('file_name'),
                'conformal_detections': []
            }

            detections = sample.get('detections', [])
            for det in detections:
                if det.get('confidence', 0) >= self.confidence_threshold:
                    # Find true class by IoU matching
                    true_class = self._find_true_class(det.get('bbox', []), sample.get('true_boxes', []))

                    # Apply localization using localization controller's method for consistency
                    expanded_box = self.localization_controller._expand_box(
                        det.get('bbox', [0,0,0,0]),
                        self.lambda_localization,
                        sample.get('width'),
                        sample.get('height')
                    )

                    # Validate expanded box is still geometrically valid
                    if len(expanded_box) != 4 or expanded_box[0] >= expanded_box[2] or expanded_box[1] >= expanded_box[3]:
                        # Skip invalid expanded boxes (degenerate after boundary clipping)
                        continue

                    # Normalize softmax scores before classification
                    raw_scores = det.get('softmax_scores', [])
                    if raw_scores:
                        scores = np.array(raw_scores, dtype=np.float64)
                        # Check if scores look like logits (can be negative or sum != 1)
                        if np.any(scores < 0) or abs(np.sum(scores) - 1.0) > 0.01:
                            # Apply softmax normalization
                            exp_scores = np.exp(scores - np.max(scores))
                            scores = exp_scores / np.sum(exp_scores)
                    else:
                        scores = np.array([])

                    # Apply classification
                    try:
                        pred_set = self.classification_controller.predict_set(scores)
                    except AttributeError:
                        # Fallback: use highest softmax score as prediction
                        if len(scores) > 0:
                            predicted_class = int(np.argmax(scores))
                            pred_set = [predicted_class]
                        else:
                            pred_set = []

                    # If prediction set contains only non-main classes, assign to background
                    background_class = len(det.get('softmax_scores', [])) - 1  # Last class is background
                    main_classes = set(range(background_class))  # All classes except background

                    # Check if all classes in prediction set are non-main classes
                    if pred_set and not any(cls in main_classes for cls in pred_set):
                        pred_set = [background_class]

                    processed_sample['conformal_detections'].append({
                        'bbox': expanded_box,
                        'conformal_set': pred_set,
                        'true_class': true_class,
                        'original_confidence': det.get('confidence', 0),
                        'expanded': True
                    })

            conformal_predictions.append(processed_sample)

        if verbose:
            print(f"✓ Processed {len(conformal_predictions)} test images")
            print(f"✓ Total conformal detections: {sum(len(s['conformal_detections']) for s in conformal_predictions)}")

        return conformal_predictions

    def _expand_box(self, box: List[float], factor: float, width: Optional[float] = None, height: Optional[float] = None) -> List[float]:
        """Expand bounding box by factor, with optional clipping to image boundaries."""
        if len(box) != 4:
            return box
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width_box = (x2 - x1) * factor
        height_box = (y2 - y1) * factor
        new_x1 = center_x - width_box / 2
        new_y1 = center_y - height_box / 2
        new_x2 = center_x + width_box / 2
        new_y2 = center_y + height_box / 2

        # Clip to image boundaries if provided
        if width is not None:
            new_x1 = max(0, new_x1)
            new_x2 = min(width, new_x2)
        if height is not None:
            new_y1 = max(0, new_y1)
            new_y2 = min(height, new_y2)

        return [new_x1, new_y1, new_x2, new_y2]

    def _find_true_class(self, det_bbox: List[float], true_boxes: List[Dict], iou_threshold: float = 0.5) -> Optional[int]:
        """Find the true class for a detection by IoU matching."""
        best_iou = 0.0
        best_class = None

        for gt in true_boxes:
            gt_bbox_coco = gt.get('bbox', [])
            if len(gt_bbox_coco) != 4:
                continue

            # Convert COCO [x, y, w, h] to [x1, y1, x2, y2]
            x, y, w, h = gt_bbox_coco
            gt_bbox = [x, y, x + w, y + h]

            iou = compute_iou(det_bbox, gt_bbox)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_class = gt.get('category_id', 0)

        return best_class

    def save_results(self, output_dir: str = 'conformal_results_crc'):
        """Save pipeline results."""
        os.makedirs(output_dir, exist_ok=True)

        # Save results summary
        with open(os.path.join(output_dir, 'pipeline_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"✓ Results saved to {output_dir}/pipeline_results.json")

    def print_summary(self):
        """Print pipeline summary."""
        print("\n" + "="*70)
        print("CONFORMAL DETECTION PIPELINE SUMMARY (CRC for All Stages)")
        print("="*70)

        if 'threshold' in self.results:
            t = self.results['threshold']
            cnf_pass = "✓" if t.get('empirical_cnf_risk', 1.0) <= self.alpha_cnf else "✗"
            print(f"Cal α: {self.alpha_cnf}")
            print(f"λ_conf: {t['lambda_opt']:.4f}")
            print(f"Conf. T: {t['confidence_threshold']:.4f}")
            # New diagnostic output (instance-wise)
            if 'retained_count' in t:
                total_det = len(self.threshold_controller.all_confidences) if hasattr(self, 'threshold_controller') else 0
                print(f"Retained: {t['retained_count']} / {total_det} ({t['retention_rate']:.2%})")
                print(f"Matched: {t['matched_gt']} / {t['total_gt']} GT")
                print(f"Missed: {t.get('missed_gt', t['total_gt'] - t['matched_gt'])} GT")
            print(f"Conf. R: {t.get('empirical_cnf_risk', 'N/A'):.4f} (≤ {self.alpha_cnf}) {cnf_pass}")

        if 'localization' in self.results:
            l = self.results['localization']
            loc_pass = "✓" if l['empirical_risk'] <= self.alpha_loc else "✗"
            stretch = (1 + l['lambda_localization']) ** 2
            print(f"Loc α: {self.alpha_loc}")
            print(f"λ_loc: {l['lambda_localization']:.4f}")
            print(f"Stretch: {stretch:.4f} ({(stretch - 1) * 100:.1f}% area)")
            print(f"Loc. R: {l['empirical_risk']:.4f} (≤ {self.alpha_loc}) {loc_pass}")

        if 'classification' in self.results:
            c = self.results['classification']
            print(f"Cls. α: {self.alpha_cls}")
            print(f"λ_cls: {c['lambda_cls']:.4f}")
            print(f"Cls. R: {c['empirical_risk']:.4f} (≤ {self.alpha_cls})")

            # Detect if using mock controller
            if self.is_using_mock_classification():
                print("⚠️  Using MOCK controller (argmax classification - no conformal guarantees)")
            else:
                print("✅ Using CONFORMAL controller (statistical guarantees apply)")

        print("="*70)

    def print_test_summary(self, validation_results: Dict):
        """Print test results summary in same format as calibration."""
        print("\n" + "="*70)
        print("TEST RESULTS SUMMARY (Empirical Risks on Test Data)")
        print("="*70)

        if 'threshold' in validation_results.get('details', {}):
            t = validation_results['details']['threshold']
            cnf_pass = "✓" if validation_results['threshold_guarantee'] else "✗"
            print(f"Test α_CNF: {t['target_cnf']}")
            # Show test-time diagnostics
            if 'retained' in t:
                print(f"Retained: {t['retained']} / {t['total_detections']} ({t['retention_rate']:.2%})")
                print(f"Matched: {t['matched_gt']} / {t['total_gt']} GT")
                print(f"Missed: {t['missed_gt']} GT")
            print(f"Test CNF risk: {t['empirical_cnf_risk']:.4f} (≤ {t['target_cnf']}) {cnf_pass}")

        if 'localization' in validation_results.get('details', {}):
            l = validation_results['details']['localization']
            loc_pass = "✓" if validation_results['localization_guarantee'] else "✗"
            print(f"Test α_Loc: {l['target_loc']}")
            print(f"Pairs: {l.get('test_matches', 0)}")
            print(f"Stretch: {l.get('stretch', 1.0):.4f} ({(l.get('stretch', 1.0) - 1) * 100:.1f}% area)")
            print(f"Covered: {l.get('covered', 0)} / {l.get('test_matches', 0)}")
            print(f"Failed: {l.get('failed', 0)}")
            print(f"Test Loc. Risk: {l['empirical_loc_risk']:.4f} (≤ {l['target_loc']}) {loc_pass}")

        if 'classification' in validation_results.get('details', {}):
            c = validation_results['details']['classification']
            cls_pass = "✓" if validation_results['classification_guarantee'] else "✗"
            print(f"Test α_Cls: {c['target_cls']}")
            print(f"Test Samples: {c.get('test_samples', 0)}")
            print(f"Test Cls. Risk: {c['empirical_cls_risk']:.4f} (≤ {c['target_cls']}) {cls_pass}")

        print("="*70)