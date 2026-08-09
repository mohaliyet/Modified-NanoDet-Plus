"""
Conformal Detection Pipeline with CRC for all stages
Threshold + Localization + Classification (All CRC)
"""

import numpy as np
import json
import os
import sys
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Add parent directory to path to import nanodet package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utilities from conformal package
from nanodet.conformal import (
    load_pred_gt_data, find_true_class, find_true_match, ConformalEvaluationCRC,
    ConformalThresholdingCRC, ConformalLocalizationCRC,
    ConformalClassificationCRC,
)

class ConformalDetectionCRC:
    """Unified conformal detection pipeline with CRC for all three stages.

    Calibrate by calling run_threshold_stage, run_localization_stage, and
    run_classification_stage in sequence (or run_pipeline which combines all
    three), then call validate_results on held-out data to audit the guarantees.
    """

    def __init__(
        self,
        calibration_data: List[Dict],
        alpha_cnf: float = 0.05,
        alpha_loc: float = 0.05,
        alpha_cls: float = 0.05,
        required_coverage: float = 0.95,
        nonconformity_type: str = 'margin',
        mode: str = 'instance',
        miss_rate: float = 0.1,
        lambda_mode: str = 'quantile',
    ):
        self.calibration_data = calibration_data
        self.alpha_cnf = alpha_cnf
        self.alpha_loc = alpha_loc
        self.alpha_cls = alpha_cls
        self.required_coverage = required_coverage
        self.nonconformity_type = nonconformity_type
        self.mode = mode
        self.miss_rate = miss_rate
        self.lambda_mode = lambda_mode

        # Controllers — populated by each run_*_stage call
        self.threshold_controller: Optional[ConformalThresholdingCRC] = None
        self.localization_controller: Optional[ConformalLocalizationCRC] = None
        self.classification_controller: Optional[ConformalClassificationCRC] = None

        # Fitted parameters — None until the corresponding stage runs
        self.confidence_threshold: Optional[float] = None
        self.lambda_localization: Optional[float] = None

        # Stage result dicts
        self.results: Dict = {}
    
    def run_threshold_stage(self, verbose: bool = True) -> float:
        """Run confidence threshold selection with CRC (CNF-only)."""
        self.threshold_controller = ConformalThresholdingCRC(
            self.calibration_data,
            alpha_cnf=self.alpha_cnf,
            mode=self.mode,
            miss_rate=self.miss_rate,
            lambda_mode=self.lambda_mode,
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
                    true_class = find_true_class(det_bbox, true_boxes)

                    # If no match (false positive), assign to background class
                    if true_class is None:
                        true_class = len(det.get('softmax_scores', [])) - 1  # Background class

                    # Normalize softmax scores (assume already probabilities)
                    scores = det.get('softmax_scores', [])
                    # No softmax application needed

                    classification_calibration.append({
                        'softmax_scores': scores,
                        'true_class': true_class
                    })

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

        self.results.update({'classification': classification_results})

    def validate_results(self, test_data: List[Dict]) -> Dict[str, bool]:
        """Validate that conformal guarantees are met on test data.

        Delegates to each stage controller's validate() method.
        """
        validation_results = {
            'threshold_guarantee': True,
            'localization_guarantee': True,
            'classification_guarantee': True,
            'details': {}
        }

        if self.confidence_threshold is None or self.lambda_localization is None:
            validation_results['threshold_guarantee'] = False
            validation_results['localization_guarantee'] = False
            validation_results['details']['error'] = 'Stages not run'
            return validation_results

        # --- Threshold stage ---
        thr_result = self.threshold_controller.validate(
            test_data, self.confidence_threshold, self.alpha_cnf
        )
        validation_results['details']['threshold'] = {
            'empirical_cnf_risk': thr_result['empirical_cnf_risk'],
            'target_cnf': self.alpha_cnf,
        }
        if not thr_result['passed']:
            validation_results['threshold_guarantee'] = False

        # --- Localization stage ---
        if hasattr(self, 'localization_controller') and self.localization_controller:
            loc_result = self.localization_controller.validate(
                test_data, self.lambda_localization, self.alpha_loc
            )
            validation_results['details']['localization'] = {
                'empirical_loc_risk': loc_result['empirical_loc_risk'],
                'target_loc': self.alpha_loc,
                'test_matches': loc_result['n_matches'],
            }
            if not loc_result['passed']:
                validation_results['localization_guarantee'] = False

            # Spot-check bbox validity on first 10 samples:
            # flag only truly degenerate boxes (zero or inverted area).
            # Negative coordinates are acceptable — _expand_box has no image
            # size, so border boxes naturally go negative; they are clipped
            # to image boundaries when applied in run_pipeline.
            for sample in test_data[:10]:
                for det in sample.get('detections', []):
                    if det.get('confidence', 0) >= self.confidence_threshold:
                        original_bbox = det.get('bbox', [])
                        if len(original_bbox) == 4:
                            expanded = self.localization_controller._expand_box(
                                original_bbox, self.lambda_localization
                            )
                            if len(expanded) != 4 or expanded[0] >= expanded[2] or expanded[1] >= expanded[3]:
                                validation_results['localization_guarantee'] = False
                                break

        # --- Classification stage ---
        if hasattr(self, 'classification_controller') and self.classification_controller:
            test_cls_data = []
            for sample in test_data:
                for det in sample.get('detections', []):
                    if det.get('confidence', 0) >= self.confidence_threshold:
                        true_class = find_true_class(
                            det.get('bbox', []), sample.get('true_boxes', [])
                        )
                        if true_class is not None:
                            scores = det.get('softmax_scores', [])
                            if scores:
                                test_cls_data.append({
                                    'softmax_scores': scores,
                                    'true_class': true_class,
                                })
            cls_result = self.classification_controller.validate(test_cls_data, self.alpha_cls)
            validation_results['details']['classification'] = {
                'empirical_cls_risk': cls_result['empirical_cls_risk'],
                'target_cls': self.alpha_cls,
                'test_samples': cls_result['n_samples'],
            }
            if not cls_result['passed']:
                validation_results['classification_guarantee'] = False

        return validation_results
    
    def apply(self, test_data: List[Dict], verbose: bool = True) -> List[Dict]:
        """Apply the already-fitted pipeline to test_data.

        All three stages must have been fitted (via run_threshold_stage,
        run_localization_stage, run_classification_stage, or run_pipeline)
        before calling this method.

        Returns a list of processed samples, each with a 'conformal_detections'
        key containing expanded boxes and conformal prediction sets.
        """
        if verbose:
            print("Applying fitted pipeline to test data...")

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
                    true_class, gt_bbox = find_true_match(det.get('bbox', []), sample.get('true_boxes', []))

                    # Expand box using fitted lambda_localization, clipped to image bounds
                    expanded_box = self.localization_controller._expand_box(
                        det.get('bbox', [0, 0, 0, 0]),
                        self.lambda_localization,
                        img_width=sample.get('width'),
                        img_height=sample.get('height'),
                    )

                    # Validate expanded box is still geometrically valid
                    if len(expanded_box) != 4 or expanded_box[0] >= expanded_box[2] or expanded_box[1] >= expanded_box[3]:
                        # Skip invalid expanded boxes (degenerate after boundary clipping)
                        continue

                    # Apply classification
                    pred_set = self.classification_controller.predict_set(
                        np.array(det.get('softmax_scores', []))
                    )

                    # If prediction set contains only non-main classes, assign to background
                    background_class = len(det.get('softmax_scores', [])) - 1  # Last class is background
                    main_classes = set(range(background_class))  # All classes except background

                    # Check if all classes in prediction set are non-main classes
                    if pred_set and not any(cls in main_classes for cls in pred_set):
                        pred_set = [background_class]

                    processed_sample['conformal_detections'].append({
                        'gt_bbox'         : gt_bbox,
                        'original_bbox'   : det.get('bbox', []),
                        'bbox'            : expanded_box,
                        'conformal_set'   : pred_set,
                        'true_class'      : true_class,
                        'original_confidence': det.get('confidence', 0),
                        'expanded'        : True
                    })

            conformal_predictions.append(processed_sample)

        if verbose:
            print(f"✓ Processed {len(conformal_predictions)} test images")
            print(f"✓ Total conformal detections: {sum(len(s['conformal_detections']) for s in conformal_predictions)}")

        return conformal_predictions

    def run_pipeline(self, test_data: List[Dict], verbose: bool = True) -> List[Dict]:
        """Calibrate all three stages then apply them to test_data.

        If you have already fitted each stage individually (run_threshold_stage,
        run_localization_stage, run_classification_stage), call apply() directly
        instead to skip redundant re-fitting.

        Returns a list of processed samples, each with a 'conformal_detections'
        key containing expanded boxes and conformal prediction sets.
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

        return self.apply(test_data, verbose=verbose)

    def save_results(self, output_dir: str = 'conformal_results_crc'):
        """Save pipeline results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results summary
        with open(os.path.join(output_dir, 'pipeline_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"✓ Results saved to {output_dir}/pipeline_results.json")
    
    def print_summary(self):
        """Print calibration summary."""
        w = 70
        print("\n" + "=" * w)
        print("CALIBRATION SUMMARY (Conformal Risk Control)")
        print("=" * w)

        if 'threshold' in self.results:
            t = self.results['threshold']
            cnf_risk = t.get('empirical_cnf_risk', float('nan'))
            print(f"  [Threshold]      alpha={self.alpha_cnf}  "
                  f"lambda={t['lambda_opt']:.4f}  "
                  f"threshold={t['confidence_threshold']:.4f}  "
                  f"CNF risk={cnf_risk:.4f}")

        if 'localization' in self.results:
            loc = self.results['localization']
            area_pct = ((1 + loc['lambda_localization']) ** 2 - 1) * 100
            print(f"  [Localization]   alpha={self.alpha_loc}  "
                  f"lambda_loc={loc['lambda_localization']:.4f}  "
                  f"loc risk={loc['empirical_risk']:.4f}  "
                  f"area increase={area_pct:.1f}%")

        if 'classification' in self.results:
            cls = self.results['classification']
            print(f"  [Classification] alpha={self.alpha_cls}  "
                  f"lambda_cls={cls['lambda_cls']:.4f}  "
                  f"cls risk={cls['empirical_risk']:.4f}")

        print("=" * w)
    
    def print_test_summary(self, validation_results: Dict):
        """Print test-set validation summary."""
        w = 70
        print("\n" + "=" * w)
        print("VALIDATION SUMMARY (Empirical Risks on Test Data)")
        print("=" * w)

        details = validation_results.get('details', {})

        if 'threshold' in details:
            t = details['threshold']
            status = 'OK' if validation_results['threshold_guarantee'] else 'FAIL'
            print(f"  [Threshold]      alpha={t['target_cnf']}  "
                  f"CNF risk={t['empirical_cnf_risk']:.4f}  {status}")

        if 'localization' in details:
            loc = details['localization']
            status = 'OK' if validation_results['localization_guarantee'] else 'FAIL'
            print(f"  [Localization]   alpha={loc['target_loc']}  "
                  f"loc risk={loc['empirical_loc_risk']:.4f}  "
                  f"matches={loc.get('test_matches', 0)}  {status}")

        if 'classification' in details:
            cls = details['classification']
            status = 'OK' if validation_results['classification_guarantee'] else 'FAIL'
            print(f"  [Classification] alpha={cls['target_cls']}  "
                  f"cls risk={cls['empirical_cls_risk']:.4f}  "
                  f"samples={cls.get('test_samples', 0)}  {status}")

        print("=" * w)


def production_calibration(calibration_data: List[Dict], clinical_requirements: Dict) -> Tuple['ConformalDetectionCRC', Dict]:
    """Calibrate all three conformal stages from a clinical requirements dict.

    Returns (detector, {'alpha_cnf': ...}) where detector is ready to call
    validate_results or run_pipeline on test data.
    """
    alpha_cnf = clinical_requirements.get('alpha_cnf', 0.05)
    print(f"Calibrating with alpha_cnf={alpha_cnf}")

    detector = ConformalDetectionCRC(
        calibration_data=calibration_data,
        alpha_cnf=alpha_cnf,
        alpha_loc=clinical_requirements.get('alpha_loc', 0.05),
        alpha_cls=clinical_requirements.get('alpha_cls', 0.10),
        required_coverage=clinical_requirements.get('required_coverage', 0.95),
        nonconformity_type=clinical_requirements.get('nonconformity_type', 'margin'),
        mode=clinical_requirements.get('mode', 'instance'),
        miss_rate=clinical_requirements.get('miss_rate', 0.1),
        lambda_mode=clinical_requirements.get('lambda_mode', 'quantile')
    )
    
    return detector, {'alpha_cnf': alpha_cnf}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Conformal Detection with CRC")
    parser.add_argument(
        "--cal_pred_json",
        required=True,
        help="Path to calibration predictions JSON"
    )
    parser.add_argument(
        "--cal_gt_json",
        required=True,
        help="Path to calibration ground truth COCO JSON"
    )
    parser.add_argument(
        "--test_pred_json",
        help="Path to test predictions JSON (optional, defaults to calibration predictions)"
    )
    parser.add_argument(
        "--test_gt_json",
        help="Path to test ground truth COCO JSON (optional, defaults to calibration ground truth)"
    )
    parser.add_argument(
        "--alpha_cnf",
        type=float,
        default=0.05,
        help="False Negative risk tolerance for threshold selection"
    )
    parser.add_argument(
        "--alpha_loc",
        type=float,
        default=0.05,
        help="Localization risk tolerance for box expansion"
    )
    parser.add_argument(
        "--alpha_cls",
        type=float,
        default=0.10,
        help="Classification risk tolerance"
    )
    parser.add_argument(
        "--output_dir",
        default="conformal_results_crc",
        help="Directory to save results"
    )
    parser.add_argument(
        "--mode",
        choices=['instance', 'image', 'relative-miss-rate'],
        default='instance',
        help="Risk calculation mode for threshold selection ('instance', 'image', or 'relative-miss-rate')"
    )
    parser.add_argument(
        "--miss-rate",
        type=float,
        default=0.1,
        help="Relative miss rate threshold for 'relative-miss-rate' mode (default: 0.1)"
    )
    parser.add_argument(
        "--lambda_mode",
        choices=['quantile', 'direct-inverted'],
        default='quantile',
        help="Lambda parameterisation for threshold: 'quantile' (default) maps lambda "
             "via sorted score quantile (non-increasing CNF risk); "
             "'direct-inverted' uses threshold = 1 - lambda (non-increasing CNF risk, "
             "lambda=0 -> threshold=1, lambda=1 -> threshold=0)"
    )
    parser.add_argument(
        "--lambda_min",
        type=float,
        default=0.0,
        help="Minimum expansion for localization"
    )
    parser.add_argument(
        "--lambda_max",
        type=float,
        default=1.0,
        help="Maximum expansion for localization"
    )
    parser.add_argument(
        "--required_coverage",
        type=float,
        default=0.95,
        help="Required pixel coverage for localization (e.g., 0.95 for 95 percent)"
    )
    parser.add_argument(
        "--nonconformity_type",
        choices=['credibility', 'margin', 'hinge', 'ratio', 'nll', 'brier'],
        default='margin',
        help="Nonconformity function for classification conformal prediction"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load calibration data
    print("Loading calibration data...")
    calibration_data = load_pred_gt_data(args.cal_pred_json, args.cal_gt_json)
    calibration_data = list(calibration_data.values())

    # Load test data (use calibration data if not specified)
    print("Loading test data...")
    if args.test_pred_json and args.test_gt_json:
        test_data = load_pred_gt_data(args.test_pred_json, args.test_gt_json)
        test_data = list(test_data.values())
    else:
        test_data = calibration_data

    print(f"Calibration: {len(calibration_data)} images, {sum(len(s['detections']) for s in calibration_data)} detections")
    print(f"Test: {len(test_data)} images, {sum(len(s['detections']) for s in test_data)} detections")

    # Use production calibration pipeline
    print("\n" + "=" * 70)
    print("CONFORMAL CALIBRATION")
    print("=" * 70)

    clinical_requirements = {
        'alpha_cnf': args.alpha_cnf,
        'alpha_loc': args.alpha_loc,
        'alpha_cls': args.alpha_cls,
        'required_coverage': args.required_coverage,
        'nonconformity_type': args.nonconformity_type,
        'mode': args.mode,
        'miss_rate': args.miss_rate,
        'lambda_mode': args.lambda_mode,
    }

    detector, actual_constraints = production_calibration(calibration_data, clinical_requirements)
    print(f"Fitted alpha_cnf = {actual_constraints['alpha_cnf']}")

    # Run full conformal detection pipeline
    print("\n" + "="*70)
    print("CONFORMAL DETECTION PIPELINE (CRC for All Stages)")
    print("="*70)

    try:
        # Run pipeline on test data
        conformal_predictions = detector.run_pipeline(
            test_data=test_data,
            verbose=True
        )

        detector.print_summary()

        # ── Test-time detection breakdown ────────────────────────────────────
        total_raw = total_retained = total_matched = total_missed_gt = total_gt = 0
        for sample in test_data:
            dets = sample.get('detections', [])
            true_boxes = sample.get('true_boxes', [])
            total_raw += len(dets)
            total_gt  += len(true_boxes)
            retained = [d for d in dets
                        if d.get('confidence', 0) >= detector.confidence_threshold]
            total_retained += len(retained)
            matched_gt = set()
            for det in retained:
                tc = find_true_class(det.get('bbox', []), true_boxes)
                if tc is not None:
                    total_matched += 1
                    for gi in range(len(true_boxes)):
                        if gi not in matched_gt:
                            matched_gt.add(gi)
                            break
            total_missed_gt += len(true_boxes) - len(matched_gt)
        fp = total_retained - total_matched
        w = 70
        print("\n" + "=" * w)
        print("TEST-TIME DETECTION BREAKDOWN")
        print("=" * w)
        print(f"  Confidence threshold (tau)   : {detector.confidence_threshold:.4f}")
        print(f"  Images                       : {len(test_data)}")
        print("-" * w)
        print(f"  Ground-truth objects (total) : {total_gt}")
        print(f"  Raw detections  (all scores) : {total_raw}")
        pct_ret = total_retained / total_raw * 100 if total_raw else 0
        print(f"  Retained   (score >= tau)    : {total_retained}  ({pct_ret:.1f}% of raw)")
        print("-" * w)
        pct_tp  = total_matched / total_retained * 100 if total_retained else 0
        pct_fp  = fp / total_retained * 100 if total_retained else 0
        pct_fn  = total_missed_gt / total_gt * 100 if total_gt else 0
        print(f"  Retained -> matched GT  (TP) : {total_matched}  ({pct_tp:.1f}% of retained)")
        print(f"  Retained -> unmatched   (FP) : {fp}  ({pct_fp:.1f}% of retained)")
        print(f"  GT objects missed       (FN) : {total_missed_gt}  ({pct_fn:.1f}% of GT)")
        print("=" * w)
        # ─────────────────────────────────────────────────────────────────────

        detector.save_results(args.output_dir)

        # Save conformal predictions
        pred_path = os.path.join(args.output_dir, 'conformal_predictions.json')
        with open(pred_path, 'w') as f:
            json.dump(conformal_predictions, f, indent=2)
        print(f"\n✓ Saved conformal predictions to: {pred_path}")

        # Evaluate conformal predictions
        evaluator = ConformalEvaluationCRC(conformal_predictions)
        evaluator.print_summary()

        # Save evaluation results
        eval_path = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(eval_path, 'w') as f:
            json.dump(evaluator.get_metrics(), f, indent=2)
        print(f"✓ Saved evaluation results to: {eval_path}")

        # Validate results
        print("\nValidating conformal guarantees...")
        validation = detector.validate_results(test_data)
        
        # Print structured test results
        detector.print_test_summary(validation)
        
        if validation['threshold_guarantee'] and validation['localization_guarantee'] and validation['classification_guarantee']:
            print("All guarantees validated.")
        else:
            print("WARNING: one or more guarantees not met (see summary above).")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during pipeline execution: {e}")
        sys.exit(1)