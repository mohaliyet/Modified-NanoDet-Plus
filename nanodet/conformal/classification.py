"""Classification conformal risk control module.

This module provides conformal risk control for classification tasks,
ensuring statistical guarantees on misclassification risk through prediction sets.
"""

import numpy as np
from typing import Dict, List


class ConformalClassificationCRC:
    """Classification controller using Conformal Risk Control (CRC).

    Calibrates a nonconformity threshold lambda such that the empirical
    misclassification risk on calibration data satisfies the given alpha
    tolerance.  At inference time, ``predict_set`` returns all class indices
    whose nonconformity score falls at or below lambda.

    Supported nonconformity types: ``'credibility'``, ``'margin'``,
    ``'hinge'``, ``'ratio'``, ``'nll'``, ``'brier'``.
    """

    def __init__(
        self,
        calibration_data: List[Dict],
        alpha: float = 0.05,
        nonconformity_type: str = 'credibility',
        verbose: bool = True,
    ):
        self.alpha = alpha
        self.n_calibration = len(calibration_data)
        self.nonconformity_type = nonconformity_type
        self.verbose = verbose

        # Determine number of classes (foreground + background)
        if calibration_data:
            self.num_classes = len(calibration_data[0]['softmax_scores'])
            self.background_class = self.num_classes - 1  # Last class is background
        else:
            self.num_classes = 0
            self.background_class = -1

        # Compute nonconformity scores for all samples
        self.calibration_scores = []
        for sample in calibration_data:
            scores = sample.get('softmax_scores', [])
            if not scores or len(scores) == 0:
                continue  # Skip samples with empty scores
            true_class = sample.get('true_class', 0)
            if true_class >= len(scores):
                continue  # Skip if true_class is out of bounds
            score = self._compute_nonconformity(scores, true_class)
            self.calibration_scores.append(score)

        # Optimize λ using binary search
        self.lambda_cls = self._optimize_lambda()
        self.empirical_risk = self._compute_risk(self.lambda_cls)

        if self.verbose and self.calibration_scores:
            print(f"✓ Classification λ: {self.lambda_cls:.4f}")
            print(f"✓ Misclassification Risk: {self.empirical_risk:.4f} (≤ {alpha})")

    def _compute_nonconformity(self, softmax_scores: List[float], true_class: int) -> float:
        """Compute the nonconformity score for a single sample.

        Args:
            softmax_scores: Per-class probability vector (must sum to ~1).
            true_class: Index of the ground-truth class.

        Returns:
            Scalar nonconformity score; higher means less conforming.
        """
        if self.nonconformity_type == 'credibility':
            return 1 - softmax_scores[true_class]
        elif self.nonconformity_type == 'margin':
            max_other = max(softmax_scores[j] for j in range(len(softmax_scores)) if j != true_class)
            margin = softmax_scores[true_class] - max_other
            return 1 - margin
        elif self.nonconformity_type == 'hinge':
            max_other = max(softmax_scores[j] for j in range(len(softmax_scores)) if j != true_class)
            margin = softmax_scores[true_class] - max_other
            return max(0.0, 1.0 - margin)
        elif self.nonconformity_type == 'ratio':
            p_true = softmax_scores[true_class]
            max_other = max(softmax_scores[j] for j in range(len(softmax_scores)) if j != true_class)
            return max_other / p_true if p_true > 0 else float('inf')
        elif self.nonconformity_type == 'nll':
            p_true = softmax_scores[true_class]
            return -np.log(p_true) if p_true > 0 else float('inf')
        elif self.nonconformity_type == 'brier':
            # Brier score: (1/K) * ||e_true - f(X)||_2^2
            k = len(softmax_scores)
            brier = 0.0
            for j in range(k):
                diff = (1.0 if j == true_class else 0.0) - softmax_scores[j]
                brier += diff ** 2
            return brier / k
        else:
            raise ValueError(f"Unknown nonconformity_type: {self.nonconformity_type}")

    def _compute_risk(self, lambda_val: float) -> float:
        """Compute empirical risk for a given λ with finite-sample correction."""
        if not self.calibration_scores:
            return 0.0
        n = len(self.calibration_scores)
        failed = sum(1 for score in self.calibration_scores if score > lambda_val)
        # Finite-sample correction for conformal guarantee
        return (failed + 1) / (n + 1)

    def _optimize_lambda(self) -> float:
        """Find the minimal λ such that risk(λ) ≤ α using binary search."""
        if not self.calibration_scores:
            return 1.0

        scores = sorted(self.calibration_scores)
        low, high = 0.0, max(scores) + 1.0

        # Binary search for the smallest λ where risk ≤ α
        while high - low > 1e-6:
            mid = (low + high) / 2
            risk = self._compute_risk(mid)
            if risk <= self.alpha:
                high = mid
            else:
                low = mid

        return high

    def predict_set(self, softmax_scores: List[float]) -> List[int]:
        """Generate prediction set using the optimized λ."""
        # Handle empty or invalid input
        if softmax_scores is None or len(softmax_scores) == 0:
            return []
        
        # Convert to list if numpy array
        if hasattr(softmax_scores, 'tolist'):
            softmax_scores = softmax_scores.tolist()

        prediction_set = []
        for c in range(len(softmax_scores)):
            score = self._compute_nonconformity(softmax_scores, c)
            if score <= self.lambda_cls:
                prediction_set.append(c)
        return prediction_set

    def validate(self, classification_test_data: List[Dict], alpha_cls: float) -> Dict:
        """Validate classification risk guarantee on held-out test data.

        Args:
            classification_test_data: List of dicts with 'softmax_scores' and 'true_class'
            alpha_cls: Classification risk tolerance to validate against

        Returns:
            Dict with keys 'passed', 'empirical_cls_risk', 'target_alpha', 'n_samples'
        """
        if not classification_test_data:
            return {
                'passed': False,
                'empirical_cls_risk': 1.0,
                'target_alpha': alpha_cls,
                'n_samples': 0,
            }

        errors = 0
        for sample in classification_test_data:
            pred_set = self.predict_set(np.array(sample['softmax_scores']))
            if sample['true_class'] not in pred_set:
                errors += 1

        empirical_cls_risk = errors / len(classification_test_data)
        return {
            'passed': empirical_cls_risk <= alpha_cls + 1e-3,
            'empirical_cls_risk': empirical_cls_risk,
            'target_alpha': alpha_cls,
            'n_samples': len(classification_test_data),
        }