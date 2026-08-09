"""
Conformal Prediction Evaluation Module

Provides evaluation metrics for conformal prediction results including
marginal coverage, set sizes, and prediction quality metrics.
"""

from typing import List, Dict


class ConformalEvaluationCRC:
    """
    Evaluation class for conformal prediction results using CRC.

    Computes key metrics:
    - Marginal coverage: fraction of true labels covered by prediction sets
    - Average set size: mean size of prediction sets
    - Singleton sets: fraction of sets containing exactly one class
    - Empty sets: fraction of sets containing no classes
    """

    def __init__(self, conformal_predictions: List[Dict]):
        """Initialize evaluator with conformal prediction results.

        Args:
            conformal_predictions: List of samples with conformal_detections
        """
        self.conformal_predictions = conformal_predictions
        self._compute_metrics()

    def _compute_metrics(self):
        """Compute all evaluation metrics."""
        total_detections = 0
        covered_detections = 0
        total_labeled_detections = 0  # Detections with known true labels
        set_sizes = []
        singleton_count = 0
        empty_count = 0

        for sample in self.conformal_predictions:
            detections = sample.get('conformal_detections', [])

            for det in detections:
                total_detections += 1
                conformal_set = det.get('conformal_set', [])
                true_class = det.get('true_class')

                # Marginal coverage - only consider detections with known true labels
                if true_class is not None:
                    total_labeled_detections += 1
                    if true_class in conformal_set:
                        covered_detections += 1

                # Set size
                set_sizes.append(len(conformal_set))

                # Singleton and empty sets
                if len(conformal_set) == 0:
                    empty_count += 1
                elif len(conformal_set) == 1:
                    singleton_count += 1

        # Compute metrics
        self.marginal_coverage = covered_detections / total_labeled_detections if total_labeled_detections > 0 else 0.0
        self.average_set_size = sum(set_sizes) / len(set_sizes) if set_sizes else 0.0
        self.singleton_fraction = singleton_count / total_detections if total_detections > 0 else 0.0
        self.empty_fraction = empty_count / total_detections if total_detections > 0 else 0.0

        # Store raw counts
        self.total_detections = total_detections
        self.covered_detections = covered_detections
        self.singleton_count = singleton_count
        self.empty_count = empty_count

    def get_metrics(self) -> Dict[str, float]:
        """Return all computed metrics.

        Returns:
            Dictionary containing all evaluation metrics
        """
        return {
            'marginal_coverage': self.marginal_coverage,
            'average_set_size': self.average_set_size,
            'singleton_fraction': self.singleton_fraction,
            'empty_fraction': self.empty_fraction,
            'total_detections': self.total_detections,
            'covered_detections': self.covered_detections,
            'singleton_count': self.singleton_count,
            'empty_count': self.empty_count
        }

    def print_summary(self):
        """Print evaluation summary."""
        print("\n" + "="*70)
        print("CONFORMAL PREDICTION EVALUATION (CRC)")
        print("="*70)
        print(f"Total Detections: {self.total_detections}")
        print(f"Marginal Coverage: {self.marginal_coverage:.4f}")
        print(f"Average Set Size: {self.average_set_size:.2f}")
        print(f"Singleton Sets: {self.singleton_count} ({self.singleton_fraction:.4f})")
        print(f"Empty Sets: {self.empty_count} ({self.empty_fraction:.4f})")
        print("="*70)