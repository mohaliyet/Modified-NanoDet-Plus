"""
nanodet Conformal Prediction Module

This package provides conformal prediction components for medical image analysis,
ensuring statistical guarantees for false negative and false positive risks.
"""

from .utils import compute_iou, load_pred_gt_data, find_true_class, find_true_match
from .evaluation import ConformalEvaluationCRC
from .thresholding import ConformalThresholdingCRC, RiskCache
from .localization import ConformalLocalizationCRC
from .classification import ConformalClassificationCRC
from .core import ConformalDetectionCRC

__version__ = "1.0.0"
__all__ = [
    'compute_iou',
    'load_pred_gt_data',
    'find_true_class',
    'find_true_match',
    'ConformalEvaluationCRC',
    'ConformalThresholdingCRC',
    'RiskCache',
    'ConformalLocalizationCRC',
    'ConformalClassificationCRC',
    'ConformalDetectionCRC'
]