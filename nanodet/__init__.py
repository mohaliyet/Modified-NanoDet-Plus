
"""
NanoDet package initialization.

Imports public metadata from __about__.py and exposes it at the package level.
Provides fallback values if __about__.py is missing.
"""

import os

try:
    from .__about__ import (
        __version__,
        __author__,
        __license__,
        __homepage__,
        __description__,
    )
except ImportError:
    # Fallback metadata (should rarely be needed)
    __version__ = "1.0.0"
    __author__ = "NanoDet Contributors"
    __license__ = "Apache-2.0"
    __homepage__ = "https://github.com/Para-Sight/Modified-NanoDet-Plus"
    __description__ = "NanoDet - A Lightweight Object Detection Designed for Edge Devices"

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "__homepage__",
    "__description__",
]

# Internal paths (optional, for package use)
_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)
