# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import warnings
from typing import Dict, Tuple

from torch.utils.data import Dataset

from .color import color_aug_and_norm
from .warp import ShapeTransform, warp_and_resize

try:
    from .torchvision_v2 import TorchvisionV2Transform
    TORCHVISION_V2_AVAILABLE = True
except ImportError:
    TORCHVISION_V2_AVAILABLE = False


class LegacyPipeline:
    def __init__(self, cfg, keep_ratio):
        warnings.warn(
            "Deprecated warning! Pipeline from nanodet v0.x has been deprecated,"
            "Please use new Pipeline and update your config!"
        )
        self.warp = functools.partial(
            warp_and_resize, warp_kwargs=cfg, keep_ratio=keep_ratio
        )
        self.color = functools.partial(color_aug_and_norm, kwargs=cfg)

    def __call__(self, meta, dst_shape):
        meta = self.warp(meta, dst_shape=dst_shape)
        meta = self.color(meta=meta)
        return meta


class Pipeline:
    """Data process pipeline. Apply augmentation and pre-processing on
    meta_data from dataset.
    
    Supports both legacy transforms and modern torchvision v2 transforms.
    Set 'use_torchvision_v2: true' in config to use the new pipeline.

    Args:
        cfg (Dict): Data pipeline config.
        keep_ratio (bool): Whether to keep aspect ratio when resizing image.

    """

    def __init__(self, cfg: Dict, keep_ratio: bool):
        
        # print(f"DEBUG Pipeline - use_torchvision_v2: {cfg.get('use_torchvision_v2')}")  # Add this
        # print(f"DEBUG Pipeline - TORCHVISION_V2_AVAILABLE: {TORCHVISION_V2_AVAILABLE}")  # Add this
        
        self.use_torchvision_v2 = cfg.get('use_torchvision_v2', True)  # Default to v2
        
        if self.use_torchvision_v2:
            if not TORCHVISION_V2_AVAILABLE:
                warnings.warn(
                    "torchvision v2 transforms requested but not available. "
                    "Falling back to legacy transforms. "
                    "Install torchvision>=0.15.0 to use v2 transforms."
                )
                self.use_torchvision_v2 = False
            else:
                # Use modern torchvision v2 pipeline
                self.transform = TorchvisionV2Transform(cfg, keep_ratio)

        if not self.use_torchvision_v2:
            # Use legacy pipeline
            self.shape_transform = ShapeTransform(keep_ratio, **cfg)
            self.color = functools.partial(color_aug_and_norm, kwargs=cfg)

    def __call__(self, dataset: Dataset, meta: Dict, dst_shape: Tuple[int, int]):
        if self.use_torchvision_v2:
            # Modern torchvision v2 path
            meta = self.transform(dataset, meta, dst_shape)
            
            return meta
        else:
            # Legacy path (backward compatible)
            meta = self.shape_transform(meta, dst_shape=dst_shape)
            meta = self.color(meta=meta)
            return meta
