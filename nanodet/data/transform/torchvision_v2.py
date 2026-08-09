# Copyright 2025 NanoDet Contributors.
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

"""Torchvision v2 transforms for object detection with bbox support."""

import random
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as T
from torchvision.tv_tensors import BoundingBoxes, Image
from .warp import warp_and_resize


class TorchvisionV2Transform:
    """Modern transform pipeline using torchvision v2 with bbox support.
    
    This provides a drop-in replacement for the legacy transform pipeline,
    using torchvision v2 which has native support for bounding boxes.
    
    Args:
        cfg (Dict): Configuration dictionary with augmentation parameters.
        keep_ratio (bool): Whether to keep aspect ratio when resizing.

    Example cfg for mosaic batch augmentation:
        cfg = {
            # ...existing keys...
            "mosaic": {
                "enabled": True,  # turn mosaic on/off
                "rows": 2,        # mosaic grid rows  (e.g. 2, 3)
                "cols": 3,        # mosaic grid cols  (e.g. 2, 3)
                "p": 0.5,         # probability per batch to use mosaic
            },
        }
    """
    
    def __init__(self, cfg: Dict, keep_ratio: bool = True):
        self.cfg = cfg
        self.keep_ratio = keep_ratio
        
        # Build geometric transforms (bbox-safe)
        # NOTE: Geometric transforms are now handled by the legacy warp_and_resize
        # to ensure correct warp_matrix calculation.
        # Build geometric transforms (bbox-safe)
        # geometric_transforms = []
        # if cfg.get('clamp_bounding_boxes', True):  # Default to True
        #     geometric_transforms.append(T.ClampBoundingBoxes())
        # self.geometric = T.Compose(geometric_transforms) if geometric_transforms else T.Identity()
        

        # Build color transforms (image-only)
        self.color = self._build_color_transforms(cfg)
        
        # Build quality transforms (image-only)
        self.quality = self._build_quality_transforms(cfg)
        
        # Medical Augmentation flags 
        self.use_microscope_artifacts = cfg.get('microscope_artifacts', False)
        self.microscope_artifacts_p = cfg.get('microscope_artifacts_p', {})
        self.use_staining_variation = cfg.get('staining_variation', False)
        self.staining_variation_p = cfg.get('staining_variation_p', {})
        self.use_illumination_variation = cfg.get('illumination_variation', False)
        self.illumination_variation_p = cfg.get('illumination_variation_p', {})
        
        # Custom Augmentation flags
        self.use_custom_augmentation = cfg.get('custom_augmentation', False)
        self.custom_augmentation_p = cfg.get('custom_augmentation_p', {})   
        
        # Normalization
        normalize_params = cfg.get('normalize', [[103.53, 116.28, 123.675], 
                                                   [57.375, 57.12, 58.395]])
        # Convert BGR mean/std to RGB for torchvision
        # These values are in [0, 255] range, need to normalize to [0, 1] for use with float images
        mean_bgr = normalize_params[0]
        std_bgr = normalize_params[1]
        self.mean = [mean_bgr[2] / 255.0, mean_bgr[1] / 255.0, mean_bgr[0] / 255.0]
        self.std = [std_bgr[2] / 255.0, std_bgr[1] / 255.0, std_bgr[0] / 255.0]
        
        # Build bbox-aware transforms
        self.bbox_transforms = self._build_bbox_transforms(cfg)

        # Mosaic augmentation config (batch-level)
        mosaic_cfg = cfg.get("mosaic", {})
        if isinstance(mosaic_cfg, bool):
            mosaic_cfg = {"enabled": mosaic_cfg}
        self.mosaic_enabled = mosaic_cfg.get("enabled", False)
        self.mosaic_rows = mosaic_cfg.get("rows", 2)
        self.mosaic_cols = mosaic_cfg.get("cols", 2)
        self.mosaic_p = mosaic_cfg.get("p", 0.0)
    
    def _append_random_apply(self, transforms, transform, cfg: Dict, prob_key: str, default: float = 1.0):
        p = cfg.get(prob_key, default)
        if p > 0:
            transforms.append(T.RandomApply([transform], p=p))

    def _build_geometric_transforms(self, cfg: Dict) -> T.Compose:
        return T.Identity()
    
    def _build_color_transforms(self, cfg: Dict) -> T.Compose:
        transforms = []
        
        if any(k in cfg for k in ['brightness', 'contrast', 'saturation', 'hue']):
            
            brightness = cfg.get('brightness', 0)
            contrast = cfg.get('contrast', [1.0, 1.0])
            saturation = cfg.get('saturation', [1.0, 1.0])
            hue = max(-0.5, min(0.5, cfg.get('hue', 0.05)))
            
            jitter_transforms = [
                (T.ColorJitter(brightness=brightness), 'brightness_p', 1.0),
                (T.ColorJitter(contrast=contrast), 'contrast_p', 1.0),
                (T.ColorJitter(saturation=saturation), 'saturation_p', 1.0),
                (T.ColorJitter(hue=hue), 'hue_p', 1.0),
            ]
            for transform, prob_key, default in jitter_transforms:
                self._append_random_apply(transforms, transform, cfg, prob_key, default)
            
        # Auto contrast (useful for microscopy)
        if cfg.get('auto_contrast', False):
            p = cfg.get('auto_contrast_p', 0.3)  # Allow custom p, default to 0.3
            transforms.append(T.RandomAutocontrast(p=p))
        
        photometric_cfg = cfg.get('photometric_distort', False)
        if photometric_cfg:
            p = cfg.get('photometric_distort_p', 1.0)  # Default to always apply
            transforms.append(T.RandomApply([T.RandomPhotometricDistort()], p=p))
        
        # Histogram equalization (useful for microscopy contrast)
        if cfg.get('random_equalize', False):
            p = cfg.get('random_equalize_p', 0.3)  # Allow custom p, default to 0.3
            transforms.append(T.RandomEqualize(p=p))
            
        # Color inversion
        if cfg.get('random_invert', False):
            p = cfg.get('random_invert_p', 0.1)  # Allow custom p, default to 0.1
            transforms.append(T.RandomInvert(p=p))
        
        return T.Compose(transforms) if transforms else T.Identity()
    
    def _build_quality_transforms(self, cfg: Dict) -> T.Compose:
        transforms = []
        
        # Gaussian blur (out of focus simulation)
        gaussian_blur_cfg = cfg.get('gaussian_blur', False)
        if gaussian_blur_cfg:
            if isinstance(gaussian_blur_cfg, dict):
                kernel_size = gaussian_blur_cfg.get('kernel_size', 5)
                sigma = gaussian_blur_cfg.get('sigma', (0.1, 2.0))
                p = gaussian_blur_cfg.get('p', 0.3)
            else:
                kernel_size = 5
                sigma = (0.1, 2.0)
                p = 0.3
            
            transforms.append(
                T.RandomApply(
                    [T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)],
                    p=p
                )
            )
        
        # Sharpness adjustment
        sharpness_cfg = cfg.get('adjust_sharpness', False)
        if sharpness_cfg:
            if isinstance(sharpness_cfg, dict):
                sharpness_factor = sharpness_cfg.get('sharpness_factor', 2.0)
                p = sharpness_cfg.get('p', 0.3)
            else:
                sharpness_factor = 2.0
                p = 0.3
            
            transforms.append(
                T.RandomAdjustSharpness(
                    sharpness_factor=sharpness_factor,
                    p=p
                )
            )
        
        # Random erasing (cutout augmentation)
        erasing_cfg = cfg.get('random_erasing', False)
        if erasing_cfg:
            if isinstance(erasing_cfg, dict):
                p = erasing_cfg.get('p', 0.3)
                scale = erasing_cfg.get('scale', (0.02, 0.1))
                ratio = erasing_cfg.get('ratio', (0.3, 3.3))
                value = erasing_cfg.get('value', 0)
            else:
                p = 0.3
                scale = (0.02, 0.1)
                ratio = (0.3, 3.3)
                value = 0
            
            transforms.append(
                T.RandomErasing(
                    p=p,
                    scale=scale,
                    ratio=ratio,
                    value=value
                )
            )
        
        # NOTE: all RandomAffine-based transforms are bbox-aware and live in _build_bbox_transforms.
        # No image-only RandomAffine here.
        
        return T.Compose(transforms) if transforms else T.Identity()
    
    
    def _build_bbox_transforms(self, cfg: Dict) -> T.Compose:
        transforms = []
        
        # cfg keys:
        #   random_iou_crop: bool | dict(...)
        #   random_zoom_out: bool | dict(...)
        #   random_affine: bool | dict(degrees, translate, scale, shear, p)
        #   flip_horizontal / flip_horizontal_p
        #   flip_vertical   / flip_vertical_p
        
        # IoU-based cropping (detection-specific)
        iou_crop_cfg = cfg.get('random_iou_crop', False)
        if iou_crop_cfg:
            if isinstance(iou_crop_cfg, dict):
                min_scale = iou_crop_cfg.get('min_scale', 0.3)
                max_scale = iou_crop_cfg.get('max_scale', 1.0)
                min_aspect_ratio = iou_crop_cfg.get('min_aspect_ratio', 0.5)
                max_aspect_ratio = iou_crop_cfg.get('max_aspect_ratio', 2.0)
                sampler_options = iou_crop_cfg.get('sampler_options', [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
            else:
                min_scale = 0.3
                max_scale = 1.0
                min_aspect_ratio = 0.5
                max_aspect_ratio = 2.0
                sampler_options = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
            
            transforms.append(
                T.RandomIoUCrop(
                    min_scale=min_scale,
                    max_scale=max_scale,
                    min_aspect_ratio=min_aspect_ratio,
                    max_aspect_ratio=max_aspect_ratio,
                    sampler_options=sampler_options
                )
            )
        
        # Zoom out augmentation (show more context)
        zoom_out_cfg = cfg.get('random_zoom_out', False)
        if zoom_out_cfg:
            if isinstance(zoom_out_cfg, dict):
                fill = zoom_out_cfg.get('fill', [0, 0, 0])
                side_range = zoom_out_cfg.get('side_range', (1.0, 4.0))
            else:
                fill = [0, 0, 0]
                side_range = (0.5, 1.0)
            
            transforms.append(
                T.RandomZoomOut(
                    fill=fill,
                    side_range=side_range
                )
            )
            
        # General bbox-aware RandomAffine (rotation / translate / scale / shear)
        random_affine_cfg = cfg.get('random_affine', False)
        if random_affine_cfg:
            if isinstance(random_affine_cfg, dict):
                degrees = random_affine_cfg.get('degrees', (-10, 10))
                translate = random_affine_cfg.get('translate', (0.02, 0.02))
                scale = random_affine_cfg.get('scale', (0.9, 1.1))
                shear = random_affine_cfg.get('shear', (-5, 5))
                p = random_affine_cfg.get('p', 0.5)
            else:
                degrees = (-10, 10)
                translate = (0.02, 0.02)
                scale = (0.9, 1.1)
                shear = (-5, 5)
                p = 0.5

            transforms.append(
                T.RandomApply(
                    [T.RandomAffine(
                        degrees=degrees,
                        translate=translate,
                        scale=scale,
                        shear=shear,
                    )],
                    p=p,
                )
            )
        
        # Add horizontal flip
        if cfg.get('flip_horizontal', False):
            p = cfg.get('flip_horizontal_p', 0.5)
            transforms.append(T.RandomHorizontalFlip(p=p))

        # Add vertical flip
        if cfg.get('flip_vertical', False):
            p = cfg.get('flip_vertical_p', 0.5)
            transforms.append(T.RandomVerticalFlip(p=p))
        
        return T.Compose(transforms) if transforms else T.Identity()
    
    
    
    # Medical Augmentations Methods
    
    def simulate_microscope_artifacts(self, img: torch.Tensor) -> torch.Tensor:
        """Simulate common microscopy artifacts (noise, blur, compression, impulse)."""
        
        if not self.use_microscope_artifacts:
            return img
        
        artifacts_cfg = self.microscope_artifacts_p  # Use the parameter dict
    
        # Gaussian noise
        if random.random() < artifacts_cfg.get('gaussian_noise_p', 0.3):
            sigma_range = artifacts_cfg.get('gaussian_sigma', [0.01, 0.05])
            sigma = random.uniform(sigma_range[0], sigma_range[1])
            img = torch.clamp(img + torch.randn_like(img) * sigma, 0, 1)
    
        # Poisson (shot) noise
        if random.random() < artifacts_cfg.get('poisson_noise_p', 0.2):
            scale_range = artifacts_cfg.get('poisson_scale', [30, 80])
            scale = random.uniform(scale_range[0], scale_range[1])
            noisy = torch.poisson(torch.clamp(img * scale, 0, None)) / scale
            img = torch.clamp(noisy, 0, 1)
    
        # Salt & pepper (impulse) noise
        if random.random() < artifacts_cfg.get('salt_pepper_p', 0.15):
            density_range = artifacts_cfg.get('salt_pepper_density', [0.001, 0.005])
            density = random.uniform(density_range[0], density_range[1])
            c, h, w = img.shape
            num = int(h * w * density)
            ys = torch.randint(0, h, (num,))
            xs = torch.randint(0, w, (num,))
            vals = torch.randint(0, 2, (num,), dtype=img.dtype, device=img.device)
            img[:, ys, xs] = vals.unsqueeze(0)
    
        # Defocus / Gaussian blur
        if random.random() < artifacts_cfg.get('defocus_blur_p', 0.25):
            kernel_sizes = artifacts_cfg.get('blur_kernel_sizes', [3, 5])
            sigma_range = artifacts_cfg.get('blur_sigma', [0.5, 1.5])
            k = random.choice(kernel_sizes)
            sigma = random.uniform(sigma_range[0], sigma_range[1])
            np_img = img.permute(1, 2, 0).cpu().numpy()
            np_img = cv2.GaussianBlur(np_img, (k, k), sigma)
            img = torch.from_numpy(np_img).to(img.device).permute(2, 0, 1)
    
        # JPEG compression artifacts
        if random.random() < artifacts_cfg.get('jpeg_compression_p', 0.15):
            quality_range = artifacts_cfg.get('jpeg_quality', [40, 85])
            quality = random.randint(quality_range[0], quality_range[1])
            np_img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            enc = cv2.imencode('.jpg', np_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])[1]
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
            img = torch.from_numpy(dec).to(img.device).permute(2, 0, 1)
    
        return torch.clamp(img, 0, 1)
    
    def simulate_staining_variation(self, img: torch.Tensor) -> torch.Tensor:
        """Simulate variations in staining intensity across samples."""
        if not self.use_staining_variation:
            return img
        staining_cfg = self.staining_variation_p
        p = staining_cfg.get('p',0.4)
        scale_range = staining_cfg.get('scale_range',[0.8,1.2])
        
        if random.random() < p:
            # Random per-channel scaling (3, 1, 1) for RGB
            scales = torch.rand(3, 1, 1) * (scale_range[1]-scale_range[0]) + scale_range[0]  
            img = img * scales.to(img.device)
            img = torch.clamp(img, 0, 1)
        
        return img
    
    def simulate_uneven_illumination(self, img: torch.Tensor) -> torch.Tensor:
        """Simulate vignetting and uneven illumination."""
        if not self.use_illumination_variation:
            return img
        
        illumination_cfg = self.illumination_variation_p
        p = illumination_cfg.get('p',0.3)
        vignette_strength = illumination_cfg.get('vignette_strength',0.2)
        
        if random.random() < p:
            _, h, w = img.shape
            y, x = torch.meshgrid(
                torch.linspace(-1, 1, h),
                torch.linspace(-1, 1, w),
                indexing='ij'
            )
            # Radial gradient for vignetting
            r = torch.sqrt(x**2 + y**2)
            vignette = 1 - vignette_strength * (r / r.max()) ** 2
            vignette = vignette.to(img.device)
            img = img * vignette.unsqueeze(0)
            img = torch.clamp(img, 0, 1)
        
        return img
    
    
    # Custom Augmentation 
    def apply_custom_augmentation(self, img: torch.Tensor) -> torch.Tensor:
        """Apply custom augmentations (user-defined)."""
        if not self.use_custom_augmentation:
            return img
        
        custom_cfg = self.custom_augmentation_p
        p = custom_cfg.get('p', 0.5)
        
        if random.random() < p:
            # Example: Random rotation (image-only, bboxes not updated)
            angle = random.uniform(-30, 30)
            np_img = img.permute(1, 2, 0).cpu().numpy()
            h, w = np_img.shape[:2]
            center = (w // 2, h // 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(np_img, rot_matrix, (w, h))
            img = torch.from_numpy(rotated).to(img.device).permute(2, 0, 1)
            img = torch.clamp(img, 0, 1)
        
        return img
    
    
    def _meta_to_torchvision(self, meta: Dict) -> Tuple[Image, BoundingBoxes, torch.Tensor]:
        """Convert NanoDet meta format to torchvision v2 format.
        
        Args:
            meta: NanoDet meta dict with 'img', 'gt_bboxes', 'gt_labels'
        
        Returns:
            Tuple of (Image tensor, BoundingBoxes, labels tensor)
        """
        # Image: numpy HWC BGR -> torch CHW RGB (keep uint8 for now)
        img = meta['img']
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1)  # Keep uint8, don't convert to float yet
        img = Image(img)
        
        # Bboxes: numpy [N, 4] (x1, y1, x2, y2) -> BoundingBoxes
        boxes = torch.from_numpy(meta['gt_bboxes'].copy()).float()
        if len(boxes) > 0:
            boxes = BoundingBoxes(
                boxes,
                format="XYXY",
                canvas_size=img.shape[-2:]
            )
        else:
            # Empty bboxes
            boxes = BoundingBoxes(
                torch.zeros((0, 4), dtype=torch.float32),
                format="XYXY",
                canvas_size=img.shape[-2:]
            )
        
        # Labels: numpy -> torch
        labels = torch.from_numpy(meta['gt_labels'].copy()).long()
        
        return img, boxes, labels
    
    def _torchvision_to_meta(
        self,
        img: Image,
        boxes: BoundingBoxes,
        labels: torch.Tensor,
        original_meta: Dict
    ) -> Dict:
        """Convert torchvision v2 format back to NanoDet meta format.
        
        Args:
            img: Image tensor (CHW RGB)
            boxes: BoundingBoxes tensor
            labels: Labels tensor
            original_meta: Original meta dict to update
        
        Returns:
            Updated meta dict
        """
        # Image: torch CHW RGB -> numpy HWC BGR
        img_np = img.permute(1, 2, 0).numpy()
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Bboxes: BoundingBoxes -> numpy [N, 4]
        if len(boxes) > 0:
            # clamp to canvas and drop invalid/degenerate boxes
            h, w = img.shape[-2:]
            clamped = boxes.clone()
            clamped[..., 0::2].clamp_(0, w - 1)  # x1, x2
            clamped[..., 1::2].clamp_(0, h - 1)  # y1, y2
            x1, y1, x2, y2 = clamped.unbind(-1)
            valid = (x2 > x1) & (y2 > y1)
            clamped = clamped[valid]
            labels = labels[valid]
            boxes_np = clamped.numpy()
        else:
            boxes_np = np.zeros((0, 4), dtype=np.float32)
            labels = labels[:0]

        # Labels: torch -> numpy
        labels_np = labels.numpy() if len(labels) > 0 else np.array([], dtype=np.int64)

        # Update meta
        original_meta['img'] = img_np
        original_meta['gt_bboxes'] = boxes_np
        original_meta['gt_labels'] = labels_np

        return original_meta

    def _transform_single(self, dataset, meta: Dict, dst_shape: Tuple[int, int]) -> Dict:
        """Apply all transforms to a single sample."""
        # --- STEP 1: Apply Bbox-Aware Transforms First ---
        # These transforms can change image size, so apply before resizing
        if len(meta['gt_bboxes']) > 0:  # Only apply if there are bounding boxes
            img, boxes, labels = self._meta_to_torchvision(meta)
            img, boxes = self.bbox_transforms(img, boxes)
            # Update meta with bbox-transformed data
            meta = self._torchvision_to_meta(img, boxes, labels, meta)

        # --- STEP 2: Apply Matrix-Based Warping (from legacy code) ---
        meta = warp_and_resize(
            meta,
            warp_kwargs=self.cfg,  # Pass the full config for geometric params
            dst_shape=dst_shape,
            keep_ratio=self.keep_ratio,
        )

        # sanitize boxes after warp_and_resize as well
        if meta['gt_bboxes'].size > 0:
            h, w = meta['img'].shape[:2]
            boxes = meta['gt_bboxes'].astype(np.float32)
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w - 1)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h - 1)
            x1, y1, x2, y2 = boxes.T
            valid = (x2 > x1) & (y2 > y1)
            meta['gt_bboxes'] = boxes[valid]
            meta['gt_labels'] = meta['gt_labels'][valid]

        # --- STEP 3: Convert to Torchvision Format ---
        img, boxes, labels = self._meta_to_torchvision(meta)

        # --- STEP 4: Convert to float [0, 1] for torchvision transforms ---
        img = img.float() / 255.0

        # --- STEP 5: Apply Color and Quality Transforms (Torchvision V2) ---
        img = self.color(img)
        img = self.quality(img)

        # Medical Augmentation Config
        if self.cfg.get('use_medical_augmentation', False):
            img = self.simulate_microscope_artifacts(img)
            img = self.simulate_staining_variation(img)
            img = self.simulate_uneven_illumination(img)

        # Custom Augmentation Config
        img = self.apply_custom_augmentation(img)

        # --- STEP 6: Normalize with ImageNet stats ---
        img = T.Normalize(mean=self.mean, std=self.std)(img)

        # --- STEP 7: Convert back to meta format ---
        img_np = img.permute(1, 2, 0).numpy()
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        meta['img'] = img_np

        # Bboxes remain unchanged (color/quality transforms don't affect bboxes)
        return meta

    def __call__(self, dataset, meta: Dict, dst_shape: Tuple[int, int]) -> Dict:
        """Apply transforms to a single meta sample (backwards-compatible API)."""
        return self._transform_single(dataset, meta, dst_shape)

    def _mosaic_augment(self, metas, dst_shape: Tuple[int, int]) -> Dict:
        """Create a single mosaic sample from a batch of metas.

        The grid is mosaic_rows x mosaic_cols, and images are resized
        to fit each cell. Bboxes and labels are translated accordingly.
        """
        if not metas:
            return None

        rows, cols = self.mosaic_rows, self.mosaic_cols
        H, W = dst_shape
        cell_h, cell_w = H // rows, W // cols

        # Canvas image + annotations
        mosaic_img = np.zeros((H, W, 3), dtype=metas[0]["img"].dtype)
        mosaic_boxes = []
        mosaic_labels = []

        # Sample indices with replacement if batch is too small
        import math, itertools
        num_cells = rows * cols
        idxs = list(range(len(metas)))
        if len(idxs) < num_cells:
            # repeat indices to fill grid
            times = math.ceil(num_cells / len(idxs))
            idxs = list(itertools.islice(itertools.cycle(idxs), num_cells))
        else:
            idxs = idxs[:num_cells]

        k = 0
        for r in range(rows):
            for c in range(cols):
                meta = metas[idxs[k]]
                k += 1

                img = meta["img"]  # H_i, W_i, 3 (BGR)
                h_i, w_i = img.shape[:2]

                # Resize to cell size
                resized = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)

                # Paste into canvas
                y0, y1 = r * cell_h, (r + 1) * cell_h
                x0, x1 = c * cell_w, (c + 1) * cell_w
                mosaic_img[y0:y1, x0:x1] = resized

                # Transform bboxes (if any)
                boxes = meta.get("gt_bboxes", np.zeros((0, 4), dtype=np.float32))
                labels = meta.get("gt_labels", np.array([], dtype=np.int64))

                if boxes.size > 0:
                    boxes = boxes.astype(np.float32)
                    # scale to cell size
                    scale_x = cell_w / float(w_i)
                    scale_y = cell_h / float(h_i)
                    boxes[:, 0] = boxes[:, 0] * scale_x + x0
                    boxes[:, 2] = boxes[:, 2] * scale_x + x0
                    boxes[:, 1] = boxes[:, 1] * scale_y + y0
                    boxes[:, 3] = boxes[:, 3] * scale_y + y0

                    mosaic_boxes.append(boxes)
                    mosaic_labels.append(labels)

        if mosaic_boxes:
            mosaic_boxes = np.concatenate(mosaic_boxes, axis=0)
            mosaic_labels = np.concatenate(mosaic_labels, axis=0)
        else:
            mosaic_boxes = np.zeros((0, 4), dtype=np.float32)
            mosaic_labels = np.array([], dtype=np.int64)

        # Clamp boxes to canvas and drop degenerate ones
        if mosaic_boxes.size > 0:
            mosaic_boxes[:, 0::2] = np.clip(mosaic_boxes[:, 0::2], 0, W - 1)
            mosaic_boxes[:, 1::2] = np.clip(mosaic_boxes[:, 1::2], 0, H - 1)
            x1, y1, x2, y2 = mosaic_boxes.T
            valid = (x2 > x1) & (y2 > y1)
            mosaic_boxes = mosaic_boxes[valid]
            mosaic_labels = mosaic_labels[valid]

        # Build a new meta dict (shallow copy of first, then override fields)
        base_meta = dict(metas[0])
        base_meta["img"] = mosaic_img
        base_meta["gt_bboxes"] = mosaic_boxes
        base_meta["gt_labels"] = mosaic_labels
        # warp_matrix etc. will be recomputed by _transform_single via warp_and_resize

        return base_meta

    def __call_batch__(self, dataset, metas, dst_shape: Tuple[int, int]):
        """Apply transforms to a batch of metas.

        If mosaic is enabled and sampled (p), a single mosaic sample is
        created from the batch, transformed, and returned as a list of
        length 1. Otherwise, per-sample transforms are applied.
        """
        # 1. Tensor Conversion is assumed to be done before this call or within _transform_single.
        
        # 2. Conditional Mosaic Augmentation
        if self.mosaic_enabled:
            # Determine how many mosaic samples to create based on probability
            num_mosaic = int(len(metas) * self.mosaic_p)
            
            if num_mosaic > 0:
                mosaic_metas = []
                # 3. Mosaic images will be created from randomly selected images from the batch.
                for _ in range(num_mosaic):
                    # Note: _mosaic_augment is designed to select its own samples, 
                    # but for batch-level replacement, we need to ensure it uses the batch 'metas'
                    # or samples from the full dataset (which is not available here).
                    # Assuming _mosaic_augment is adapted to sample from the batch or dataset as needed.
                    mosaic_meta = self._mosaic_augment(metas, dst_shape)
                    if mosaic_meta is not None:
                        mosaic_metas.append(mosaic_meta)
                
                if mosaic_metas:
                    # 4. The mosaic images replace images in the batch randomly.
                    # Select indices of original samples to replace
                    replace_indices = random.sample(range(len(metas)), len(mosaic_metas))
                    
                    # Replace original samples with mosaic composites
                    for i, meta in zip(replace_indices, mosaic_metas):
                        metas[i] = meta
        
        # 5. Geometric, 6. Image-Only, and 7. Normalization (Per-Sample Transforms)
        # These are applied within _transform_single, which is called for every sample in the batch.
        return [self._transform_single(dataset, m, dst_shape) for m in metas]
