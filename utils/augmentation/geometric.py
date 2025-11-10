#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025-11-10 09:52:49
@author  : William_Trouvaille
@function: 几何类数据增强操作的实现，包含裁剪、翻转与旋转
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
from loguru import logger
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .base import BaseTransform, validate_probability, validate_range


# ========================================================================
# 1. Inception风格随机裁剪与缩放
# ========================================================================

class RandomCropAndResize(BaseTransform):
    """执行SimCLR常用的随机裁剪并缩放至指定尺寸。"""

    def __init__(
        self,
        output_size: Tuple[int, int],
        scale_range: Tuple[float, float] = (0.08, 1.0),
        ratio_range: Tuple[float, float] = (0.75, 1.33),
        num_attempts: int = 100,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        p: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        self.output_size = output_size
        self.scale_range = scale_range
        self.ratio_range = ratio_range
        self.num_attempts = num_attempts
        self.interpolation = interpolation
        validate_range(scale_range[0], 0.0, 1.0, "scale范围下限")
        validate_range(scale_range[1], 0.0, 1.0, "scale范围上限")
        validate_range(ratio_range[0], 0.1, 10.0, "ratio范围下限")
        validate_range(ratio_range[1], 0.1, 10.0, "ratio范围上限")
        assert scale_range[0] <= scale_range[1], "scale范围必须满足下限<=上限"
        assert ratio_range[0] <= ratio_range[1], "ratio范围必须满足下限<=上限"
        self.register_config(
            output_size=output_size,
            scale_range=scale_range,
            ratio_range=ratio_range,
            num_attempts=num_attempts,
            interpolation=interpolation,
        )

    def _apply(self, image: torch.Tensor) -> torch.Tensor:
        c, height, width = image.shape
        top, left, crop_h, crop_w = self._sample_crop_params(height, width, image.device)
        logger.debug(
            f"随机裁剪参数: top={top}, left={left}, crop_h={crop_h}, crop_w={crop_w}, 原始尺寸=({height},{width})"
        )
        cropped = TF.resized_crop(
            image,
            top=top,
            left=left,
            height=crop_h,
            width=crop_w,
            size=self.output_size,
            interpolation=self.interpolation,
        )
        return torch.clamp(cropped, 0.0, 1.0)

    # --- 1.1 采样裁剪区域参数 ---
    def _sample_crop_params(self, height: int, width: int, device: torch.device) -> tuple[int, int, int, int]:
        area = height * width
        log_ratio = (math.log(self.ratio_range[0]), math.log(self.ratio_range[1]))
        for _ in range(self.num_attempts):
            target_area = torch.empty(1, device=device).uniform_(self.scale_range[0], self.scale_range[1]) * area
            aspect_ratio = torch.exp(torch.empty(1, device=device).uniform_(*log_ratio))
            w = torch.round(torch.sqrt(target_area / aspect_ratio)).int().item()
            h = torch.round(torch.sqrt(target_area * aspect_ratio)).int().item()
            if 0 < w <= width and 0 < h <= height:
                top = torch.randint(0, height - h + 1, (1,), device=device).item()
                left = torch.randint(0, width - w + 1, (1,), device=device).item()
                return top, left, h, w
        min_side = min(height, width)
        crop_size = int(min_side * 0.875)
        top = max((height - crop_size) // 2, 0)
        left = max((width - crop_size) // 2, 0)
        return top, left, crop_size, crop_size


# ========================================================================
# 2. 左右翻转操作
# ========================================================================

class RandomHorizontalFlip(BaseTransform):
    """以固定概率执行水平翻转。"""

    def __init__(self, p: float = 0.5, seed: int | None = None) -> None:
        super().__init__(p=p, seed=seed)
        self.register_config()

    def _apply(self, image: torch.Tensor) -> torch.Tensor:
        return torch.flip(image, dims=(-1,))


# ========================================================================
# 3. 任意角度旋转
# ========================================================================

class RandomRotation(BaseTransform):
    """随机角度旋转，支持多种插值与填充值。"""

    def __init__(
        self,
        degrees: float | Sequence[float],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        expand: bool = False,
        center: Optional[Tuple[float, float]] = None,
        fill: float | Sequence[float] = 0.0,
        p: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        if isinstance(degrees, (int, float)):
            self.min_degree = float(-abs(degrees))
            self.max_degree = float(abs(degrees))
        else:
            assert len(degrees) == 2, "degrees需要包含上下限"
            self.min_degree = float(degrees[0])
            self.max_degree = float(degrees[1])
        validate_range(self.max_degree, self.min_degree, self.max_degree, "角度范围")
        self.interpolation = interpolation
        self.expand = expand
        self.center = center
        self.fill = fill
        self.register_config(
            degrees=(self.min_degree, self.max_degree),
            interpolation=interpolation,
            expand=expand,
            center=center,
            fill=fill,
        )

    def _apply(self, image: torch.Tensor) -> torch.Tensor:
        angle = torch.empty(1, device=image.device).uniform_(self.min_degree, self.max_degree).item()
        rotated = TF.rotate(
            image,
            angle=angle,
            interpolation=self.interpolation,
            expand=self.expand,
            center=self.center,
            fill=self.fill,
        )
        return torch.clamp(rotated, 0.0, 1.0)


# ========================================================================
# 4. 中心裁剪
# ========================================================================

class CenterCrop(BaseTransform):
    """按比例执行中心裁剪，并可选地缩放到目标尺寸。"""

    def __init__(
        self,
        target_size: Optional[Tuple[int, int]] = None,
        crop_proportion: float = 0.875,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        seed: int | None = None,
    ) -> None:
        super().__init__(p=1.0, seed=seed)
        validate_probability(crop_proportion)
        self.crop_proportion = float(crop_proportion)
        self.target_size = target_size
        self.interpolation = interpolation
        self.register_config(
            target_size=target_size,
            crop_proportion=crop_proportion,
            interpolation=interpolation,
        )

    def _apply(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        crop_h, crop_w = self._compute_crop_size(height, width)
        cropped = TF.center_crop(image, output_size=[crop_h, crop_w])
        if self.target_size is not None:
            cropped = TF.resize(cropped, size=self.target_size, interpolation=self.interpolation, antialias=True)
        return torch.clamp(cropped, 0.0, 1.0)

    # --- 4.1 依据目标宽高比计算裁剪尺寸 ---
    def _compute_crop_size(self, height: int, width: int) -> tuple[int, int]:
        base = int(min(height, width) * self.crop_proportion)
        base = max(base, 1)
        if self.target_size is None:
            return min(base, height), min(base, width)
        target_ratio = self.target_size[0] / self.target_size[1]
        crop_h = base
        crop_w = max(int(crop_h / target_ratio), 1)
        if crop_w > width:
            crop_w = width
            crop_h = max(int(crop_w * target_ratio), 1)
        crop_h = min(crop_h, height)
        crop_w = min(crop_w, width)
        return crop_h, crop_w
