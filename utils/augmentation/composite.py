#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025-11-10 09:52:49
@author  : William_Trouvaille
@function: 复合类数据增强操作，组合多种基础变换
"""

from __future__ import annotations

import torch

from .base import BaseTransform, RandomApply, validate_probability
from .color import ColorJitter, ToGrayscale
from .filter import GaussianBlur
from .geometric import RandomCropAndResize, RandomHorizontalFlip


# ========================================================================
# 1. 随机裁剪-缩放-翻转链路
# ========================================================================

class RandomCropResizeFlip(BaseTransform):
    """组合随机裁剪缩放与翻转，符合SimCLR预处理。"""

    def __init__(
        self,
        output_size: tuple[int, int],
        flip_p: float = 0.5,
        p: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        validate_probability(flip_p)
        self.crop_resize = RandomCropAndResize(output_size=output_size, p=1.0, seed=seed)
        self.flip = RandomHorizontalFlip(p=flip_p, seed=seed)
        self.register_config(output_size=output_size, flip_p=flip_p)

    def _apply(self, image: torch.Tensor) -> torch.Tensor:
        image = self.crop_resize(image)
        image = self.flip(image)
        return image


# ========================================================================
# 2. 颜色扰动复合策略
# ========================================================================

class RandomColorDistortion(BaseTransform):
    """SimCLR风格颜色失真，含颜色扰动与灰度化。"""

    def __init__(
        self,
        strength: float = 1.0,
        color_jitter_prob: float = 0.8,
        grayscale_prob: float = 0.2,
        impl: str = "simclrv2",
        p: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        validate_probability(color_jitter_prob)
        validate_probability(grayscale_prob)
        self.color_jitter = ColorJitter(strength=strength, random_order=True, impl=impl, p=1.0, seed=seed)
        self.to_gray = ToGrayscale(keep_channels=True, p=1.0, seed=seed)
        self._color_apply = RandomApply(self.color_jitter._apply, color_jitter_prob)
        self._gray_apply = RandomApply(self.to_gray._apply, grayscale_prob)
        self.register_config(
            strength=strength,
            color_jitter_prob=color_jitter_prob,
            grayscale_prob=grayscale_prob,
            impl=impl,
        )

    def _apply(self, image: torch.Tensor) -> torch.Tensor:
        image = self._color_apply(image)
        image = self._gray_apply(image)
        return image


# ========================================================================
# 3. 随机模糊
# ========================================================================

class RandomBlur(BaseTransform):
    """以一定概率施加高斯模糊，核尺寸依赖输入高度。"""

    def __init__(self, sigma_range: tuple[float, float] = (0.1, 2.0), p: float = 0.5, seed: int | None = None) -> None:
        super().__init__(p=p, seed=seed)
        self.blur = GaussianBlur(kernel_size=None, sigma_range=sigma_range, p=1.0, seed=seed)
        self.register_config(sigma_range=sigma_range)

    def _apply(self, image: torch.Tensor) -> torch.Tensor:
        dynamic_kernel = max(3, int(image.shape[1] / 10))
        if dynamic_kernel % 2 == 0:
            dynamic_kernel += 1
        original_kernel = self.blur.kernel_size
        self.blur.kernel_size = dynamic_kernel
        result = self.blur._apply(image)
        self.blur.kernel_size = original_kernel
        return torch.clamp(result, 0.0, 1.0)
