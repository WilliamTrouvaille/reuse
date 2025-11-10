#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025-11-10 09:52:49
@author  : William_Trouvaille
@function: 颜色空间相关的数据增强操作，涵盖亮度、对比度等变换
"""

from __future__ import annotations

from typing import Sequence

import torch
from loguru import logger
from torchvision.transforms import functional as TF

from .base import BaseTransform, validate_probability, validate_range


# ========================================================================
# 1. 亮度与对比度调整
# ========================================================================

class RandomBrightness(BaseTransform):
    """提供SimCLRv1/v2两种亮度扰动策略。"""

    def __init__(self, max_delta: float, impl: str = "simclrv2", p: float = 1.0, seed: int | None = None) -> None:
        super().__init__(p=p, seed=seed)
        validate_range(max_delta, 0.0, 1.0, "亮度扰动幅度")
        assert impl in {"simclrv1", "simclrv2"}, "impl仅支持simclrv1/simclrv2"
        self.max_delta = float(max_delta)
        self.impl = impl
        self.register_config(max_delta=max_delta, impl=impl)

    def _apply(self, image: torch.Tensor) -> torch.Tensor:
        if self.impl == "simclrv2":
            lower = max(1.0 - self.max_delta, 0.0)
            upper = 1.0 + self.max_delta
            factor = torch.empty(1, device=image.device).uniform_(lower, upper).item()
            adjusted = image * factor
        else:
            delta = torch.empty(1, device=image.device).uniform_(-self.max_delta, self.max_delta).item()
            adjusted = image + delta
        return torch.clamp(adjusted, 0.0, 1.0)


class RandomContrast(BaseTransform):
    """通过torchvision函数调整对比度。"""

    def __init__(self, contrast_range: tuple[float, float], p: float = 1.0, seed: int | None = None) -> None:
        super().__init__(p=p, seed=seed)
        validate_range(contrast_range[0], 0.0, 2.0, "对比度下限")
        validate_range(contrast_range[1], 0.0, 2.0, "对比度上限")
        assert contrast_range[0] <= contrast_range[1], "对比度范围必须合法"
        self.contrast_range = contrast_range
        self.register_config(contrast_range=contrast_range)

    def _apply(self, image: torch.Tensor) -> torch.Tensor:
        factor = torch.empty(1, device=image.device).uniform_(*self.contrast_range).item()
        adjusted = TF.adjust_contrast(image, factor)
        return torch.clamp(adjusted, 0.0, 1.0)


# ========================================================================
# 2. 饱和度、色调与灰度
# ========================================================================

class RandomSaturation(BaseTransform):
    """随机调节饱和度。"""

    def __init__(self, saturation_range: tuple[float, float], p: float = 1.0, seed: int | None = None) -> None:
        super().__init__(p=p, seed=seed)
        validate_range(saturation_range[0], 0.0, 2.0, "饱和度下限")
        validate_range(saturation_range[1], 0.0, 2.0, "饱和度上限")
        assert saturation_range[0] <= saturation_range[1], "饱和度范围必须合法"
        self.saturation_range = saturation_range
        self.register_config(saturation_range=saturation_range)

    def _apply(self, image: torch.Tensor) -> torch.Tensor:
        factor = torch.empty(1, device=image.device).uniform_(*self.saturation_range).item()
        adjusted = TF.adjust_saturation(image, factor)
        return torch.clamp(adjusted, 0.0, 1.0)


class RandomHue(BaseTransform):
    """随机调整色调，相当于HSV空间中的H轴偏移。"""

    def __init__(self, max_delta: float, p: float = 1.0, seed: int | None = None) -> None:
        super().__init__(p=p, seed=seed)
        validate_range(max_delta, 0.0, 0.5, "色调偏移幅度")
        self.max_delta = float(max_delta)
        self.register_config(max_delta=max_delta)

    def _apply(self, image: torch.Tensor) -> torch.Tensor:
        delta = torch.empty(1, device=image.device).uniform_(-self.max_delta, self.max_delta).item()
        adjusted = TF.adjust_hue(image, delta)
        return torch.clamp(adjusted, 0.0, 1.0)


class ToGrayscale(BaseTransform):
    """转换为灰度图，可选择保留三通道。"""

    def __init__(self, keep_channels: bool = True, p: float = 1.0, seed: int | None = None) -> None:
        super().__init__(p=p, seed=seed)
        self.keep_channels = keep_channels
        self.register_config(keep_channels=keep_channels)

    def _apply(self, image: torch.Tensor) -> torch.Tensor:
        gray = TF.rgb_to_grayscale(image)
        if self.keep_channels:
            gray = gray.repeat(3, 1, 1)
        return gray


# ========================================================================
# 3. 组合颜色扰动
# ========================================================================

class ColorJitter(BaseTransform):
    """SimCLR式颜色扰动，支持随机顺序。"""

    def __init__(
        self,
        strength: float = 1.0,
        random_order: bool = True,
        impl: str = "simclrv2",
        p: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        validate_range(strength, 0.0, 2.0, "颜色扰动强度")
        self.strength = strength
        self.random_order = random_order
        self.impl = impl
        self.brightness = 0.8 * strength
        self.contrast = 0.8 * strength
        self.saturation = 0.8 * strength
        self.hue = 0.2 * strength
        self.register_config(strength=strength, random_order=random_order, impl=impl)

    def _apply(self, image: torch.Tensor) -> torch.Tensor:
        transforms = [
            self._brightness_transform,
            self._contrast_transform,
            self._saturation_transform,
            self._hue_transform,
        ]
        order = list(range(len(transforms)))
        if self.random_order:
            perm = torch.randperm(len(transforms), device=image.device).tolist()
            order = perm
        logger.debug(f"颜色扰动顺序: {order}")
        for idx in order:
            image = transforms[idx](image)
            image = torch.clamp(image, 0.0, 1.0)
        return image

    # --- 3.1 亮度扰动 ---
    def _brightness_transform(self, image: torch.Tensor) -> torch.Tensor:
        if self.brightness <= 0:
            return image
        if self.impl == "simclrv2":
            lower = max(1.0 - self.brightness, 0.0)
            upper = 1.0 + self.brightness
            factor = torch.empty(1, device=image.device).uniform_(lower, upper).item()
            return image * factor
        delta = torch.empty(1, device=image.device).uniform_(-self.brightness, self.brightness).item()
        return image + delta

    # --- 3.2 对比度扰动 ---
    def _contrast_transform(self, image: torch.Tensor) -> torch.Tensor:
        if self.contrast <= 0:
            return image
        factor = torch.empty(1, device=image.device).uniform_(1 - self.contrast, 1 + self.contrast).item()
        return TF.adjust_contrast(image, factor)

    # --- 3.3 饱和度扰动 ---
    def _saturation_transform(self, image: torch.Tensor) -> torch.Tensor:
        if self.saturation <= 0:
            return image
        factor = torch.empty(1, device=image.device).uniform_(1 - self.saturation, 1 + self.saturation).item()
        return TF.adjust_saturation(image, factor)

    # --- 3.4 色调扰动 ---
    def _hue_transform(self, image: torch.Tensor) -> torch.Tensor:
        if self.hue <= 0:
            return image
        delta = torch.empty(1, device=image.device).uniform_(-self.hue, self.hue).item()
        return TF.adjust_hue(image, delta)
