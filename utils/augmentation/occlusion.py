#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025-11-10 09:52:49
@author  : William_Trouvaille
@function: 遮挡类数据增强操作，包括随机擦除与CutOut
"""

from __future__ import annotations

import math
from typing import Literal, Tuple

import torch

from .base import BaseTransform, validate_range


# ========================================================================
# 1. 随机擦除
# ========================================================================

class RandomErasing(BaseTransform):
    """随机遮挡图像区域，可配置填充值策略。"""

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.02, 0.33),
        ratio_range: tuple[float, float] = (0.3, 3.3),
        fill_mode: Literal["constant", "random", "mean"] = "constant",
        value: float = 0.0,
        p: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        validate_range(scale_range[0], 0.0, 1.0, "擦除面积下限")
        validate_range(scale_range[1], 0.0, 1.0, "擦除面积上限")
        validate_range(ratio_range[0], 0.1, 10.0, "宽高比下限")
        validate_range(ratio_range[1], 0.1, 10.0, "宽高比上限")
        assert scale_range[0] <= scale_range[1]
        assert ratio_range[0] <= ratio_range[1]
        assert fill_mode in {"constant", "random", "mean"}, "填充模式非法"
        self.scale_range = scale_range
        self.ratio_range = ratio_range
        self.fill_mode = fill_mode
        self.value = value
        self.register_config(scale_range=scale_range, ratio_range=ratio_range, fill_mode=fill_mode, value=value)

    def _apply(self, image: torch.Tensor) -> torch.Tensor:
        c, height, width = image.shape
        area = height * width
        for _ in range(10):
            target_area = torch.empty(1, device=image.device).uniform_(*self.scale_range) * area
            aspect_ratio = torch.exp(torch.empty(1, device=image.device).uniform_(math.log(self.ratio_range[0]), math.log(self.ratio_range[1])))
            erase_h = int(torch.round(torch.sqrt(target_area * aspect_ratio)).item())
            erase_w = int(torch.round(torch.sqrt(target_area / aspect_ratio)).item())
            if erase_h < height and erase_w < width:
                top = torch.randint(0, height - erase_h + 1, (1,), device=image.device).item()
                left = torch.randint(0, width - erase_w + 1, (1,), device=image.device).item()
                fill = self._build_fill(image, erase_h, erase_w)
                image[:, top : top + erase_h, left : left + erase_w] = fill
                return image
        return image

    # --- 1.1 构造填充值 ---
    def _build_fill(self, image: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if self.fill_mode == "constant":
            return torch.full((image.shape[0], h, w), self.value, device=image.device, dtype=image.dtype)
        if self.fill_mode == "random":
            return torch.rand((image.shape[0], h, w), device=image.device, dtype=image.dtype)
        mean = image.mean(dim=(1, 2), keepdim=True)
        return mean.expand(-1, h, w)


# ========================================================================
# 2. CutOut
# ========================================================================

class CutOut(BaseTransform):
    """在随机位置放置固定大小的遮挡块。"""

    def __init__(
        self,
        num_holes: int = 1,
        hole_size: Tuple[int, int] = (32, 32),
        fill_value: float = 0.0,
        p: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        assert num_holes > 0, "遮挡块数量必须大于0"
        self.num_holes = num_holes
        self.hole_size = hole_size
        self.fill_value = fill_value
        self.register_config(num_holes=num_holes, hole_size=hole_size, fill_value=fill_value)

    def _apply(self, image: torch.Tensor) -> torch.Tensor:
        channels, height, width = image.shape
        hole_h, hole_w = self.hole_size
        device = image.device
        for _ in range(self.num_holes):
            center_y = int(torch.randint(0, height, (1,), device=device).item())
            center_x = int(torch.randint(0, width, (1,), device=device).item())
            top = max(center_y - hole_h // 2, 0)
            left = max(center_x - hole_w // 2, 0)
            bottom = min(top + hole_h, height)
            right = min(left + hole_w, width)
            image[:, top:bottom, left:right] = self.fill_value
        return image
