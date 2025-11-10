#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025-11-10 09:52:49
@author  : William_Trouvaille
@function: 滤波相关的数据增强操作，包含高斯模糊与Sobel算子
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from loguru import logger

from .base import BaseTransform, validate_range


# ========================================================================
# 1. 高斯模糊
# ========================================================================

class GaussianBlur(BaseTransform):
    """可分离卷积实现的高斯模糊，核大小可自适应。"""

    def __init__(
        self,
        kernel_size: int | None = None,
        sigma_range: tuple[float, float] = (0.1, 2.0),
        p: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        validate_range(sigma_range[0], 0.01, 10.0, "sigma下限")
        validate_range(sigma_range[1], 0.01, 10.0, "sigma上限")
        assert sigma_range[0] <= sigma_range[1], "sigma范围必须合法"
        if kernel_size is not None:
            assert kernel_size % 2 == 1, "kernel_size必须为奇数"
            assert kernel_size > 1, "kernel_size需大于1"
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        self.register_config(kernel_size=kernel_size, sigma_range=sigma_range)

    def _apply(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        kernel_size = self._resolve_kernel_size(height, width)
        sigma = torch.empty(1, device=image.device).uniform_(*self.sigma_range).item()
        kernel_1d = self._build_kernel(kernel_size, sigma, image.device, image.dtype)
        logger.debug(f"GaussianBlur kernel_size={kernel_size}, sigma={sigma:.4f}")
        padding = kernel_size // 2
        image = image.unsqueeze(0)
        weight_y = kernel_1d.view(1, 1, kernel_size, 1)
        weight_x = kernel_1d.view(1, 1, 1, kernel_size)
        image = F.conv2d(image, weight_y.expand(image.size(1), -1, -1, -1), padding=(padding, 0), groups=image.size(1))
        image = F.conv2d(image, weight_x.expand(image.size(1), -1, -1, -1), padding=(0, padding), groups=image.size(1))
        return image.squeeze(0)

    # --- 1.1 构建一维高斯核 ---
    def _build_kernel(self, kernel_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        radius = kernel_size // 2
        coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-coords.pow(2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel

    # --- 1.2 动态确定kernel size ---
    def _resolve_kernel_size(self, height: int, width: int) -> int:
        if self.kernel_size is not None:
            return self.kernel_size
        base = max(3, int(min(height, width) / 10))
        if base % 2 == 0:
            base += 1
        return max(base, 3)


# ========================================================================
# 2. Sobel边缘检测
# ========================================================================

class SobelFilter(BaseTransform):
    """Sobel算子，可输出水平/垂直/组合结果。"""

    def __init__(
        self,
        mode: Literal["horizontal", "vertical", "combined"] = "combined",
        keep_rgb: bool = True,
        p: float = 1.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(p=p, seed=seed)
        assert mode in {"horizontal", "vertical", "combined"}, "Sobel模式非法"
        self.mode = mode
        self.keep_rgb = keep_rgb
        self.register_config(mode=mode, keep_rgb=keep_rgb)

    def _apply(self, image: torch.Tensor) -> torch.Tensor:
        kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=image.dtype, device=image.device)
        kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=image.dtype, device=image.device)
        weight_x = kernel_x.view(1, 1, 3, 3)
        weight_y = kernel_y.view(1, 1, 3, 3)
        padding = 1
        image_batch = image.unsqueeze(0)
        grad_x = F.conv2d(image_batch, weight_x.expand(image.size(0), -1, -1, -1), padding=padding, groups=image.size(0))
        grad_y = F.conv2d(image_batch, weight_y.expand(image.size(0), -1, -1, -1), padding=padding, groups=image.size(0))
        grad_x = grad_x.squeeze(0)
        grad_y = grad_y.squeeze(0)
        if self.mode == "horizontal":
            result = grad_x
        elif self.mode == "vertical":
            result = grad_y
        else:
            result = torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-12)
        if not self.keep_rgb:
            mean = result.mean(dim=0, keepdim=True)
            result = mean.expand_as(result)
        result = (result - result.min()) / (result.max() - result.min() + 1e-12)
        return torch.clamp(result, 0.0, 1.0)
