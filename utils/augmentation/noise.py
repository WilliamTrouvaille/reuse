#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025-11-10 09:52:49
@author  : William_Trouvaille
@function: 噪声类数据增强操作
"""

from __future__ import annotations

import torch

from .base import BaseTransform, validate_range


# ========================================================================
# 1. 高斯噪声
# ========================================================================

class GaussianNoise(BaseTransform):
    """向图像添加高斯噪声并裁剪到合法范围。"""

    def __init__(self, mean: float = 0.0, std: float = 0.1, p: float = 1.0, seed: int | None = None) -> None:
        super().__init__(p=p, seed=seed)
        validate_range(std, 0.0, 1.0, "噪声标准差")
        self.mean = float(mean)
        self.std = float(std)
        self.register_config(mean=mean, std=std)

    def _apply(self, image: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(image) * self.std + self.mean
        noisy = image + noise
        return torch.clamp(noisy, 0.0, 1.0)
