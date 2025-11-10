#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025-11-10 09:52:49
@author  : William_Trouvaille
@function: 增强管道及预设配置，便于统一管理与序列化
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Type

import torch
import torch.nn as nn
from loguru import logger

from .base import BaseTransform
from .color import ColorJitter, RandomBrightness, RandomContrast, RandomHue, RandomSaturation, ToGrayscale
from .composite import RandomBlur, RandomColorDistortion, RandomCropResizeFlip
from .filter import GaussianBlur, SobelFilter
from .geometric import CenterCrop, RandomCropAndResize, RandomHorizontalFlip, RandomRotation
from .noise import GaussianNoise
from .occlusion import CutOut, RandomErasing


# ========================================================================
# 1. 变换注册表
# ========================================================================

TRANSFORM_REGISTRY: Dict[str, Type[BaseTransform]] = {
    cls.__name__: cls
    for cls in [
        RandomBrightness,
        RandomContrast,
        RandomSaturation,
        RandomHue,
        ToGrayscale,
        ColorJitter,
        GaussianBlur,
        SobelFilter,
        GaussianNoise,
        RandomErasing,
        CutOut,
        RandomCropAndResize,
        RandomHorizontalFlip,
        RandomRotation,
        CenterCrop,
        RandomCropResizeFlip,
        RandomColorDistortion,
        RandomBlur,
    ]
}


# ========================================================================
# 2. 增强管道定义
# ========================================================================

class AugmentationPipeline(nn.Module):
    """顺序执行多个BaseTransform的容器。"""

    def __init__(self, transforms: Sequence[BaseTransform] | None = None, name: str = "augmentation_pipeline") -> None:
        super().__init__()
        self.name = name
        self.transforms = nn.ModuleList(transforms or [])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # noqa: D401
        logger.info("=" * 40)
        logger.info(f"增强管道 '{self.name}' 开始执行")
        logger.info("=" * 40)
        output = inputs
        for transform in self.transforms:
            output = transform(output)
        return output

    # --- 2.1 变换维护 ---
    def add_transform(self, transform: BaseTransform) -> None:
        logger.info(f"管道新增变换: {transform.__class__.__name__}")
        self.transforms.append(transform)

    def remove_transform(self, class_name: str) -> None:
        remaining = [t for t in self.transforms if t.__class__.__name__ != class_name]
        if len(remaining) == len(self.transforms):
            logger.warning(f"未找到需要移除的变换: {class_name}")
            return
        self.transforms = nn.ModuleList(remaining)
        logger.info(f"已移除变换: {class_name}")

    # --- 2.2 序列化与反序列化 ---
    def to_config(self) -> Dict[str, Any]:
        return {"name": self.name, "transforms": [t.to_config() for t in self.transforms]}

    @classmethod
    def from_config(cls, config: Dict[str, Any], registry: Dict[str, Type[BaseTransform]] | None = None) -> "AugmentationPipeline":
        registry = registry or TRANSFORM_REGISTRY
        transforms = []
        for item in config.get("transforms", []):
            class_name = item.get("class")
            if class_name not in registry:
                raise ValueError(f"未注册的变换: {class_name}")
            transform_cls = registry[class_name]
            transforms.append(transform_cls.from_config(item))
        return cls(transforms=transforms, name=config.get("name", "augmentation_pipeline"))


# ========================================================================
# 3. 预设配置
# ========================================================================

def get_simclr_augmentation(image_size: tuple[int, int], strength: float = 1.0) -> AugmentationPipeline:
    """SimCLR风格的强增广组合。"""
    transforms: List[BaseTransform] = [
        RandomCropResizeFlip(output_size=image_size, flip_p=0.5, p=1.0),
        RandomColorDistortion(strength=strength, color_jitter_prob=0.8, grayscale_prob=0.2, p=1.0),
        RandomBlur(p=0.5),
    ]
    return AugmentationPipeline(transforms=transforms, name="simclr")


def get_basic_augmentation(image_size: tuple[int, int]) -> AugmentationPipeline:
    """基础版增广：中心裁剪+翻转+轻微颜色抖动。"""
    transforms: List[BaseTransform] = [
        CenterCrop(target_size=image_size, crop_proportion=0.9),
        RandomHorizontalFlip(p=0.5),
        ColorJitter(strength=0.3, random_order=False),
    ]
    return AugmentationPipeline(transforms=transforms, name="basic")


def get_strong_augmentation(image_size: tuple[int, int]) -> AugmentationPipeline:
    """强增强配置，适合对比学习或半监督任务。"""
    transforms: List[BaseTransform] = [
        RandomCropResizeFlip(output_size=image_size, flip_p=0.5),
        RandomColorDistortion(strength=1.2, color_jitter_prob=0.8, grayscale_prob=0.2),
        RandomBlur(p=0.8),
        GaussianNoise(std=0.05),
        RandomErasing(scale_range=(0.02, 0.2), ratio_range=(0.3, 3.3), p=0.5),
    ]
    return AugmentationPipeline(transforms=transforms, name="strong")
