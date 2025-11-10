#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025-11-10 09:52:49
@author  : William_Trouvaille
@function: 数据增强子工具箱统一导出入口
"""

from .base import BaseTransform, RandomApply, TorchSeedContext, validate_probability, validate_range
from .color import (
    ColorJitter,
    RandomBrightness,
    RandomContrast,
    RandomHue,
    RandomSaturation,
    ToGrayscale,
)
from .composite import RandomBlur, RandomColorDistortion, RandomCropResizeFlip
from .filter import GaussianBlur, SobelFilter
from .geometric import CenterCrop, RandomCropAndResize, RandomHorizontalFlip, RandomRotation
from .noise import GaussianNoise
from .occlusion import CutOut, RandomErasing
from .pipeline import AugmentationPipeline, TRANSFORM_REGISTRY, get_basic_augmentation, get_simclr_augmentation, get_strong_augmentation

__all__ = [
    "BaseTransform",
    "RandomApply",
    "TorchSeedContext",
    "validate_probability",
    "validate_range",
    "RandomBrightness",
    "RandomContrast",
    "RandomSaturation",
    "RandomHue",
    "ToGrayscale",
    "ColorJitter",
    "RandomCropAndResize",
    "RandomHorizontalFlip",
    "RandomRotation",
    "CenterCrop",
    "GaussianBlur",
    "SobelFilter",
    "GaussianNoise",
    "RandomErasing",
    "CutOut",
    "RandomCropResizeFlip",
    "RandomColorDistortion",
    "RandomBlur",
    "AugmentationPipeline",
    "TRANSFORM_REGISTRY",
    "get_simclr_augmentation",
    "get_basic_augmentation",
    "get_strong_augmentation",
]
