#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/07
@author  : William_Trouvaille
@function: Callbacks 模块 - 提供训练回调工具
@detail:
    本模块包含从 PyTorch Lightning 集成的训练回调工具：
    - Timer: 训练计时器
    - LearningRateMonitor: 学习率监控器
    - BatchSizeFinder: 批大小查找器

    所有模块均已解耦，可独立使用或集成到自定义训练循环中。
"""

from utils.callbacks.batch_size_finder import BatchSizeFinder, find_optimal_batch_size
from utils.callbacks.lr_monitor import LearningRateMonitor
from utils.callbacks.timer import Interval, Stage, Timer

__all__ = [
    # 计时器
    "Timer",
    "Interval",
    "Stage",
    # 学习率监控
    "LearningRateMonitor",
    # 批大小查找
    "BatchSizeFinder",
    "find_optimal_batch_size",
]
