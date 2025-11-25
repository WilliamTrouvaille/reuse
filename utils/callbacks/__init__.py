#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/07
@author  : William_Trouvaille
@function: Callbacks 模块 - 提供训练回调工具
@detail:
    本模块提供轻量级的训练回调系统：
    - Callback: 回调基类，定义生命周期钩子
    - Timer: 训练计时器，跟踪训练耗时并支持时间限制

    所有模块均已解耦，可独立使用或集成到自定义训练循环中。
"""

from utils.callbacks.base import Callback
from utils.callbacks.timer import Interval, Stage, Timer

__all__ = [
    # 基类
    "Callback",
    # 计时器
    "Timer",
    "Interval",
    "Stage",
]
