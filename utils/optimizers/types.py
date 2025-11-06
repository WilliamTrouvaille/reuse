#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025-11-06 14:50:45
@author  : William_Trouvaille
@function: 提供优化器常用的类型别名，保持不同优化器间的接口一致
"""

from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

from torch import Tensor

# ========================================================================
# 1. 类型别名定义
# ========================================================================

# --- 1.1 参数与状态类型 ---
Params = Union[Iterable[Tensor], Iterable[Dict[str, Any]]]
State = Dict[str, Any]

# --- 1.2 超参数类型 ---
Betas2 = Tuple[float, float]
Nus2 = Tuple[float, float]

# --- 1.3 闭包类型 ---
LossClosure = Callable[[], float]
OptLossClosure = Optional[LossClosure]
OptFloat = Optional[float]
