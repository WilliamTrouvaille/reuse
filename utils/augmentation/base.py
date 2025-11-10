#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025-11-10 09:52:49
@author  : William_Trouvaille
@function: 定义数据增强所需的基础抽象、概率控制与随机上下文工具
"""

from __future__ import annotations

import abc
from contextlib import ContextDecorator
from typing import Any, Callable, Dict

import torch
import torch.nn as nn
from loguru import logger


# ========================================================================
# 1. 参数验证与随机控制
# ========================================================================

# --- 1.1 概率与取值范围校验 ---

def validate_probability(p: float) -> None:
    """验证概率取值，确保后续随机逻辑稳定。"""
    assert 0.0 <= float(p) <= 1.0, "概率必须在[0, 1]范围内"


def validate_range(value: float, min_val: float, max_val: float, name: str) -> None:
    """统一的范围断言，便于排查参数配置错误。"""
    assert min_val <= float(value) <= max_val, f"{name}必须在[{min_val}, {max_val}]范围内"


# --- 1.2 Torch随机状态上下文 ---

class TorchSeedContext(ContextDecorator):
    """用于在局部代码块中临时设置随机种子。"""

    def __init__(self, seed: int | None) -> None:
        self.seed = seed
        self._state: torch.Tensor | None = None

    def __enter__(self) -> None:
        if self.seed is None:
            return None
        self._state = torch.random.get_rng_state()
        torch.manual_seed(int(self.seed))
        logger.debug(f"设置局部随机种子: {self.seed}")
        return None

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        if self._state is not None:
            torch.random.set_rng_state(self._state)
            logger.debug("恢复进入上下文前的随机状态")
        return None


# ========================================================================
# 2. 概率封装模块
# ========================================================================

class RandomApply(nn.Module):
    """负责以固定概率执行传入的变换函数。"""

    def __init__(self, transform: Callable[[torch.Tensor], torch.Tensor], p: float) -> None:
        super().__init__()
        validate_probability(p)
        self.transform = transform
        self.p = float(p)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:  # noqa: D401 保持nn.Module接口
        logger.debug(f"RandomApply开始执行, 概率={self.p}")
        rand_val = torch.rand(1, device=tensor.device)
        if rand_val <= self.p:
            logger.debug("触发变换函数")
            return self.transform(tensor)
        logger.debug("跳过当前变换")
        return tensor


# ========================================================================
# 3. 统一的变换抽象
# ========================================================================

class BaseTransform(nn.Module, metaclass=abc.ABCMeta):
    """所有数据增强操作的抽象基类。"""

    def __init__(self, p: float = 1.0, seed: int | None = None) -> None:
        super().__init__()
        validate_probability(p)
        self.p = float(p)
        self.seed = seed
        self._random_apply = RandomApply(self._apply, self.p)
        self._config_params: Dict[str, Any] = {}

    # ====================================================================
    # 3.1 前向入口与输入校验
    # ====================================================================

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """统一的前向逻辑，封装输入校验、概率控制与日志。"""
        tensor = self._validate_input(inputs)
        logger.info("=" * 40)
        logger.info(f"执行变换 '{self.__class__.__name__}'")
        logger.info("=" * 40)
        with TorchSeedContext(self.seed):
            try:
                if tensor.dim() == 3:
                    return self._random_apply(tensor)
                # --- 3.1.1 批量输入逐样本处理 ---
                outputs = [self._random_apply(sample) for sample in tensor]
                return torch.stack(outputs, dim=0)
            except Exception as exc:
                logger.exception(f"变换 {self.__class__.__name__} 执行失败: {exc}")
                raise

    # ====================================================================
    # 3.2 输入张量校验
    # ====================================================================

    def _validate_input(self, tensor: torch.Tensor) -> torch.Tensor:
        """保证张量维度、类型与取值满足预期。"""
        assert isinstance(tensor, torch.Tensor), "输入必须是torch.Tensor"
        assert tensor.dim() in (3, 4), "输入张量必须是(C,H,W)或(B,C,H,W)"
        assert tensor.shape[-3] == 3, "输入张量必须包含3个通道(RGB)"
        assert tensor.is_floating_point(), "图像张量必须为浮点类型"
        min_val = torch.min(tensor).item()
        max_val = torch.max(tensor).item()
        assert -1e-3 <= min_val <= 1.0 + 1e-3, "像素值应位于[0,1]附近"
        assert -1e-3 <= max_val <= 1.0 + 1e-3, "像素值应位于[0,1]附近"
        return tensor.contiguous()

    # ====================================================================
    # 3.3 子类需实现的核心逻辑
    # ====================================================================

    @abc.abstractmethod
    def _apply(self, image: torch.Tensor) -> torch.Tensor:
        """子类实现具体变换，形状固定为(C,H,W)。"""

    def extra_repr(self) -> str:
        return f"p={self.p}, seed={self.seed}"

    # ====================================================================
    # 3.4 配置导出与还原
    # ====================================================================

    def register_config(self, **params: Any) -> None:
        """记录用于重建当前变换的参数集合。"""
        self._config_params = params

    def to_config(self) -> Dict[str, Any]:
        """导出可序列化配置，供增强管道记录。"""
        return {"class": self.__class__.__name__, "params": self._config_params | {"p": self.p, "seed": self.seed}}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseTransform":
        """根据配置重新构造实例，子类可按需重载。"""
        params = dict(config.get("params", {}))
        return cls(**params)
