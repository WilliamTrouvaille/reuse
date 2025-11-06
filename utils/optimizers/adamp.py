#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025-11-06 14:50:45
@author  : William_Trouvaille
@function: 提供 AdamP 优化器实现，结合权重方向投影降低权重范数膨胀
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import torch
from loguru import logger
from torch import Tensor
from torch.optim.optimizer import Optimizer

from .types import Betas2, OptFloat, OptLossClosure, Params

__all__: Tuple[str, ...] = ("AdamP",)


# ========================================================================
# 1. AdamP 优化器核心实现
# ========================================================================


class AdamP(Optimizer):
    r"""
    AdamP 在 Adam 的基础上引入对尺度不变参数的正交投影，限制权重范数膨胀现象。

    该算法最初发表于 `Slowing Down the Weight Norm Increase in Momentum-based Optimizers`
    （https://arxiv.org/abs/2006.08217），默认超参取值与原论文保持一致以确保复现性。
    """

    # --- 1.1 初始化 ---
    def __init__(
        self,
        params: Params,
        lr: float = 1e-3,
        betas: Betas2 = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        delta: float = 0.1,
        wd_ratio: float = 0.1,
        nesterov: bool = False,
    ) -> None:
        logger.info("=" * 60)
        logger.info("初始化 AdamP 优化器".center(80))
        logger.info("=" * 60)
        logger.debug(
            "AdamP 超参数: lr=%s, betas=%s, eps=%s, weight_decay=%s, delta=%s, "
            "wd_ratio=%s, nesterov=%s",
            lr,
            betas,
            eps,
            weight_decay,
            delta,
            wd_ratio,
            nesterov,
        )

        assert lr > 0.0, "学习率必须为正数以避免训练停滞"
        assert eps >= 0.0, "数值稳定系数不得为负数"
        assert 0.0 <= betas[0] < 1.0, "第一阶动量 β1 需位于 [0, 1)"
        assert 0.0 <= betas[1] < 1.0, "第二阶动量 β2 需位于 [0, 1)"
        assert weight_decay >= 0.0, "权重衰减系数不得为负"
        assert delta >= 0.0, "尺度阈值必须非负"
        assert wd_ratio >= 0.0, "尺度不变参数的额外权重衰减比例必须非负"

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            delta=delta,
            wd_ratio=wd_ratio,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    # --- 1.2 辅助视图函数 ---
    @staticmethod
    def _channel_view(tensor: Tensor) -> Tensor:
        """
        调整张量为 [out_channels, -1] 的形状以便计算通道级相似度。
        """
        return tensor.view(tensor.size(0), -1)

    @staticmethod
    def _layer_view(tensor: Tensor) -> Tensor:
        """
        调整张量为 [1, -1] 的形状以便计算层级相似度。
        """
        return tensor.view(1, -1)

    # --- 1.3 余弦相似度 ---
    @staticmethod
    def _cosine_similarity(
        tensor_x: Tensor,
        tensor_y: Tensor,
        eps: float,
        view_fn: Callable[[Tensor], Tensor],
    ) -> Tensor:
        """
        通过指定视图函数计算两张量的逐向量余弦相似度。
        """
        reshaped_x = view_fn(tensor_x)
        reshaped_y = view_fn(tensor_y)

        x_norm = reshaped_x.norm(dim=1).add_(eps)  # torch.norm 计算范数
        y_norm = reshaped_y.norm(dim=1).add_(eps)  # torch.norm 计算范数
        dot_product = (reshaped_x * reshaped_y).sum(dim=1)  # torch.sum 累加向量积

        return dot_product.abs() / x_norm / y_norm

    # --- 1.4 投影操作 ---
    def _projection(
        self,
        parameter: Tensor,
        gradient: Tensor,
        perturb: Tensor,
        delta: float,
        wd_ratio: float,
        eps: float,
    ) -> Tuple[Tensor, float]:
        """
        将扰动向量在尺度不变空间内执行正交投影，并返回投影后的扰动及权重衰减比例。
        """
        weight_decay_multiplier = 1.0
        expand_size = [-1] + [1] * (len(parameter.shape) - 1)

        for view_fn in (self._channel_view, self._layer_view):
            cosine_sim = self._cosine_similarity(
                gradient, parameter.data, eps, view_fn
            )

            max_similarity = cosine_sim.max()
            threshold = delta / math.sqrt(view_fn(parameter.data).size(1))

            if max_similarity < threshold:
                parameter_norm = view_fn(parameter.data).norm(dim=1).view(
                    expand_size
                ).add_(eps)
                projected_weight = parameter.data / parameter_norm

                projected_component = view_fn(projected_weight * perturb).sum(
                    dim=1
                ).view(expand_size)
                perturb -= projected_weight * projected_component
                weight_decay_multiplier = wd_ratio

                logger.debug("触发 AdamP 投影并调整 weight decay 比例")
                return perturb, weight_decay_multiplier

        return perturb, weight_decay_multiplier

    # --- 1.5 单步更新 ---
    def step(self, closure: OptLossClosure = None) -> OptFloat:
        """
        执行一次优化步骤，若提供 closure 将在异常时记录日志并回退。
        """
        logger.debug("AdamP 开始执行 step")
        loss: OptFloat = None

        if closure is not None:
            try:
                loss = closure()
            except RuntimeError as runtime_error:
                logger.exception(
                    "AdamP 闭包执行失败，保留当前参数；请检查模型或数据"
                )
                raise runtime_error

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            nesterov_enabled = group["nesterov"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                gradient = parameter.grad.data
                assert torch.isfinite(gradient).all(), (
                    "检测到非有限梯度，AdamP 无法继续优化"
                )

                state = self.state[parameter]

                if len(state) == 0:
                    state["step"] = 0

                    state["exp_avg"] = torch.zeros_like(
                        parameter.data, memory_format=torch.preserve_format
                    )  # torch.zeros_like 生成与参数相同形状的零张量
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter.data, memory_format=torch.preserve_format
                    )  # torch.zeros_like 生成与参数相同形状的零张量

                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]

                state["step"] += 1
                bias_correction1 = 1 - beta1**state["step"]
                bias_correction2 = 1 - beta2**state["step"]

                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    gradient, gradient, value=1 - beta2
                )  # Tensor.addcmul_ 做指数平方动量累积

                denom = (
                    exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                ).add_(group["eps"])
                step_size = group["lr"] / bias_correction1

                if nesterov_enabled:
                    perturb = (beta1 * exp_avg + (1 - beta1) * gradient) / denom
                else:
                    perturb = exp_avg / denom

                weight_decay_multiplier = 1.0
                if len(parameter.shape) > 1:
                    perturb, weight_decay_multiplier = self._projection(
                        parameter,
                        gradient,
                        perturb,
                        group["delta"],
                        group["wd_ratio"],
                        group["eps"],
                    )

                if group["weight_decay"] > 0:
                    decay = (
                        1 - group["lr"] * group["weight_decay"] * weight_decay_multiplier
                    )
                    parameter.data.mul_(decay)

                parameter.data.add_(perturb, alpha=-step_size)

        logger.debug("AdamP 完成 step 更新")
        return loss
