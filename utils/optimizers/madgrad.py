#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025-11-06 16:25:30
@author  : William_Trouvaille
@function: 提供 MADGRAD 优化器实现，引入动量化的自适应双平均策略提升收敛稳定性
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from loguru import logger
from torch import Tensor
from torch.optim.optimizer import Optimizer

from ..decorators import no_grad
from .types import OptFloat, OptLossClosure, Params

__all__: Tuple[str, ...] = ("MadGrad",)


# ========================================================================
# 1. MADGRAD 优化器核心实现
# ========================================================================


class MadGrad(Optimizer):
    r"""
    MADGRAD (Momentumized, Adaptive, Dual Averaged Gradient) 使用双平均思想
    结合动量项，能够在保持鲁棒性的同时获得更快的下降速度。

    该算法提出于 `Adaptivity without Compromise: A Momentumized, Adaptive,
    Dual Averaged Gradient Method for Stochastic Optimization`
    (https://arxiv.org/abs/2101.11075)，其核心包含三部分：
        1. 梯度平方的累积（grad_sum_sq），提供自适应的缩放
        2. 梯度的加权求和（s），用于追踪一阶信息
        3. 可选的动量平滑（momentum），缓解更新噪声
    """

    # --- 1.1 初始化 ---
    def __init__(
        self,
        params: Params,
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        """
        初始化 MADGRAD 优化器，并对核心超参数进行合法性校验。

        参数说明:
            params: 待优化的参数集合或参数组
            lr: 学习率，控制整体步长，需为正数
            momentum: 动量系数，范围 [0, 1)，用于平滑参数更新
            weight_decay: 解耦式权重衰减系数
            eps: 数值稳定项，避免分母为零
        """
        logger.info("=" * 60)
        logger.info("初始化 MADGRAD 优化器".center(80))
        logger.info("=" * 60)
        logger.debug(
            "MADGRAD 超参数: lr=%s, momentum=%s, weight_decay=%s, eps=%s",
            lr,
            momentum,
            weight_decay,
            eps,
        )

        # --- 1.1.1 超参数合法性检查 ---
        assert 0.0 <= momentum < 1.0, "动量系数需处于 [0, 1)，否则会导致发散"
        assert lr > 0.0, "学习率必须大于 0，才能确保参数向最优值移动"
        assert weight_decay >= 0.0, "权重衰减必须非负，以避免梯度方向被反转"
        assert eps >= 0.0, "数值稳定项需非负，否则无法保证分母稳定"

        # --- 1.1.2 构造默认超参数并调用父类 ---
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            eps=eps,
            k=0,
        )
        super().__init__(params, defaults)

        # --- 1.1.3 为每个参数建立状态缓存 ---
        for param_group in self.param_groups:
            for parameter in param_group["params"]:
                if parameter.requires_grad is False:
                    logger.debug(
                        "检测到 requires_grad=False 的参数，MADGRAD 将跳过该参数"
                    )
                    continue
                state = self.state[parameter]
                # grad_sum_sq 累积梯度平方用于自适应缩放，初值为 0
                # 创建与参数同形状的零张量，保持原有内存格式以提升性能
                state["grad_sum_sq"] = torch.zeros_like(
                    parameter.data, memory_format=torch.preserve_format
                )
                # s 保存梯度的加权和（一阶动量信息），初值同样为 0
                state["s"] = torch.zeros_like(
                    parameter.data, memory_format=torch.preserve_format
                )
                # 当存在动量时需要保存初始参数 x0 以做平滑插值
                if momentum != 0.0:
                    state["x0"] = torch.clone(
                        parameter.data, memory_format=torch.preserve_format
                    )

        logger.info("MADGRAD 优化器初始化完成")

    # --- 1.2 单步更新 ---
    @no_grad
    def step(self, closure: OptLossClosure = None) -> OptFloat:
        """
        执行一次优化步骤。

        该方法实现了 MADGRAD 的核心更新规则：将梯度累积、自适应缩放与动量平滑应用到所有参数。
        遍历所有参数组并更新参数状态，必要时执行闭包以重新计算损失。
        """
        logger.debug("MADGRAD 正在执行 step")
        loss: OptFloat = None

        # --- 1.2.1 执行闭包（如有） ---
        if closure is not None:
            try:
                with torch.enable_grad():
                    loss = closure()
            except RuntimeError as runtime_error:
                logger.exception("MADGRAD 闭包执行失败，保留当前参数状态")
                raise runtime_error

        # --- 1.2.2 遍历参数组 ---
        for group in self.param_groups:
            eps = group["eps"]
            iteration_index: int = group["k"]
            effective_lr = group["lr"] + eps  # 避免 lr 为 0 导致除法出错，确保数值稳定性
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            ck = 1.0 - momentum  # ck 控制动量平滑的强度，值越大则更新越激进
            lamb = effective_lr * math.sqrt(iteration_index + 1.0)  # 学习率衰减因子，随迭代递增

            # --- 1.2.2.1 遍历参数 ---
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue

                # --- 1.2.2.1.1 梯度有限性检查 ---
                assert torch.isfinite(gradient).all(), (
                    "MADGRAD 检测到非有限梯度值（NaN/Inf），请检查前向或数据"
                )

                state = self.state[parameter]
                grad_sum_sq: Tensor = state["grad_sum_sq"]
                s_buffer: Tensor = state["s"]

                if momentum != 0.0 and gradient.is_sparse:
                    raise RuntimeError(
                        "MADGRAD 在使用动量时不支持稀疏梯度，请关闭动量或改用稠密梯度"
                    )

                # --- 1.2.2.1.2 权重衰减 ---
                if weight_decay != 0.0:
                    if gradient.is_sparse:
                        raise RuntimeError(
                            "MADGRAD 的权重衰减不支持稀疏梯度参数，请改用稠密优化"
                        )
                    # 对梯度直接加上 λ * w，实现解耦式 L2 正则化
                    gradient = gradient.add(parameter.data, alpha=weight_decay)

                if gradient.is_sparse:
                    # --- 1.2.2.1.3 稀疏梯度分支 ---
                    # 对稀疏梯度进行合并优化，减少内存碎片
                    gradient = gradient.coalesce()
                    grad_values = gradient._values()

                    # 只对有梯度的位置进行掩码操作，提升计算效率
                    parameter_masked = parameter.sparse_mask(gradient)
                    grad_sum_sq_masked = grad_sum_sq.sparse_mask(gradient)
                    s_masked = s_buffer.sparse_mask(gradient)

                    # 使用梯度平方的立方根作为自适应缩放因子，eps 确保数值稳定
                    rms_masked_vals = (
                        grad_sum_sq_masked._values().pow(1.0 / 3.0).add_(eps)
                    )
                    # 计算参考点 x0 = w + s / rms，用于后续参数更新
                    x0_masked_vals = parameter_masked._values().addcdiv(
                        s_masked._values(), rms_masked_vals, value=1.0
                    )

                    # 累积梯度平方用于更新缩放因子，实现自适应性
                    grad_sq = gradient * gradient
                    grad_sum_sq.add_(grad_sq, alpha=lamb)
                    grad_sum_sq_masked.add_(grad_sq, alpha=lamb)

                    # 重新计算缩放因子，因为 grad_sum_sq 已更新
                    rms_masked_vals = (
                        grad_sum_sq_masked._values().pow_(1.0 / 3.0).add_(eps)
                    )

                    # 累积梯度和 s，维持双平均结构
                    s_buffer.add_(gradient, alpha=lamb)
                    s_masked._values().add_(grad_values, alpha=lamb)

                    # 计算更新后的参数值：p_next = x0 - s / rms
                    p_next_masked_vals = x0_masked_vals.addcdiv(
                        s_masked._values(), rms_masked_vals, value=-1.0
                    )
                    # 应用参数更新，仅更新有梯度的位置
                    parameter_masked._values().add_(p_next_masked_vals, alpha=-1.0)
                    parameter.data.add_(parameter_masked, alpha=-1.0)
                else:
                    # --- 1.2.2.1.4 稠密梯度分支 ---
                    if momentum == 0.0:
                        # 当无动量时需显式构造参考点 x0 = w + s / rms
                        rms = grad_sum_sq.pow(1.0 / 3.0).add_(eps)
                        x0_value = parameter.data.addcdiv(s_buffer, rms, value=1.0)
                    else:
                        x0_value = state["x0"]  # 使用保存的初始参数作为参考点

                    # 累积梯度平方确保自适应性，使用 lamb 作为权重
                    grad_sum_sq.addcmul_(gradient, gradient, value=lamb)
                    # 重新计算均方根缩放因子，立方根提供更温和的缩放
                    rms = grad_sum_sq.pow(1.0 / 3.0).add_(eps)

                    # 累积梯度和 s，维持双平均结构的核心思想
                    s_buffer.add_(gradient, alpha=lamb)

                    if momentum == 0.0:
                        # 无动量时直接更新：w_new = x0 - s / rms
                        parameter.data.copy_(x0_value.addcdiv(s_buffer, rms, value=-1.0))
                    else:
                        # 计算原始 MADGRAD 更新方向
                        z_value = x0_value.addcdiv(s_buffer, rms, value=-1.0)
                        # 使用动量平滑当前参数，降低更新噪声：w_new = (1-ck)*w + ck*z
                        parameter.data.mul_(1.0 - ck).add_(z_value, alpha=ck)

            group["k"] = iteration_index + 1

        logger.debug("MADGRAD 完成 step 更新")
        return loss
