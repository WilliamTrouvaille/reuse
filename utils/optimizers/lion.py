#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025-11-06 15:00:00
@author  : William_Trouvaille
@function: 提供 Lion 优化器实现，通过符号动量降低内存开销并提升训练效率
"""

from __future__ import annotations

from typing import Tuple

import torch
from loguru import logger
from torch import Tensor
from torch.optim.optimizer import Optimizer

from ..decorators import no_grad
from .types import Betas2, OptFloat, OptLossClosure, Params

__all__: Tuple[str, ...] = ("Lion",)


# ========================================================================
# 1. Lion 优化器核心实现
# ========================================================================


class Lion(Optimizer):
    r"""
    Lion (EvoLved Sign Momentum) 通过符号函数更新降低内存开销并提升训练效率。

    该算法最初发表于 `Symbolic Discovery of Optimization Algorithms`
    (https://arxiv.org/abs/2302.06675)，核心思想是仅跟踪一阶动量而不跟踪二阶动量，
    并使用 sign() 函数进行参数更新，从而在保持优化性能的同时显著降低内存消耗。

    相较于 Adam/AdamW：
        - 内存效率：仅需存储一阶动量（节省约 1/2 显存）
        - 更新方式：使用 sign(interpolation) 替代自适应学习率
        - 超参调整：需要更小的学习率（约为 Adam 的 1/3 到 1/10）
        - 正则化：需要更大的 weight_decay（约为 Adam 的 3 到 10 倍）

    使用注意事项：
        1. 学习率建议：lr = 1e-4（Adam 通常使用 1e-3）
        2. 权重衰减建议：weight_decay = 0.01 到 0.1（Adam 通常使用 0.01）
        3. 批量大小：Lion 的增益随 batch size 增大而增大
        4. 适用场景：在某些大规模语言模型和视觉任务中表现优异
        5. 局限性：在某些大型语言模型和文本/图像数据集上未能超越 AdamW
    """

    # --- 1.1 初始化 ---
    def __init__(
        self,
        params: Params,
        lr: float = 1e-4,
        betas: Betas2 = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        """
        初始化 Lion 优化器。

        参数说明:
            params: 待优化的参数（或参数组）
            lr: 学习率，建议使用比 Adam 小的值（例如 1e-4）
                - Lion 对学习率更敏感，过大的学习率会导致训练不稳定
            betas: 动量插值系数 (β1, β2)
                - β1: 用于计算更新方向的插值系数（默认 0.9）
                - β2: 用于更新动量的指数衰减率（默认 0.99）
                - 注意：与 Adam 不同，这里的 β1 和 β2 含义不同
            weight_decay: 解耦权重衰减系数，建议使用比 Adam 大的值
                - 由于 Lion 的更新方式不同，需要更强的正则化来防止过拟合
        """
        logger.info("=" * 60)
        logger.info("初始化 Lion 优化器".center(60))
        logger.info("=" * 60)
        logger.debug(
            "Lion 超参数: lr=%s, betas=%s, weight_decay=%s",
            lr,
            betas,
            weight_decay,
        )

        # --- 1.1.1 超参数合法性检查 ---
        assert lr > 0.0, "学习率必须为正数以避免训练停滞"
        assert 0.0 <= betas[0] < 1.0, "第一个 beta 参数 β1 需位于 [0, 1)"
        assert 0.0 <= betas[1] < 1.0, "第二个 beta 参数 β2 需位于 [0, 1)"
        assert weight_decay >= 0.0, "权重衰减系数不得为负"

        # --- 1.1.2 构建默认超参数字典并调用父类初始化 ---
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

        logger.info("Lion 优化器初始化完成")

    # --- 1.2 单步更新 ---
    @no_grad
    def step(self, closure: OptLossClosure = None) -> OptFloat:
        """
        执行一次优化步骤，更新所有参数。

        Why: 实现 Lion 的核心更新逻辑，包括动量插值、符号更新和权重衰减。
        What: 遍历所有参数组和参数，应用 Lion 更新规则：
              1. 计算插值动量：c_t = β1 * m_{t-1} + (1 - β1) * g_t
              2. 使用符号更新参数：w_t = w_{t-1} - lr * sign(c_t) - lr * λ * w_{t-1}
              3. 更新动量：m_t = β2 * m_{t-1} + (1 - β2) * g_t

        参数:
            closure: 可选的闭包函数，用于重新计算损失（用于某些优化算法）

        返回:
            损失值（如果提供了 closure）
        """
        logger.debug("Lion 开始执行 step")
        loss: OptFloat = None

        # --- 1.2.1 执行闭包（如果提供） ---
        if closure is not None:
            try:
                with torch.enable_grad():
                    loss = closure()
            except RuntimeError as runtime_error:
                logger.exception(
                    "Lion 闭包执行失败，保留当前参数；请检查模型或数据"
                )
                raise runtime_error

        # --- 1.2.2 遍历所有参数组 ---
        for group in self.param_groups:
            # --- 1.2.2.1 提取当前组的超参数 ---
            beta1, beta2 = group["betas"]

            # --- 1.2.2.2 遍历当前组的所有参数 ---
            for parameter in group["params"]:
                # 跳过没有梯度的参数（如冻结层）
                if parameter.grad is None:
                    continue

                # --- 1.2.2.2.1 梯度合法性检查 ---
                gradient = parameter.grad
                assert torch.isfinite(gradient).all(), (
                    "检测到非有限梯度（NaN或Inf），Lion 无法继续优化"
                )

                # --- 1.2.2.2.2 应用解耦权重衰减 ---
                # AdamW 风格的权重衰减：w = w * (1 - lr * λ)
                # 在参数更新之前先应用权重衰减，确保与梯度更新解耦
                if group["weight_decay"] > 0:
                    decay_factor = 1 - group["lr"] * group["weight_decay"]
                    parameter.data.mul_(decay_factor)

                # --- 1.2.2.2.3 初始化或获取状态字典 ---
                state = self.state[parameter]
                if len(state) == 0:
                    # 初始化一阶动量 m_0 = 0
                    # Lion 只需要跟踪一阶动量，不需要二阶动量（相比 Adam 节省内存）
                    state["exp_avg"] = torch.zeros_like(
                        parameter.data, memory_format=torch.preserve_format
                    )  # 保持与参数相同的内存格式以提高效率

                # --- 1.2.2.2.4 获取动量状态 ---
                exp_avg: Tensor = state["exp_avg"]  # m_{t-1}

                # --- 1.2.2.2.5 计算插值更新方向 ---
                # c_t = β1 * m_{t-1} + (1 - β1) * g_t
                # 这是 Lion 的核心创新之一：使用不同的 beta 系数进行插值
                # 与标准动量不同，这里的插值仅用于确定更新方向，不用于更新动量
                update = exp_avg * beta1 + gradient * (1 - beta1)

                # --- 1.2.2.2.6 使用符号函数更新参数 ---
                # w_t = w_{t-1} - lr * sign(c_t)
                # sign() 函数将更新方向归一化为 {-1, 0, 1}，这是 Lion 的另一个核心创新
                # 优点：1) 降低方差，提高训练稳定性
                #       2) 减少计算量（不需要计算平方根）
                #       3) 使得优化器对梯度尺度不敏感
                parameter.data.add_(torch.sign(update), alpha=-group["lr"])

                # --- 1.2.2.2.7 更新动量 ---
                # m_t = β2 * m_{t-1} + (1 - β2) * g_t
                # 注意：这里使用 β2（而非 β1）来更新动量
                # 通常 β2 > β1，意味着动量衰减得更慢
                # 这种设计使得 Lion 能够"记住"更长历史的梯度信息
                exp_avg.mul_(beta2).add_(gradient, alpha=1 - beta2)

        logger.debug("Lion 完成 step 更新")
        return loss
