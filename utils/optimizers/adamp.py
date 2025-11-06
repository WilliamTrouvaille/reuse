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
    （https://arxiv.org/abs/2006.08217），默认超参取值与原论文保持一致。
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
        """
        初始化 AdamP 优化器。

        参数说明:
            params: 待优化的参数（或参数组）
            lr: 学习率，控制每次更新的步长大小
            betas: 一阶和二阶动量的指数衰减率 (β1, β2)
                - β1 用于梯度的指数移动平均
                - β2 用于梯度平方的指数移动平均
            eps: 数值稳定项，防止除零错误
            weight_decay: L2 正则化系数，用于防止过拟合
            delta: 投影触发阈值，当余弦相似度低于此值时触发投影
                - 较小的 delta 会更频繁地触发投影，增强正则化效果
            wd_ratio: 尺度不变参数的额外权重衰减比例
                - 当触发投影时，权重衰减会乘以此比例
            nesterov: 是否使用 Nesterov 动量加速
                - 开启后会在当前位置"向前看一步"来计算梯度
        """
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

        # --- 1.1.1 超参数合法性检查 ---
        assert lr > 0.0, "学习率必须为正数以避免训练停滞"
        assert eps >= 0.0, "数值稳定系数不得为负数"
        assert 0.0 <= betas[0] < 1.0, "第一阶动量 β1 需位于 [0, 1)"
        assert 0.0 <= betas[1] < 1.0, "第二阶动量 β2 需位于 [0, 1)"
        assert weight_decay >= 0.0, "权重衰减系数不得为负"
        assert delta >= 0.0, "尺度阈值必须非负"
        assert wd_ratio >= 0.0, "尺度不变参数的额外权重衰减比例必须非负"

        # --- 1.1.2 构建默认超参数字典并调用父类初始化 ---
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

        Why: 在卷积层等参数中，需要按输出通道分别计算梯度与权重的相似度。
        What: 将形状 [C_out, C_in, K_h, K_w] 的权重张量重塑为 [C_out, C_in*K_h*K_w]。

        示例:
            输入形状 [64, 3, 3, 3] (64个输出通道，3x3卷积核，3个输入通道)
            输出形状 [64, 27] (每个输出通道对应一个27维向量)
        """
        return tensor.view(tensor.size(0), -1)

    @staticmethod
    def _layer_view(tensor: Tensor) -> Tensor:
        """
        调整张量为 [1, -1] 的形状以便计算层级相似度。

        Why: 对于小参数（如bias）或需要全局投影时，应将整个参数视为单一向量。
        What: 将任意形状的张量展平为 [1, N]，其中 N 是参数总数。

        示例:
            输入形状 [64, 3, 3, 3]
            输出形状 [1, 1728] (所有参数展平为一个1728维向量)
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

        Why: 余弦相似度衡量梯度与权重的方向一致性，用于判断是否需要投影。
        What: 计算 cos(θ) = |<x, y>| / (||x|| * ||y||)，返回每个向量对的相似度。

        参数:
            tensor_x: 第一个张量（通常是梯度）
            tensor_y: 第二个张量（通常是权重）
            eps: 数值稳定项，避免范数为零导致除零
            view_fn: 视图函数，用于重塑张量形状（channel_view 或 layer_view）

        返回:
            形状为 [N] 的张量，其中 N 取决于 view_fn 的输出第一维
        """
        # --- 1.3.1 重塑张量为统一的二维形式 [N, D] ---
        reshaped_x = view_fn(tensor_x)  # [N, D]，N个D维向量
        reshaped_y = view_fn(tensor_y)  # [N, D]

        # --- 1.3.2 计算每个向量的L2范数 ||x|| 和 ||y|| ---
        # dim=1 表示沿着特征维度（D）计算范数，得到 [N] 的向量
        x_norm = reshaped_x.norm(dim=1).add_(eps)  # [N]，add_ 是原地加法，防止除零
        y_norm = reshaped_y.norm(dim=1).add_(eps)  # [N]

        # --- 1.3.3 计算内积 <x, y> ---
        # 逐元素相乘后沿特征维度求和，得到 [N] 的向量
        dot_product = (reshaped_x * reshaped_y).sum(dim=1)  # [N]

        # --- 1.3.4 计算余弦相似度并取绝对值 ---
        # abs() 确保相似度为正值，因为我们只关心方向一致性的强度
        return dot_product.abs() / x_norm / y_norm  # [N]

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

        Why: 当梯度与权重方向过于一致时，参数更新会导致权重范数快速增长。
             通过投影去除扰动在权重方向上的分量，可以有效抑制范数膨胀。

        What: 检查梯度与权重的余弦相似度，若低于阈值则执行正交投影:
             perturb_new = perturb - (perturb · w_normalized) * w_normalized

        参数:
            parameter: 当前参数张量 w
            gradient: 当前梯度张量 g
            perturb: 待投影的扰动向量（Adam更新方向）
            delta: 投影触发阈值
            wd_ratio: 触发投影后的权重衰减缩放比例
            eps: 数值稳定项

        返回:
            (投影后的扰动, 权重衰减乘数)
        """
        # --- 1.4.1 初始化权重衰减乘数（默认不改变） ---
        weight_decay_multiplier = 1.0

        # --- 1.4.2 构建广播形状 ---
        # expand_size 用于将 [N] 的向量扩展为与 parameter 相同的形状
        # 例如: parameter.shape = [64, 3, 3, 3]，则 expand_size = [64, 1, 1, 1]
        expand_size = [-1] + [1] * (len(parameter.shape) - 1)

        # --- 1.4.3 尝试两种粒度的投影：通道级 -> 层级 ---
        # 先尝试细粒度（通道级），若不满足条件再尝试粗粒度（层级）
        for view_fn in (self._channel_view, self._layer_view):
            # --- 1.4.3.1 计算梯度与权重的余弦相似度 ---
            cosine_sim = self._cosine_similarity(
                gradient, parameter.data, eps, view_fn
            )  # [N]

            # --- 1.4.3.2 判断是否需要投影 ---
            max_similarity = cosine_sim.max()  # 取最大相似度
            # 阈值根据向量维度调整：维度越高，阈值越小
            # sqrt(D) 归一化确保阈值在不同维度下具有可比性
            threshold = delta / math.sqrt(view_fn(parameter.data).size(1))

            # --- 1.4.3.3 若最大相似度低于阈值，触发投影 ---
            if max_similarity < threshold:
                # --- 1.4.3.3.1 归一化权重向量 w_normalized = w / ||w|| ---
                # 计算每个向量的L2范数 [N] -> [N, 1, 1, ...] 以便广播
                parameter_norm = view_fn(parameter.data).norm(dim=1).view(
                    expand_size
                ).add_(eps)  # 形状匹配 parameter，例如 [64, 1, 1, 1]

                # 归一化：每个权重向量除以其范数
                projected_weight = parameter.data / parameter_norm  # 与 parameter 同形

                # --- 1.4.3.3.2 计算扰动在归一化权重方向上的投影分量 ---
                # projected_component = <perturb, w_normalized>
                # 先计算内积 perturb · w_normalized，然后沿特征维度求和
                projected_component = view_fn(projected_weight * perturb).sum(
                    dim=1
                ).view(expand_size)  # [N] -> [N, 1, 1, ...]

                # --- 1.4.3.3.3 从扰动中减去投影分量，得到正交部分 ---
                # perturb_new = perturb - projected_component * w_normalized
                # 这样 perturb_new 与 w_normalized 正交，不会增加权重范数
                perturb -= projected_weight * projected_component

                # --- 1.4.3.3.4 调整权重衰减比例 ---
                # 投影后权重衰减会减弱，乘以 wd_ratio 进行补偿
                weight_decay_multiplier = wd_ratio

                logger.debug("触发 AdamP 投影并调整 weight decay 比例")
                return perturb, weight_decay_multiplier

        # --- 1.4.4 若两种粒度都未触发投影，返回原始扰动 ---
        return perturb, weight_decay_multiplier

    # --- 1.5 单步更新 ---
    def step(self, closure: OptLossClosure = None) -> OptFloat:
        """
        执行一次优化步骤，若提供 closure 将在异常时记录日志并回退。

        Why: 实现 AdamP 的核心更新逻辑，包括动量更新、偏差修正、投影和权重衰减。
        What: 遍历所有参数组和参数，应用 AdamP 更新规则。

        参数:
            closure: 可选的闭包函数，用于重新计算损失（用于某些优化算法）

        返回:
            损失值（如果提供了 closure）
        """
        logger.debug("AdamP 开始执行 step")
        loss: OptFloat = None

        # --- 1.5.1 执行闭包（如果提供） ---
        if closure is not None:
            try:
                loss = closure()
            except RuntimeError as runtime_error:
                logger.exception(
                    "AdamP 闭包执行失败，保留当前参数；请检查模型或数据"
                )
                raise runtime_error

        # --- 1.5.2 遍历所有参数组 ---
        for group in self.param_groups:
            # --- 1.5.2.1 提取当前组的超参数 ---
            beta1, beta2 = group["betas"]
            nesterov_enabled = group["nesterov"]

            # --- 1.5.2.2 遍历当前组的所有参数 ---
            for parameter in group["params"]:
                # 跳过没有梯度的参数（如冻结层）
                if parameter.grad is None:
                    continue

                # --- 1.5.2.2.1 梯度合法性检查 ---
                gradient = parameter.grad.data
                assert torch.isfinite(gradient).all(), (
                    "检测到非有限梯度（NaN或Inf），AdamP 无法继续优化"
                )

                # --- 1.5.2.2.2 初始化状态字典（首次访问该参数时） ---
                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0  # 步数计数器，用于偏差修正

                    # 一阶动量：梯度的指数移动平均（EMA）
                    state["exp_avg"] = torch.zeros_like(
                        parameter.data, memory_format=torch.preserve_format
                    )  # 保持与参数相同的内存格式以提高效率

                    # 二阶动量：梯度平方的指数移动平均
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter.data, memory_format=torch.preserve_format
                    )

                # --- 1.5.2.2.3 获取状态变量 ---
                exp_avg: Tensor = state["exp_avg"]  # m_t
                exp_avg_sq: Tensor = state["exp_avg_sq"]  # v_t

                # --- 1.5.2.2.4 更新步数并计算偏差修正系数 ---
                state["step"] += 1
                # Adam 的偏差修正：修正初始化为零导致的偏差
                # bias_correction1 = 1 - β1^t，随着 t 增大趋近于 1
                bias_correction1 = 1 - beta1**state["step"]
                bias_correction2 = 1 - beta2**state["step"]

                # --- 1.5.2.2.5 更新一阶和二阶动量 ---
                # m_t = β1 * m_{t-1} + (1 - β1) * g_t
                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)

                # v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
                # addcmul_ 执行 v_t = β2*v_{t-1} + (1-β2)*g_t*g_t
                exp_avg_sq.mul_(beta2).addcmul_(
                    gradient, gradient, value=1 - beta2
                )

                # --- 1.5.2.2.6 计算自适应学习率的分母 ---
                # denom = sqrt(v_t / (1 - β2^t)) + ε
                denom = (
                    exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                ).add_(group["eps"])

                # 计算有效步长（包含偏差修正）
                step_size = group["lr"] / bias_correction1

                # --- 1.5.2.2.7 计算扰动方向 ---
                if nesterov_enabled:
                    # Nesterov 动量："向前看"一步
                    # perturb = (β1 * m_t + (1 - β1) * g_t) / denom
                    perturb = (beta1 * exp_avg + (1 - beta1) * gradient) / denom
                else:
                    # 标准 Adam 更新方向
                    # perturb = m_t / denom
                    perturb = exp_avg / denom

                # --- 1.5.2.2.8 应用投影（仅对多维参数） ---
                weight_decay_multiplier = 1.0
                if len(parameter.shape) > 1:
                    # 对于卷积核、全连接层权重等多维参数，执行投影
                    # 一维参数（如bias）跳过投影
                    perturb, weight_decay_multiplier = self._projection(
                        parameter,
                        gradient,
                        perturb,
                        group["delta"],
                        group["wd_ratio"],
                        group["eps"],
                    )

                # --- 1.5.2.2.9 应用权重衰减（解耦的L2正则化） ---
                if group["weight_decay"] > 0:
                    # AdamW 风格的权重衰减：w = w * (1 - lr * λ * multiplier)
                    # 直接缩放权重，而非将正则项加到梯度中
                    decay = (
                        1 - group["lr"] * group["weight_decay"] * weight_decay_multiplier
                    )
                    parameter.data.mul_(decay)

                # --- 1.5.2.2.10 更新参数 ---
                # w = w - step_size * perturb
                parameter.data.add_(perturb, alpha=-step_size)

        logger.debug("AdamP 完成 step 更新")
        return loss