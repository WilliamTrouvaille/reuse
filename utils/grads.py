#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/07
@author  : William_Trouvaille
@function: 梯度工具函数
@detail:
    本文件源自 PyTorch Lightning 项目
    原始许可证: Apache License 2.0
    原始版权: Copyright The Lightning AI team.
    原始仓库: https://github.com/Lightning-AI/pytorch-lightning
    原始文件: lightning/pytorch/utilities/grads.py
"""

# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union, Dict, Optional

import torch
from loguru import logger
from torch.nn import Module


def grad_norm(
        module: Module,
        norm_type: Union[float, int, str] = 2.0,
        group_separator: str = "/"
) -> Dict[str, float]:
    """
    计算模型各参数的梯度范数和总梯度范数。

    总梯度范数是对所有梯度一起计算的，就像它们被串联成一个向量一样。
    这对于监控训练过程中的梯度爆炸/消失非常有用。

    参数:
        module (nn.Module): 要检查的 PyTorch 模型
        norm_type (float | int | str): 使用的 p-范数类型，会转换为 float。
                                        可以是 'inf' 表示无穷范数。
                                        默认为 2.0 (L2 范数)
        group_separator (str): 日志记录器用于分组梯度范数的分隔符字符串，
                               用于将它们放在自己的子文件夹中。默认为 "/"

    返回:
        dict[str, float]: 包含以下内容的字典:
            - 每个参数梯度的 p-范数（键名格式: "grad_{norm_type}_norm{separator}{param_name}"）
            - 总梯度 p-范数（键名: "grad_{norm_type}_norm_total"）

    引发:
        ValueError: 如果 norm_type <= 0

    示例:
        >>> import torch.nn as nn
        >>> model = nn.Linear(10, 5)
        >>> loss = model(torch.randn(3, 10)).sum()
        >>> loss.backward()
        >>> norms = grad_norm(model, norm_type=2.0)
        >>> print(f"总梯度范数: {norms['grad_2.0_norm_total']:.6f}")
        >>> print(f"权重梯度范数: {norms['grad_2.0_norm/weight']:.6f}")

    注意:
        - 只会计算 `p.grad is not None` 的参数（即参与了反向传播的参数）
        - 如果没有任何参数有梯度，返回空字典
        - 对于大型模型，建议使用 norm_type=2.0（L2 范数）以平衡计算效率和信息量
    """
    # 将 norm_type 转换为 float（支持传入 'inf' 字符串）
    norm_type = float(norm_type)

    # 验证 norm_type 的有效性
    if norm_type <= 0:
        raise ValueError(
            f"`norm_type` 必须是正数或 'inf'（无穷范数）。得到: {norm_type}"
        )

    # 计算每个参数的梯度范数
    # 使用字典推导式，只处理有梯度的参数
    norms = {
        f"grad_{norm_type}_norm{group_separator}{name}": p.grad.data.norm(norm_type).item()
        for name, p in module.named_parameters()
        if p.grad is not None
    }

    # 如果有梯度，计算总梯度范数
    if norms:
        # 将所有梯度范数作为张量，然后计算它们的范数
        # 这相当于将所有梯度展平为一个向量后计算范数
        total_norm = torch.tensor(list(norms.values())).norm(norm_type).item()
        norms[f"grad_{norm_type}_norm_total"] = total_norm

    return norms


def log_grad_norm(
        module: Module,
        norm_type: Union[float, int, str] = 2.0,
        log_per_parameter: bool = False
) -> Optional[float]:
    """
    计算并记录梯度范数到日志。

    这是 grad_norm() 的便捷封装，自动将结果记录到日志中。

    参数:
        module (nn.Module): 要检查的 PyTorch 模型
        norm_type (float | int | str): p-范数类型（默认: 2.0）
        log_per_parameter (bool): 是否记录每个参数的梯度范数（默认: False）
                                  如果为 True，会为每个参数输出一行日志

    返回:
        float 或 None: 总梯度范数，如果没有梯度则返回 None

    示例:
        >>> # 在训练循环中使用
        >>> loss.backward()
        >>> total_norm = log_grad_norm(model, norm_type=2.0)
        >>> optimizer.step()
    """
    norms = grad_norm(module, norm_type=norm_type)

    if not norms:
        logger.warning("模型中没有参数有梯度，跳过梯度范数记录")
        return None

    # 提取总梯度范数
    total_key = f"grad_{float(norm_type)}_norm_total"
    total_norm = norms.get(total_key)

    if total_norm is not None:
        logger.debug(f"总梯度范数 (L{norm_type}): {total_norm:.6f}")

        # (可选) 记录每个参数的梯度范数
        if log_per_parameter:
            for name, value in norms.items():
                if name != total_key:
                    # 移除前缀以获得简洁的参数名
                    param_name = name.split("/")[-1] if "/" in name else name
                    logger.debug(f"  {param_name}: {value:.6f}")

    return total_norm


def check_grad_anomaly(
        module: Module,
        max_norm_threshold: float = 10.0,
        norm_type: Union[float, int, str] = 2.0
) -> bool:
    """
    检查梯度是否出现异常（爆炸或消失）。

    参数:
        module (nn.Module): 要检查的 PyTorch 模型
        max_norm_threshold (float): 梯度范数的阈值，超过此值认为发生梯度爆炸（默认: 10.0）
        norm_type (float | int | str): p-范数类型（默认: 2.0）

    返回:
        bool: True 表示检测到梯度异常（爆炸或消失），False 表示正常

    示例:
        >>> loss.backward()
        >>> if check_grad_anomaly(model, max_norm_threshold=10.0):
        ...     logger.warning("检测到梯度异常！")
        ...     optimizer.zero_grad()  # 跳过这一步
        ... else:
        ...     optimizer.step()
    """
    norms = grad_norm(module, norm_type=norm_type)

    if not norms:
        logger.warning("模型中没有参数有梯度")
        return False

    total_key = f"grad_{float(norm_type)}_norm_total"
    total_norm = norms.get(total_key)

    if total_norm is None:
        return False

    # 检查梯度爆炸
    if total_norm > max_norm_threshold:
        logger.error(
            f"检测到梯度爆炸！总梯度范数: {total_norm:.6f} > 阈值: {max_norm_threshold}"
        )
        return True

    # 检查梯度消失（总范数接近 0）
    if total_norm < 1e-7:
        logger.error(f"检测到梯度消失！总梯度范数: {total_norm:.10f} ≈ 0")
        return True

    # 检查是否有 NaN 或 Inf
    for name, p in module.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any():
                logger.error(f"参数 '{name}' 的梯度包含 NaN！")
                return True
            if torch.isinf(p.grad).any():
                logger.error(f"参数 '{name}' 的梯度包含 Inf！")
                return True

    return False
