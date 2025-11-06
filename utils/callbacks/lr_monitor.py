#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/07
@author  : William_Trouvaille
@function: 学习率监控 - 自动监控和记录优化器的学习率、动量等超参数
@detail:
    本文件部分代码源自 PyTorch Lightning 项目
    原始许可证: Apache License 2.0
    原始版权: Copyright The Lightning AI team.
    原始仓库: https://github.com/Lightning-AI/pytorch-lightning
    原始文件: lightning/pytorch/callbacks/lr_monitor.py
"""

import itertools
from collections import defaultdict
from typing import Any, Literal, Optional

import torch
from loguru import logger
from torch.optim.optimizer import Optimizer


# ========================================================================
# 1. LearningRateMonitor 类
# ========================================================================

class LearningRateMonitor:
    """学习率监控器 - 自动监控和记录学习率、动量、权重衰减等优化器超参数。

    功能特性:
    - 多优化器支持: 自动处理多个优化器和调度器
    - 多参数组支持: 支持优化器的多个参数组（如分层学习率）
    - 灵活记录: 可选记录 momentum、weight_decay 等超参数
    - 智能命名: 自动为优化器和参数组生成唯一名称

    使用示例:
        # 创建学习率监控器
        lr_monitor = LearningRateMonitor(
            logging_interval='epoch',  # 每个 epoch 记录一次
            log_momentum=True,         # 记录动量
            log_weight_decay=True      # 记录权重衰减
        )

        # 初始化（传入优化器和调度器）
        lr_monitor.setup(optimizers=[optimizer], schedulers=[scheduler])

        # 在训练循环中提取学习率信息
        lr_stats = lr_monitor.extract_lr_stats(interval='epoch')
        # lr_stats: {'lr-Adam': 0.001, 'lr-Adam-momentum': 0.9, ...}

    Args:
        logging_interval: 记录间隔，可选值:
            - 'step': 每个训练步记录
            - 'epoch': 每个训练轮次记录
            - None: 根据调度器的间隔自动决定
        log_momentum: 是否记录动量值（默认 False）
        log_weight_decay: 是否记录权重衰减值（默认 False）

    Raises:
        ValueError: 如果 logging_interval 不是 'step'、'epoch' 或 None
    """

    def __init__(
        self,
        logging_interval: Optional[Literal["step", "epoch"]] = None,
        log_momentum: bool = False,
        log_weight_decay: bool = False,
    ) -> None:
        # --- 1.1 参数验证 ---
        if logging_interval not in (None, "step", "epoch"):
            raise ValueError("logging_interval 应为 'step'、'epoch' 或 None")

        # --- 1.2 初始化配置 ---
        self.logging_interval = logging_interval
        self.log_momentum = log_momentum
        self.log_weight_decay = log_weight_decay

        # --- 1.3 初始化内部状态 ---
        self.lrs: dict[str, list[float]] = {}                          # 学习率历史记录
        self.last_momentum_values: dict[str, Optional[float]] = {}     # 最新动量值
        self.last_weight_decay_values: dict[str, Optional[float]] = {} # 最新权重衰减值
        self._optimizer_names: list[list[str]] = []                    # 优化器参数组名称列表

    # ========================================================================
    # 2. 初始化方法
    # ========================================================================

    def setup(
        self,
        optimizers: list[Optimizer],
        schedulers: Optional[list[Any]] = None,
    ) -> None:
        """初始化学习率监控器 - 为所有优化器和调度器生成唯一名称。

        Args:
            optimizers: 优化器列表
            schedulers: 调度器列表（可选），每个调度器应为包含以下键的字典:
                - 'scheduler': torch.optim.lr_scheduler 对象
                - 'interval': 'step' 或 'epoch'（可选）
                - 'name': 自定义名称（可选）

        Raises:
            ValueError: 如果同一优化器的参数组有重复名称
        """
        if schedulers is None:
            schedulers = []

        # --- 2.1 检查动量配置 ---
        if self.log_momentum:
            def _check_no_key(key: str) -> bool:
                # 检查是否有优化器缺少指定的键
                if schedulers:
                    return any(
                        key not in config.get("scheduler").optimizer.defaults
                        for config in schedulers
                    )
                return any(key not in optimizer.defaults for optimizer in optimizers)

            if _check_no_key("momentum") and _check_no_key("betas"):
                logger.warning(
                    "log_momentum=True，但部分优化器不支持 momentum。"
                    "这些优化器的 momentum 将记录为 0"
                )

        # --- 2.2 从调度器和优化器中提取名称 ---
        names: list[list[str]] = []

        # 处理调度器
        (
            sched_hparam_keys,
            optimizers_with_scheduler,
            optimizers_with_scheduler_types,
        ) = self._find_names_from_schedulers(schedulers)
        names.extend(sched_hparam_keys)

        # 处理剩余的优化器（没有调度器的优化器）
        optimizer_hparam_keys, _ = self._find_names_from_optimizers(
            optimizers,
            seen_optimizers=optimizers_with_scheduler,
            seen_optimizer_types=optimizers_with_scheduler_types,
        )
        names.extend(optimizer_hparam_keys)

        # --- 2.3 初始化存储 ---
        names_flatten = list(itertools.chain.from_iterable(names))
        self.lrs = {name: [] for name in names_flatten}
        self.last_momentum_values = {name + "-momentum": None for name in names_flatten}
        self.last_weight_decay_values = {name + "-weight_decay": None for name in names_flatten}
        self._optimizer_names = names

    # ========================================================================
    # 3. 学习率提取方法
    # ========================================================================

    def extract_lr_stats(
        self,
        optimizers: list[Optimizer],
        schedulers: Optional[list[Any]] = None,
        interval: str = "any",
    ) -> dict[str, float]:
        """提取当前的学习率和其他优化器统计信息。

        Args:
            optimizers: 优化器列表
            schedulers: 调度器列表（可选）
            interval: 记录间隔过滤，可选值:
                - 'step': 仅记录 interval='step' 的调度器
                - 'epoch': 仅记录 interval='epoch' 的调度器
                - 'any': 记录所有调度器和优化器

        Returns:
            包含学习率等统计信息的字典，格式如:
            {
                'lr-Adam': 0.001,
                'lr-Adam-momentum': 0.9,
                'lr-Adam-weight_decay': 0.0001,
                ...
            }
        """
        if schedulers is None:
            schedulers = []

        latest_stat = {}

        # --- 3.1 从调度器中提取 ---
        (
            scheduler_hparam_keys,
            optimizers_with_scheduler,
            optimizers_with_scheduler_types,
        ) = self._find_names_from_schedulers(schedulers)
        self._remap_keys(scheduler_hparam_keys)

        for name, config in zip(scheduler_hparam_keys, schedulers):
            config_interval = config.get("interval", "epoch")
            if interval in [config_interval, "any"]:
                opt = config["scheduler"].optimizer
                current_stat = self._get_optimizer_stats(opt, name)
                latest_stat.update(current_stat)

        # --- 3.2 从剩余优化器中提取 ---
        optimizer_hparam_keys, optimizers_without_scheduler = self._find_names_from_optimizers(
            optimizers,
            seen_optimizers=optimizers_with_scheduler,
            seen_optimizer_types=optimizers_with_scheduler_types,
        )
        self._remap_keys(optimizer_hparam_keys)

        for opt, names in zip(optimizers_without_scheduler, optimizer_hparam_keys):
            current_stat = self._get_optimizer_stats(opt, names)
            latest_stat.update(current_stat)

        return latest_stat

    def _get_optimizer_stats(self, optimizer: Optimizer, names: list[str]) -> dict[str, float]:
        """从优化器中提取学习率、动量、权重衰减等统计信息。

        Args:
            optimizer: 优化器对象
            names: 参数组名称列表

        Returns:
            包含统计信息的字典
        """
        stats = {}
        param_groups = optimizer.param_groups
        use_betas = "betas" in optimizer.defaults  # 判断是否使用 betas（如 Adam）

        for pg, name in zip(param_groups, names):
            # 提取学习率
            lr = self._extract_lr(pg, name)
            stats.update(lr)

            # 提取动量
            momentum = self._extract_momentum(
                param_group=pg,
                name=f"{name}-momentum",
                use_betas=use_betas
            )
            stats.update(momentum)

            # 提取权重衰减
            weight_decay = self._extract_weight_decay(pg, f"{name}-weight_decay")
            stats.update(weight_decay)

        return stats

    # ========================================================================
    # 4. 辅助方法 - 提取单个超参数
    # ========================================================================

    def _extract_lr(self, param_group: dict[str, Any], name: str) -> dict[str, float]:
        """提取学习率"""
        lr = param_group["lr"]
        self.lrs[name].append(lr)
        return {name: lr}

    def _extract_momentum(
        self,
        param_group: dict[str, Any],
        name: str,
        use_betas: bool
    ) -> dict[str, float]:
        """提取动量值（支持 momentum 和 betas[0]）"""
        if not self.log_momentum:
            return {}

        # Adam 等优化器使用 betas[0]，SGD 使用 momentum
        momentum = param_group["betas"][0] if use_betas else param_group.get("momentum", 0)
        self.last_momentum_values[name] = momentum
        return {name: momentum}

    def _extract_weight_decay(self, param_group: dict[str, Any], name: str) -> dict[str, float]:
        """提取权重衰减值"""
        if not self.log_weight_decay:
            return {}

        weight_decay = param_group["weight_decay"]
        self.last_weight_decay_values[name] = weight_decay
        return {name: weight_decay}

    # ========================================================================
    # 5. 辅助方法 - 名称生成与管理
    # ========================================================================

    def _remap_keys(self, names: list[list[str]], token: str = "/pg1") -> None:
        """重新映射键名（处理参数组增加的情况）

        当优化器的参数组数量增加时，需要重新映射键名以保持一致性。
        例如，从 "lr-Adam" 变为 "lr-Adam/pg1"。

        Args:
            names: 参数组名称列表
            token: 用于标识参数组的标记（默认 "/pg1"）
        """
        for group_new_names in names:
            for new_name in group_new_names:
                old_name = new_name.replace(token, "")
                if token in new_name and old_name in self.lrs:
                    # 将旧键名的数据迁移到新键名
                    self.lrs[new_name] = self.lrs.pop(old_name)
                elif new_name not in self.lrs:
                    # 创建新键
                    self.lrs[new_name] = []

    def _add_prefix(
        self,
        name: str,
        optimizer_cls: type[Optimizer],
        seen_optimizer_types: defaultdict[type[Optimizer], int]
    ) -> str:
        """为优化器名称添加前缀（处理同类型多优化器的情况）

        如果存在多个相同类型的优化器，添加编号后缀。
        例如: "lr-Adam", "lr-Adam-1", "lr-Adam-2"

        Args:
            name: 原始名称
            optimizer_cls: 优化器类型
            seen_optimizer_types: 已见过的优化器类型计数

        Returns:
            添加前缀后的名称
        """
        if optimizer_cls not in seen_optimizer_types:
            return name
        count = seen_optimizer_types[optimizer_cls]
        return name + f"-{count - 1}" if count > 1 else name

    def _add_suffix(
        self,
        name: str,
        param_groups: list[dict],
        param_group_index: int,
        use_names: bool = True
    ) -> str:
        """为参数组名称添加后缀

        Args:
            name: 优化器名称
            param_groups: 参数组列表
            param_group_index: 参数组索引
            use_names: 是否使用参数组的自定义名称

        Returns:
            添加后缀后的名称
        """
        if len(param_groups) > 1:
            if not use_names:
                return f"{name}/pg{param_group_index + 1}"
            pg_name = param_groups[param_group_index].get("name", f"pg{param_group_index + 1}")
            return f"{name}/{pg_name}"
        if use_names:
            pg_name = param_groups[param_group_index].get("name")
            return f"{name}/{pg_name}" if pg_name else name
        return name

    def _duplicate_param_group_names(self, param_groups: list[dict]) -> set[str]:
        """检测参数组中是否有重复的名称

        Args:
            param_groups: 参数组列表

        Returns:
            重复的名称集合
        """
        names = [pg.get("name", f"pg{i}") for i, pg in enumerate(param_groups, start=1)]
        unique = set(names)
        if len(names) == len(unique):
            return set()
        return {n for n in names if names.count(n) > 1}

    # ========================================================================
    # 6. 辅助方法 - 从调度器和优化器中查找名称
    # ========================================================================

    def _find_names_from_schedulers(
        self,
        schedulers: list[Any],
    ) -> tuple[list[list[str]], list[Optimizer], defaultdict[type[Optimizer], int]]:
        """从调度器列表中提取优化器名称

        Args:
            schedulers: 调度器列表

        Returns:
            (参数组名称列表, 已见优化器列表, 已见优化器类型计数)
        """
        names = []
        seen_optimizers: list[Optimizer] = []
        seen_optimizer_types: defaultdict[type[Optimizer], int] = defaultdict(int)

        for config in schedulers:
            sch = config["scheduler"]
            name = config.get("name") if config.get("name") is not None else "lr-" + sch.optimizer.__class__.__name__

            updated_names = self._check_duplicates_and_update_name(
                sch.optimizer, name, seen_optimizers, seen_optimizer_types, config
            )
            names.append(updated_names)

        return names, seen_optimizers, seen_optimizer_types

    def _find_names_from_optimizers(
        self,
        optimizers: list[Optimizer],
        seen_optimizers: list[Optimizer],
        seen_optimizer_types: defaultdict[type[Optimizer], int],
    ) -> tuple[list[list[str]], list[Optimizer]]:
        """从优化器列表中提取名称（排除已处理的优化器）

        Args:
            optimizers: 优化器列表
            seen_optimizers: 已见优化器列表
            seen_optimizer_types: 已见优化器类型计数

        Returns:
            (参数组名称列表, 未处理的优化器列表)
        """
        names = []
        optimizers_without_scheduler = []

        for optimizer in optimizers:
            if optimizer in seen_optimizers:
                continue

            name = "lr-" + optimizer.__class__.__name__
            updated_names = self._check_duplicates_and_update_name(
                optimizer, name, seen_optimizers, seen_optimizer_types, None
            )
            names.append(updated_names)
            optimizers_without_scheduler.append(optimizer)

        return names, optimizers_without_scheduler

    def _check_duplicates_and_update_name(
        self,
        optimizer: Optimizer,
        name: str,
        seen_optimizers: list[Optimizer],
        seen_optimizer_types: defaultdict[type[Optimizer], int],
        scheduler_config: Optional[dict],
    ) -> list[str]:
        """检查重复并更新名称

        Args:
            optimizer: 优化器对象
            name: 基础名称
            seen_optimizers: 已见优化器列表
            seen_optimizer_types: 已见优化器类型计数
            scheduler_config: 调度器配置（如果来自调度器）

        Returns:
            参数组名称列表

        Raises:
            ValueError: 如果参数组有重复名称
        """
        seen_optimizers.append(optimizer)
        optimizer_cls = type(optimizer)

        if scheduler_config is None or scheduler_config.get("name") is None:
            seen_optimizer_types[optimizer_cls] += 1

        # 检查参数组名称重复
        param_groups = optimizer.param_groups
        duplicates = self._duplicate_param_group_names(param_groups)
        if duplicates:
            raise ValueError(
                f"单个优化器的参数组不能有重复的 'name'。"
                f"{name} 的参数组名称重复: {duplicates}"
            )

        # 添加前缀和后缀
        name = self._add_prefix(name, optimizer_cls, seen_optimizer_types)
        return [self._add_suffix(name, param_groups, i) for i in range(len(param_groups))]
