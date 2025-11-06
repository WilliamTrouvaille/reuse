#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/07
@author  : William_Trouvaille
@function: 批大小查找器 - 自动查找最大可用的批大小
@detail:
    本文件部分代码源自 PyTorch Lightning 项目
    原始许可证: Apache License 2.0
    原始版权: Copyright The Lightning AI team.
    原始仓库: https://github.com/Lightning-AI/pytorch-lightning
    原始文件:
        - lightning/pytorch/callbacks/batch_size_finder.py
        - lightning/pytorch/tuner/batch_size_scaling.py
"""

from typing import Any, Callable, Optional

from loguru import logger

from utils.helpers import garbage_collection_cuda, is_oom_error


# ========================================================================
# 1. BatchSizeFinder 类
# ========================================================================

class BatchSizeFinder:
    """批大小查找器 - 自动查找最大可用的批大小（在不触发 OOM 的前提下）。

    功能特性:
    - 智能搜索: 支持两种搜索策略（power 和 binsearch）
    - OOM 检测: 自动检测内存溢出错误
    - 安全边界: 提供安全边界（margin）以避免边缘 OOM
    - 灵活集成: 通过回调接口集成到任意训练循环

    使用示例:
        # 创建批大小查找器
        finder = BatchSizeFinder(
            mode='binsearch',      # 使用二分搜索
            steps_per_trial=3,     # 每次尝试运行 3 步
            init_val=2,            # 初始批大小为 2
            max_trials=25,         # 最多尝试 25 次
            margin=0.05            # 5% 的安全边界
        )

        # 定义一个试运行函数（模拟训练步骤）
        def trial_fn(batch_size: int) -> None:
            # 更新数据加载器的批大小
            train_loader.batch_size = batch_size
            # 运行若干步训练
            for i, batch in enumerate(train_loader):
                if i >= finder._steps_per_trial:
                    break
                outputs = model(batch)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        # 查找最优批大小
        optimal_batch_size = finder.find_batch_size(trial_fn)
        print(f"找到最优批大小: {optimal_batch_size}")

    Args:
        mode: 搜索策略，可选值:
            - 'power': 持续乘以 2 直到 OOM
            - 'binsearch': 先指数增长，遇到 OOM 后进行二分搜索
        steps_per_trial: 每次尝试运行的步数（理论上 1 步即可检测 OOM，但实践中建议 2-3 步）
        init_val: 初始批大小
        max_trials: 最大尝试次数
        margin: 安全边界（仅用于 binsearch 模式），范围 [0, 1)。
                例如 0.05 表示在找到的最大批大小基础上减少 5%
        max_val: 批大小上限（默认 8192），避免测试过大或低效的批大小

    Raises:
        ValueError: 如果 mode 不是 'power' 或 'binsearch'
        AssertionError: 如果 margin 不在 [0, 1) 范围内
    """

    SUPPORTED_MODES = ("power", "binsearch")

    def __init__(
        self,
        mode: str = "power",
        steps_per_trial: int = 3,
        init_val: int = 2,
        max_trials: int = 25,
        margin: float = 0.05,
        max_val: int = 8192,
    ) -> None:
        # --- 1.1 参数验证 ---
        mode = mode.lower()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(f"mode 应为 {self.SUPPORTED_MODES} 之一，当前值: {mode}")

        assert 0.0 <= margin < 1.0, f"margin 应在 [0, 1) 范围内，当前值: {margin}"

        # --- 1.2 初始化配置 ---
        self._mode = mode
        self._steps_per_trial = steps_per_trial
        self._init_val = init_val
        self._max_trials = max_trials
        self._margin = margin
        self._max_val = max_val

        # --- 1.3 初始化状态 ---
        self.optimal_batch_size: Optional[int] = None  # 找到的最优批大小

    # ========================================================================
    # 2. 主要接口
    # ========================================================================

    def find_batch_size(
        self,
        trial_fn: Callable[[int], None],
        get_batch_size_fn: Optional[Callable[[], int]] = None,
        set_batch_size_fn: Optional[Callable[[int], None]] = None,
    ) -> int:
        """查找最优批大小。

        Args:
            trial_fn: 试运行函数，接受批大小作为参数，执行若干步训练。
                     函数内部应调用 set_batch_size_fn 设置批大小，然后运行训练步骤。
            get_batch_size_fn: 获取当前批大小的函数（可选）
            set_batch_size_fn: 设置批大小的函数（可选）

        Returns:
            找到的最优批大小

        Raises:
            RuntimeError: 如果遇到非 OOM 的异常
        """
        logger.info("=" * 80)
        logger.info("开始批大小查找".center(80))
        logger.info(f"搜索模式: {self._mode}, 初始值: {self._init_val}, 最大尝试: {self._max_trials}")
        logger.info("=" * 80)

        try:
            # 根据搜索模式选择算法
            if self._mode == "power":
                new_size = self._run_power_scaling(trial_fn, set_batch_size_fn)
            elif self._mode == "binsearch":
                new_size = self._run_binsearch_scaling(trial_fn, set_batch_size_fn)
            else:
                raise ValueError(f"不支持的搜索模式: {self._mode}")

            garbage_collection_cuda()
            self.optimal_batch_size = new_size

            logger.success("=" * 80)
            logger.success(f"批大小查找完成，最优批大小: {new_size}".center(80))
            logger.success("=" * 80)

            return new_size

        except Exception as ex:
            logger.error(f"批大小查找失败: {ex}")
            raise

    # ========================================================================
    # 3. 搜索算法实现
    # ========================================================================

    def _run_power_scaling(
        self,
        trial_fn: Callable[[int], None],
        set_batch_size_fn: Optional[Callable[[int], None]] = None,
    ) -> int:
        """指数增长搜索算法 - 持续乘以 2 直到遇到 OOM。

        Args:
            trial_fn: 试运行函数
            set_batch_size_fn: 设置批大小的函数

        Returns:
            找到的最大批大小
        """
        new_size = self._init_val
        any_success = False  # 是否至少有一次成功
        last_successful_size = new_size

        for i in range(self._max_trials):
            garbage_collection_cuda()

            # 检查是否达到上限
            if new_size >= self._max_val:
                logger.info(f"已达到批大小上限 {self._max_val}，停止搜索")
                break

            try:
                # 设置批大小并运行试验
                if set_batch_size_fn:
                    set_batch_size_fn(new_size)
                logger.info(f"尝试批大小: {new_size}")
                trial_fn(new_size)

                # 成功，记录当前大小
                last_successful_size = new_size
                any_success = True

                # 检查是否是最后一次尝试
                if i + 1 >= self._max_trials:
                    new_size = last_successful_size
                    break

                # 加倍批大小
                new_size = int(new_size * 2)
                if new_size > self._max_val:
                    new_size = self._max_val
                logger.info(f"批大小 {last_successful_size} 成功，尝试更大的批大小 {new_size}")

            except RuntimeError as exception:
                if is_oom_error(exception):
                    # OOM 错误，减半并返回
                    logger.warning(f"批大小 {new_size} 触发 OOM，回退到 {last_successful_size}")
                    garbage_collection_cuda()
                    new_size = last_successful_size
                    if any_success:
                        break
                else:
                    # 其他错误，重新抛出
                    raise

        return new_size

    def _run_binsearch_scaling(
        self,
        trial_fn: Callable[[int], None],
        set_batch_size_fn: Optional[Callable[[int], None]] = None,
    ) -> int:
        """二分搜索算法 - 先指数增长，遇到 OOM 后进行二分搜索。

        Args:
            trial_fn: 试运行函数
            set_batch_size_fn: 设置批大小的函数

        Returns:
            找到的最大批大小（应用 margin 后）
        """
        assert 0.0 <= self._margin < 1.0, f"margin 应在 [0, 1) 范围内，当前值: {self._margin}"

        low = 1
        high = None
        new_size = self._init_val
        count = 0
        last_successful_size = new_size

        while True:
            garbage_collection_cuda()

            # 检查是否达到上限
            if new_size >= self._max_val:
                logger.info(f"已达到批大小上限 {self._max_val}，停止搜索")
                break

            try:
                # 设置批大小并运行试验
                if set_batch_size_fn:
                    set_batch_size_fn(new_size)
                logger.info(f"尝试批大小: {new_size}")
                trial_fn(new_size)

                # 成功，更新下界
                last_successful_size = new_size
                count += 1

                # 检查是否达到最大尝试次数
                if count >= self._max_trials:
                    new_size = last_successful_size
                    break

                # 更新搜索范围
                low = new_size
                if high:
                    # 二分搜索阶段
                    if high - low <= 1:
                        break
                    midval = (high + low) // 2
                    new_size = min(midval, self._max_val)
                    logger.info(f"批大小 {low} 成功，继续二分搜索 [{low}, {high}]，下次尝试 {new_size}")
                else:
                    # 指数增长阶段
                    new_size = min(int(new_size * 2), self._max_val)
                    logger.info(f"批大小 {low} 成功，继续指数增长，下次尝试 {new_size}")

            except RuntimeError as exception:
                if is_oom_error(exception):
                    # OOM 错误，设置上界
                    logger.warning(f"批大小 {new_size} 触发 OOM")
                    garbage_collection_cuda()

                    high = new_size
                    midval = (high + low) // 2
                    new_size = midval
                    logger.info(f"进入二分搜索，范围 [{low}, {high}]，下次尝试 {new_size}")

                    if high - low <= 1:
                        break
                else:
                    # 其他错误，重新抛出
                    raise

        # 应用安全边界
        if self._margin > 0:
            margin_reduced_size = max(1, int(new_size * (1 - self._margin)))
            if margin_reduced_size != new_size:
                logger.info(
                    f"应用安全边界 {self._margin:.1%}，"
                    f"批大小从 {new_size} 降低到 {margin_reduced_size}"
                )
                new_size = margin_reduced_size
                if set_batch_size_fn:
                    set_batch_size_fn(new_size)

        return new_size


# ========================================================================
# 4. 辅助函数（供高级用法使用）
# ========================================================================

def find_optimal_batch_size(
    trial_fn: Callable[[int], None],
    mode: str = "binsearch",
    steps_per_trial: int = 3,
    init_val: int = 2,
    max_trials: int = 25,
    margin: float = 0.05,
    max_val: int = 8192,
    set_batch_size_fn: Optional[Callable[[int], None]] = None,
) -> int:
    """便捷函数 - 查找最优批大小。

    这是 BatchSizeFinder 的快捷方式，适合一次性使用。

    Args:
        trial_fn: 试运行函数，接受批大小作为参数
        mode: 搜索模式 ('power' 或 'binsearch')
        steps_per_trial: 每次尝试运行的步数
        init_val: 初始批大小
        max_trials: 最大尝试次数
        margin: 安全边界（仅用于 binsearch）
        max_val: 批大小上限
        set_batch_size_fn: 设置批大小的函数（可选）

    Returns:
        找到的最优批大小

    示例:
        >>> def trial(batch_size):
        ...     # 运行若干步训练
        ...     for i in range(3):
        ...         model.train_step(batch_size)
        >>> optimal_bs = find_optimal_batch_size(trial, mode='binsearch')
    """
    finder = BatchSizeFinder(
        mode=mode,
        steps_per_trial=steps_per_trial,
        init_val=init_val,
        max_trials=max_trials,
        margin=margin,
        max_val=max_val,
    )
    return finder.find_batch_size(trial_fn, set_batch_size_fn=set_batch_size_fn)
