#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/07
@author  : William_Trouvaille
@function: 训练计时器 - 跟踪训练、验证、测试各阶段的耗时
@detail:
    本文件部分代码源自 PyTorch Lightning 项目
    原始许可证: Apache License 2.0
    原始版权: Copyright The Lightning AI team.
    原始仓库: https://github.com/Lightning-AI/pytorch-lightning
    原始文件: lightning/pytorch/callbacks/timer.py
"""

import re
import time
from datetime import timedelta
from typing import Any, Optional, Union

from loguru import logger


# ========================================================================
# 1. 常量定义
# ========================================================================

class Interval:
    """时间检查间隔枚举"""
    STEP = "step"    # 每步检查
    EPOCH = "epoch"  # 每轮检查


class Stage:
    """训练阶段枚举"""
    TRAINING = "training"      # 训练阶段
    VALIDATING = "validating"  # 验证阶段
    TESTING = "testing"        # 测试阶段


# ========================================================================
# 2. Timer 类
# ========================================================================

class Timer:
    """训练计时器 - 跟踪训练、验证、测试各阶段的耗时，并支持时间限制功能。

    功能特性:
    - 分阶段计时: 独立跟踪训练、验证、测试阶段的时间
    - 时间限制: 可设置训练时间上限，超时自动停止
    - 灵活配置: 支持多种时间格式（字符串、timedelta、字典）
    - 状态持久化: 支持保存和加载计时器状态

    使用示例:
        # 创建计时器，设置12小时训练时间限制
        timer = Timer(duration="00:12:00:00")

        # 或使用 timedelta
        from datetime import timedelta
        timer = Timer(duration=timedelta(weeks=1))

        # 或使用字典
        timer = Timer(duration=dict(weeks=4, days=2))

        # 在训练循环中使用
        timer.on_train_start()
        # ... 训练代码 ...
        if timer.should_stop():
            logger.info("训练时间已到，停止训练")
            break
        timer.on_train_end()

        # 查询各阶段耗时（秒）
        train_time = timer.time_elapsed("training")
        val_time = timer.time_elapsed("validating")

    Args:
        duration: 训练时间上限，支持三种格式:
            - 字符串: "DD:HH:MM:SS" 格式（天:小时:分钟:秒）
            - timedelta: datetime.timedelta 对象
            - 字典: 兼容 timedelta 的键值对（如 {"weeks": 1, "days": 2}）
        interval: 时间检查间隔，可选值:
            - "step": 每个训练步检查一次
            - "epoch": 每个训练轮次检查一次
        verbose: 是否输出详细日志（默认 True）

    Raises:
        ValueError: 如果 duration 格式不正确
        ValueError: 如果 interval 不是 "step" 或 "epoch"
    """

    def __init__(
        self,
        duration: Optional[Union[str, timedelta, dict[str, int]]] = None,
        interval: str = Interval.STEP,
        verbose: bool = True,
    ) -> None:
        # --- 1.1 参数验证与解析 ---
        # 解析时间限制参数
        if isinstance(duration, str):
            duration_match = re.fullmatch(r"(\d+):(\d\d):(\d\d):(\d\d)", duration.strip())
            if not duration_match:
                raise ValueError(
                    f"Timer(duration={duration!r}) 格式不正确。"
                    "期望格式为 DD:HH:MM:SS（天:小时:分钟:秒）"
                )
            duration = timedelta(
                days=int(duration_match.group(1)),
                hours=int(duration_match.group(2)),
                minutes=int(duration_match.group(3)),
                seconds=int(duration_match.group(4)),
            )
        elif isinstance(duration, dict):
            duration = timedelta(**duration)

        # 验证时间检查间隔
        if interval not in {Interval.STEP, Interval.EPOCH}:
            raise ValueError(
                f"不支持的 interval 参数: {interval}。"
                f"可选值: {Interval.STEP}, {Interval.EPOCH}"
            )

        # --- 1.2 初始化内部状态 ---
        self._duration = duration.total_seconds() if duration is not None else None  # 时间限制（秒）
        self._interval = interval      # 检查间隔
        self._verbose = verbose         # 是否输出日志

        # 各阶段的开始和结束时间（使用 time.monotonic() 获取单调时钟）
        self._start_time: dict[str, Optional[float]] = {
            Stage.TRAINING: None,
            Stage.VALIDATING: None,
            Stage.TESTING: None,
        }
        self._end_time: dict[str, Optional[float]] = {
            Stage.TRAINING: None,
            Stage.VALIDATING: None,
            Stage.TESTING: None,
        }
        self._offset = 0  # 时间偏移量（用于从检查点恢复时累加之前的时间）

    # ========================================================================
    # 3. 时间查询接口
    # ========================================================================

    def start_time(self, stage: str = Stage.TRAINING) -> Optional[float]:
        """返回指定阶段的开始时间（秒，基于 time.monotonic()）

        Args:
            stage: 阶段名称，可选值: "training", "validating", "testing"

        Returns:
            开始时间（秒），如果该阶段尚未开始则返回 None
        """
        return self._start_time.get(stage)

    def end_time(self, stage: str = Stage.TRAINING) -> Optional[float]:
        """返回指定阶段的结束时间（秒，基于 time.monotonic()）

        Args:
            stage: 阶段名称，可选值: "training", "validating", "testing"

        Returns:
            结束时间（秒），如果该阶段尚未结束则返回 None
        """
        return self._end_time.get(stage)

    def time_elapsed(self, stage: str = Stage.TRAINING) -> float:
        """返回指定阶段已经过的时间（秒）

        计算逻辑:
        - 如果阶段尚未开始: 返回偏移量（从检查点恢复的时间）
        - 如果阶段正在进行: 返回从开始到当前的时间 + 偏移量
        - 如果阶段已结束: 返回从开始到结束的时间 + 偏移量

        Args:
            stage: 阶段名称，可选值: "training", "validating", "testing"

        Returns:
            已经过的时间（秒）
        """
        start = self.start_time(stage)
        end = self.end_time(stage)
        offset = self._offset if stage == Stage.TRAINING else 0  # 仅训练阶段使用偏移量

        if start is None:
            return offset
        if end is None:
            return time.monotonic() - start + offset  # 正在进行中
        return end - start + offset  # 已结束

    def time_remaining(self, stage: str = Stage.TRAINING) -> Optional[float]:
        """返回指定阶段剩余的时间（秒）

        Args:
            stage: 阶段名称，可选值: "training", "validating", "testing"

        Returns:
            剩余时间（秒），如果未设置时间限制则返回 None
        """
        if self._duration is not None:
            return self._duration - self.time_elapsed(stage)
        return None

    # ========================================================================
    # 4. 生命周期回调方法（供外部训练循环调用）
    # ========================================================================

    def on_train_start(self) -> None:
        """训练开始时调用 - 记录训练开始时间"""
        self._start_time[Stage.TRAINING] = time.monotonic()
        if self._verbose:
            logger.info("计时器: 训练阶段开始")

    def on_train_end(self) -> None:
        """训练结束时调用 - 记录训练结束时间"""
        self._end_time[Stage.TRAINING] = time.monotonic()
        if self._verbose:
            elapsed = timedelta(seconds=int(self.time_elapsed(Stage.TRAINING)))
            logger.info(f"计时器: 训练阶段结束，总耗时 {elapsed}")

    def on_validation_start(self) -> None:
        """验证开始时调用 - 记录验证开始时间"""
        self._start_time[Stage.VALIDATING] = time.monotonic()

    def on_validation_end(self) -> None:
        """验证结束时调用 - 记录验证结束时间"""
        self._end_time[Stage.VALIDATING] = time.monotonic()

    def on_test_start(self) -> None:
        """测试开始时调用 - 记录测试开始时间"""
        self._start_time[Stage.TESTING] = time.monotonic()

    def on_test_end(self) -> None:
        """测试结束时调用 - 记录测试结束时间"""
        self._end_time[Stage.TESTING] = time.monotonic()

    def on_train_batch_end(self) -> bool:
        """每个训练批次结束时调用 - 检查是否应该停止训练

        Returns:
            如果应该停止训练返回 True，否则返回 False
        """
        if self._interval != Interval.STEP or self._duration is None:
            return False
        return self._check_time_remaining()

    def on_train_epoch_end(self) -> bool:
        """每个训练轮次结束时调用 - 检查是否应该停止训练

        Returns:
            如果应该停止训练返回 True，否则返回 False
        """
        if self._interval != Interval.EPOCH or self._duration is None:
            return False
        return self._check_time_remaining()

    # ========================================================================
    # 5. 辅助方法
    # ========================================================================

    def should_stop(self) -> bool:
        """检查是否应该停止训练（基于时间限制）

        Returns:
            如果已超过时间限制返回 True，否则返回 False
        """
        if self._duration is None:
            return False
        return self.time_elapsed(Stage.TRAINING) >= self._duration

    def _check_time_remaining(self) -> bool:
        """内部方法 - 检查剩余时间并输出日志

        Returns:
            如果应该停止训练返回 True，否则返回 False
        """
        assert self._duration is not None
        should_stop = self.time_elapsed(Stage.TRAINING) >= self._duration

        if should_stop and self._verbose:
            elapsed = timedelta(seconds=int(self.time_elapsed(Stage.TRAINING)))
            logger.warning(f"训练时间已达上限 {elapsed}，发出停止信号")

        return should_stop

    # ========================================================================
    # 6. 状态持久化
    # ========================================================================

    def state_dict(self) -> dict[str, Any]:
        """保存计时器状态到字典

        Returns:
            包含各阶段已用时间的字典
        """
        return {
            "time_elapsed": {
                Stage.TRAINING: self.time_elapsed(Stage.TRAINING),
                Stage.VALIDATING: self.time_elapsed(Stage.VALIDATING),
                Stage.TESTING: self.time_elapsed(Stage.TESTING),
            }
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """从字典恢复计时器状态

        Args:
            state_dict: 包含各阶段已用时间的字典
        """
        time_elapsed = state_dict.get("time_elapsed", {})
        self._offset = time_elapsed.get(Stage.TRAINING, 0)  # 恢复训练阶段的时间偏移量
        if self._verbose:
            logger.info(f"计时器: 从检查点恢复，已累计训练时间 {timedelta(seconds=int(self._offset))}")
