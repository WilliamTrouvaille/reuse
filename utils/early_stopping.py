#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/2 01:28
@author  : William_Trouvaille
@function: 早停 (Early Stopping) 逻辑
@detail: 提供 EarlyStopper 类，用于在训练期间监控验证指标，并在指标不再改善时停止训练。
"""

from typing import Literal

import numpy as np
from loguru import logger


class EarlyStopper:
    """
    封装早停 (Early Stopping) 逻辑。

    在 Trainer 中使用：
        stopper = EarlyStopper(patience=10, mode='max')

        for epoch in ...:
            val_metric = evaluate(...)

            # 传入指标
            is_best = stopper.step(val_metric)

            # 检查是否应保存
            if is_best:
                checkpoint_manager.save_best_model(...)

            # 检查是否应停止
            if stopper.should_stop:
                logger.info("早停触发！")
                break
    """

    def __init__(
            self,
            patience: int,
            mode: Literal['min', 'max'] = 'max',
            min_delta: float = 0.0,
            verbose: bool = True
    ):
        """
        初始化 EarlyStopper。

        参数:
            patience (int): 在触发停止前，指标可以不改善的 epoch 数。
            mode (str): 'max' 或 'min'。
                        'max' 模式下，当指标停止*增加*时触发 (例如 Accuracy)。
                        'min' 模式下，当指标停止*减少*时触发 (例如 Loss)。
            min_delta (float): (可选) 为被视为“改善”所需的最小变化量。
                                 默认为 0.0。
            verbose (bool): 是否在指标改善/恶化时打印日志。
        """
        if patience <= 0:
            logger.warning(f"EarlyStopper patience (={patience}) <= 0，早停功能已禁用。")
        if mode not in ['min', 'max']:
            raise ValueError("EarlyStopper mode 必须是 'min' 或 'max'")

        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose

        # 内部状态
        self.patience_counter = 0
        self.best_metric = -np.inf if self.mode == 'max' else np.inf
        self._should_stop = False

        if self.verbose:
            logger.debug(
                f"EarlyStopper 初始化: Patience={patience}, Mode={mode}, MinDelta={min_delta}"
            )

    @property
    def should_stop(self) -> bool:
        """(只读) 检查是否应触发早停。"""
        if self.patience <= 0:
            return False  # 早停被禁用

        if self.patience_counter >= self.patience:
            if not self._should_stop and self.verbose:
                # 仅在第一次触发时记录 Critical
                logger.critical(
                    f"EarlyStopper: 早停触发！"
                    f"连续 {self.patience_counter} 个 epoch 未（显著）改善。"
                )
            self._should_stop = True

        return self._should_stop

    def step(self, current_metric: float) -> bool:
        """
        (在每个评估 epoch 后调用) 使用新指标更新早停状态。

        参数:
            current_metric (float): 最新的验证指标 (例如 val_acc 或 val_loss)。

        返回:
            bool: (is_best) True 表示当前 epoch 是*新*的最佳模型。
        """

        if self.patience <= 0:
            return False  # 早停被禁用

        # 1. 检查指标是否显著改善
        is_best = False
        if self.mode == 'max':
            is_best = (current_metric > self.best_metric + self.min_delta)
        elif self.mode == 'min':
            is_best = (current_metric < self.best_metric - self.min_delta)

        # 2. 更新状态
        if is_best:
            # 好的情况：指标改善了
            if self.verbose:
                logger.success(
                    f"EarlyStopper: 指标改善 {self.best_metric:.5f} -> {current_metric:.5f}"
                )
            self.best_metric = current_metric
            self.patience_counter = 0
        else:
            # 坏的情况：指标未改善
            self.patience_counter += 1
            if self.verbose:
                logger.warning(
                    f"EarlyStopper: 指标未改善。Patience: {self.patience_counter}/{self.patience}"
                )

        # (调用 should_stop 属性来更新内部的 _should_stop 标志)
        _ = self.should_stop

        return is_best

    def state_dict(self) -> dict:
        """
        返回早停器的状态字典，用于保存到检查点。
        """
        return {
            'patience_counter': self.patience_counter,
            'best_metric': self.best_metric,
            'should_stop': self._should_stop,
            'patience': self.patience,
            'mode': self.mode,
            'min_delta': self.min_delta
        }

    def load_state_dict(self, state_dict: dict):
        """
        从检查点加载早停器的状态。
        """
        self.patience_counter = state_dict.get('patience_counter', 0)
        self.best_metric = state_dict.get('best_metric', -np.inf if self.mode == 'max' else np.inf)
        self._should_stop = state_dict.get('should_stop', False)

        # (可选) 从检查点恢复超参数
        self.patience = state_dict.get('patience', self.patience)
        self.mode = state_dict.get('mode', self.mode)
        self.min_delta = state_dict.get('min_delta', self.min_delta)

        if self.verbose:
            logger.info(
                f"EarlyStopper 状态已加载: "
                f"counter={self.patience_counter}, best_metric={self.best_metric:.5f}"
            )
