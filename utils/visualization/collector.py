#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/4 00:10
@author  : William_Trouvaille
@function: 训练指标收集器
@description: 提供 MetricsCollector 类，用于收集和存储训练过程中的历史指标数据，
              包括损失、准确率、耗时以及最终的预测结果，供后续可视化使用。
"""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import numpy as np
from loguru import logger


class MetricsCollector:
    """
    训练指标收集器，负责收集和存储训练过程中的所有关键指标。

    核心功能:
        - 记录每个 epoch 的训练/验证损失和准确率
        - 记录每个 epoch 的耗时
        - 存储最终的预测结果、真实标签和预测概率
        - 提供汇总统计和数据导出功能

    设计理念:
        - 与训练循环解耦，只负责数据存储
        - 支持增量更新（每个 epoch 调用一次）
        - 内置数据验证，确保数据一致性
    """

    def __init__(self):
        """
        初始化指标收集器。

        所有指标列表初始化为空列表，预测相关数据初始化为 None。
        """
        # 训练和验证指标历史
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.train_accs: List[float] = []
        self.val_accs: List[float] = []

        # 时间统计
        self.epoch_times: List[float] = []  # 每个 epoch 的耗时（秒）
        self.total_time: float = 0.0  # 总训练时间（秒）

        # 预测结果（用于生成混淆矩阵、ROC 曲线等）
        self.predictions: Optional[np.ndarray] = None  # 预测标签
        self.targets: Optional[np.ndarray] = None  # 真实标签
        self.probabilities: Optional[np.ndarray] = None  # 预测概率（用于 ROC）

        logger.debug("MetricsCollector 已初始化。")

    def add_train_loss(self, loss: float):
        """添加训练损失"""
        self.train_losses.append(loss)

    def add_val_loss(self, loss: float):
        """添加验证损失"""
        self.val_losses.append(loss)

    def add_train_acc(self, acc: float):
        """添加训练准确率"""
        self.train_accs.append(acc)

    def add_val_acc(self, acc: float):
        """添加验证准确率"""
        self.val_accs.append(acc)

    def add_epoch_time(self, time: float):
        """添加epoch耗时"""
        self.epoch_times.append(time)
        self.total_time += time

    def update_epoch_metrics(
            self,
            epoch: int,
            train_loss: float,
            train_acc: float,
            val_loss: Optional[float] = None,
            val_acc: Optional[float] = None,
            epoch_time: Optional[float] = None
    ):
        """
        更新单个 epoch 的指标记录。

        通常在每个 epoch 结束时调用，记录训练和验证的损失、准确率以及耗时。

        参数:
            epoch (int): 当前 epoch 编号（从 0 开始）。
            train_loss (float): 训练集平均损失。
            train_acc (float): 训练集准确率（百分比，0-100）。
            val_loss (float, optional): 验证集平均损失。
            val_acc (float, optional): 验证集准确率（百分比，0-100）。
            epoch_time (float, optional): 当前 epoch 耗时（秒）。

        异常:
            AssertionError: 如果 epoch 编号与当前记录数不匹配。
        """
        # 验证 epoch 编号连续性（防止跳过或重复记录）
        expected_epoch = len(self.train_losses)
        assert epoch == expected_epoch, (
            f"Epoch 编号不连续：期望 {expected_epoch}，实际 {epoch}。"
        )

        # 记录训练指标
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)

        # 记录验证指标（可能为 None，例如只在某些 epoch 验证）
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if val_acc is not None:
            self.val_accs.append(val_acc)

        # 记录耗时
        if epoch_time is not None:
            self.epoch_times.append(epoch_time)
            self.total_time += epoch_time

        logger.debug(
            f"Epoch {epoch} 指标已记录: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
            f"val_loss={val_loss:.4f if val_loss is not None else 'N/A'}, "
            f"val_acc={val_acc:.2f if val_acc is not None else 'N/A'}%, "
            f"time={epoch_time:.2f if epoch_time is not None else 'N/A'}s"
        )

    def set_predictions(
            self,
            predictions: Union[np.ndarray, List],
            targets: Union[np.ndarray, List],
            probabilities: Optional[Union[np.ndarray, List]] = None
    ):
        """
        设置最终的预测结果（用于生成混淆矩阵、ROC 曲线等）。

        通常在训练结束后，对测试集或验证集进行一次完整预测后调用。

        参数:
            predictions (ndarray | list): 预测标签，形状 [N]。
            targets (ndarray | list): 真实标签，形状 [N]。
            probabilities (ndarray | list, optional): 预测概率，形状 [N, C]（C 为类别数）。

        异常:
            AssertionError: 如果 predictions 和 targets 的长度不匹配。
        """
        # 转换为 NumPy 数组（如果输入是列表）
        self.predictions = np.asarray(predictions)
        self.targets = np.asarray(targets)

        # 验证形状一致性
        assert self.predictions.shape[0] == self.targets.shape[0], (
            f"预测标签和真实标签数量不匹配：{self.predictions.shape[0]} vs {self.targets.shape[0]}。"
        )

        # 处理概率数组（可选）
        if probabilities is not None:
            self.probabilities = np.asarray(probabilities)
            assert self.probabilities.shape[0] == self.targets.shape[0], (
                f"预测概率和真实标签数量不匹配：{self.probabilities.shape[0]} vs {self.targets.shape[0]}。"
            )
            logger.debug(
                f"预测结果已记录: {len(self.targets)} 个样本，"
                f"概率形状 {self.probabilities.shape}。"
            )
        else:
            self.probabilities = None
            logger.debug(f"预测结果已记录: {len(self.targets)} 个样本（无概率信息）。")

    def get_summary(self) -> Dict[str, Any]:
        """
        获取训练过程的汇总统计。

        返回:
            dict: 包含以下字段的字典：
                - best_train_acc: 最佳训练准确率
                - best_val_acc: 最佳验证准确率
                - final_train_loss: 最终训练损失
                - final_val_loss: 最终验证损失
                - total_epochs: 总训练轮数
                - total_time: 总训练时间（秒）
                - avg_epoch_time: 平均每轮耗时（秒）
        """
        summary = {
            'total_epochs': len(self.train_losses),
            'best_train_acc': max(self.train_accs) if self.train_accs else 0.0,
            'best_val_acc': max(self.val_accs) if self.val_accs else 0.0,
            'final_train_loss': self.train_losses[-1] if self.train_losses else 0.0,
            'final_val_loss': self.val_losses[-1] if self.val_losses else 0.0,
            'total_time': self.total_time,
            'avg_epoch_time': (
                self.total_time / len(self.epoch_times)
                if self.epoch_times else 0.0
            ),
        }

        logger.debug(f"生成训练摘要: {summary}")
        return summary

    def export_to_dict(self) -> Dict[str, Any]:
        """
        导出所有收集的数据为字典格式（用于序列化或进一步处理）。

        返回:
            dict: 包含所有历史指标和预测结果的字典。
        """
        data = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'epoch_times': self.epoch_times,
            'total_time': self.total_time,
            'summary': self.get_summary(),
        }

        # 添加预测结果（转换为列表以便 JSON 序列化）
        if self.predictions is not None:
            data['predictions'] = self.predictions.tolist()
            data['targets'] = self.targets.tolist()
            if self.probabilities is not None:
                data['probabilities'] = self.probabilities.tolist()

        return data

    def save_to_json(self, save_path: Union[str, Path]):
        """
        将收集的数据持久化为 JSON 文件。

        参数:
            save_path (str | Path): JSON 文件保存路径。
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.export_to_dict(), f, indent=2, ensure_ascii=False)
            logger.success(f"指标数据已保存至: {save_path}")
        except Exception as e:
            logger.error(f"保存指标数据失败: {save_path}，错误: {e}")
            raise

    def __repr__(self) -> str:
        """返回对象的字符串表示，方便调试。"""
        return (
            f"MetricsCollector("
            f"epochs={len(self.train_losses)}, "
            f"has_predictions={self.predictions is not None})"
        )
