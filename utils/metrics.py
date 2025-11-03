#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/2 01:02
@author  : William_Trouvaille
@function: 高性能指标跟踪模块
@description: 提供 MetricTracker 类，用于在 GPU/设备 上高效累积指标，并在最后统一计算，以避免循环中的 .item() 同步瓶颈。
"""
from typing import Dict, Union

import torch
from loguru import logger


class MetricTracker:
    """
        高性能指标跟踪器，在 GPU/设备 上累积指标以避免频繁同步。

        核心优势:
            - 在训练循环中，所有指标累积都在 GPU 上进行（非阻塞）。
            - 只在调用 compute() 时才执行一次 GPU->CPU 同步。
            - 内存占用 O(1)，避免了 `torch.cat` 导致的 OOM 风险。
        """

    def __init__(
            self,
            device: Union[str, torch.device],
            compute_top5: bool = False
    ):
        """
        初始化指标跟踪器。

        参数:
            device (str | torch.device): 累加器所在的设备。
            compute_top5 (bool): 是否计算 Top-5 准确率。
        """
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.compute_top5 = compute_top5

        # 初始化累加器
        self.reset()
        logger.debug(f"MetricTracker 初始化，设备: {self.device}，Top-5: {self.compute_top5}")

    def reset(self):
        """
        重置所有累加器 (通常在每个 epoch 开始时调用)。
        """
        # (关键) 在目标设备上创建累加器
        self.loss_sum = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        self.correct_top1 = torch.tensor(0, device=self.device, dtype=torch.int64)
        if self.compute_top5:
            self.correct_top5 = torch.tensor(0, device=self.device, dtype=torch.int64)

        self.total_samples = 0
        logger.debug("MetricTracker 已重置。")

    def update(
            self,
            loss: torch.Tensor,
            outputs: torch.Tensor,
            targets: torch.Tensor
    ):
        """
        (核心方法) 更新累加器，所有操作在 GPU 上进行（非阻塞）。

        参数:
            loss (Tensor): 当前 batch 的*平均*损失 (标量)。
            outputs (Tensor): 模型输出的 logits (shape: [B, C])。
            targets (Tensor): 真实标签 (shape: [B])。
        """
        batch_size = targets.size(0)

        # 1. 累积损失 (在 GPU 上，非阻塞)
        #    (detach() 防止梯度累积，避免内存泄漏)
        self.loss_sum += loss.detach() * batch_size

        # 2. 计算 Top-1 准确率 (在 GPU 上)
        predictions_top1 = outputs.argmax(dim=1)
        self.correct_top1 += predictions_top1.eq(targets).sum()

        # 3. (可选) 计算 Top-5 准确率 (在 GPU 上)
        if self.compute_top5:
            _, predictions_top5 = outputs.topk(k=5, dim=1, largest=True, sorted=True)
            targets_expanded = targets.view(-1, 1)  # 变为 [B, 1]
            self.correct_top5 += predictions_top5.eq(targets_expanded).any(dim=1).sum()

        # 4. 累积样本总数 (在 CPU 上)
        self.total_samples += batch_size

    def compute(self) -> Dict[str, float]:
        """
        (唯一的同步点) 计算最终的平均指标。
        (在循环结束后调用)

        返回:
            dict: 包含 'loss', 'acc', 和 (可选) 'top5'。
        """
        if self.total_samples == 0:
            logger.warning("MetricTracker.compute() 被调用，但 total_samples = 0。")
            return {'loss': 0.0, 'acc': 0.0}

        try:
            # 1. (同步) 计算平均损失
            avg_loss = (self.loss_sum / self.total_samples).item()

            # 2. (同步) 计算 Top-1 准确率 (百分比)
            acc_top1 = (self.correct_top1.float() / self.total_samples).item() * 100.0

            metrics = {'loss': avg_loss, 'acc': acc_top1}

            # 3. (同步) 计算 Top-5 准确率
            if self.compute_top5:
                acc_top5 = (self.correct_top5.float() / self.total_samples).item() * 100.0
                metrics['top5'] = acc_top5

            logger.debug(f"MetricTracker 计算完成: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"在 MetricTracker.compute() 中计算指标失败: {e}")
            return {'loss': -1.0, 'acc': -1.0}


class AverageMeter:
    """
    (轻量级) 简单的平均值计算器 (用于 CPU 标量)。

    用于跟踪非 Tensor 值（例如学习率、数据加载时间）。
    每次 update() 都会同步 (.item())。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: Union[float, torch.Tensor], n: int = 1):
        """
        更新统计量。

        参数:
            val (float | Tensor): 要添加的值 (如果是 Tensor，会自动 .item())。
            n (int): 样本数量 (用于加权平均)。
        """
        if isinstance(val, torch.Tensor):
            val = val.item()  # (同步点)

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"AverageMeter(avg={self.avg:.4f}, count={self.count})"
