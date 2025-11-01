#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/2 01:02
@author  : William_Trouvaille
@function: 高性能指标跟踪模块
@description: 提供 MetricTracker 类，用于在 GPU/设备 上高效累积指标，并在最后统一计算，以避免循环中的 .item() 同步瓶颈。
"""
from typing import Dict, List, Optional
import torch
from loguru import logger

class MetricTracker:
    """
    在训练/评估循环中高效跟踪指标。

    它通过在设备上累积原始 Tensors (logits, labels, loss)
    并在最后调用 .compute() 时才执行同步，
    从而避免了在 'hot loop' 中使用 .item()。
    """

    def __init__(self, device: torch.device):
        """
        初始化指标跟踪器。

        参数:
            device (torch.device): 指标累积应在的设备 (例如 'cuda:0')。
        """
        self.device = device
        self.reset()
        logger.debug(f"MetricTracker 初始化，将在 {self.device} 上累积。")

    def reset(self):
        """
        清空所有累积的指标，为下一个 epoch 做准备。
        """
        self.all_logits: List[torch.Tensor] = []
        self.all_labels: List[torch.Tensor] = []

        # 用于加权平均 loss
        self.weighted_loss_sum: float = 0.0
        self.total_samples: int = 0

        logger.debug("MetricTracker 已重置。")

    def update(
            self,
            logits: torch.Tensor,
            labels: torch.Tensor,
            loss: Optional[torch.Tensor] = None
    ) -> None:
        """
        (在循环内调用) 更新当前批次的指标。
        这是一个廉价的操作。

        参数:
            logits (Tensor): 模型的原始输出 (B, NumClasses)。
            labels (Tensor): 真实标签 (B,)。
            loss (Tensor, optional): 当前批次的*平均*损失 (标量 Tensor)。
        """
        # 1. 累积 Logits 和 Labels
        # (我们 detach 它们以防止保留计算图)
        self.all_logits.append(logits.detach())
        self.all_labels.append(labels.detach())

        # 2. 累积 Loss
        batch_size = labels.size(0)
        if loss is not None:
            # (关键) 我们在这里使用 .item()
            # 因为 loss 已经是*单个*标量，并且与 logits/labels 不同，
            # 它不能在最后被 .cat() 和重新计算。
            # 这是“计算与存储”的权衡。
            # 相比于累积数百万个 logits，
            # 每步同步一个标量 loss 是可以接受的。
            #
            # (替代方案是累积 loss_tensor * batch_size，但 .item() 更简单)
            self.weighted_loss_sum += loss.item() * batch_size

        self.total_samples += batch_size

    def compute(self) -> Dict[str, float]:
        """
        (在循环后调用) 计算所有累积指标。
        这是一个昂贵的操作，会触发 CUDA 同步。

        返回:
            dict: 包含 'loss' 和 'acc' 的字典。
        """
        if self.total_samples == 0:
            logger.warning("MetricTracker.compute() 被调用，但没有样本。")
            return {}

        logger.debug(f"正在 {self.total_samples} 个样本上计算最终指标...")

        try:
            # --- 1. (昂贵) 聚合所有 Tensors ---
            # (我们不需要将它们移至 self.device，因为它们已经在那里了)
            all_logits_cat = torch.cat(self.all_logits, dim=0)
            all_labels_cat = torch.cat(self.all_labels, dim=0)

            # --- 2. 计算指标 ---

            # (a) 计算 Loss (加权平均)
            final_loss = self.weighted_loss_sum / self.total_samples

            # (b) 计算 Top-1 准确率
            _, predicted = all_logits_cat.max(1)
            correct_sum = predicted.eq(all_labels_cat).sum().item()
            final_acc = (correct_sum / self.total_samples) * 100.0

            # (c) (可选) 计算 Top-5 准确率
            #    (我们在这里只实现了 Top-1，但 Top-5 可以在此添加)

            logger.debug(f"指标计算完成: Loss={final_loss:.4f}, Acc={final_acc:.2f}%")

            return {
                'loss': final_loss,
                'acc': final_acc
            }

        except Exception as e:
            logger.error(f"在 MetricTracker.compute() 中计算指标失败: {e}")
            return {'loss': -1.0, 'acc': -1.0}