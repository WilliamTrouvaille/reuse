#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025-11-04T00:00:00
@author  : William_Trouvaille
@function: 简单的分类网络模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


# ========================================================================
# 1. 基础模块
# ========================================================================

class BasicBlock(nn.Module):
    """
    ResNet 风格的基础残差块。

    结构:
        x -> Conv -> BN -> ReLU -> Conv -> BN -> (+x) -> ReLU
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        初始化基础残差块。

        参数:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            stride (int): 第一个卷积的步长（用于降采样）
        """
        super().__init__()

        # --- 1.1 主路径（两层 3x3 卷积）---
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # --- 1.2 捷径路径（shortcut）---
        # 如果维度不匹配，使用 1x1 卷积进行调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x (Tensor): 输入张量，形状 [batch, in_channels, H, W]

        返回:
            Tensor: 输出张量，形状 [batch, out_channels, H', W']
        """
        # 主路径
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 残差连接
        out += self.shortcut(x)
        out = F.relu(out)

        return out


# ========================================================================
# 2. 主分类网络
# ========================================================================

class SimpleCNN(nn.Module):
    """
    简单的卷积神经网络，适用于 CIFAR10 等小图像分类任务。

    网络结构:
        - 初始卷积层
        - 多个 BasicBlock 残差块
        - 全局平均池化
        - 全连接分类层
    """

    def __init__(
            self,
            num_classes: int = 10,
            num_blocks: int = 3,
            dropout_rate: float = 0.3
    ):
        """
        初始化网络。

        参数:
            num_classes (int): 分类类别数
            num_blocks (int): 每个阶段的残差块数量
            dropout_rate (float): Dropout 比率
        """
        super().__init__()

        logger.info(f"正在构建 SimpleCNN: num_classes={num_classes}, "
                    f"num_blocks={num_blocks}, dropout_rate={dropout_rate}")

        # --- 2.1 初始卷积层 ---
        # CIFAR10: 32x32x3 -> 32x32x64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # --- 2.2 残差块层 ---
        # 第一组: 64 -> 64 (32x32)
        self.layer1 = self._make_layer(64, 64, num_blocks, stride=1)

        # 第二组: 64 -> 128 (16x16)
        self.layer2 = self._make_layer(64, 128, num_blocks, stride=2)

        # 第三组: 128 -> 256 (8x8)
        self.layer3 = self._make_layer(128, 256, num_blocks, stride=2)

        # --- 2.3 分类头 ---
        # 使用全局平均池化替代全连接层，减少参数量
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)

        # --- 2.4 初始化权重 ---
        self._initialize_weights()

        logger.success("SimpleCNN 构建完成")

    def _make_layer(
            self,
            in_channels: int,
            out_channels: int,
            num_blocks: int,
            stride: int
    ) -> nn.Sequential:
        """
        构建包含多个 BasicBlock 的层。

        参数:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            num_blocks (int): BasicBlock 数量
            stride (int): 第一个块的步长（用于降采样）

        返回:
            nn.Sequential: 残差块序列
        """
        layers = []

        # 第一个块可能改变通道数和空间尺寸
        layers.append(BasicBlock(in_channels, out_channels, stride))

        # 后续块保持通道数和空间尺寸不变
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """
        初始化网络权重（使用 Kaiming 初始化）。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming 初始化（适用于 ReLU）
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # BN 层的权重初始化为 1，偏置初始化为 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层使用正态分布初始化
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x (Tensor): 输入图像张量，形状 [batch, 3, H, W]

        返回:
            Tensor: 分类 logits，形状 [batch, num_classes]
        """
        # 初始卷积
        out = F.relu(self.bn1(self.conv1(x)))  # [batch, 64, H, W]

        # 残差块
        out = self.layer1(out)  # [batch, 64, H, W]
        out = self.layer2(out)  # [batch, 128, H/2, W/2]
        out = self.layer3(out)  # [batch, 256, H/4, W/4]

        # 全局平均池化
        out = self.avgpool(out)  # [batch, 256, 1, 1]
        out = torch.flatten(out, 1)  # [batch, 256]

        # 分类
        out = self.dropout(out)
        out = self.fc(out)  # [batch, num_classes]

        return out


# ========================================================================
# 3. 工厂函数
# ========================================================================

def create_model(config) -> SimpleCNN:
    """
    从配置创建模型的工厂函数。

    参数:
        config: 配置对象，需包含 model 子配置

    返回:
        SimpleCNN: 实例化的模型
    """
    model = SimpleCNN(
        num_classes=config.model.num_classes,
        num_blocks=config.model.num_blocks,
        dropout_rate=config.model.dropout_rate
    )
    return model
