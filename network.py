#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/04
@author  : William_Trouvaille
@function: CIFAR-10 分类网络定义
@description: 提供针对 CIFAR-10 数据集优化的卷积神经网络模型
"""

import torch
import torch.nn as nn
from loguru import logger


# ========================================================================
# 1. ResNet 基础模块
# ========================================================================

class BasicBlock(nn.Module):
    """
    ResNet 基础残差块。

    包含两个 3x3 卷积层，使用批归一化和 ReLU 激活函数。
    使用跳跃连接（shortcut）来缓解深层网络的梯度消失问题。
    """
    expansion = 1  # 输出通道数相对于输入通道数的扩展倍数

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        """
        初始化残差块。

        参数:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            stride (int): 步长（用于下采样，默认为 1）
        """
        super().__init__()

        # --- 主路径：两个 3x3 卷积 ---
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # --- 跳跃连接（shortcut）---
        # 当输入输出维度不匹配时，使用 1x1 卷积进行维度变换
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x (Tensor): 输入张量，形状 [batch, in_channels, height, width]

        返回:
            Tensor: 输出张量，形状 [batch, out_channels, height//stride, width//stride]
        """
        # --- 主路径 ---
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # --- 跳跃连接 ---
        out += self.shortcut(identity)
        out = self.relu(out)

        return out


# ========================================================================
# 2. CIFAR-10 分类网络
# ========================================================================

class CIFAR10Net(nn.Module):
    """
    针对 CIFAR-10 数据集优化的 ResNet 风格网络。

    网络结构:
        - 初始卷积层：64 通道
        - 3 个残差块组：[64, 128, 256]，每组包含 num_blocks 个残差块
        - 全局平均池化
        - 全连接层：输出 10 个类别

    特性:
        - 使用残差连接提升训练稳定性
        - 针对 32x32 的 CIFAR-10 图像优化
        - 支持 Dropout 正则化
    """

    def __init__(self, num_classes: int = 10, num_blocks: int = 2, dropout_rate: float = 0.3):
        """
        初始化网络。

        参数:
            num_classes (int): 分类类别数（CIFAR-10 为 10）
            num_blocks (int): 每个残差块组的块数（默认为 2）
            dropout_rate (float): Dropout 概率（默认为 0.3）
        """
        super().__init__()

        self.in_channels = 64  # 当前通道数（用于构建残差块）

        # --- 1. 初始卷积层 ---
        # 将 3 通道 RGB 图像映射到 64 通道特征图
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # --- 2. 残差块组 ---
        # 逐步增加通道数，提取多尺度特征
        self.layer1 = self._make_layer(64, num_blocks, stride=1)    # 输出: 64 通道
        self.layer2 = self._make_layer(128, num_blocks, stride=2)   # 输出: 128 通道，下采样
        self.layer3 = self._make_layer(256, num_blocks, stride=2)   # 输出: 256 通道，下采样

        # --- 3. 全局平均池化 ---
        # 将特征图 [batch, 256, H, W] 池化为 [batch, 256]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # --- 4. 分类头 ---
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(256, num_classes)

        # --- 5. 权重初始化 ---
        self._initialize_weights()

        logger.info(f"CIFAR10Net 已初始化: 类别数={num_classes}, 残差块数={num_blocks}, Dropout={dropout_rate}")

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        构建一个残差块组。

        参数:
            out_channels (int): 输出通道数
            num_blocks (int): 残差块数量
            stride (int): 第一个残差块的步长（用于下采样）

        返回:
            nn.Sequential: 包含多个残差块的序列模块
        """
        layers = []

        # --- 第一个残差块（可能包含下采样）---
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels

        # --- 后续残差块（不改变空间维度）---
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """
        初始化网络权重。

        使用 Kaiming 初始化（He 初始化）来缓解梯度消失/爆炸问题。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 对卷积层使用 Kaiming Normal 初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                # 批归一化层权重初始化为 1，偏置初始化为 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 全连接层权重使用正态分布初始化，偏置初始化为 0
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        logger.debug("网络权重已初始化（Kaiming Normal）")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x (Tensor): 输入图像，形状 [batch, 3, 32, 32]

        返回:
            Tensor: 类别 logits，形状 [batch, num_classes]
        """
        # --- 1. 初始卷积 ---
        # [batch, 3, 32, 32] -> [batch, 64, 32, 32]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # --- 2. 残差块组 ---
        # [batch, 64, 32, 32] -> [batch, 64, 32, 32]
        x = self.layer1(x)

        # [batch, 64, 32, 32] -> [batch, 128, 16, 16]（下采样）
        x = self.layer2(x)

        # [batch, 128, 16, 16] -> [batch, 256, 8, 8]（下采样）
        x = self.layer3(x)

        # --- 3. 全局平均池化 ---
        # [batch, 256, 8, 8] -> [batch, 256, 1, 1]
        x = self.avgpool(x)

        # [batch, 256, 1, 1] -> [batch, 256]
        x = torch.flatten(x, 1)

        # --- 4. 分类头 ---
        x = self.dropout(x)
        x = self.fc(x)  # [batch, 256] -> [batch, num_classes]

        return x


# ========================================================================
# 3. 模型工厂函数
# ========================================================================

def create_cifar10_model(
        num_classes: int = 10,
        num_blocks: int = 2,
        dropout_rate: float = 0.3,
        device: str = 'cuda'
) -> nn.Module:
    """
    创建并初始化 CIFAR-10 分类模型。

    参数:
        num_classes (int): 分类类别数
        num_blocks (int): 每组残差块数量
        dropout_rate (float): Dropout 概率
        device (str): 计算设备（'cuda' 或 'cpu'）

    返回:
        nn.Module: 初始化并移到指定设备的模型
    """
    logger.info("=" * 60)
    logger.info("正在创建 CIFAR-10 分类模型...".center(60))
    logger.info("=" * 60)

    # --- 创建模型 ---
    model = CIFAR10Net(
        num_classes=num_classes,
        num_blocks=num_blocks,
        dropout_rate=dropout_rate
    )

    # --- 移到设备 ---
    model = model.to(device)
    logger.info(f"模型已移动到设备: {device}")

    # --- 统计参数量 ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,}")

    logger.success("模型创建完成！")
    logger.info("=" * 60)

    return model


if __name__ == '__main__':
    """测试模型结构"""
    # 设置日志
    from utils import setup_logging
    setup_logging(log_file='logs/network_test.log')

    # 创建测试输入
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 32, 32)

    logger.info(f"测试输入形状: {test_input.shape}")

    # 创建模型
    model = create_cifar10_model(device='cpu')

    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(test_input)

    logger.info(f"输出形状: {output.shape}")
    logger.success("网络测试通过！")
