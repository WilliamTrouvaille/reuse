#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/1 23:40
@author  : William_Trouvaille
@function: TODO
"""

import sys
import os
import time
import argparse
import traceback
from loguru import logger
from tqdm import tqdm

# 1. --- PyTorch 核心导入 ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 2. --- 导入我们所有的 utils 工具 ---
from utils import (
    # 日志
    setup_logging,

    # 配置
    setup_config, print_config, save_config_to_yaml,

    # 通知
    NtfyNotifier,

    # 检查点
    CheckpointManager,

    # 数据
    load_dataset_info,

    # 辅助
    set_random_seed, get_device, clear_memory, count_parameters,

    # 进度条
    Progress
)


# 3. --- 项目特定定义 (本应在 model.py, config.py 中) ---

class SimpleMNISTConvNet(nn.Module):
    """一个为 MNIST (1x28x28) 设计的简单 CNN"""
    def __init__(self, num_classes=10):
        super().__init__()
        # (Batch, 1, 28, 28)
        self.conv1 = nn.Conv2d(1, 16, 5, 1, 2) # (Batch, 16, 28, 28)
        self.pool1 = nn.MaxPool2d(2)          # (Batch, 16, 14, 14)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2) # (Batch, 32, 14, 14)
        self.pool2 = nn.MaxPool2d(2)          # (Batch, 32, 7, 7)
        self.fc_input_features = 32 * 7 * 7
        self.fc1 = nn.Linear(self.fc_input_features, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.conv1(x)))
        x = self.pool2(self.relu(self.conv2(x)))
        x = x.view(-1, self.fc_input_features)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def get_project_defaults() -> dict:
    """定义本项目（MNIST 实验）的默认参数"""
    return {
        'experiment': {
            'name': 'mnist_tracker_comparison',
            'seed': 42,
        },
        'dataset': {
            'name': 'MNIST',
            'data_path': './data',
        },
        'dataloader': {
            # (性能) 推荐使用 > 0
            'num_workers': 4 if sys.platform != "win32" else 0,
            'pin_memory': True,
        },
        'model': {
            'name': 'SimpleMNISTConvNet',
        },
        'training': {
            'epochs': 20,  # 保持较短时间以便于测试
            'lr': 0.01,
            'batch_size': 128,
            'optimizer': 'SGD'
        },
        'logging': {
            'log_dir': './logs',
            'console_level': 'INFO',
            'file_level': 'DEBUG'
        },
        'checkpoint': {
            'save_dir': './checkpoints',
            'max_to_keep': 3
        },
        'ntfy': {
            'enabled': True # 设为 False 可禁用 ntfy
        }
    }

def parse_arguments() -> dict:
    """定义和解析命令行参数"""
    parser = argparse.ArgumentParser(description="MNIST 训练与进度条对比实验")

    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config.yaml',
        help='配置文件的路径'
    )
    parser.add_argument(
        '--training.epochs',
        type=int,
        help='覆盖训练轮数'
    )
    parser.add_argument(
        '--training.batch_size',
        type=int,
        help='覆盖批次大小'
    )

    args = parser.parse_args()
    return vars(args)


# 4. --- 核心训练与评估逻辑 ---

def train_epoch_with_tracker(epoch: int, model: nn.Module, loader: DataLoader,
                             optimizer: optim.Optimizer, criterion: nn.Module,
                             device: torch.device):
    """(实验 A) 使用高性能 Progress 进行训练"""

    model.train() # 设置为训练模式

    # 1. (关键) 包装 DataLoader
    with Progress(
            loader,
            description=f"Epoch {epoch+1} (A: Tracker)",
            leave=False,
            device=device
    ) as tracker:

        for images, labels in tracker:
            # 2. 将数据移至设备
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            # 3. 标准训练步骤
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels) # (这是一个 Tensor)
            loss.backward()
            optimizer.step()

            # 4. (关键) 更新 Tracker
            # 这是一个廉价、非阻塞的调用
            tracker.update({'loss': loss})

    # 5. 返回最终的平均 loss
    return tracker.get_final_metrics().get('loss', 0.0)


def train_epoch_without_tracker(epoch: int, model: nn.Module, loader: DataLoader,
                                optimizer: optim.Optimizer, criterion: nn.Module,
                                device: torch.device):
    """(实验 B) 不使用任何进度条，仅在内部循环"""

    model.train() # 设置为训练模式
    total_loss = 0.0

    # 1. (关键) 直接迭代
    for images, labels in loader:
        # 2. 将数据移至设备
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        # 3. 标准训练步骤
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 4. (关键) 必须调用 .item() 来同步并累加
        total_loss += loss.item()

    # 5. 返回最终的平均 loss
    return total_loss / len(loader)


def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: torch.device) -> (float, float):
    """在测试集上评估模型"""
    model.eval() # 设置为评估模式
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 评估时不需要计算梯度
    with torch.no_grad():
        # 评估循环通常很快，我们可以使用一个简单的 TQDM
        for images, labels in tqdm(loader, desc="Validating", leave=False, ncols=100):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_loss += loss.item() * images.size(0)
            total_samples += labels.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = (total_correct / total_samples) * 100
    return avg_loss, avg_acc

def main():
    """
    主执行函数
    """
    # --- 1. 初始化设置 (日志, 配置, 通知) ---

    # (关键) 必须先设置一个临时日志
    setup_logging(log_dir="../logs", console_level="INFO")

    cmd_args = parse_arguments()
    default_config = get_project_defaults()

    # (关键) `setup_config` 会加载 YAML 和 CMD 参数，覆盖默认值
    config = setup_config(
        default_config=default_config,
        yaml_config_path=cmd_args['config'],
        cmd_args=cmd_args
    )

    # (关键) `setup_config` 完成后，我们有了最终的日志路径
    # 再次调用 setup_logging 以使用*正确*的配置
    setup_logging(
        log_dir=config.logging.log_dir,
        console_level=config.logging.console_level,
        file_level=config.logging.file_level
    )

    # (可选) 存档本次运行的最终配置
    # run_config_path = os.path.join(config.logging.log_dir, "run_config.yaml")
    # save_config_to_yaml(config, run_config_path)

    # 初始化 Ntfy
    notifier = NtfyNotifier()
    ntfy_enabled = config.ntfy.enabled

    # --- 2. 核心训练逻辑 (使用完整的错误处理) ---

    # 初始化检查点管理器
    ckpt_manager = CheckpointManager(
        save_dir=config.checkpoint.save_dir,
        max_to_keep=config.checkpoint.max_to_keep
    )

    # 准备保存中断时的状态
    interrupt_state = {}

    try:
        # --- 3. 实验设置 (设备, 种子, 数据) ---

        if ntfy_enabled:
            notifier.notify_start(f"实验 {config.experiment.name} 已开始。")

        set_random_seed(config.experiment.seed)
        device = get_device() # 'auto'
        print_config(config, "MNIST 实验配置")

        # 加载数据
        data_info = load_dataset_info(
            dataset_name=config.dataset.name,
            data_path=config.dataset.data_path
        )

        # (关键) main.py 负责创建 DataLoader，使用 config 中的性能参数
        train_loader = DataLoader(
            data_info['dst_train'],
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.dataloader.num_workers,
            pin_memory=config.dataloader.pin_memory,
            persistent_workers=True if config.dataloader.num_workers > 0 else False
        )
        test_loader = DataLoader(
            data_info['dst_test'],
            batch_size=config.training.batch_size * 2, # 评估时 batch
            shuffle=False,
            num_workers=config.dataloader.num_workers,
            pin_memory=config.dataloader.pin_memory
        )

        # --- 4. 实验 A: 使用 Progress ---
        logger.info("=" * 60)
        logger.info("🚀 开始实验 A: 使用 Progress")
        logger.info("=" * 60)

        # 初始化模型
        model_a = SimpleMNISTConvNet(num_classes=data_info['num_classes']).to(device)
        optimizer_a = optim.SGD(model_a.parameters(), lr=config.training.lr)
        criterion = nn.CrossEntropyLoss().to(device)

        logger.info(f"模型参数: {count_parameters(model_a):,}")

        # 尝试恢复
        start_epoch = 0
        best_acc = 0.0

        logger.info("正在检查实验 A 的检查点...")
        latest_ckpt = ckpt_manager.load_latest_checkpoint()
        if latest_ckpt:
            try:
                model_a.load_state_dict(latest_ckpt['model_state'])
                optimizer_a.load_state_dict(latest_ckpt['optimizer_state'])
                start_epoch = latest_ckpt['epoch'] + 1
                best_acc = latest_ckpt.get('best_acc', 0.0)
                logger.success(f"成功从 Epoch {start_epoch - 1} 恢复训练。")
            except Exception as e:
                logger.error(f"加载检查点失败: {e}。将从头开始。")

        # 记录时间
        exp_a_start_time = time.monotonic()

        for epoch in range(start_epoch, config.training.epochs):

            # (调用 A)
            train_loss = train_epoch_with_tracker(
                epoch, model_a, train_loader, optimizer_a, criterion, device
            )

            val_loss, val_acc = evaluate(model_a, test_loader, criterion, device)

            logger.info(
                f"Epoch {epoch+1}/{config.training.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}%"
            )

            # (关键) 准备 state 字典
            state = {
                'epoch': epoch,
                'model_state': model_a.state_dict(),
                'optimizer_state': optimizer_a.state_dict(),
                'best_acc': best_acc,
                'config': config.to_dict() # 保存配置快照
            }
            # (关键) 更新中断状态
            interrupt_state = state

            # 保存滚动检查点
            ckpt_manager.save_epoch_checkpoint(state, epoch)

            # 保存最佳模型
            if val_acc > best_acc:
                logger.success(f"新高分! 准确率从 {best_acc:.2f}% 提升到 {val_acc:.2f}%")
                best_acc = val_acc
                state['best_acc'] = best_acc # 更新 state
                ckpt_manager.save_best_model(state, best_acc)

        exp_a_end_time = time.monotonic()
        time_a = exp_a_end_time - exp_a_start_time


        # --- 5. 实验 B: 不使用 Progress ---
        logger.info("=" * 60)
        logger.info("🐢 开始实验 B: 不使用任何进度条")
        logger.info("=" * 60)

        # (关键) 重置所有状态
        logger.warning("正在重置模型和优化器以进行公平比较...")
        clear_memory()

        model_b = SimpleMNISTConvNet(num_classes=data_info['num_classes']).to(device)
        optimizer_b = optim.SGD(model_b.parameters(), lr=config.training.lr)

        exp_b_start_time = time.monotonic()

        for epoch in range(config.training.epochs):
            # (调用 B)
            train_loss = train_epoch_without_tracker(
                epoch, model_b, train_loader, optimizer_b, criterion, device
            )
            # (不打印日志，模拟纯粹的、无 I/O 的训练)
            # logger.info(f"Epoch {epoch+1} (B) | Train Loss: {train_loss:.4f}")

        exp_b_end_time = time.monotonic()
        time_b = exp_b_end_time - exp_b_start_time


        # --- 6. 实验结论 ---
        logger.info("=" * 60)
        logger.info("📊 实验对比结论")
        logger.info("=" * 60)

        logger.info(f"实验 A (使用 Progress) 总耗时: {time_a:.3f} 秒")
        logger.info(f"实验 B (无 I/O)             总耗时: {time_b:.3f} 秒")

        overhead = time_a - time_b
        overhead_percent = (overhead / time_b) * 100

        if overhead > 0:
            logger.warning(f"Progress 带来了 {overhead:.3f} 秒的开销 ({overhead_percent:+.2f}%)")
        else:
            logger.success(f"Progress 开销可忽略不计 ({overhead:.3f}s)")

        if ntfy_enabled:
            notifier.notify_success(
                f"实验 {config.experiment.name} 已完成。\n\n"
                f"**Tracker (A):** {time_a:.3f}s\n"
                f"**No I/O (B):** {time_b:.3f}s\n"
                f"**Overhead:** {overhead_percent:+.2f}%"
            )

    except KeyboardInterrupt:
        # --- 7. (关键) 处理 Ctrl+C ---
        logger.critical("检测到用户中断 (KeyboardInterrupt)！")
        if interrupt_state:
            logger.info("正在保存最后的中断检查点...")
            ckpt_manager.save_interrupt_checkpoint(interrupt_state)

        if ntfy_enabled:
            notifier.notify_error(
                message=f"实验 {config.experiment.name} 被用户手动中断。",
                error_details="KeyboardInterrupt"
            )

    except Exception as e:
        # --- 8. (关键) 处理所有其他异常 ---
        logger.error(f"实验 {config.experiment.name} 因未捕获的异常而失败！")

        # 获取完整的堆栈跟踪
        error_details = traceback.format_exc()
        logger.exception(error_details) # loguru.exception 会自动记录堆栈

        if ntfy_enabled:
            notifier.notify_error(
                message=f"实验 {config.experiment.name} 失败: {e}",
                error_details=error_details
            )

    finally:
        # --- 9. 最终清理 ---
        clear_memory()
        logger.info("=" * 60)
        logger.info(f"实验 {config.experiment.name} 执行完毕。")
        logger.info("=" * 60)


if __name__ == '__main__':
    main()
