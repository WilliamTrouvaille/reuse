#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025-11-04T00:00:00
@author  : William_Trouvaille
@function: CIFAR10 分类任务主程序

功能说明:
    1. 指标跟踪 (utils/metrics.py):
       - 使用 MetricTracker 在训练循环中高效跟踪指标（GPU端累积，避免频繁同步）
       - 使用 MetricsCollector 收集训练历史数据，用于后续可视化

    2. 通知推送 (utils/ntfy_notifier.py):
       - 支持通过 NTFY 服务推送训练进度通知到手机
       - 启用方式：在 config.yaml 中设置 ntfy.enabled: true
       - 自动发送训练开始、成功和失败通知

    3. 早停机制 (utils/early_stopping.py):
       - 使用 EarlyStopper 自动监控验证指标并在不再改善时停止训练
       - 配置方式：在 config.yaml 中设置 training.patience（>0 启用）

    4. 可视化报告 (utils/visualization):
       - 使用 MetricsCollector 收集所有训练数据
       - 使用 TrainingReporter.generate_full_report() 一键生成：
         * 训练曲线（损失和准确率）
         * 混淆矩阵
         * ROC 曲线
         * 分类报告热图
         * Epoch 耗时分布
         * Markdown 格式的文本摘要
       - 所有报告保存在 reports/ 目录下，支持时间戳子目录
"""

import sys

import torch
import torch.nn as nn
from loguru import logger
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# 导入工具包
from utils import (
    setup_logging,
    setup_config,
    set_random_seed,
    get_device,
    load_dataset_info,
    count_parameters,
    Trainer,
    MetricsCollector,
    TrainingReporter
)

# 导入网络
from network import create_model


# ========================================================================
# 1. 主程序入口
# ========================================================================

def main():
    """主训练流程"""

    # --- 1.1 配置加载 ---
    config = setup_config(
        yaml_config_path="config.yaml",
        cmd_args=None,
        default_config=None
    )

    # --- 1.2 日志配置 ---
    setup_logging(
        log_dir=config.logging.log_dir
    )

    logger.info("=" * 80)
    logger.info(f"{config.experiment.name}".center(80))
    logger.info("=" * 80)

    # --- 1.3 设置随机种子 ---
    set_random_seed(config.experiment.seed)

    # --- 1.4 获取计算设备 ---
    device = get_device(config.experiment.device)

    # --- 1.5 加载数据集 ---
    logger.info("=" * 80)
    logger.info("加载数据集".center(80))
    logger.info("=" * 80)

    dataset_info = load_dataset_info('CIFAR10', config.dataset.dataset_path)

    logger.info(f"数据集信息:")
    logger.info(f"  训练集大小: {len(dataset_info['dst_train'])}")
    logger.info(f"  测试集大小: {len(dataset_info['dst_test'])}")
    logger.info(f"  图像尺寸: {dataset_info['im_size']}")
    logger.info(f"  通道数: {dataset_info['channel']}")
    logger.info(f"  类别数: {dataset_info['num_classes']}")
    logger.info(f"  类别名称: {dataset_info['class_names']}")

    # --- 1.6 创建数据加载器 ---
    train_loader = DataLoader(
        dataset_info['dst_train'],
        batch_size=config.dataloader.batch_size,
        shuffle=True,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory,
        drop_last=True,  # 避免最后一个 batch 大小不一致
        persistent_workers=config.dataloader.persistent_workers
    )

    val_loader = DataLoader(
        dataset_info['dst_test'],
        batch_size=config.dataloader.batch_size,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory
    )

    logger.info(f"数据加载器已创建:")
    logger.info(f"  训练批次数: {len(train_loader)}")
    logger.info(f"  验证批次数: {len(val_loader)}")
    logger.info(f"  批次大小: {config.dataloader.batch_size}")

    # --- 1.7 创建模型 ---
    logger.info("=" * 80)
    logger.info("创建模型".center(80))
    logger.info("=" * 80)

    model = create_model(config)
    model = model.to(device)

    # 打印模型参数量
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    logger.info(f"模型参数统计:")
    logger.info(f"  总参数量: {total_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,}")

    # --- 1.8 可选: 使用 torch.compile 优化 ---
    if config.training.use_compile:
        try:
            logger.info("正在使用 torch.compile() 优化模型...")
            # torch.compile 仅在 PyTorch 2.0+ 可用
            model = torch.compile(model)
            logger.success("torch.compile() 已启用")
        except Exception as e:
            logger.warning(f"torch.compile() 失败，继续使用原始模型: {e}")

    # --- 1.9 创建优化器 ---
    optimizer = SGD(
        model.parameters(),
        lr=config.training.lr,
        momentum=config.training.momentum,
        weight_decay=config.training.weight_decay
    )

    logger.info(f"优化器: SGD")
    logger.info(f"  学习率: {config.training.lr}")
    logger.info(f"  动量: {config.training.momentum}")
    logger.info(f"  权重衰减: {config.training.weight_decay}")

    # --- 1.10 创建学习率调度器 ---
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.training.epochs,
        eta_min=1e-6
    )

    logger.info(f"学习率调度器: CosineAnnealingLR")
    logger.info(f"  T_max: {config.training.epochs}")
    logger.info(f"  eta_min: 1e-6")

    # --- 1.11 创建损失函数 ---
    criterion = nn.CrossEntropyLoss()

    logger.info(f"损失函数: CrossEntropyLoss")

    # --- 1.11.5 创建 Callbacks ---
    callbacks = []
    if config.callbacks.timer.enabled:
        from utils.callbacks import Timer
        timer = Timer(
            duration=config.callbacks.timer.duration,
            interval=config.callbacks.timer.interval,
            verbose=config.callbacks.timer.verbose
        )
        callbacks.append(timer)
        logger.info(f"已启用 Timer callback")

    # --- 1.12 创建训练器 ---
    logger.info("=" * 80)
    logger.info("初始化训练器".center(80))
    logger.info("=" * 80)

    trainer = Trainer.from_config(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        config=config,
        scheduler=scheduler,
        callbacks=callbacks
    )

    # --- 1.13 开始训练 ---
    logger.info("=" * 80)
    logger.info("开始训练".center(80))
    logger.info("=" * 80)

    result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs
    )

    # --- 1.14 生成训练报告 ---
    logger.info("=" * 80)
    logger.info("生成训练报告".center(80))
    logger.info("=" * 80)

    try:
        # --- 1.14.1 创建指标收集器并收集训练历史数据 ---
        collector = MetricsCollector()

        # 从训练历史中收集指标
        history = result['history']
        for epoch_data in history:
            epoch = epoch_data['epoch']

            # 使用 update_epoch_metrics 方法统一记录训练和验证指标
            collector.update_epoch_metrics(
                epoch=epoch,
                train_loss=epoch_data.get('train_loss', 0.0),
                train_acc=epoch_data.get('train_acc', 0.0),
                val_loss=epoch_data.get('val_loss', None),
                val_acc=epoch_data.get('val_acc', None),
                epoch_time=epoch_data.get('epoch_time', None)
            )

        # --- 1.14.2 在测试集上进行完整预测（用于生成混淆矩阵、ROC曲线等）---
        logger.info("正在收集测试集预测结果...")
        model.eval()  # 设置为评估模式

        all_predictions = []
        all_targets = []
        all_probabilities = []

        import torch.nn.functional as F

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                # 前向传播
                outputs = model(images)

                # 获取预测标签
                predictions = outputs.argmax(dim=1)
                all_predictions.extend(predictions.cpu().numpy())

                # 获取真实标签
                all_targets.extend(labels.cpu().numpy())

                # 获取预测概率（用于ROC曲线）
                probabilities = F.softmax(outputs, dim=1)
                all_probabilities.extend(probabilities.cpu().numpy())

        # 将预测结果存储到收集器中
        collector.set_predictions(
            predictions=all_predictions,
            targets=all_targets,
            probabilities=all_probabilities
        )

        logger.success(f"已收集 {len(all_targets)} 个样本的预测结果")

        # --- 1.14.3 创建报告生成器并一键生成完整报告 ---
        reporter = TrainingReporter(config.reporter)

        # 使用 generate_full_report 一键生成所有图表和摘要
        reporter.generate_full_report(
            collector=collector,
            class_names=dataset_info['class_names']
        )

        logger.success(f"训练报告已保存至: {reporter.report_dir}")

    except Exception as e:
        logger.error(f"生成训练报告时出错: {e}")
        logger.exception("详细错误信息:")

    # --- 1.15 训练完成 ---
    logger.info("=" * 80)
    logger.info("训练流程全部完成".center(80))
    logger.info(f"最佳指标 ({config.training.metric_to_track}): {result['best_metric']:.4f}".center(80))
    logger.info("=" * 80)


# ========================================================================
# 2. 程序入口
# ========================================================================

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.critical("程序被用户中断 (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"程序执行失败: {e}")
        logger.exception("详细错误信息:")
        sys.exit(1)
