#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/04
@author  : William_Trouvaille
@function: CIFAR-10 分类训练主程序
@description: 完整的训练流程，检验 utils 工具包的运行效果
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from network import create_cifar10_model
from utils import (
    setup_logging,
    setup_config,
    ConfigNamespace,
    set_random_seed,
    get_device,
    count_parameters,
    Trainer,
    MetricsCollector,
    TrainingReporter
)


# ========================================================================
# 1. 默认配置
# ========================================================================

DEFAULT_CONFIG = {
    'experiment': {
        'name': 'cifar10_baseline',
        'seed': 42,
        'device': 'cuda'
    },

    'data': {
        'root': './data',
        'batch_size': 128,
        'num_workers': 4,
        'pin_memory': True,
        'download': True
    },

    'model': {
        'num_classes': 10,
        'num_blocks': 3,
        'dropout_rate': 0.3
    },

    'training': {
        'epochs': 50,
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,

        # 性能优化
        'use_amp': True,
        'grad_accum_steps': 1,
        'max_grad_norm': None,
        'use_compile': True,

        # 早停与指标
        'patience': 10,
        'metric_to_track': 'acc',
        'metric_mode': 'max',
        'compute_top5': False,

        # 日志与验证间隔
        'log_interval': 1,
        'val_interval': 1,

        # 进度条配置
        'show_progress': True,
        'progress_update_interval': 0.5
    },

    'checkpoint': {
        'enabled': True,
        'save_dir': './checkpoints',
        'max_to_keep': 3
    },

    'logging': {
        'log_dir': './logs',
        'log_file': 'training.log'
    },

    'report': {
        'save_dir': './reports',
        'use_timestamp': True,

        'plots': {
            'style': 'seaborn-v0_8-darkgrid',
            'dpi': 150,
            'figure_size': [12, 8]
        },

        'enable_plots': {
            'training_curves': True,
            'confusion_matrix': True,
            'roc_curve': True,
            'classification_report': True,
            'epoch_time_distribution': True
        },

        'confusion_matrix': {
            'normalize': True,
            'cmap': 'Blues'
        },

        'roc': {
            'multi_class': 'ovr'
        }
    },

    'ntfy': {
        'enabled': False
    }
}


# ========================================================================
# 2. 数据加载
# ========================================================================

class CIFAR10DataModule:
    """
    CIFAR-10 数据模块，负责数据加载和预处理。

    职责:
        - 定义数据增强策略
        - 创建训练集和测试集 DataLoader
        - 管理数据集元信息
    """

    # CIFAR-10 数据集统计信息（用于归一化）
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2470, 0.2435, 0.2616)

    # CIFAR-10 类别名称
    CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    def __init__(self, config: ConfigNamespace):
        """
        初始化数据模块。

        参数:
            config (ConfigNamespace): 数据配置对象
        """
        self.config = config
        self.train_loader = None
        self.test_loader = None

        logger.info("CIFAR10DataModule 初始化完成")

    def setup(self):
        """
        准备数据集和数据加载器。
        """
        logger.info("=" * 60)
        logger.info("准备 CIFAR-10 数据集...".center(60))
        logger.info("=" * 60)

        # --- 1. 定义数据增强 ---
        # 训练集增强：随机裁剪、随机水平翻转、归一化
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # 随机裁剪（填充后）
            transforms.RandomHorizontalFlip(),      # 随机水平翻转
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD)
        ])

        # 测试集增强：仅归一化
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD)
        ])

        logger.info("数据增强策略已定义")

        # --- 2. 加载数据集 ---
        train_dataset = datasets.CIFAR10(
            root=self.config.root,
            train=True,
            download=self.config.download,
            transform=train_transform
        )

        test_dataset = datasets.CIFAR10(
            root=self.config.root,
            train=False,
            download=self.config.download,
            transform=test_transform
        )

        logger.success(f"训练集样本数: {len(train_dataset)}")
        logger.success(f"测试集样本数: {len(test_dataset)}")

        # --- 3. 创建 DataLoader ---
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=True if self.config.num_workers > 0 else False
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=True if self.config.num_workers > 0 else False
        )

        logger.info(f"训练批次数: {len(self.train_loader)}")
        logger.info(f"测试批次数: {len(self.test_loader)}")
        logger.success("数据加载器创建完成！")
        logger.info("=" * 60)

    def get_class_names(self):
        """获取类别名称列表"""
        return self.CLASSES


# ========================================================================
# 3. 自定义 Trainer（扩展功能）
# ========================================================================

class CIFAR10Trainer(Trainer):
    """
    扩展 Trainer 类，添加指标收集功能。

    新增功能:
        - 与 MetricsCollector 集成
        - 在训练和评估结束时自动收集指标
    """

    def __init__(self, *args, collector: MetricsCollector = None, **kwargs):
        """
        初始化训练器。

        参数:
            collector (MetricsCollector): 指标收集器实例
            *args, **kwargs: 传递给父类的参数
        """
        super().__init__(*args, **kwargs)
        self.collector = collector

    def _on_train_epoch_end(self, epoch: int, train_metrics: dict):
        """
        （钩子）训练 epoch 结束时，记录指标到收集器。
        """
        if self.collector:
            # 记录训练 Loss 和准确率
            self.collector.add_train_loss(train_metrics.get('loss', 0.0))
            self.collector.add_train_acc(train_metrics.get('acc', 0.0))

    def _on_eval_epoch_end(self, epoch: int, val_metrics: dict):
        """
        （钩子）评估 epoch 结束时，记录指标到收集器。
        """
        if self.collector:
            # 记录验证 Loss 和准确率
            self.collector.add_val_loss(val_metrics.get('loss', 0.0))
            self.collector.add_val_acc(val_metrics.get('acc', 0.0))


# ========================================================================
# 4. 主训练流程
# ========================================================================

def main():
    """主程序入口"""

    # --- 1. 命令行参数解析 ---
    parser = argparse.ArgumentParser(description='CIFAR-10 分类训练')
    parser.add_argument('--config', type=str, default='', help='YAML 配置文件路径')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--lr', type=float, help='学习率')
    parser.add_argument('--device', type=str, help='计算设备（cuda/cpu）')
    args = parser.parse_args()

    # 转换为字典（过滤 None 值）
    args_dict = {k: v for k, v in vars(args).items() if v is not None and k != 'config'}

    # --- 2. 配置管理 ---
    config = setup_config(
        default_config=DEFAULT_CONFIG,
        yaml_config_path=args.config,
        cmd_args=args_dict
    )

    # --- 3. 设置日志 ---
    log_dir = Path(config.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(log_dir=str(log_dir))

    logger.info("=" * 80)
    logger.info(f"开始 CIFAR-10 分类训练实验".center(80))
    logger.info(f"实验名称: {config.experiment.name}".center(80))
    logger.info("=" * 80)

    # --- 4. 设置随机种子 ---
    set_random_seed(config.experiment.seed)

    # --- 5. 设备选择 ---
    device = get_device(config.experiment.device)
    logger.info(f"使用设备: {device}")

    # --- 6. 数据加载 ---
    data_module = CIFAR10DataModule(config.data)
    data_module.setup()

    # --- 7. 模型创建 ---
    model = create_cifar10_model(
        num_classes=config.model.num_classes,
        num_blocks=config.model.num_blocks,
        dropout_rate=config.model.dropout_rate,
        device=device
    )

    # （可选）使用 torch.compile 优化（PyTorch 2.0+）
    if config.training.use_compile:
        try:
            logger.info("正在使用 torch.compile() 优化模型...")
            model = torch.compile(model)
            logger.success("torch.compile() 优化已启用")
        except Exception as e:
            logger.warning(f"torch.compile() 失败，使用原始模型: {e}")

    # 统计参数量
    logger.info(f"模型参数量: {count_parameters(model):,}")

    # --- 8. 损失函数与优化器 ---
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=config.training.lr,
        momentum=config.training.momentum,
        weight_decay=config.training.weight_decay
    )

    # --- 9. 学习率调度器 ---
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.training.epochs,
        eta_min=1e-5
    )

    logger.info(f"优化器: SGD (lr={config.training.lr}, momentum={config.training.momentum})")
    logger.info(f"学习率调度: CosineAnnealingLR (T_max={config.training.epochs})")

    # --- 10. 初始化指标收集器 ---
    collector = MetricsCollector()

    # --- 11. 创建训练器 ---
    from utils import CheckpointManager, EarlyStopper

    # 创建检查点管理器
    ckpt_mgr = None
    if config.checkpoint.enabled:
        ckpt_mgr = CheckpointManager(
            save_dir=config.checkpoint.save_dir,
            device=device,
            max_to_keep=config.checkpoint.max_to_keep
        )

    # 创建早停器
    early_stop = None
    if config.training.get('patience', 0) > 0:
        early_stop = EarlyStopper(
            patience=config.training.patience,
            mode=config.training.metric_mode
        )

    # 创建自定义训练器
    trainer = CIFAR10Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        checkpoint_manager=ckpt_mgr,
        early_stopper=early_stop,
        scheduler=scheduler,
        collector=collector,
        use_amp=config.training.use_amp,
        grad_accum_steps=config.training.grad_accum_steps,
        max_grad_norm=config.training.get('max_grad_norm', None),
        metric_to_track=config.training.metric_to_track,
        metric_mode=config.training.metric_mode,
        compute_top5=config.training.compute_top5,
        log_interval=config.training.log_interval,
        val_interval=config.training.val_interval,
        show_progress=config.training.show_progress,
        progress_update_interval=config.training.progress_update_interval
    )

    logger.info(f"训练器初始化完成: {trainer}")

    # --- 12. 开始训练 ---
    start_time = time.monotonic()

    try:
        result = trainer.fit(
            train_loader=data_module.train_loader,
            val_loader=data_module.test_loader,
            epochs=config.training.epochs
        )

        total_time = time.monotonic() - start_time

        logger.info("=" * 80)
        logger.success(f"训练完成！总耗时: {total_time:.2f} 秒".center(80))
        logger.success(f"最佳验证准确率: {result['best_metric']:.2f}%".center(80))
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"训练过程中发生错误: {e}")
        raise

    # --- 13. 最终评估（在测试集上）---
    logger.info("=" * 80)
    logger.info("开始最终评估（测试集）...".center(80))
    logger.info("=" * 80)

    # 加载最佳模型
    if trainer.checkpoint_manager:
        best_checkpoint = trainer.checkpoint_manager.load_best_model()
        if best_checkpoint:
            model.load_state_dict(best_checkpoint['model_state'])
            logger.success("已加载最佳模型")

    # 评估模型并收集预测结果
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, targets in data_module.test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # 前向传播
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)

            # 收集结果
            predictions = outputs.argmax(dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # 记录预测结果到收集器
    collector.set_predictions(all_predictions, all_targets)
    collector.set_probabilities(all_probabilities)

    logger.success("最终评估完成！")

    # --- 14. 生成训练报告 ---
    logger.info("=" * 80)
    logger.info("开始生成训练报告...".center(80))
    logger.info("=" * 80)

    reporter = TrainingReporter(config.report)
    reporter.generate_full_report(
        collector=collector,
        class_names=data_module.get_class_names()
    )

    logger.info("=" * 80)
    logger.success(f"所有任务完成！".center(80))
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
