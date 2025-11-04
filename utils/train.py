#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/03
@author  : William_Trouvaille
@function: 训练循环模板（最终版）
@detail: 提供可选的、高度可定制的 Trainer 类，用于标准的 PyTorch 训练流程
         整合了依赖注入模式和配置驱动模式，支持完整的性能优化和健壮性保障
"""

import time
import traceback
from typing import Dict, Any, Optional, Union, Literal

import torch
import torch.nn as nn
from loguru import logger
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader

# 导入我们已有的工具
from .checkpoint_manager import CheckpointManager
from .early_stopping import EarlyStopper
from .helpers import clear_memory, format_time, log_memory_usage
from .metrics import MetricTracker, AverageMeter
from .ntfy_notifier import NtfyNotifier
from .progress import Progress


class Trainer:
    """
    可选的、可复用的 PyTorch 训练协调器 (Coordinator)。

    设计理念:
        - 职责分离: Trainer 负责"如何训练"(How)，main.py 负责"训练什么"(What)
        - 依赖注入: 通过 __init__ 接收所有已实例化的工具，最大化灵活性
        - 模板方法: 训练逻辑可通过重写 _train_step 等方法来定制
        - 性能优先: 默认集成高性能工具（MetricTracker, Progress, AMP）
    """

    # ========================================================================
    # 1. 初始化与构造
    # ========================================================================

    def __init__(
            self,
            model: nn.Module,
            optimizer: Optimizer,
            criterion: nn.Module,
            device: Union[str, torch.device],

            # (可选工具) 通过依赖注入传入
            checkpoint_manager: Optional[CheckpointManager] = None,
            early_stopper: Optional[EarlyStopper] = None,
            notifier: Optional[NtfyNotifier] = None,
            scheduler: Optional[_LRScheduler] = None,

            # (性能优化) 配置选项
            use_amp: bool = False,
            grad_accum_steps: int = 1,
            max_grad_norm: Optional[float] = None,

            # (指标与日志) 配置选项
            metric_to_track: str = 'acc',
            metric_mode: Literal['min', 'max'] = 'max',
            compute_top5: bool = False,
            log_interval: int = 1,
            val_interval: int = 1,

            # (新增) 进度条配置
            show_progress: bool = True,
            progress_update_interval: float = 1.5
    ):
        """
        初始化训练器（依赖注入模式）。

        参数:
            model (nn.Module): PyTorch 模型（应已 .to(device)）
            optimizer (Optimizer): PyTorch 优化器
            criterion (nn.Module): 损失函数
            device (str | torch.device): 计算设备

            checkpoint_manager (CheckpointManager, optional): 检查点管理器
            early_stopper (EarlyStopper, optional): 早停器
            notifier (NtfyNotifier, optional): Ntfy 通知器
            scheduler (_LRScheduler, optional): 学习率调度器

            use_amp (bool): 是否使用自动混合精度（默认: False）
            grad_accum_steps (int): 梯度累积步数（默认: 1）
            max_grad_norm (float, optional): 梯度裁剪的最大范数（默认: None）

            metric_to_track (str): 早停/最佳模型所跟踪的指标键（默认: 'acc'）
            metric_mode (str): 'max' (准确率) 或 'min' (损失)（默认: 'max'）
            compute_top5 (bool): 是否计算 Top-5 准确率（默认: False）
            log_interval (int): 每隔多少个 epoch 记录一次详细日志（默认: 1）
            val_interval (int): 每隔多少个 epoch 验证一次（默认: 1）

            show_progress (bool): 是否显示进度条（默认: True）
            progress_update_interval (float): 进度条更新间隔（秒）（默认: 1.5）
        """
        logger.info("Trainer 初始化...")

        # --- 1. 核心组件 ---
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        # 设备处理
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.scheduler = scheduler

        # --- 2. 可选工具（通过依赖注入）---
        self.checkpoint_manager = checkpoint_manager
        self.early_stopper = early_stopper
        self.notifier = notifier

        # --- 3. 性能优化配置 ---
        self.use_amp = use_amp
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.max_grad_norm = max_grad_norm

        # 初始化 GradScaler（用于 AMP）
        self.scaler = GradScaler() if (use_amp and self.device.type == 'cuda') else None

        if self.scaler:
            logger.info("AMP (自动混合精度) 已启用")
        if self.grad_accum_steps > 1:
            logger.info(f"梯度累积已启用，累积步数: {self.grad_accum_steps}")
        if self.max_grad_norm is not None:
            logger.info(f"梯度裁剪已启用，最大范数: {self.max_grad_norm}")

        # --- 4. 指标与日志配置 ---
        self.metric_to_track = metric_to_track
        self.metric_mode = metric_mode
        self.compute_top5 = compute_top5
        self.log_interval = max(1, log_interval)
        self.val_interval = max(1, val_interval)

        # --- 5. (新增) 进度条配置 ---
        self.show_progress = show_progress
        self.progress_update_interval = progress_update_interval

        if not self.show_progress:
            logger.info("进度条已禁用")

        # --- 6. 自动实例化内部工具 ---
        self.metric_tracker = MetricTracker(self.device, compute_top5=self.compute_top5)
        self.lr_meter = AverageMeter()

        # --- 7. 内部状态 ---
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = -float('inf') if self.metric_mode == 'max' else float('inf')
        self.training_history = []
        self.start_epoch = 0

        # --- 8. 自动恢复检查点 ---
        if self.checkpoint_manager:
            self._load_checkpoint()

        logger.success(f"Trainer 初始化完成（设备: {self.device}）")

    @classmethod
    def from_config(
            cls,
            model: nn.Module,
            optimizer: Optimizer,
            criterion: nn.Module,
            device: Union[str, torch.device],
            config: Any,
            scheduler: Optional[_LRScheduler] = None
    ) -> 'Trainer':
        """
        (可选) 配置驱动模式：从 config 对象自动实例化所有工具。

        参数:
            model, optimizer, criterion, device, scheduler: (同 __init__)
            config (Any): 完整配置对象，必须包含以下子配置:

                config.training (必需):
                    - use_amp (bool): 是否启用 AMP
                    - grad_accum_steps (int): 梯度累积步数
                    - max_grad_norm (float | None): 梯度裁剪范数
                    - metric_to_track (str): 跟踪的指标名称（如 'acc', 'loss'）
                    - metric_mode (str): 'max' 或 'min'
                    - compute_top5 (bool): 是否计算 Top-5 准确率
                    - log_interval (int): 日志记录间隔（epoch）
                    - val_interval (int): 验证间隔（epoch）
                    - patience (int): 早停容忍度（>0 启用早停）
                    - show_progress (bool, optional): 是否显示进度条
                    - progress_update_interval (float, optional): 进度条更新间隔

                config.checkpoint (可选):
                    - enabled (bool): 是否启用检查点保存
                    - save_dir (str): 检查点保存目录
                    - max_to_keep (int): 保留的检查点数量

                config.ntfy (可选):
                    - enabled (bool): 是否启用通知

        返回:
            Trainer: 已配置的 Trainer 实例

        示例:
            config = load_config_from_yaml("config.yaml")
            trainer = Trainer.from_config(model, optimizer, criterion, device, config)
            trainer.fit(train_loader, val_loader)
        """
        logger.info("Trainer (from_config 模式) 初始化...")

        # --- 1. 从 config 自动实例化所有 utils 工具 ---

        # 检查点
        ckpt_mgr = None
        if hasattr(config, 'checkpoint') and config.checkpoint.enabled:
            ckpt_mgr = CheckpointManager(
                save_dir=config.checkpoint.save_dir,
                device=device,
                max_to_keep=config.checkpoint.get('max_to_keep', 3)
            )

        # 早停
        early_stop = None
        if hasattr(config, 'training') and config.training.get('patience', 0) > 0:
            early_stop = EarlyStopper(
                patience=config.training.patience,
                mode=config.training.get('metric_mode', 'max'),
            )

        # 通知
        notifier = None
        if hasattr(config, 'ntfy') and config.ntfy.get('enabled', False):
            notifier = NtfyNotifier()

        # --- 2. 调用标准的 __init__（依赖注入模式）---
        return cls(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            checkpoint_manager=ckpt_mgr,
            early_stopper=early_stop,
            notifier=notifier,
            scheduler=scheduler,
            use_amp=config.training.get('use_amp', False),
            grad_accum_steps=config.training.get('grad_accum_steps', 1),
            max_grad_norm=config.training.get('max_grad_norm', None),
            metric_to_track=config.training.get('metric_to_track', 'acc'),
            metric_mode=config.training.get('metric_mode', 'max'),
            compute_top5=config.training.get('compute_top5', False),
            log_interval=config.training.get('log_interval', 1),
            val_interval=config.training.get('val_interval', 1),
            show_progress=config.training.get('show_progress', True),
            progress_update_interval=config.training.get('progress_update_interval', 0.5)
        )

    # ========================================================================
    # 2. 公共方法 - 主训练循环
    # ========================================================================

    def fit(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            epochs: int = 100
    ) -> Dict[str, Any]:
        """
        主训练循环。这是唯一的公共方法。

        参数:
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader, optional): 验证数据加载器
            epochs (int): 总训练轮数（默认: 100）

        返回:
            dict: 包含训练历史和最佳指标的字典
                - 'history': 每个 epoch 的指标列表
                - 'best_metric': 最佳验证指标值
        """
        total_start_time = time.monotonic()

        # 计算实际训练的轮次数
        num_epochs_to_train = epochs - self.start_epoch

        logger.info("=" * 80)
        logger.info(f"🚀 开始训练: Epoch {self.start_epoch + 1} -> {epochs}（共 {num_epochs_to_train} 轮）".center(80))
        logger.info(f"   跟踪指标: '{self.metric_to_track}' (模式: {self.metric_mode})".center(80))
        if self.early_stopper:
            logger.info(f"   早停容忍: {self.early_stopper.patience} epochs".center(80))
        log_memory_usage("训练开始前")
        logger.info("=" * 80)

        # 发送开始通知
        if self.notifier:
            self.notifier.notify_start(
                f"训练开始\nEpochs: {self.start_epoch + 1}..{epochs}\n"
                f"指标: {self.metric_to_track} ({self.metric_mode})"
            )

        try:
            # --- 主训练循环 ---
            self._main_training_loop(train_loader, val_loader, epochs)

        except KeyboardInterrupt:
            # --- 处理 Ctrl+C ---
            logger.critical("检测到键盘中断 (Ctrl+C)，训练被中断")
            self._handle_interrupt()
            if self.notifier:
                self.notifier.notify_error("训练被用户中断", "KeyboardInterrupt")
            # 重新抛出异常，确保程序直接退出而不是继续执行后续流程
            raise

        except Exception as e:
            # --- 处理其他异常 ---
            logger.error(f"训练过程中发生未捕获的异常: {type(e).__name__}")
            error_details = traceback.format_exc()
            logger.exception(error_details)
            if self.notifier:
                self.notifier.notify_error(f"训练失败: {e}", error_details)
            raise  # 重新抛出异常，让 main.py 知道

        else:
            # --- 如果训练正常完成 ---
            logger.success("=" * 80)
            logger.success(f"训练在 Epoch {epochs} 正常完成")
            logger.success(f"最佳指标 ({self.metric_to_track}): {self.best_metric:.4f}")
            logger.success("=" * 80)

            if self.notifier:
                self.notifier.notify_success(
                    f"训练已正常完成\n"
                    f"Epochs: {epochs}\n"
                    f"最佳 {self.metric_to_track}: {self.best_metric:.4f}"
                )

        finally:
            # --- 最终清理 ---
            total_duration = time.monotonic() - total_start_time
            logger.info(f"总训练耗时: {format_time(total_duration)}")
            self._cleanup()

        return {
            'history': self.training_history,
            'best_metric': self.best_metric
        }

    # ========================================================================
    # 3. 核心私有方法 - 训练和评估的一个 epoch
    # ========================================================================

    def _main_training_loop(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader],
            total_epochs: int
    ):
        """(私有) 包含主 for 循环的内部编排器"""

        for epoch in range(self.start_epoch, total_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.monotonic()

            # --- 1. 训练阶段 ---
            train_metrics = self._train_epoch(train_loader, epoch)

            # 调用钩子（供子类扩展）
            self._on_train_epoch_end(epoch, train_metrics)

            # --- 2. 验证阶段 ---
            val_metrics = {}
            if val_loader and (epoch % self.val_interval == 0 or epoch == total_epochs - 1):
                val_metrics = self._eval_epoch(val_loader, epoch)

                # 调用钩子（供子类扩展）
                self._on_eval_epoch_end(epoch, val_metrics)

            # --- 3. 学习率调度 ---
            if self.scheduler:
                self._step_scheduler(val_metrics)

            # --- 4. 日志记录 ---
            if epoch % self.log_interval == 0:
                self._log_epoch_metrics(epoch, train_metrics, val_metrics, epoch_start_time)

            # --- 5. 检查点保存与早停检查 ---
            should_stop = self._save_and_check_stop(epoch, val_metrics)
            if should_stop:
                logger.warning(f"早停触发，训练终止于 Epoch {epoch + 1}")
                break

    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        (私有) 训练一个 epoch。

        参数:
            train_loader (DataLoader): 训练数据加载器
            epoch (int): 当前 epoch 编号

        返回:
            dict: 包含平均指标的字典（例如 {'loss': 0.123, 'acc': 95.2}）
        """
        self.model.train()  # 设置为训练模式

        # 重置指标跟踪器
        self.metric_tracker.reset()
        self.lr_meter.reset()

        # (可选) 使用 Progress 包装数据加载器
        if self.show_progress:
            loader_wrapper = Progress(
                train_loader,
                description=f"Epoch {epoch + 1} [Train]",
                leave=False,
                update_interval_sec=self.progress_update_interval,
                device=self.device
            )
        else:
            loader_wrapper = train_loader

        # 使用上下文管理器（如果是 Progress）或直接迭代
        if self.show_progress:
            with loader_wrapper as pbar:
                self._train_epoch_inner_loop(pbar)
        else:
            self._train_epoch_inner_loop(loader_wrapper)

        # 计算平均指标
        metrics = self.metric_tracker.compute()
        metrics['lr'] = self.lr_meter.avg

        return metrics

    def _train_epoch_inner_loop(self, loader):
        """(私有) 训练 epoch 的内部循环"""
        for step, batch in enumerate(loader):
            # ========== 1. 执行训练步骤 ==========
            step_result = self._train_step(batch)

            loss = step_result['loss']
            outputs = step_result['outputs']
            targets = step_result['targets']

            # ========== 2. 反向传播 ==========
            scaled_loss = loss / self.grad_accum_steps

            if self.scaler:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # ========== 3. 优化器更新 ==========
            if (step + 1) % self.grad_accum_steps == 0:
                # (可选) 梯度裁剪
                if self.max_grad_norm is not None:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )

                # 优化器步骤
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                # 清空梯度
                self.optimizer.zero_grad(set_to_none=True)

                # 更新全局步数
                self.global_step += 1

            # ========== 4. 指标跟踪 ==========
            self.metric_tracker.update(loss, outputs, targets)

            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_meter.update(current_lr)

            # ========== 5. 更新进度条 ==========
            if self.show_progress and hasattr(loader, 'update'):
                loader.update({'loss': loss, 'lr': current_lr})

    def _eval_epoch(self, eval_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        (私有) 评估一个 epoch。

        参数:
            eval_loader (DataLoader): 评估数据加载器
            epoch (int): 当前 epoch 编号

        返回:
            dict: 包含平均指标的字典
        """
        self.model.eval()  # 设置为评估模式

        # 重置指标跟踪器
        self.metric_tracker.reset()

        # (可选) 使用 Progress 包装数据加载器
        if self.show_progress:
            loader_wrapper = Progress(
                eval_loader,
                description=f"Epoch {epoch + 1} [Val]",
                leave=False,
                update_interval_sec=self.progress_update_interval,
                device=self.device
            )
        else:
            loader_wrapper = eval_loader

        # 禁用梯度计算
        with torch.no_grad():
            if self.show_progress:
                with loader_wrapper as pbar:
                    self._eval_epoch_inner_loop(pbar)
            else:
                self._eval_epoch_inner_loop(loader_wrapper)

        # 计算平均指标
        metrics = self.metric_tracker.compute()

        return metrics

    def _eval_epoch_inner_loop(self, loader):
        """(私有) 评估 epoch 的内部循环"""
        for batch in loader:
            # ========== 1. 执行评估步骤 ==========
            step_result = self._eval_step(batch)

            loss = step_result['loss']
            outputs = step_result['outputs']
            targets = step_result['targets']

            # ========== 2. 指标跟踪 ==========
            self.metric_tracker.update(loss, outputs, targets)

            # ========== 3. 更新进度条 ==========
            if self.show_progress and hasattr(loader, 'update'):
                loader.update({'loss': loss})

    # ========================================================================
    # 4. (关键) 可重写的保护方法 - 供子类定制
    # ========================================================================

    def _train_step(self, batch) -> Dict[str, torch.Tensor]:
        """
        (可重写) 单个训练步骤。

        默认实现: 标准的分类任务
            - 输入: (images, labels)
            - 输出: logits
            - 损失: criterion(logits, labels)

        子类可以重写以支持:
            - 多任务学习（多个输出和损失）
            - GAN 训练（生成器和判别器）
            - 对比学习（SimCLR, MoCo 等）
            等等

        参数:
            batch: 来自 DataLoader 的一个批次（通常是 (inputs, targets)）

        返回:
            dict: 必须包含以下键
                - 'loss' (Tensor): 当前 batch 的损失（标量）
                - 'outputs' (Tensor): 模型输出的 logits (shape: [batch, num_classes])
                - 'targets' (Tensor): 真实标签 (shape: [batch])

        示例（多任务学习）:
            def _train_step(self, batch):
                inputs, target_cls, target_seg = batch
                inputs = inputs.to(self.device, non_blocking=True)
                target_cls = target_cls.to(self.device, non_blocking=True)
                target_seg = target_seg.to(self.device, non_blocking=True)

                with autocast('cuda', enabled=(self.scaler is not None)):
                    out_cls, out_seg = self.model(inputs)
                    loss_cls = self.criterion[0](out_cls, target_cls)
                    loss_seg = self.criterion[1](out_seg, target_seg)
                    total_loss = loss_cls + 0.5 * loss_seg

                return {
                    'loss': total_loss,
                    'outputs': out_cls,
                    'targets': target_cls
                }
        """
        # 解包 batch（假设是标准的 (inputs, targets)）
        inputs, targets = batch

        # 移到设备（使用 non_blocking 优化）
        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        # 前向传播（使用 autocast 支持 AMP）
        # 注意：autocast 仅在 CUDA 设备上启用
        if self.scaler is not None:
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
        else:
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

        return {
            'loss': loss,
            'outputs': outputs,
            'targets': targets
        }

    def _eval_step(self, batch) -> Dict[str, torch.Tensor]:
        """
        (可重写) 单个评估步骤。

        默认实现: 与 _train_step 相同的逻辑（但在 eval 模式和 no_grad 下）

        参数:
            batch: 来自 DataLoader 的一个批次

        返回:
            dict: 与 _train_step 相同的格式
        """
        # 默认实现：与训练步骤相同（但已在 no_grad 和 eval 模式下）
        inputs, targets = batch

        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        # 评估时也可以使用 AMP（加速推理）
        if self.scaler is not None:
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
        else:
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

        return {
            'loss': loss,
            'outputs': outputs,
            'targets': targets
        }

    def _on_train_epoch_end(self, epoch: int, train_metrics: Dict[str, float]):
        """
        (可重写) 训练 epoch 结束时的钩子。

        用途:
            - 记录额外的信息（例如权重直方图）
            - 执行自定义的逻辑（例如更新可视化）

        参数:
            epoch (int): 当前 epoch 编号
            train_metrics (dict): 训练指标
        """
        pass  # 默认实现：什么都不做

    def _on_eval_epoch_end(self, epoch: int, val_metrics: Dict[str, float]):
        """
        (可重写) 评估 epoch 结束时的钩子。

        参数:
            epoch (int): 当前 epoch 编号
            val_metrics (dict): 验证指标
        """
        pass  # 默认实现：什么都不做

    # ========================================================================
    # 5. 辅助私有方法 - 检查点、早停、学习率调度等
    # ========================================================================

    def _save_and_check_stop(
            self,
            epoch: int,
            val_metrics: Dict[str, float]
    ) -> bool:
        """
        (私有) 封装检查点保存和早停逻辑。

        参数:
            epoch (int): 当前 epoch
            val_metrics (dict): 验证指标

        返回:
            bool: True 表示应该停止训练
        """
        if not self.checkpoint_manager and not self.early_stopper:
            return False  # 没有工具，什么都不做

        # is_best = False

        # --- 1. 准备 state 字典 ---
        state = self._build_checkpoint_state(epoch, val_metrics)

        # --- 2. 保存滚动检查点 ---
        if self.checkpoint_manager:
            self.checkpoint_manager.save_epoch_checkpoint(state, epoch)

        # --- 3. 早停与最佳模型 ---
        should_stop = False
        if self.early_stopper and val_metrics:
            current_metric = val_metrics.get(self.metric_to_track)

            if current_metric is None:
                logger.warning(
                    f"验证指标中未找到 '{self.metric_to_track}'，跳过早停检查。"
                    f"可用指标: {list(val_metrics.keys())}"
                )
            else:
                # (关键) is_best 由 EarlyStopper 返回
                is_best = self.early_stopper.step(current_metric)

                if is_best:
                    self.best_metric = current_metric
                    if self.checkpoint_manager:
                        logger.success(
                            f"Epoch {epoch + 1}: 发现新的最佳模型！"
                            f"{self.metric_to_track} = {self.best_metric:.4f}"
                        )
                        self.checkpoint_manager.save_best_model(state, self.best_metric)

                should_stop = self.early_stopper.should_stop

        return should_stop

    def _load_checkpoint(self):
        """(私有) 从检查点恢复训练"""
        logger.info("尝试从检查点恢复训练...")

        checkpoint = self.checkpoint_manager.load_latest_checkpoint()

        if checkpoint is None:
            logger.info("未找到检查点，从头开始训练")
            return

        try:
            # 恢复模型和优化器
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])

            # 恢复 epoch
            self.start_epoch = checkpoint['epoch'] + 1

            # (可选) 恢复 scheduler
            if self.scheduler and 'scheduler_state' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
                logger.debug("学习率调度器状态已恢复")

            # (可选) 恢复早停器
            if self.early_stopper and 'early_stopper_state' in checkpoint:
                self.early_stopper.load_state_dict(checkpoint['early_stopper_state'])
                logger.debug("早停器状态已恢复")

            # (可选) 恢复 scaler (AMP)
            if self.scaler and 'scaler_state' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state'])
                logger.debug("GradScaler 状态已恢复")

            # (可选) 恢复最佳指标
            if 'best_metric' in checkpoint:
                self.best_metric = checkpoint['best_metric']

            # (可选) 恢复训练历史
            if 'history' in checkpoint:
                self.training_history = checkpoint['history']

            logger.success(f"训练已从 Epoch {self.start_epoch} 恢复")
            logger.info(f"已恢复的最佳指标 ({self.metric_to_track}): {self.best_metric:.4f}")
            logger.info(f"提示: 训练将从 Epoch {self.start_epoch + 1} 继续，请在 fit() 中指定目标 epoch 数")

        except Exception as e:
            logger.error(f"加载检查点时发生错误: {e}")
            logger.warning("将从头开始训练")
            self.start_epoch = 0

    def _build_checkpoint_state(
            self,
            epoch: int,
            val_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        (私有) 构建检查点状态字典。

        参数:
            epoch (int): 当前 epoch
            val_metrics (dict): 验证指标

        返回:
            dict: 完整的状态字典
        """
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'history': self.training_history
        }

        # (可选) 添加验证指标
        if val_metrics:
            state['val_metrics'] = val_metrics

        # (可选) 保存 scheduler 状态
        if self.scheduler:
            state['scheduler_state'] = self.scheduler.state_dict()

        # (可选) 保存早停器状态
        if self.early_stopper:
            state['early_stopper_state'] = self.early_stopper.state_dict()

        # (可选) 保存 scaler 状态（AMP）
        if self.scaler:
            state['scaler_state'] = self.scaler.state_dict()

        return state

    def _step_scheduler(self, val_metrics: Dict[str, float]):
        """
        (私有) 执行学习率调度器步骤。

        参数:
            val_metrics (dict): 验证指标
        """
        # 判断 scheduler 的类型
        if isinstance(self.scheduler, ReduceLROnPlateau):
            # ReduceLROnPlateau 需要传入指标
            metric_value = val_metrics.get(self.metric_to_track)
            if metric_value is None:
                logger.warning(
                    f"ReduceLROnPlateau 需要验证指标 '{self.metric_to_track}'，但未找到"
                )
            else:
                self.scheduler.step(metric_value)
        else:
            # 其他 scheduler（如 StepLR, CosineAnnealingLR）
            self.scheduler.step()

    def _handle_interrupt(self):
        """
        (私有) 处理训练中断（Ctrl+C）。
        """
        if self.checkpoint_manager:
            logger.warning("正在保存中断检查点...")

            state = self._build_checkpoint_state(self.current_epoch, {})

            self.checkpoint_manager.save_interrupt_checkpoint(state)
            logger.success("中断检查点已保存")
        else:
            logger.warning("未配置 CheckpointManager，中断检查点未保存")

    def _cleanup(self):
        """
        (私有) 训练结束后的资源清理。
        """
        # 清理 GPU 缓存
        if self.device.type == 'cuda':
            clear_memory()
            logger.debug("GPU 缓存已清理")

    def _log_epoch_metrics(
            self,
            epoch: int,
            train_metrics: Dict[str, float],
            val_metrics: Dict[str, float],
            epoch_start_time: float
    ):
        """
        (私有) 格式化并记录 epoch 指标到日志。

        参数:
            epoch (int): 当前 epoch
            train_metrics (dict): 训练指标
            val_metrics (dict): 验证指标
            epoch_start_time (float): epoch 开始时间
        """
        duration = time.monotonic() - epoch_start_time

        # 构建日志消息
        msg_parts = [f"Epoch {epoch + 1:03d}"]
        msg_parts.append(f"Time: {format_time(duration)}")
        msg_parts.append(f"Train Loss: {train_metrics.get('loss', 0):.4f}")
        msg_parts.append(f"Train Acc: {train_metrics.get('acc', 0):.2f}%")

        if val_metrics:
            msg_parts.append(f"Val Loss: {val_metrics.get('loss', 0):.4f}")
            msg_parts.append(f"Val Acc: {val_metrics.get('acc', 0):.2f}%")

        msg_parts.append(f"LR: {train_metrics.get('lr', 0):.2e}")

        # (可选) 添加内存使用
        if self.device.type == 'cuda':
            from .helpers import get_memory_usage
            mem_info = get_memory_usage()
            if mem_info:
                msg_parts.append(f"Mem: {mem_info['allocated']}")

        # 输出日志
        log_msg = " | ".join(msg_parts)
        logger.success(log_msg)

        # 记录详细历史
        epoch_history = {
            'epoch': epoch,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()}
        }
        self.training_history.append(epoch_history)

    # ========================================================================
    # 6. 实用方法 - 供外部调用
    # ========================================================================

    def get_current_lr(self) -> float:
        """
        获取当前学习率。

        返回:
            float: 当前学习率
        """
        return self.optimizer.param_groups[0]['lr']

    def get_training_history(self) -> list:
        """
        获取训练历史。

        返回:
            list: 包含每个 epoch 指标的列表
        """
        return self.training_history

    def __repr__(self) -> str:
        return (
            f"Trainer(\n"
            f"  model={type(self.model).__name__},\n"
            f"  device={self.device},\n"
            f"  use_amp={self.use_amp},\n"
            f"  grad_accum_steps={self.grad_accum_steps},\n"
            f"  current_epoch={self.current_epoch},\n"
            f"  best_metric={self.best_metric:.4f},\n"
            f"  show_progress={self.show_progress}\n"
            f")"
        )
