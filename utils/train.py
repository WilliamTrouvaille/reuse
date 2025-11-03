#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/03
@author  : William_Trouvaille
@function: 训练协调器模块
@detail: 提供高度可复用、可扩展的 Trainer 类，用于标准 PyTorch 训练流程
"""

import time
import traceback
from typing import Dict, Any, Optional, Union, List, Literal

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from loguru import logger

from .checkpoint_manager import CheckpointManager
from .early_stopping import EarlyStopper
from .helpers import clear_memory, get_memory_usage, log_memory_usage, format_time
from .metrics import MetricTracker, AverageMeter
from .ntfy_notifier import NtfyNotifier
from .progress import Progress


class Trainer:
    """
    可复用的 PyTorch 训练协调器。

    设计理念:
        - 职责分离: 通过依赖注入接收所有核心组件
        - 模板方法: 提供可重写的 _train_step/_eval_step 支持自定义逻辑
        - 性能优先: 集成 AMP、梯度累积、高性能指标跟踪
        - 双构造模式: 支持 DI 模式和 from_config 模式

    使用场景:
        场景1 - DI模式 (推荐, 最大灵活性):
            ckpt_mgr = CheckpointManager(...)
            stopper = EarlyStopper(...)

            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                checkpoint_manager=ckpt_mgr,
                early_stopper=stopper,
                use_amp=True
            )
            trainer.fit(train_loader, val_loader, epochs=100)

        场景2 - from_config模式 (推荐, 大型项目):
            trainer = Trainer.from_config(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                config=config
            )
            trainer.fit(train_loader, val_loader)

        场景3 - 自定义训练 (继承):
            class MyTrainer(Trainer):
                def _train_step(self, batch):
                    # 多任务、GAN、对比学习等自定义逻辑
                    ...
    """

    def __init__(
            self,
            model: nn.Module,
            optimizer: Optimizer,
            criterion: nn.Module,
            device: Union[str, torch.device],

            # 可选工具 (通过依赖注入)
            checkpoint_manager: Optional[CheckpointManager] = None,
            early_stopper: Optional[EarlyStopper] = None,
            notifier: Optional[NtfyNotifier] = None,
            scheduler: Optional[_LRScheduler] = None,

            # 性能优化选项
            use_amp: bool = False,
            grad_accum_steps: int = 1,
            max_grad_norm: Optional[float] = None,

            # 指标与日志配置
            metric_to_track: str = 'acc',
            metric_mode: Literal['min', 'max'] = 'max',
            compute_top5: bool = False,
            log_interval: int = 1,
            val_interval: int = 1,

            # 其他配置
            auto_resume: bool = True
    ):
        """
        初始化训练器 (依赖注入模式)。

        参数:
            model (nn.Module): PyTorch 模型 (应已移到目标设备)
            optimizer (Optimizer): PyTorch 优化器
            criterion (nn.Module): 损失函数
            device (str | torch.device): 计算设备

            checkpoint_manager (CheckpointManager, optional): 检查点管理器
            early_stopper (EarlyStopper, optional): 早停器
            notifier (NtfyNotifier, optional): Ntfy 通知器
            scheduler (_LRScheduler, optional): 学习率调度器

            use_amp (bool): 是否使用自动混合精度 (默认: False)
            grad_accum_steps (int): 梯度累积步数 (默认: 1)
            max_grad_norm (float, optional): 梯度裁剪的最大范数

            metric_to_track (str): 早停和最佳模型跟踪的指标 (默认: 'acc')
            metric_mode (str): 'max' 或 'min' (默认: 'max')
            compute_top5 (bool): 是否计算 Top-5 准确率 (默认: False)
            log_interval (int): 每隔多少个 epoch 记录详细日志 (默认: 1)
            val_interval (int): 每隔多少个 epoch 验证一次 (默认: 1)

            auto_resume (bool): 是否自动从检查点恢复训练 (默认: True)
        """
        logger.info("Trainer 初始化 (DI 模式)...")

        # ========== 1. 核心组件 ==========
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.scheduler = scheduler

        logger.info(f"设备: {self.device}")

        # ========== 2. 可选工具 ==========
        self.checkpoint_manager = checkpoint_manager
        self.early_stopper = early_stopper
        self.notifier = notifier

        # ========== 3. 性能优化 ==========
        self.use_amp = use_amp
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.max_grad_norm = max_grad_norm

        self.scaler = GradScaler(enabled=(use_amp and self.device.type == 'cuda'))

        if self.scaler.is_enabled():
            logger.info("AMP (自动混合精度) 已启用")
        if self.grad_accum_steps > 1:
            logger.info(f"梯度累积已启用，累积步数: {self.grad_accum_steps}")
        if self.max_grad_norm is not None:
            logger.info(f"梯度裁剪已启用，最大范数: {self.max_grad_norm}")

        # ========== 4. 指标与日志 ==========
        self.metric_to_track = metric_to_track
        self.metric_mode = metric_mode
        self.log_interval = max(1, log_interval)
        self.val_interval = max(1, val_interval)

        # 自动实例化内部工具
        self.metric_tracker = MetricTracker(self.device, compute_top5=compute_top5)
        self.lr_meter = AverageMeter()

        # ========== 5. 内部状态 ==========
        self.start_epoch = 0
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = -float('inf') if self.metric_mode == 'max' else float('inf')
        self.training_history: List[Dict[str, float]] = []
        self.interrupt_state: Dict[str, Any] = {}

        # ========== 6. 自动恢复检查点 ==========
        if auto_resume and self.checkpoint_manager:
            self._load_checkpoint()

        logger.success("Trainer 初始化完成")

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
        从配置对象创建 Trainer 实例 (适合大型项目)。

        参数:
            model, optimizer, criterion, device, scheduler: 同 __init__
            config (ConfigNamespace): 包含 training/checkpoint/ntfy 子配置的完整配置对象

        返回:
            Trainer: 配置好的 Trainer 实例

        示例:
            config = setup_config(...)
            trainer = Trainer.from_config(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                config=config
            )
        """
        logger.info("Trainer 初始化 (from_config 模式)...")

        # 从 config 自动实例化所有工具
        try:
            train_cfg = config.training
            ckpt_cfg = config.checkpoint
            ntfy_cfg = config.ntfy
        except AttributeError as e:
            logger.error(f"配置对象缺少必需的子配置: {e}")
            raise ValueError(f"配置对象不完整: {e}")

        # 实例化工具
        ckpt_mgr = CheckpointManager(
            save_dir=ckpt_cfg.save_dir,
            max_to_keep=ckpt_cfg.max_to_keep,
            device=device
        )

        stopper = EarlyStopper(
            patience=train_cfg.patience,
            mode=train_cfg.metric_mode,
            min_delta=getattr(train_cfg, 'min_delta', 0.0)
        )

        notifier = NtfyNotifier(
            server_url = ntfy_cfg.server_url,
            topic = ntfy_cfg.topic,
        ) if ntfy_cfg.enabled else None

        # 调用标准构造函数
        return cls(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            checkpoint_manager=ckpt_mgr,
            early_stopper=stopper,
            notifier=notifier,
            scheduler=scheduler,
            use_amp=train_cfg.use_amp,
            grad_accum_steps=train_cfg.grad_accum_steps,
            max_grad_norm=getattr(train_cfg, 'max_grad_norm', None),
            metric_to_track=train_cfg.metric_to_track,
            metric_mode=train_cfg.metric_mode,
            compute_top5=getattr(train_cfg, 'compute_top5', False),
            log_interval=getattr(train_cfg, 'log_interval', 1),
            val_interval=getattr(train_cfg, 'val_interval', 1)
        )

    # ============================================================
    # 公共方法 - 主训练循环
    # ============================================================

    def fit(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            epochs: int = 100
    ) -> Dict[str, Any]:
        """
        主训练循环。

        参数:
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader, optional): 验证数据加载器
            epochs (int): 总训练轮数 (默认: 100)

        返回:
            dict: 训练历史和最佳指标
        """
        total_start_time = time.monotonic()

        logger.info("=" * 70)
        logger.info(f"🚀 开始训练: Epoch {self.start_epoch + 1} -> {epochs}".center(70))
        logger.info(f"   跟踪指标: '{self.metric_to_track}' (模式: {self.metric_mode})".center(70))
        if self.early_stopper:
            logger.info(f"   早停耐心: {self.early_stopper.patience} epochs".center(70))
        log_memory_usage("训练开始前")
        logger.info("=" * 70)

        if self.notifier:
            self.notifier.notify_start(
                f"训练开始\n"
                f"Epochs: {self.start_epoch + 1} → {epochs}\n"
                f"跟踪指标: {self.metric_to_track}"
            )

        try:
            for epoch in range(self.start_epoch, epochs):
                self.current_epoch = epoch
                epoch_start_time = time.monotonic()

                # ========== 1. 训练阶段 ==========
                train_metrics = self._train_epoch(train_loader)
                self._on_train_epoch_end(epoch, train_metrics)

                # ========== 2. 验证阶段 ==========
                val_metrics = {}
                if val_loader and (epoch % self.val_interval == 0 or epoch == epochs - 1):
                    val_metrics = self._eval_epoch(val_loader)
                    self._on_eval_epoch_end(epoch, val_metrics)

                # ========== 3. 日志记录 ==========
                if epoch % self.log_interval == 0:
                    self._log_epoch_metrics(epoch, epochs, train_metrics, val_metrics, epoch_start_time)

                # 记录历史
                epoch_history = {
                    'epoch': epoch,
                    **{f'train_{k}': v for k, v in train_metrics.items()},
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                }
                self.training_history.append(epoch_history)

                # ========== 4. 学习率调度 ==========
                self._step_scheduler(val_metrics)

                # ========== 5. 检查点保存与早停 ==========
                should_stop = self._save_and_check_stop(epoch, val_metrics)
                if should_stop:
                    logger.warning(f"早停触发，训练终止于 Epoch {epoch + 1}")
                    break

        except KeyboardInterrupt:
            logger.critical(f"检测到键盘中断 (Ctrl+C)，训练被中断于 Epoch {self.current_epoch + 1}")
            self._handle_interrupt()
            if self.notifier:
                self.notifier.notify_error(
                    "训练被用户手动中断",
                    "KeyboardInterrupt"
                )

        except Exception as e:
            logger.error(f"训练过程中发生未捕获的异常: {type(e).__name__}")
            error_details = traceback.format_exc()
            logger.exception(error_details)
            if self.notifier:
                self.notifier.notify_error(
                    f"训练失败: {type(e).__name__}",
                    error_details
                )
            raise

        else:
            logger.success("=" * 70)
            logger.success(f"训练在 Epoch {epochs} 正常完成")
            logger.success(f"最佳指标 ({self.metric_to_track}): {self.best_metric:.4f}")
            logger.success("=" * 70)

            if self.notifier:
                self.notifier.notify_success(
                    f"训练已正常完成\n\n"
                    f"**总轮数:** {epochs}\n"
                    f"**最佳指标:** {self.best_metric:.4f}"
                )

        finally:
            total_duration = time.monotonic() - total_start_time
            logger.info(f"总训练耗时: {format_time(total_duration)}")
            self._cleanup()

        return {
            'history': self.training_history,
            'best_metric': self.best_metric
        }

    # ============================================================
    # 核心私有方法 - 训练和评估的一个 epoch
    # ============================================================

    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        训练一个 epoch。

        参数:
            train_loader (DataLoader): 训练数据加载器

        返回:
            dict: 包含平均指标的字典 (例如 {'loss': 0.123, 'acc': 95.2})
        """
        self.model.train()

        self.metric_tracker.reset()
        self.lr_meter.reset()

        with Progress(
                train_loader,
                description=f"Epoch {self.current_epoch + 1} [Train]",
                device=self.device,
                leave=False
        ) as pbar:

            for step, batch in enumerate(pbar):
                # ========== 1. 执行训练步骤 ==========
                step_result = self._train_step(batch)

                loss = step_result['loss']
                outputs = step_result['outputs']
                targets = step_result['targets']

                # ========== 2. 反向传播 ==========
                scaled_loss = loss / self.grad_accum_steps

                if self.scaler.is_enabled():
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                # ========== 3. 优化器更新 ==========
                if (step + 1) % self.grad_accum_steps == 0:
                    if self.max_grad_norm is not None:
                        if self.scaler.is_enabled():
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )

                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                # ========== 4. 指标跟踪 ==========
                self.metric_tracker.update(loss, outputs, targets)

                current_lr = self.optimizer.param_groups[0]['lr']
                self.lr_meter.update(current_lr)

                pbar.update({'loss': loss, 'lr': current_lr})

        # ========== 5. 计算平均指标 ==========
        metrics = self.metric_tracker.compute()
        metrics['lr'] = self.lr_meter.avg
        return metrics

    def _eval_epoch(self, eval_loader: DataLoader) -> Dict[str, float]:
        """
        评估一个 epoch。

        参数:
            eval_loader (DataLoader): 评估数据加载器

        返回:
            dict: 包含平均指标的字典
        """
        self.model.eval()

        self.metric_tracker.reset()

        with Progress(
                eval_loader,
                description=f"Epoch {self.current_epoch + 1} [Eval]",
                device=self.device,
                leave=False
        ) as pbar:

            with torch.no_grad():
                for batch in pbar:
                    # ========== 1. 执行评估步骤 ==========
                    step_result = self._eval_step(batch)

                    loss = step_result['loss']
                    outputs = step_result['outputs']
                    targets = step_result['targets']

                    # ========== 2. 指标跟踪 ==========
                    self.metric_tracker.update(loss, outputs, targets)
                    pbar.update({'loss': loss})

        # ========== 3. 计算平均指标 ==========
        return self.metric_tracker.compute()

    # ============================================================
    # 可重写的保护方法 - 供子类定制
    # ============================================================

    def _train_step(self, batch) -> Dict[str, torch.Tensor]:
        """
        (可重写) 单个训练步骤。

        默认实现: 标准的分类任务
            - 输入: (images, labels)
            - 输出: logits
            - 损失: criterion(logits, labels)

        子类可以重写以支持:
            - 多任务学习 (多个输出和损失)
            - GAN 训练 (生成器和判别器)
            - 对比学习
            - 自监督学习

        参数:
            batch: 来自 DataLoader 的一个批次 (通常是 (inputs, targets))

        返回:
            dict: 必须包含以下键
                - 'loss' (Tensor): 当前 batch 的损失 (标量)
                - 'outputs' (Tensor): 模型输出的 logits (shape: [batch, num_classes])
                - 'targets' (Tensor): 真实标签 (shape: [batch])

        示例 (多任务学习):
            def _train_step(self, batch):
                inputs, target_cls, target_seg = batch
                inputs = inputs.to(self.device, non_blocking=True)
                target_cls = target_cls.to(self.device, non_blocking=True)
                target_seg = target_seg.to(self.device, non_blocking=True)

                with autocast(self.device.type, enabled=self.scaler.is_enabled()):
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
        inputs, targets = batch

        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=self.scaler.is_enabled()):
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

        默认实现: 与 _train_step 相同的逻辑 (但在 eval 模式和 no_grad 下)

        参数:
            batch: 来自 DataLoader 的一个批次

        返回:
            dict: 与 _train_step 相同的格式
        """
        inputs, targets = batch

        inputs = inputs.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=self.scaler.is_enabled()):
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
            - 记录额外的信息 (例如权重直方图)
            - 执行自定义的逻辑 (例如更新可视化)

        参数:
            epoch (int): 当前 epoch 编号
            train_metrics (dict): 训练指标
        """
        pass

    def _on_eval_epoch_end(self, epoch: int, val_metrics: Dict[str, float]):
        """
        (可重写) 评估 epoch 结束时的钩子。

        参数:
            epoch (int): 当前 epoch 编号
            val_metrics (dict): 验证指标
        """
        pass

    # ============================================================
    # 辅助私有方法 - 检查点、早停、学习率调度等
    # ============================================================

    def _save_and_check_stop(
            self,
            epoch: int,
            val_metrics: Dict[str, float]
    ) -> bool:
        """
        封装检查点保存和早停逻辑。

        参数:
            epoch (int): 当前 epoch
            val_metrics (dict): 验证指标

        返回:
            bool: 是否应该停止训练
        """
        if not self.checkpoint_manager and not self.early_stopper:
            return False

        is_best = False
        should_stop = False

        # 构建状态字典
        state = self._build_checkpoint_state(epoch, val_metrics)
        self.interrupt_state = state

        # 保存滚动检查点
        if self.checkpoint_manager:
            self.checkpoint_manager.save_epoch_checkpoint(state, epoch)

        # 早停与最佳模型
        if self.early_stopper and val_metrics:
            current_metric = val_metrics.get(self.metric_to_track)

            if current_metric is not None:
                is_best = self.early_stopper.step(current_metric)

                if is_best:
                    self.best_metric = self.early_stopper.best_metric
                    if self.checkpoint_manager:
                        logger.success(
                            f"Epoch {epoch + 1}: 发现新的最佳模型，"
                            f"{self.metric_to_track}={self.best_metric:.4f}"
                        )
                        self.checkpoint_manager.save_best_model(state, self.best_metric)

                should_stop = self.early_stopper.should_stop

        return should_stop

    def _load_checkpoint(self):
        """
        从检查点恢复训练。
        """
        logger.info("尝试从检查点恢复训练...")

        checkpoint = self.checkpoint_manager.load_latest_checkpoint()

        if checkpoint is None:
            logger.info("未找到检查点，从头开始训练")
            return

        try:
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])

            self.start_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint.get('global_step', 0)

            if self.scheduler and 'scheduler_state' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
                logger.debug("学习率调度器状态已恢复")

            if self.early_stopper and 'early_stopper_state' in checkpoint:
                self.early_stopper.load_state_dict(checkpoint['early_stopper_state'])
                logger.debug("早停器状态已恢复")

            if self.scaler.is_enabled() and 'scaler_state' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state'])
                logger.debug("GradScaler 状态已恢复")

            if 'best_metric' in checkpoint:
                self.best_metric = checkpoint['best_metric']

            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']

            logger.success(f"训练已从 Epoch {self.start_epoch} 恢复")

        except Exception as e:
            logger.error(f"加载检查点失败: {e}，将从头开始训练")
            self.start_epoch = 0

    def _build_checkpoint_state(
            self,
            epoch: int,
            val_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        构建检查点状态字典。

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
            'training_history': self.training_history
        }

        if val_metrics:
            state['val_metrics'] = val_metrics

        if self.scheduler:
            state['scheduler_state'] = self.scheduler.state_dict()

        if self.early_stopper:
            state['early_stopper_state'] = self.early_stopper.state_dict()

        if self.scaler.is_enabled():
            state['scaler_state'] = self.scaler.state_dict()

        return state

    def _step_scheduler(self, val_metrics: Dict[str, float]):
        """
        执行学习率调度器步骤。

        参数:
            val_metrics (dict): 验证指标
        """
        if not self.scheduler:
            return

        scheduler_name = type(self.scheduler).__name__

        if scheduler_name == 'ReduceLROnPlateau':
            metric_val = val_metrics.get(self.metric_to_track)
            if metric_val is not None:
                self.scheduler.step(metric_val)
            else:
                logger.warning("ReduceLROnPlateau 需要验证指标，但未提供")
        else:
            self.scheduler.step()

    def _handle_interrupt(self):
        """
        处理训练中断 (Ctrl+C)。
        """
        if self.checkpoint_manager and self.interrupt_state:
            logger.warning("正在保存中断检查点...")
            self.checkpoint_manager.save_interrupt_checkpoint(self.interrupt_state)
            logger.success("中断检查点已保存")
        else:
            logger.warning("未配置 CheckpointManager 或状态为空，中断检查点未保存")

    def _cleanup(self):
        """
        训练结束后的资源清理。
        """
        log_memory_usage("训练结束后")
        if self.device.type == 'cuda':
            clear_memory()
            logger.debug("GPU 缓存已清理")

    def _log_epoch_metrics(
            self,
            epoch: int,
            total_epochs: int,
            train_metrics: Dict[str, float],
            val_metrics: Dict[str, float],
            epoch_start_time: float
    ):
        """
        格式化并记录指标到日志。

        参数:
            epoch (int): 当前 epoch
            total_epochs (int): 总 epoch 数
            train_metrics (dict): 训练指标
            val_metrics (dict): 验证指标
            epoch_start_time (float): epoch 开始时间
        """
        duration = time.monotonic() - epoch_start_time

        msg_parts = [f"Epoch {epoch + 1:03d}/{total_epochs}"]
        msg_parts.append(f"Time: {format_time(duration)}")
        msg_parts.append(f"Loss: {train_metrics['loss']:.4f}")
        msg_parts.append(f"Acc: {train_metrics.get('acc', 0):.2f}%")

        if val_metrics:
            msg_parts.append(f"Val Loss: {val_metrics['loss']:.4f}")
            msg_parts.append(f"Val Acc: {val_metrics.get('acc', 0):.2f}%")

        if 'top5' in train_metrics:
            msg_parts.append(f"Top5: {train_metrics['top5']:.2f}%")

        msg_parts.append(f"LR: {train_metrics.get('lr', 0):.2e}")

        if self.device.type == 'cuda':
            mem_info = get_memory_usage()
            if mem_info:
                msg_parts.append(f"Mem: {mem_info['allocated']}")

        log_msg = " | ".join(msg_parts)
        logger.success(log_msg)

    # ============================================================
    # 实用方法 - 供外部调用
    # ============================================================

    def get_current_lr(self) -> float:
        """
        获取当前学习率。

        返回:
            float: 当前学习率
        """
        return self.optimizer.param_groups[0]['lr']

    def get_training_history(self) -> List[Dict[str, float]]:
        """
        获取训练历史。

        返回:
            list: 包含所有训练和验证指标的历史记录
        """
        return self.training_history

    def get_best_metric(self) -> float:
        """
        获取最佳指标值。

        返回:
            float: 最佳指标值
        """
        return self.best_metric

    def __repr__(self) -> str:
        return (
            f"Trainer(\n"
            f"  model={type(self.model).__name__},\n"
            f"  device={self.device},\n"
            f"  use_amp={self.use_amp},\n"
            f"  grad_accum_steps={self.grad_accum_steps},\n"
            f"  current_epoch={self.current_epoch},\n"
            f"  best_metric={self.best_metric:.4f}\n"
            f")"
        )