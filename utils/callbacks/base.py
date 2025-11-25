#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/07
@author  : William_Trouvaille
@function: Callback 基类 - 训练回调系统的核心抽象
@detail:
    参考 PyTorch Lightning 的 Callback 设计，但简化为适合本项目的钩子集合。
    提供完整的生命周期钩子，支持状态持久化，完全解耦。

    原始来源: PyTorch Lightning
    原始许可证: Apache License 2.0
    原始版权: Copyright The Lightning AI team.
    原始仓库: https://github.com/Lightning-AI/pytorch-lightning
    原始文件: lightning/pytorch/callbacks/callback.py
"""

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from utils.train import Trainer


# ========================================================================
# 1. Callback 基类
# ========================================================================

class Callback:
    """训练回调基类，提供完整的生命周期钩子。

    Callback 是一种设计模式，允许在训练过程的关键节点插入自定义逻辑，
    而无需修改核心训练代码。这实现了训练逻辑与辅助功能的完全解耦。

    核心特性:
    - 生命周期钩子: 覆盖训练、验证、测试的各个阶段
    - 状态持久化: 支持保存和加载回调状态到检查点
    - 完全解耦: 通过 trainer 参数访问训练状态，不侵入训练代码

    使用示例:
        class MyCallback(Callback):
            def on_train_start(self, trainer: "Trainer") -> None:
                print(f"Training started on device {trainer.device}")

            def on_train_epoch_end(self, trainer: "Trainer") -> None:
                print(f"Epoch {trainer.current_epoch} completed")

        # 使用回调
        trainer = Trainer(..., callbacks=[MyCallback()])
        trainer.fit(train_loader, val_loader)

    钩子调用顺序（典型训练流程）:
        1. setup(trainer)                       # Trainer 初始化完成后
        2. on_train_start(trainer)              # 训练开始
        3. FOR each epoch:
            4. on_train_epoch_start(trainer)       # 训练 epoch 开始
            5. ... 训练循环 ...
            6. on_train_epoch_end(trainer)         # 训练 epoch 结束
            7. on_validation_epoch_start(trainer)  # 验证 epoch 开始
            8. ... 验证循环 ...
            9. on_validation_epoch_end(trainer)    # 验证 epoch 结束
        10. on_train_end(trainer)               # 训练结束
        11. teardown(trainer)                   # 清理资源

    状态持久化:
        如果回调有状态需要保存到检查点，重写以下方法：
        - state_dict(): 返回包含状态的字典
        - load_state_dict(state_dict): 从字典恢复状态

        状态会自动保存到检查点，并在恢复训练时加载。
    """

    # ========================================================================
    # 1.1 状态管理
    # ========================================================================

    @property
    def state_key(self) -> str:
        """回调状态的唯一标识符。

        用于在检查点中存储和检索回调状态：
        checkpoint["callbacks"][state_key] = callback.state_dict()

        默认使用类的全限定名（包括模块路径），确保唯一性。
        如果需要在同一个 Trainer 中使用多个相同类型的回调实例，
        应该重写此属性以生成唯一的键。

        Returns:
            回调的唯一标识符字符串

        示例:
            # 默认行为
            callback = EarlyStoppingCallback()
            print(callback.state_key)
            # 输出: "EarlyStoppingCallback"

            # 自定义唯一键（用于多实例场景）
            class MyCallback(Callback):
                def __init__(self, name: str):
                    self.name = name

                @property
                def state_key(self) -> str:
                    return f"{self.__class__.__qualname__}_{self.name}"
        """
        return self.__class__.__qualname__

    def _generate_state_key(self, **kwargs: Any) -> str:
        """生成带参数的状态键（辅助方法）。

        将一组键值对格式化为状态键字符串，用于区分同类型的多个实例。

        Args:
            **kwargs: 键值对参数，必须可序列化为字符串

        Returns:
            格式化的状态键字符串

        示例:
            state_key = self._generate_state_key(monitor='val_loss', patience=10)
            # 返回: "MyCallback{'monitor': 'val_loss', 'patience': 10}"
        """
        return f"{self.__class__.__qualname__}{repr(kwargs)}"

    def state_dict(self) -> dict[str, Any]:
        """保存回调状态到字典。

        当保存检查点时调用，用于序列化回调的内部状态。
        默认返回空字典（无状态回调）。

        Returns:
            包含回调状态的字典，键值对应该是可序列化的

        示例:
            def state_dict(self) -> dict[str, Any]:
                return {
                    'best_score': self.best_score,
                    'patience_counter': self.patience_counter,
                    'stopped': self.stopped
                }
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """从字典恢复回调状态。

        当从检查点恢复训练时调用，用于反序列化回调状态。
        默认实现为空（无状态回调）。

        Args:
            state_dict: 由 state_dict() 生成的状态字典

        示例:
            def load_state_dict(self, state_dict: dict[str, Any]) -> None:
                self.best_score = state_dict.get('best_score', -float('inf'))
                self.patience_counter = state_dict.get('patience_counter', 0)
                self.stopped = state_dict.get('stopped', False)
        """
        pass

    # ========================================================================
    # 2. 生命周期钩子 - 训练器级别
    # ========================================================================

    def setup(self, trainer: "Trainer") -> None:
        """Trainer 初始化完成后调用。

        此钩子在 Trainer.__init__() 完成后立即调用，适合执行：
        - 访问 trainer 的属性进行初始化
        - 设置依赖于 trainer 配置的内部状态
        - 执行一次性的准备工作

        Args:
            trainer: Trainer 实例，可以访问其所有属性（model、optimizer等）

        示例:
            def setup(self, trainer: "Trainer") -> None:
                # 根据 trainer 配置初始化
                self.device = trainer.device
                self.log_dir = Path(trainer.checkpoint_manager.save_dir) / "logs"
                self.log_dir.mkdir(parents=True, exist_ok=True)
        """
        pass

    def teardown(self, trainer: "Trainer") -> None:
        """训练完全结束时调用（无论成功、失败还是中断）。

        此钩子在训练流程的最后调用（finally 块中），适合执行：
        - 清理临时资源
        - 关闭文件句柄
        - 释放内存
        - 保存最终报告

        Args:
            trainer: Trainer 实例

        示例:
            def teardown(self, trainer: "Trainer") -> None:
                # 关闭日志文件
                if hasattr(self, 'log_file'):
                    self.log_file.close()
                # 清理临时目录
                if hasattr(self, 'temp_dir'):
                    shutil.rmtree(self.temp_dir, ignore_errors=True)
        """
        pass

    # ========================================================================
    # 3. 生命周期钩子 - 训练阶段
    # ========================================================================

    def on_train_start(self, trainer: "Trainer") -> None:
        """训练开始时调用（在第一个 epoch 之前）。

        适合执行：
        - 记录训练开始时间
        - 发送通知
        - 初始化统计指标

        Args:
            trainer: Trainer 实例

        示例:
            def on_train_start(self, trainer: "Trainer") -> None:
                self.start_time = time.time()
                logger.info(f"Training started with {trainer.device}")
        """
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        """训练正常结束时调用（在最后一个 epoch 之后）。

        注意：此钩子仅在训练正常完成时调用，不会在异常或中断时调用。

        适合执行：
        - 生成训练报告
        - 发送完成通知
        - 保存最终结果

        Args:
            trainer: Trainer 实例

        示例:
            def on_train_end(self, trainer: "Trainer") -> None:
                duration = time.time() - self.start_time
                logger.success(f"Training completed in {duration:.2f}s")
                self.generate_report(trainer)
        """
        pass

    # ========================================================================
    # 4. 生命周期钩子 - Epoch 级别
    # ========================================================================

    def on_train_epoch_start(self, trainer: "Trainer") -> None:
        """训练 epoch 开始时调用（在第一个 batch 之前）。

        适合执行：
        - 重置 epoch 级别的统计
        - 创建进度条
        - 记录 epoch 开始时间

        Args:
            trainer: Trainer 实例，可通过 trainer.current_epoch 获取当前 epoch

        示例:
            def on_train_epoch_start(self, trainer: "Trainer") -> None:
                self.epoch_start_time = time.time()
                logger.info(f"Epoch {trainer.current_epoch + 1} started")
        """
        pass

    def on_train_epoch_end(self, trainer: "Trainer") -> None:
        """训练 epoch 结束时调用（在最后一个 batch 之后）。

        适合执行：
        - 记录 epoch 指标
        - 更新学习率
        - 保存检查点
        - 检查早停条件

        Args:
            trainer: Trainer 实例

        示例:
            def on_train_epoch_end(self, trainer: "Trainer") -> None:
                metrics = trainer.metric_tracker.compute()
                self.history.append(metrics)
                logger.info(f"Epoch {trainer.current_epoch + 1} metrics: {metrics}")
        """
        pass

    def on_validation_epoch_start(self, trainer: "Trainer") -> None:
        """验证 epoch 开始时调用。

        Args:
            trainer: Trainer 实例
        """
        pass

    def on_validation_epoch_end(self, trainer: "Trainer") -> None:
        """验证 epoch 结束时调用。

        适合执行：
        - 记录验证指标
        - 检查早停条件
        - 更新最佳模型

        Args:
            trainer: Trainer 实例

        示例:
            def on_validation_epoch_end(self, trainer: "Trainer") -> None:
                val_metrics = trainer.val_metrics  # 假设 trainer 存储了验证指标
                if val_metrics['acc'] > self.best_acc:
                    self.best_acc = val_metrics['acc']
                    logger.info(f"New best accuracy: {self.best_acc:.2f}%")
        """
        pass

    # ========================================================================
    # 5. 生命周期钩子 - 异常处理
    # ========================================================================

    def on_exception(self, trainer: "Trainer", exception: BaseException) -> None:
        """当训练被异常中断时调用。

        注意：此钩子在 except 块中调用，然后异常会被重新抛出。

        适合执行：
        - 发送错误通知
        - 保存调试信息
        - 记录异常日志

        Args:
            trainer: Trainer 实例
            exception: 捕获的异常对象

        示例:
            def on_exception(self, trainer, exception) -> None:
                logger.error(f"Training interrupted by {type(exception).__name__}")
                # 保存当前状态用于调试
                self.save_debug_info(trainer, exception)
                # 发送错误通知
                if hasattr(self, 'notifier'):
                    self.notifier.send_error(str(exception))
        """
        pass

    # ========================================================================
    # 6. 字符串表示
    # ========================================================================

    def __repr__(self) -> str:
        """返回回调的字符串表示。

        Returns:
            格式化的字符串，包含回调类名和状态键
        """
        return f"{self.__class__.__name__}(state_key='{self.state_key}')"
