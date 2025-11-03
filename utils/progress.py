#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/1 22:51
@author  : William_Trouvaille
@function: 高性能进度条包装器
@description: 提供了 Progress 类，它封装了 TQDM，并使用 "基于时间的节流" 策略来更新指标，以避免在快速的 CUDA 循环中因 I/O 和同步 (.item()) 而导致的性能瓶颈。
"""
import time
from typing import Dict, Any, Optional, Union

import torch
from loguru import logger
from tqdm import tqdm


class Progress:
    """
    一个对性能友好的 TQDM 包装器，用于 PyTorch 训练。

    它通过在 GPU 上累积指标（Tensors）并在后台进行“时间节流”来
    最小化 .item() 同步和 I/O (set_postfix) 开销。

    它始终显示"整个周期的运行平均值"，而不是"最后 N 秒的平均值"。
    """

    def __init__(
            self,
            iterable,
            description: str = "Processing",
            leave: bool = False,
            update_interval_sec: float = 1.5,
            device: Optional[torch.device] = None
    ):
        """
        初始化 TQDM 包装器。

        参数:
            iterable: 要迭代的对象 (例如 DataLoader)。
            description (str): 进度条的描述文本。
            leave (bool): 结束后是否保留进度条 (True=保留, False=清除)。
            update_interval_sec (float): 更新 `set_postfix` (I/O) 的最小间隔（秒）。
            device (torch.device, optional): 指标累加器所在的设备。如果为 None，将从第一个 tensor 自动推断。
        """

        # 1. 创建 TQDM 实例并将其暴露
        self.tqdm_bar = tqdm(
            iterable,
            desc=description,
            leave=leave,
            ncols=120,
            dynamic_ncols=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        )
        self.iterable = self.tqdm_bar

        # 2. 节流 (Throttling) 控制
        self.update_interval = update_interval_sec
        # (使用 time.monotonic() 更适合测量时间间隔)
        self.last_update_time = time.monotonic()

        # 3. 指标累加状态
        self.device = device
        self.is_initialized = False

        # (关键) 这些状态用于计算 *整个 epoch* 的运行平均值
        # 它们在 .close() 之前不会被重置
        self.metric_accumulators: Dict[str, torch.Tensor] = {}
        self.total_steps = 0
        self.current_non_tensor_metrics: Dict[str, Any] = {}

        logger.debug(f"Progress '{description}' 已初始化 (更新间隔: {update_interval_sec}s)。")

    def _initialize_accumulators(self, tensor_metrics: Dict[str, torch.Tensor]):
        """在第一次调用时，根据传入的 Tensors 初始化累加器。"""
        logger.debug("Progress 正在初始化累加器...")

        # 1. 自动推断设备 (如果未提供)
        if self.device is None:
            self.device = next(iter(tensor_metrics.values())).device
            logger.debug(f"自动推断设备: {self.device}")

        # 2. 在目标设备上创建累加器
        for key, tensor in tensor_metrics.items():
            self.metric_accumulators[key] = torch.tensor(
                0.0,
                device=self.device,
                dtype=tensor.dtype
            )

        self.is_initialized = True

    def update(self, metrics_dict: Dict[str, Union[torch.Tensor, float, int]]):
        """
        (在循环内调用) 更新当前步骤的指标。

        此方法在绝大多数情况下是非阻塞的。

        参数:
            metrics_dict (dict): 包含当前步骤值的字典 (例如 {'loss': tensor, 'lr': 0.01})
        """

        # 1. (仅第一次) 初始化累加器
        if not self.is_initialized:
            tensor_metrics = {k: v for k, v in metrics_dict.items() if isinstance(v, torch.Tensor)}
            if tensor_metrics:
                self._initialize_accumulators(tensor_metrics)
            else:
                self.is_initialized = True  # 也许没有 tensor

        # 2. 累积指标 (为整个 epoch)
        self.total_steps += 1

        for key, value in metrics_dict.items():
            if isinstance(value, torch.Tensor):
                # (关键) 在 GPU 上累积
                if key in self.metric_accumulators:
                    self.metric_accumulators[key] += value.detach()
                else:
                    # mid-epoch 出现了新的 tensor
                    logger.warning(f"在 {key} 累加器未初始化时收到了该 tensor。")
            else:
                # 存储非 Tensor 指标 (例如 'lr') 的最新值
                self.current_non_tensor_metrics[key] = value

        # 3. (关键) 节流 I/O
        current_time = time.monotonic()
        if (current_time - self.last_update_time) < self.update_interval:
            # 时间未到，跳过昂贵的 I/O，立即返回
            return

        # --- 4. 时间到了，执行昂贵的更新 ---
        self.last_update_time = current_time

        postfix_display = {}

        # a. 计算 Tensor 的运行平均值
        if self.total_steps > 0:
            for key, tensor_sum in self.metric_accumulators.items():
                avg_tensor = tensor_sum / self.total_steps

                # b. *** 唯一的 CUDA 同步点 ***
                avg_float = avg_tensor.item()

                # c. 准备显示 (例如 'loss': "0.1234")
                postfix_display[key] = f"{avg_float:.4f}"

        # d. 添加非 Tensor 指标 (例如 'lr': 1.0e-04)
        for key, value in self.current_non_tensor_metrics.items():
            if isinstance(value, float):
                postfix_display[key] = f"{value:.1e}"
            else:
                postfix_display[key] = value

        # e. (昂贵的 I/O) 更新 TQDM 显示
        try:
            self.tqdm_bar.set_postfix(postfix_display)
        except Exception as e:
            # 进度条可能已关闭或出现其他异常，记录但不中断训练
            logger.debug(f"更新进度条失败（可能已关闭）: {e}")

    def close(self) -> None:
        """
        (在循环后调用) 关闭进度条。
        """
        logger.debug(f"Progress '{self.tqdm_bar.desc}' 正在关闭...")

        # (可选) 在关闭前设置一次最终的、精确的 postfix
        final_metrics = self.get_final_metrics()
        formatted_postfix = {
            key: f"{value:.4f}"
            for key, value in final_metrics.items()
        }
        # 合并最终的非 tensor 指标
        for key, value in self.current_non_tensor_metrics.items():
            if key not in formatted_postfix:
                if isinstance(value, float):
                    formatted_postfix[key] = f"{value:.1e}"
                else:
                    formatted_postfix[key] = value

        self.tqdm_bar.set_postfix(formatted_postfix)

        self.tqdm_bar.close()

    def get_final_metrics(self) -> Dict[str, float]:
        """
        计算并返回整个周期的最终平均指标 (Tensors Only)。
        """
        if self.total_steps == 0:
            return {}

        final_avg_metrics = {
            key: (tensor_sum / self.total_steps).item()
            for key, tensor_sum in self.metric_accumulators.items()
        }
        return final_avg_metrics

    # --- 支持 'with' 语句 和 'for' 循环 ---

    def __iter__(self):
        """允许 'for item in tracker:' """
        return self.iterable.__iter__()

    def __enter__(self):
        """允许 'with tracker:' """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """自动调用 .close() """
        self.close()
