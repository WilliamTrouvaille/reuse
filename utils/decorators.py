#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/2 00:22
@author  : William_Trouvaille
@function: 通用装饰器
@description: 提供可复用的通用装饰器，用于处理日志、计时、PyTorch 状态管理和错误处理等"横切关注点"。
"""
import functools
import time
import traceback
from typing import Callable, Any, Optional

import torch
import torch.nn as nn
from loguru import logger

# 单向依赖
from .helpers import format_time
from .ntfy_notifier import NtfyNotifier


def time_it(func: Callable) -> Callable:
    """
    装饰器：测量并记录函数的执行时间。

    用法:
        @time_it
        def my_expensive_function():
            ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.monotonic()

        # 执行原始函数
        result = func(*args, **kwargs)

        end_time = time.monotonic()
        duration = end_time - start_time

        # 使用我们之前创建的 format_time 辅助函数
        logger.info(f"函数 '{func.__name__}' 执行完毕，耗时: {format_time(duration)}")
        return result

    return wrapper


def no_grad(func: Callable) -> Callable:
    """
    装饰器：在 `torch.no_grad()` 上下文中执行函数。

    (用于评估、验证或推理函数)

    用法:
        @no_grad
        def evaluate_step(model, data):
            ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        with torch.no_grad():
            return func(*args, **kwargs)

    return wrapper


def _model_mode_setter(mode: bool, model_attr: str = 'model') -> Callable:
    """(私有) 创建 train() 或 eval() 装饰器的工厂函数。"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 假设被装饰的函数是一个类方法 (第一个参数是 self)
            if not args or not hasattr(args[0], model_attr):
                logger.warning(
                    f"@{'train_mode' if mode else 'eval_mode'} 装饰器无法在 "
                    f"'{args[0].__class__.__name__}' 中找到 '{model_attr}' 属性。"
                )
                return func(*args, **kwargs)

            # 1. 获取模型
            self_obj = args[0]
            model = getattr(self_obj, model_attr)

            if not isinstance(model, nn.Module):
                logger.warning(
                    f"'{model_attr}' 属性不是一个 nn.Module，跳过模式设置。"
                )
                return func(*args, **kwargs)

            # 2. (关键) 保存原始模式
            original_mode = model.training

            # 3. 设置新模式
            model.train(mode)

            # 4. 执行函数
            try:
                result = func(*args, **kwargs)
            finally:
                # 5. (关键) 无论如何都恢复原始模式
                # (防止在 eval 函数中调用了带 @train_mode 的辅助函数)
                model.train(original_mode)

            return result

        return wrapper

    return decorator


def train_mode(model_attr: str = 'model') -> Callable:
    """
    装饰器：在执行函数前将模型设置为 .train() 模式，
    并在结束后恢复其原始模式。

    (假设用于类方法，例如 Trainer.train_step())

    用法:
        @train_mode(model_attr='my_net') # 可选参数
        def train_step(self, data):
            # self.my_net 自动处于 .train() 模式
            ...
    """
    return _model_mode_setter(mode=True, model_attr=model_attr)


def eval_mode(model_attr: str = 'model') -> Callable:
    """
    装饰器：在执行函数前将模型设置为 .eval() 模式，
    并在结束后恢复其原始模式。

    (假设用于类方法，例如 Trainer.evaluate())

    用法:
        @eval_mode(model_attr='model')
        @no_grad # 装饰器可以堆叠
        def evaluate(self):
            # self.model 自动处于 .eval() 模式
            ...
    """
    return _model_mode_setter(mode=False, model_attr=model_attr)


def log_errors(
        notifier: Optional[NtfyNotifier] = None,
        re_raise: bool = True
) -> Callable:
    """
    装饰器（工厂）：自动捕获异常，使用 logger.exception() 记录，
    并（可选）通过 Ntfy 发送通知。

    用法:
        # 在 main.py 中:
        my_notifier = NtfyNotifier()

        @log_errors(notifier=my_notifier, re_raise=False)
        def my_risky_function():
            # ...
            # 如果这里出错，程序不会崩溃，但会记录日志+发送通知
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 1. (关键) 使用 logger.exception 记录完整的堆栈跟踪
                func_name = func.__name__
                logger.error(f"函数 '{func_name}' 中发生未捕获的异常！")
                logger.exception(e)  # 这会自动打印堆栈

                # 2. (可选) 发送 Ntfy 通知
                if notifier:
                    error_details = traceback.format_exc()
                    notifier.notify_error(
                        message=f"函数 '{func_name}' 失败: {type(e).__name__}",
                        error_details=error_details
                    )

                # 3. (可选) 重新抛出异常（使程序崩溃）
                if re_raise:
                    raise e

                # 如果 re_raise=False，则返回 None 或默认值
                return None

        return wrapper

    return decorator
