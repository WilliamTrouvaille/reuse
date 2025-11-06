#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/1 21:45
@author  : William_Trouvaille
@function: 通用辅助函数
@detail:
    本文件部分功能源自 PyTorch Lightning 项目
    原始许可证: Apache License 2.0
    原始版权: Copyright The Lightning AI team.
    原始仓库: https://github.com/Lightning-AI/pytorch-lightning
    源文件:
        - lightning/pytorch/utilities/memory.py (内存管理功能)
        - lightning/fabric/utilities/seed.py (随机种子隔离功能)
"""

import gc
import json
import math
import os
import random
import time
from contextlib import contextmanager
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state
from typing import Dict, Any, Optional, Union, Generator

import numpy as np
import torch
from loguru import logger


def get_time(format_str: str = "[%Y-%m-%d %H:%M:%S]") -> str:
    """
    获取格式化的当前时间。

    参数:
        format_str (str): 时间格式字符串 (例如 "[%Y-%m-%d %H:%M:%S]")

    返回:
        str: 格式化的时间字符串
    """
    return str(time.strftime(format_str, time.localtime()))


def format_time(seconds: float) -> str:
    """
    格式化时间（秒）为可读的字符串 (例如 "1h 15m 30.1s")。

    参数:
        seconds (float): 秒数

    返回:
        str: 格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {remaining_minutes}m {remaining_seconds:.1f}s"


# ========================================================================
# 2. 随机种子管理（源自 PyTorch Lightning）
# ========================================================================

def _collect_rng_states(include_cuda: bool = True) -> Dict[str, Any]:
    """
    (私有) 收集 torch、torch.cuda、numpy 和 Python 的全局随机状态。

    参数:
        include_cuda (bool): 是否收集 CUDA 随机状态

    返回:
        dict: 包含各库随机状态的字典
    """
    states = {
        "torch": torch.get_rng_state(),
        "python": python_get_rng_state(),
        "numpy": np.random.get_state(),
    }
    if include_cuda and torch.cuda.is_available():
        states["torch.cuda"] = torch.cuda.get_rng_state_all()
    return states


def _set_rng_states(rng_state_dict: Dict[str, Any]) -> None:
    """
    (私有) 设置 torch、torch.cuda、numpy 和 Python 的全局随机状态。

    参数:
        rng_state_dict (dict): 包含各库随机状态的字典
    """
    torch.set_rng_state(rng_state_dict["torch"])

    # 恢复 CUDA 随机状态（如果存在）
    if "torch.cuda" in rng_state_dict and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state_dict["torch.cuda"])

    # 恢复 NumPy 随机状态
    if "numpy" in rng_state_dict:
        np.random.set_state(rng_state_dict["numpy"])

    # 恢复 Python 随机状态
    version, state, gauss = rng_state_dict["python"]
    python_set_rng_state((version, tuple(state), gauss))


@contextmanager
def isolate_rng(include_cuda: bool = True) -> Generator[None, None, None]:
    """
    上下文管理器：在退出时将全局随机状态重置为进入前的状态。

    支持隔离 PyTorch、NumPy 和 Python 内置随机数生成器的状态。
    这对于需要固定随机性的代码块（如数据增强测试）非常有用。

    参数:
        include_cuda (bool): 是否控制 `torch.cuda` 随机数生成器。
                            在 fork 进程中禁止 CUDA 重新初始化时，应设置为 False。

    示例:
        >>> import torch
        >>> torch.manual_seed(1)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> with isolate_rng():
        ...     [torch.rand(1) for _ in range(3)]
        [tensor([0.7576]), tensor([0.2793]), tensor([0.4031])]
        >>> torch.rand(1)  # 全局状态未受影响
        tensor([0.7576])

    注意:
        - 这个上下文管理器会捕获进入时的随机状态，并在退出时恢复
        - 在上下文内部的任何随机操作都不会影响外部的随机状态
        - 适用于需要可复现的测试或特定随机行为的场景
    """
    # 保存当前随机状态
    states = _collect_rng_states(include_cuda)

    try:
        yield
    finally:
        # 恢复随机状态
        _set_rng_states(states)


def get_device(requested_device: str = 'auto') -> torch.device:
    """
    获取并验证 PyTorch 计算设备。

    - 'auto': 自动选择 GPU (如果可用)，否则 CPU
    - 'cuda': 尝试使用 'cuda:0'
    - 'cuda:1': 尝试使用 'cuda:1'
    - 'cpu': 使用 'cpu'

    参数:
        requested_device (str): 'auto', 'cpu', 'cuda', 'cuda:1' 等

    返回:
        torch.device: 验证后的 PyTorch 设备对象
    """
    logger.debug(f"请求的设备: '{requested_device}'")

    if requested_device == 'auto':
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_str = requested_device

    # 核心验证逻辑
    if device_str.startswith('cuda'):
        if not torch.cuda.is_available():
            logger.warning("CUDA 不可用。回退到 CPU。")
            device_str = 'cpu'
        else:
            # 检查 GPU 数量是否满足请求
            # (处理 'cuda:1' 这样的多 GPU 情况)
            if ":" in device_str:
                try:
                    gpu_id = int(device_str.split(':')[-1])
                    gpu_count = torch.cuda.device_count()
                    if gpu_id >= gpu_count:
                        logger.warning(f"请求的 GPU {device_str} 不可用 (仅找到 {gpu_count} 块 GPU)。")
                        logger.warning(f"回退到 'cuda:0'。")
                        device_str = 'cuda:0'
                except ValueError:
                    logger.error(f"无效的设备字符串: {device_str}。回退到 'cuda:0'。")
                    device_str = 'cuda:0'
            # 如果只是 'cuda'，则默认为 'cuda:0'
            else:
                device_str = 'cuda:0'

    # 创建设备对象
    device_obj = torch.device(device_str)

    # 打印最终设备信息
    if device_obj.type == 'cuda':
        # torch.cuda.get_device_name() 需要一个设备索引或对象
        logger.success(f"计算设备已设置为: {device_obj} ({torch.cuda.get_device_name(device_obj)})")
    else:
        logger.success("计算设备已设置为: CPU")

    return device_obj


def clear_memory() -> None:
    """
    尝试清理 GPU 缓存。
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("已清理 GPU 缓存 (torch.cuda.empty_cache())。")


# ========================================================================
# 4. 内存管理增强（源自 PyTorch Lightning）
# ========================================================================

def recursive_detach(in_dict: Any, to_cpu: bool = False) -> Any:
    """
    递归分离字典中的所有张量。

    可以递归处理嵌套字典中的 Tensor 实例。字典中的其他类型不受影响。

    参数:
        in_dict (Any): 包含张量的字典或任意数据结构
        to_cpu (bool): 是否将张量移动到 CPU（默认: False）

    返回:
        Any: 分离后的数据结构

    示例:
        >>> data = {
        ...     'loss': torch.tensor(1.5, requires_grad=True),
        ...     'metrics': {'acc': torch.tensor(0.95, requires_grad=True)}
        ... }
        >>> detached = recursive_detach(data, to_cpu=True)
        >>> print(detached['loss'].requires_grad)  # False
    """
    from torch import Tensor

    def _detach_and_move(t: Tensor, to_cpu: bool) -> Tensor:
        """内部函数：分离并可选地移动张量到 CPU"""
        t = t.detach()
        if to_cpu:
            t = t.cpu()
        return t

    # 递归处理字典
    if isinstance(in_dict, dict):
        return {k: recursive_detach(v, to_cpu) for k, v in in_dict.items()}

    # 递归处理列表
    if isinstance(in_dict, (list, tuple)):
        result = [recursive_detach(item, to_cpu) for item in in_dict]
        return type(in_dict)(result)  # 保持原始类型（list 或 tuple）

    # 处理张量
    if isinstance(in_dict, Tensor):
        return _detach_and_move(in_dict, to_cpu)

    # 其他类型直接返回
    return in_dict


def is_cuda_out_of_memory(exception: BaseException) -> bool:
    """
    检测是否为 CUDA 内存溢出（OOM）错误。

    参数:
        exception (BaseException): 捕获的异常

    返回:
        bool: True 表示是 CUDA OOM 错误

    示例:
        >>> try:
        ...     # 尝试分配过大的张量
        ...     x = torch.randn(10000, 10000, 10000, device='cuda')
        ... except RuntimeError as e:
        ...     if is_cuda_out_of_memory(e):
        ...         print("CUDA 内存不足！")
    """
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "CUDA" in exception.args[0]
        and "out of memory" in exception.args[0]
    )


def is_cudnn_snafu(exception: BaseException) -> bool:
    """
    检测是否为 cuDNN 特定错误。

    这是一个已知的 PyTorch 问题：https://github.com/pytorch/pytorch/issues/4107

    参数:
        exception (BaseException): 捕获的异常

    返回:
        bool: True 表示是 cuDNN 错误
    """
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )


def is_out_of_cpu_memory(exception: BaseException) -> bool:
    """
    检测是否为 CPU 内存溢出错误。

    参数:
        exception (BaseException): 捕获的异常

    返回:
        bool: True 表示是 CPU OOM 错误
    """
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
    )


def is_oom_error(exception: BaseException) -> bool:
    """
    检测是否为任意类型的内存溢出（OOM）错误。

    包括 CUDA OOM、cuDNN 错误和 CPU OOM。

    参数:
        exception (BaseException): 捕获的异常

    返回:
        bool: True 表示是 OOM 错误

    示例:
        >>> try:
        ...     # 训练代码
        ...     outputs = model(inputs)
        ... except RuntimeError as e:
        ...     if is_oom_error(e):
        ...         logger.error("检测到 OOM 错误！")
        ...         garbage_collection_cuda()
    """
    return is_cuda_out_of_memory(exception) or is_cudnn_snafu(exception) or is_out_of_cpu_memory(exception)


def garbage_collection_cuda() -> None:
    """
    安全地执行 CUDA 垃圾回收。

    这个函数会先进行 Python 垃圾回收，然后清空 CUDA 缓存。
    如果清空缓存时出现 OOM 错误，会捕获并忽略该错误。

    示例:
        >>> # 在捕获 OOM 错误后使用
        >>> try:
        ...     model(large_batch)
        ... except RuntimeError as e:
        ...     if is_oom_error(e):
        ...         garbage_collection_cuda()
        ...         # 尝试使用更小的 batch size
    """
    gc.collect()
    try:
        # 这是最后可能导致 OOM 错误的操作，但实际上也可能会发生
        torch.cuda.empty_cache()
        logger.debug("CUDA 垃圾回收完成")
    except RuntimeError as exception:
        if not is_oom_error(exception):
            # 只处理 OOM 错误，其他错误重新抛出
            raise
        logger.warning("在清空 CUDA 缓存时遇到 OOM 错误，已忽略")


def get_memory_usage(device: Optional[Union[int, torch.device]] = None) -> Optional[Dict[str, str]]:
    """
    获取指定 CUDA 设备的内存使用情况。

    参数:
        device (int | torch.device, optional): 目标设备。
            - None: 使用当前设备（默认）
            - int: 设备编号（如 0, 1）
            - torch.device: PyTorch 设备对象

    返回:
        dict 或 None: 包含内存使用信息的字典，如果不是 CUDA 设备则返回 None
    """
    if not torch.cuda.is_available():
        logger.warning("get_memory_usage() 仅适用于 CUDA。")
        return None

    # 确定设备 ID
    if device is None:
        device_id = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        device_id = device.index if device.index is not None else 0
    else:
        device_id = device

    allocated = torch.cuda.memory_allocated(device_id)
    reserved = torch.cuda.memory_reserved(device_id)  # PyTorch 缓存
    total = torch.cuda.get_device_properties(device_id).total_memory

    return {
        "allocated": format_size(allocated),
        "reserved": format_size(reserved),
        "total": format_size(total),
        "percent_used": f"{(allocated / total * 100):.2f}%"
    }


def log_memory_usage(description: str = "当前内存使用"):
    """
    记录内存使用情况到日志。
    """
    usage = get_memory_usage()
    if usage:
        logger.debug(f"{description}: {usage['allocated']} / {usage['total']} ({usage['percent_used']})")


# --- 4. 张量与模型辅助函数 ---

def validate_tensor(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """
    验证张量是否有效（无 NaN 或 Inf）。

    参数:
        tensor (Tensor): 要验证的张量
        name (str): 张量名称 (用于日志)

    返回:
        bool: 张量是否有效 (True = 有效)
    """
    if torch.isnan(tensor).any():
        logger.error(f"张量 {name} 包含 NaN 值！")
        return False

    if torch.isinf(tensor).any():
        logger.error(f"张量 {name} 包含 Inf 值！")
        return False

    return True


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """
    计算模型参数数量。

    参数:
        model (nn.Module): 模型
        trainable_only (bool): 是否只计算可训练参数

    返回:
        int: 参数数量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


# --- 5. 格式化与 IO ---

def format_size(size_bytes: int) -> str:
    """格式化字节大小为可读格式 (KB, MB, GB)"""
    if size_bytes == 0:
        return "0B"
    size_names = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    # 使用 math.log 替代 np.log
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def save_dict_to_json(data: Dict[str, Any], file_path: str) -> None:
    """保存字典到 JSON 文件。"""
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.info(f"数据已保存到 JSON 文件: {file_path}")
    except Exception as e:
        logger.error(f"保存 JSON 文件失败: {file_path}。错误: {e}")
        raise  # 重新抛出异常，让调用方知道失败了


def load_dict_from_json(file_path: str) -> Optional[Dict[str, Any]]:
    """从 JSON 文件加载字典。"""
    if not os.path.exists(file_path):
        logger.error(f"JSON 文件未找到: {file_path}")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"已从 JSON 文件加载数据: {file_path}")
        return data
    except Exception as e:
        logger.error(f"加载 JSON 文件失败: {file_path}。错误: {e}")
        return None
