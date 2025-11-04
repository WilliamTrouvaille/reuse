#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/1 21:45
@author  : William_Trouvaille
@function: 通用辅助函数
"""

import json
import math
import os
import random
import time
from typing import Dict, Any, Optional, Union

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


def set_random_seed(seed: int = 42, enable_cudnn_benchmark: bool = True) -> None:
    """
    设置所有相关库的随机种子以确保实验的可复现性。

    参数:
        seed (int): 随机种子值
        enable_cudnn_benchmark (bool): 是否启用 cuDNN 自动调优。
            - False (默认): 禁用 benchmark，保证完全可复现
            - True: 启用 benchmark，在固定输入尺寸时可提升 20-30% 性能，但会牺牲可复现性
    """
    logger.info(f"正在设置全局随机种子: {seed}")

    # 设置 Python 内置随机数生成器种子
    random.seed(seed)

    # 设置 NumPy 随机数生成器种子
    np.random.seed(seed)

    # 设置 PyTorch 随机数生成器种子 (CPU)
    torch.manual_seed(seed)

    # 如果使用 CUDA
    if torch.cuda.is_available():
        logger.debug("设置 CUDA 随机种子...")
        # 设置 PyTorch CUDA 随机数生成器种子 (当前 GPU)
        torch.cuda.manual_seed(seed)
        # 设置 PyTorch CUDA 随机数生成器种子 (所有 GPU)
        torch.cuda.manual_seed_all(seed)

        # (重要) 确保 cuDNN 使用确定性算法
        torch.backends.cudnn.deterministic = True

        # cuDNN benchmark 配置（可选性能优化）
        torch.backends.cudnn.benchmark = enable_cudnn_benchmark

        if enable_cudnn_benchmark:
            logger.warning(
                "cuDNN benchmark 已启用，将牺牲可复现性以换取性能提升。"
                "如需完全可复现，请在配置中设置 enable_cudnn_benchmark: false"
            )
        else:
            logger.debug("cuDNN benchmark 已禁用（保证可复现性）。")

    logger.success(f"全局随机种子 {seed} 设置完毕。")


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
    (保留) 记录内存使用情况到日志。
    """
    usage = get_memory_usage()
    if usage:
        logger.debug(f"{description}: {usage['allocated']} / {usage['total']} ({usage['percent_used']})")


# --- 4. (保留) 张量与模型辅助函数 ---

def validate_tensor(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """
    (保留) 验证张量是否有效（无 NaN 或 Inf）。

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
    (保留) 计算模型参数数量。

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


# --- 5. (保留) 格式化与 IO ---

def format_size(size_bytes: int) -> str:
    """
    (保留并重构) 格式化字节大小为可读格式 (KB, MB, GB)。
    (移除了对 NumPy 的依赖)
    """
    if size_bytes == 0:
        return "0B"
    size_names = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    # 使用 math.log 替代 np.log
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def save_dict_to_json(data: Dict[str, Any], file_path: str) -> None:
    """
    (保留并重构) 保存字典到 JSON 文件。
    (移除了对 `ensure_dir` 的依赖)
    """
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
    """
    (保留) 从 JSON 文件加载字典。
    """
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
