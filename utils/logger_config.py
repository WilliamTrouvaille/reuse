#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/1 15:12
@version : 1.0.0
@author  : William_Trouvaille
@function: 日志配置模块
"""

import os
import sys
from loguru import logger


def setup_logging(
        log_dir: str = "logs",
        console_level: str = "INFO",
        file_level: str = "DEBUG"
):
    """
    配置 Loguru 日志记录器，设置控制台和文件输出。

    此函数是幂等的：它会首先移除所有现有的日志处理器，
    然后再添加新的处理器，确保不会重复配置。

    参数:
        log_dir (str): 保存日志文件的目录。默认为 "logs"。
        console_level (str): 控制台输出的最低日志级别。
                             默认为 "INFO"。
        file_level (str): 文件输出的最低日志级别。
                          默认为 "DEBUG"，以便在文件中保留更详细的日志。
    """

    # 1. (重要) 移除所有已存在的 handlers，确保幂等性
    # 这样可以避免因重复调用此函数而导致的日志重复输出
    # 同时也移除了对全局变量 `_logger_configured` 的依赖
    try:
        logger.remove()
    except ValueError:
        # 如果没有 handler，logger.remove() 会抛出 ValueError，直接忽略
        pass

    # 2. 定义统一的日志格式
    # 这种格式包含了时间、级别、模块、函数和行号，易于调试
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # 3. 添加控制台 (stderr) Sink
    logger.add(
        sys.stderr,  # 输出到标准错误流
        level=console_level.upper(),  # 使用传入的控制台日志级别
        format=log_format,  # 使用定义的格式
        colorize=True,  # 在控制台中启用彩色输出
        enqueue=True  # 异步处理，防止日志I/O阻塞主线程
    )

    # 4. 确保日志目录存在
    try:
        os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        # 如果创建目录失败（例如权限问题），记录一个错误并继续
        # 此时日志只会输出到控制台，控制台 sink 已经被添加
        logger.error(f"无法创建日志目录: {log_dir}。错误: {e}")
        return

    # 5. 定义日志文件路径
    # 使用 {time:YYYYMMDD} 确保每天一个日志文件
    log_file_path = os.path.join(log_dir, "log_{time:YYYYMMDD}.log")

    # 6. 添加文件 Sink
    logger.add(
        log_file_path,  # 文件路径模式
        level=file_level.upper(),  # 使用传入的文件日志级别
        format=log_format,  # 使用相同的定义格式
        rotation="10 MB",  # 当日志文件达到 10 MB 时进行轮转（创建新文件）
        retention="10 days",  # 最多保留最近 10 天的日志文件
        encoding="utf-8",  # 指定文件编码
        enqueue=True,  # 启用异步日志记录
        backtrace=True,  # (推荐) 显示完整的堆栈跟踪
        diagnose=True  # (推荐) 提供更详细的错误诊断信息
    )

    # 7. 配置完成日志
    logger.info("Loguru 日志记录器配置完成。")
    logger.debug(f"控制台日志级别: {console_level.upper()}")
    logger.debug(f"文件日志级别: {file_level.upper()}")
    logger.debug(f"日志文件目录: {os.path.abspath(log_dir)}")