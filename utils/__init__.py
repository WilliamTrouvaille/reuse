#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/1 15:16
@version : 1.0.0
@author  : William_Trouvaille
@function: 工具包初始化模块
"""

from .logger_config import setup_logging
from .config import (
    setup_config,
    load_config_from_yaml,
    save_config_to_yaml,
    print_config,
    ConfigNamespace
)
from .ntfy_notifier import NtfyNotifier
from .checkpoint_manager import CheckpointManager

# 版本信息
__version__ = "0.0.1"
__author__ = "William_Trouvaille"

# 导出主要接口
__all__ = [
    # logger_config.py
    'setup_logging',

    # config.py
    'setup_config',
    'load_config_from_yaml',
    'save_config_to_yaml',
    'print_config',
    'ConfigNamespace',

    # ntfy_notifier.py
    'NtfyNotifier',

    # checkpoint_manager.py
    'CheckpointManager',
]