#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/1 15:37
@author  : William_Trouvaille
@function: 通用配置管理模块
@detail: 提供从 YAML 加载配置、深度合并、命令行参数覆盖以及转换为属性访问对象 (Namespace) 的功能。
"""
import os
from collections.abc import Mapping

import yaml
from loguru import logger


class ConfigNamespace:
    """
    配置命名空间类，将字典转换为对象，以便通过属性访问配置项。

    支持嵌套字典的递归转换和属性更新。
    """

    def __init__(self, config_dict: dict):
        """
        初始化配置命名空间。

        参数:
            config_dict (dict): 配置字典
        """
        if not isinstance(config_dict, dict):
            raise ValueError("ConfigNamespace 必须使用字典进行初始化")

        logger.debug("正在创建 ConfigNamespace...")
        for key, value in config_dict.items():
            if isinstance(value, dict):
                # 如果值是字典，递归创建 ConfigNamespace 对象
                setattr(self, key, ConfigNamespace(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        """
        定义对象的打印形式，方便日志记录。
        """
        return str(vars(self))

    def to_dict(self) -> dict:
        """
        将 ConfigNamespace 对象递归转换回字典。
        """
        result = {}
        for key, value in vars(self).items():
            if isinstance(value, ConfigNamespace):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def get(self, key: str, default=None):
        """
        安全地获取配置项，如果不存在则返回默认值。
        """
        return getattr(self, key, default)

    def update(self, new_config: dict):
        """
        使用字典递归更新 ConfigNamespace。
        """
        logger.debug(f"正在更新 ConfigNamespace...")
        for key, value in new_config.items():
            if isinstance(value, dict) and hasattr(self, key) and isinstance(getattr(self, key), ConfigNamespace):
                # 如果现有属性是 ConfigNamespace 对象，递归更新
                getattr(self, key).update(value)
            elif isinstance(value, dict):
                # 如果是新字典或现有属性不是 Namespace，创建新的
                setattr(self, key, ConfigNamespace(value))
            else:
                # 覆盖或设置新值
                setattr(self, key, value)
        logger.debug(f"ConfigNamespace 更新完毕。")


def _deep_merge_dict(base_dict: dict, override_dict: dict) -> dict:
    """
    递归（深度）合并两个字典。
    `override_dict` 中的值将覆盖 `base_dict` 中的值。
    """
    merged = base_dict.copy()
    for key, value in override_dict.items():
        if (key in merged and isinstance(merged[key], Mapping)
                and isinstance(value, Mapping)):
            # 如果键存在且值都是字典，则递归合并
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            # 否则，直接覆盖（或添加）
            merged[key] = value
    return merged


def load_config_from_yaml(config_path: str) -> dict:
    """
    加载 YAML 配置文件并返回一个字典。

    参数:
        config_path (str): YAML 配置文件的路径。

    返回:
        dict: 包含配置参数的字典。如果文件不存在或解析失败，则返回空字典。
    """
    resolved_path = os.path.abspath(config_path)

    if not os.path.exists(resolved_path):
        logger.warning(f"配置文件未找到: {resolved_path}。跳过加载。")
        return {}

    logger.debug(f"尝试从 '{resolved_path}' 加载配置...")
    try:
        with open(resolved_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)  # 使用 safe_load 防止执行任意代码

        if config is None:
            logger.warning(f"配置文件为空: {resolved_path}")
            return {}

        logger.success(f"成功加载配置文件: {resolved_path}")
        logger.debug(f"加载的配置内容: {config}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"解析 YAML 文件时出错: {resolved_path}\n错误详情: {e}")
        return {}
    except Exception as e:
        logger.error(f"加载配置文件时发生未知错误: {resolved_path}\n错误详情: {e}")
        return {}


def update_config_from_args(config_dict: dict, args_dict: dict) -> dict:
    """
    从命令行参数字典更新配置字典。
    支持点分键 (dot-separated keys) 来访问嵌套字典。

    参数:
        config_dict (dict): 原始配置字典
        args_dict (dict): 命令行参数字典 (例如来自 argparse)

    返回:
        dict: 更新后的配置字典
    """
    updated_config = config_dict.copy()

    # 过滤掉值为 None 的参数，以防止它们覆盖有效的默认值
    valid_args = {k: v for k, v in args_dict.items() if v is not None}
    if not valid_args:
        return updated_config

    logger.debug(f"正在从命令行参数更新配置: {list(valid_args.keys())}")

    for key, value in valid_args.items():
        # 处理嵌套键，如 "dataset.name"
        if '.' in key:
            keys = key.split('.')
            current = updated_config
            try:
                # 遍历到倒数第二个键
                for k in keys[:-1]:
                    if k not in current or not isinstance(current[k], dict):
                        current[k] = {}
                    current = current[k]
                # 设置最后一个键的值
                current[keys[-1]] = value
            except Exception as e:
                logger.error(f"更新嵌套参数 '{key}' 时出错: {e}")
        else:
            # 非嵌套键，直接更新
            updated_config[key] = value

    logger.info(f"配置已从 {len(valid_args)} 个命令行参数更新。")
    return updated_config


def save_config_to_yaml(config: (dict | ConfigNamespace), config_path: str):
    """
    保存配置到 YAML 文件。

    参数:
        config (dict or ConfigNamespace): 要保存的配置
        config_path (str): 保存路径
    """
    if isinstance(config, ConfigNamespace):
        config_dict = config.to_dict()
    else:
        config_dict = config

    resolved_path = os.path.abspath(config_path)

    # 确保保存目录存在
    try:
        os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
    except OSError as e:
        logger.error(f"无法创建用于保存配置的目录: {os.path.dirname(resolved_path)}。错误: {e}")
        return

    logger.debug(f"正在保存配置至: {resolved_path}")
    try:
        with open(resolved_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                config_dict,
                f,
                default_flow_style=False,
                allow_unicode=True,
                indent=2,
                sort_keys=False  # 保持原始顺序
            )
        logger.success(f"配置文件已保存至: {resolved_path}")
    except Exception as e:
        logger.error(f"保存配置文件失败: {resolved_path}, 错误: {e}")


def print_config(config: (dict | ConfigNamespace), title: str = "当前配置信息"):
    """
    以格式化的方式打印配置信息到日志 (INFO 级别)。

    参数:
        config (dict or ConfigNamespace): 配置
        title (str): 打印的标题
    """
    logger.info("=" * 60)
    logger.info(f"{title}".center(60))
    logger.info("=" * 60)

    def print_recursive(d: dict, indent: int = 0):
        for key, value in d.items():
            if key.startswith('_'):  # 跳过内部或注释字段
                continue

            prefix = "  " * indent + f"{key}:"
            if isinstance(value, dict):
                logger.info(prefix)
                print_recursive(value, indent + 1)
            else:
                logger.info(f"{prefix} {value}")

    if isinstance(config, ConfigNamespace):
        config_dict = config.to_dict()
    else:
        config_dict = config

    print_recursive(config_dict)
    logger.info("=" * 60)


def setup_config(
        default_config: dict,
        yaml_config_path: str,
        cmd_args: dict
) -> ConfigNamespace:
    """
    优雅的配置编排函数，处理三阶段覆盖。

    覆盖优先级: 命令行参数 > YAML 文件 > 默认配置

    参数:
        default_config (dict): 在代码中定义的默认配置字典。
        yaml_config_path (str): 用户 YAML 配置文件的路径。
        cmd_args (dict): 从 argparse.parse_args() 得到的参数字典。

    返回:
        ConfigNamespace: 包含最终合并配置的命名空间对象。
    """
    logger.info("开始配置加载程序...")

    # 1. 加载 YAML 配置
    yaml_config = load_config_from_yaml(yaml_config_path)

    # 2. 合并：默认 < YAML
    config_step_1 = _deep_merge_dict(default_config, yaml_config)

    # 3. 合并：(默认 + YAML) < 命令行参数
    final_config_dict = update_config_from_args(config_step_1, cmd_args)

    # 4. 打印最终配置
    print_config(final_config_dict, "最终合并配置")

    # 5. 转换为 Namespace
    final_config_namespace = ConfigNamespace(final_config_dict)

    logger.success("配置加载完成并转换为 ConfigNamespace。")
    return final_config_namespace
