#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/1 15:12
@version : 1.0.0
@author  : William_Trouvaille
@function: 主文件，用于测试
"""

# 运行代码请放到根目录运行

import argparse
from loguru import logger
from utils import \
    setup_logging, \
    setup_config

def get_project_defaults() -> dict:
    """定义本项目（Dataset Condensation）的默认参数"""
    return {
        'experiment': {
            'name': 'dataset_condensation',
            'seed': 42,
        },
        'dataset': {
            'name': 'CIFAR10',
            'data_path': './data',
            'ipc': 1,
            'num_workers': 4
        },
        'model': {
            'name': 'ConvNet',
        },
        'training': {
            'epochs': 1000,
            'lr_img': 1.0,
            'lr_net': 0.01,
        },
        'logging': {
            'log_dir': './logs',
            'console_level': 'INFO',
            'file_level': 'DEBUG'
        }
    }

def parse_arguments() -> dict:
    """定义和解析命令行参数"""
    parser = argparse.ArgumentParser(description="数据集压缩实验")

    # 定义参数，注意 dest 的命名应与配置字典匹配
    # 使用点分key (dot-notation) 来覆盖嵌套设置
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='config.yaml',
        help='配置文件的路径'
    )
    parser.add_argument(
        '--dataset.name',
        type=str,
        help='覆盖数据集名称 (例如: MNIST)'
    )
    parser.add_argument(
        '--dataset.ipc',
        type=int,
        help='覆盖每类图像数 (IPC)'
    )
    parser.add_argument(
        '--training.epochs',
        type=int,
        help='覆盖训练轮数'
    )

    args = parser.parse_args()

    # 返回字典形式的参数
    return vars(args)


def main():
    # 1. 解析命令行参数
    # cmd_args = {'config': 'config.yaml', 'dataset.name': 'MNIST', 'dataset.ipc': 10, ...}
    cmd_args = parse_arguments()

    # 2. 获取项目默认配置
    default_config = get_project_defaults()

    # 3. (注意) 在 setup_config 之前设置日志
    # 我们使用默认配置中的日志设置来初始化
    # setup_config 稍后可能会加载 YAML/Args 中新的日志级别，
    # 但初始日志记录需要现在开始。
    setup_logging(
        log_dir=default_config['logging']['log_dir'],
        console_level=default_config['logging']['console_level'],
        file_level=default_config['logging']['file_level']
    )

    # 4. 🔥 核心：加载和合并配置
    #    这将按 (Default -> YAML -> CMD) 的顺序自动合并
    config = setup_config(
        default_config=default_config,
        yaml_config_path=cmd_args['config'], # 告知 YAML 路径
        cmd_args=cmd_args                 # 传入所有命令行参数
    )

    # 5. 开始使用配置 (通过属性访问)
    logger.info(f"实验开始: {config.experiment.name}")
    logger.info(f"数据集: {config.dataset.name} (IPC={config.dataset.ipc})")
    logger.info(f"模型: {config.model.name}")
    logger.info(f"图像学习率: {config.training.lr_img}")

    # 6. (可选) 验证项目特定配置
    #    这部分逻辑也从 utils 中移除了
    if config.dataset.ipc < 1:
        logger.error("IPC 必须大于 0。")
        # raise ValueError("IPC 必须大于 0")

    # ... 您的训练代码 ...
    logger.success("实验完成。")

if __name__ == "__main__":
    main()
