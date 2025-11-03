#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/1 21:20
@author  : William_Trouvaille
@function: 数据处理模块
@detail:
    - 包含一个数据注册表 (_DATASET_REGISTRY) 来存储元数据
    - 提供 `load_dataset_info` 函数以便加载数据集和元数据。
"""

from typing import Dict, Any, List

from loguru import logger
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# 使用注册表模式来分离“配置元数据”和“加载逻辑”
_DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    'MNIST': {
        'torchvision_class': datasets.MNIST,
        'im_size': (28, 28),
        'channel': 1,
        'num_classes': 10,
        'mean': [0.1307],
        'std': [0.3081],
    },
    'FashionMNIST': {
        'torchvision_class': datasets.FashionMNIST,
        'im_size': (28, 28),
        'channel': 1,
        'num_classes': 10,
        'mean': [0.2861],
        'std': [0.3530],
    },
    'CIFAR10': {
        'torchvision_class': datasets.CIFAR10,
        'im_size': (32, 32),
        'channel': 3,
        'num_classes': 10,
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2023, 0.1994, 0.2010],
    },
    'CIFAR100': {
        'torchvision_class': datasets.CIFAR100,
        'im_size': (32, 32),
        'channel': 3,
        'num_classes': 100,
        'mean': [0.5071, 0.4866, 0.4409],
        'std': [0.2673, 0.2564, 0.2762],
    },
}


def load_dataset_info(dataset_name: str, data_path: str) -> Dict[str, Any]:
    """
    加载指定的数据集并返回包含数据集对象和元数据的字典。

    参数:
        dataset_name (str): 数据集名称 (必须在 _DATASET_REGISTRY 中)
        data_path (str): 数据集存储路径

    返回:
        dict: 包含以下键
            - 'dst_train' (Dataset): 训练集
            - 'dst_test' (Dataset): 测试集
            - 'im_size' (tuple): 图像尺寸 (H, W)
            - 'channel' (int): 通道数
            - 'num_classes' (int): 类别数
            - 'class_names' (list): 类别名称
            - 'mean' (list): 均值
            - 'std' (list): 标准差

    引发:
        ValueError: 如果 `dataset_name` 不在注册表中。
    """
    logger.info(f"开始加载数据集: {dataset_name} from {data_path}")

    # 1. 查找元数据
    if dataset_name not in _DATASET_REGISTRY:
        logger.error(f"未知的数据集: {dataset_name}")
        raise ValueError(f"未知的数据集: {dataset_name}。支持的数据集: {list(_DATASET_REGISTRY.keys())}")

    meta = _DATASET_REGISTRY[dataset_name]

    # 2. 定义默认变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=meta['mean'], std=meta['std'])
    ])

    dst_train: Dataset
    dst_test: Dataset
    class_names: List[str]

    # 3. 加载数据集 (已简化)
    # (已移除 SVHN 和 TinyImageNet 的特殊处理)
    # 现在所有数据集都使用标准 torchvision 接口
    try:
        logger.debug(f"使用 torchvision.datasets.{meta['torchvision_class'].__name__} 加载...")
        tv_class = meta['torchvision_class']

        # 使用 train=True 加载训练集
        dst_train = tv_class(data_path, train=True, download=True, transform=transform)

        # 使用 train=False 加载测试集
        dst_test = tv_class(data_path, train=False, download=True, transform=transform)

        # 尝试获取 class_names
        if hasattr(dst_train, 'classes'):
            class_names = dst_train.classes
        else:
            class_names = [str(c) for c in range(meta['num_classes'])]

    except Exception as e:
        logger.error(f"加载数据集 {dataset_name} 时失败。请检查 data_path 或网络连接。")
        logger.error(f"错误详情: {e}")
        raise

    # 4. 返回一个干净的字典
    logger.success(f"成功加载数据集: {dataset_name}")
    return {
        'dst_train': dst_train,
        'dst_test': dst_test,
        'im_size': meta['im_size'],
        'channel': meta['channel'],
        'num_classes': meta['num_classes'],
        'class_names': class_names,
        'mean': meta['mean'],
        'std': meta['std']
    }
