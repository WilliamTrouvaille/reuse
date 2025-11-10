#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/1 15:16
@version : 1.0.0
@author  : William_Trouvaille
@function: 工具包初始化模块
"""

from .checkpoint_manager import CheckpointManager
from .config import (
    setup_config,
    load_config_from_yaml,
    save_config_to_yaml,
    print_config,
    ConfigNamespace
)
from .data import load_dataset_info
from .decorators import (
    time_it,
    no_grad,
    train_mode,
    eval_mode,
    log_errors
)
from .early_stopping import EarlyStopper
from .grads import (
    grad_norm,
    log_grad_norm,
    check_grad_anomaly
)
from .helpers import (
    get_time,
    format_time,
    isolate_rng,
    get_device,
    clear_memory,
    recursive_detach,
    is_cuda_out_of_memory,
    is_cudnn_snafu,
    is_out_of_cpu_memory,
    is_oom_error,
    garbage_collection_cuda,
    get_memory_usage,
    log_memory_usage,
    validate_tensor,
    count_parameters,
    format_size,
    save_dict_to_json,
    load_dict_from_json
)
from .logger_config import setup_logging
from .metrics import MetricTracker, AverageMeter
from .ntfy_notifier import NtfyNotifier
from .progress import Progress
from .train import Trainer
from .visualization import (
    MetricsCollector,
    MetricsVisualizer,
    TrainingReporter
)
from .callbacks import (
    Timer,
    Interval,
    Stage,
    LearningRateMonitor,
    BatchSizeFinder,
    find_optimal_batch_size
)
from .optimizers import AdamP, Lion, MadGrad
from .augmentation import (
    AugmentationPipeline,
    BaseTransform,
    CutOut,
    CenterCrop,
    ColorJitter,
    GaussianBlur,
    GaussianNoise,
    RandomApply,
    RandomBlur,
    RandomBrightness,
    RandomColorDistortion,
    RandomContrast,
    RandomCropAndResize,
    RandomCropResizeFlip,
    RandomErasing,
    RandomHorizontalFlip,
    RandomHue,
    RandomRotation,
    RandomSaturation,
    SobelFilter,
    ToGrayscale,
    TorchSeedContext,
    TRANSFORM_REGISTRY,
    get_basic_augmentation,
    get_simclr_augmentation,
    get_strong_augmentation,
    validate_probability,
    validate_range,
)

# 版本信息
__version__ = "0.1.0"
__author__ = "William_Trouvaille"

# 导出主要接口
__all__: list[str] = [
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

    # data.py
    'load_dataset_info',

    # progress.py
    'Progress',

    # grads.py
    'grad_norm',
    'log_grad_norm',
    'check_grad_anomaly',

    # helpers.py
    'get_time',
    'format_time',
    'isolate_rng',
    'get_device',
    'clear_memory',
    'recursive_detach',
    'is_cuda_out_of_memory',
    'is_cudnn_snafu',
    'is_out_of_cpu_memory',
    'is_oom_error',
    'garbage_collection_cuda',
    'get_memory_usage',
    'log_memory_usage',
    'validate_tensor',
    'count_parameters',
    'format_size',
    'save_dict_to_json',
    'load_dict_from_json',

    # decorators.py
    'time_it',
    'no_grad',
    'train_mode',
    'eval_mode',
    'log_errors',

    # metrics.py
    'MetricTracker',
    'AverageMeter',

    # early_stopping.py
    'EarlyStopper',

    # train.py
    'Trainer',

    # visualization/
    'MetricsCollector',
    'MetricsVisualizer',
    'TrainingReporter',

    # callbacks/
    'Timer',
    'Interval',
    'Stage',
    'LearningRateMonitor',
    'BatchSizeFinder',
    'find_optimal_batch_size',

    # optimizers/
    'AdamP',
    'Lion',
    'MadGrad',

    # augmentation/
    'AugmentationPipeline',
    'BaseTransform',
    'CutOut',
    'CenterCrop',
    'ColorJitter',
    'GaussianBlur',
    'GaussianNoise',
    'RandomApply',
    'RandomBlur',
    'RandomBrightness',
    'RandomColorDistortion',
    'RandomContrast',
    'RandomCropAndResize',
    'RandomCropResizeFlip',
    'RandomErasing',
    'RandomHorizontalFlip',
    'RandomHue',
    'RandomRotation',
    'RandomSaturation',
    'SobelFilter',
    'ToGrayscale',
    'TorchSeedContext',
    'TRANSFORM_REGISTRY',
    'get_basic_augmentation',
    'get_simclr_augmentation',
    'get_strong_augmentation',
    'validate_probability',
    'validate_range',
    
]
