# Utils 工具包使用指南

本文档是 `utils` 工具包的总览索引，提供所有工具模块的快速导航。

## 目录结构

```
utils/
├── config.py              # 配置管理
├── logger_config.py       # 日志配置
├── data.py                # 数据集加载
├── helpers.py             # 辅助工具函数
├── metrics.py             # 指标跟踪
├── progress.py            # 高性能进度条
├── decorators.py          # 装饰器
├── checkpoint_manager.py  # 检查点管理
├── early_stopping.py      # 早停
├── ntfy_notifier.py       # Ntfy 通知
└── train.py               # 训练器

readme/                    # 工具文档（仅说明）
├── config-guide.md
├── logger_config-guide.md
├── data-guide.md
├── helpers-guide.md
├── metrics-guide.md
├── progress-guide.md
├── decorators-guide.md
├── checkpoint_manager-guide.md
├── early_stopping-guide.md
├── ntfy_notifier-guide.md
└── train-guide.md

example/                   # 使用示例（含测试代码）
├── config-example.py
├── logger_config-example.py
├── progress-example.py
└── train-example.py
```

## 核心工具模块

### 配置管理

**模块**: `utils.config`
**文档**: [config-guide.md](./config-guide.md)
**示例**: [example/config-example.py](../example/config-example.py)

提供配置文件管理功能，支持：
- YAML 配置文件加载和保存
- 命令行参数覆盖（优先级：命令行 > YAML > 默认配置）
- `ConfigNamespace` 类支持属性访问（`config.dataset.name`）
- 配置打印和验证

**核心函数**:
- `setup_config(default_config, yaml_config_path, cmd_args)` - 推荐使用
- `ConfigNamespace` - 字典到属性访问的转换

### 日志配置

**模块**: `utils.logger_config`
**文档**: [logger_config-guide.md](./logger_config-guide.md)
**示例**: [example/logger_config-example.py](../example/logger_config-example.py)

基于 Loguru 的日志配置工具，支持：
- 控制台和文件输出的独立级别控制
- 自动日志轮转（10 MB）和清理（10 天）
- 异步日志记录（不阻塞主线程）
- 完整堆栈跟踪和错误诊断

**核心函数**:
- `setup_logging(log_dir="logs", console_level="INFO", file_level="DEBUG")`

### 训练器

**模块**: `utils.train`
**文档**: [train-guide.md](./train-guide.md)
**示例**: [example/train-example.py](../example/train-example.py)

可复用的 PyTorch 训练协调器，支持：
- 标准训练流程编排
- 自动混合精度（AMP）
- 梯度累积和梯度裁剪
- 检查点管理和自动恢复
- 早停和通知集成
- 可定制的训练步骤（支持多任务学习、对比学习等）

**核心类**:
- `Trainer` - 训练器主类
- `Trainer.from_config(...)` - 配置驱动模式

### 进度条

**模块**: `utils.progress`
**文档**: [progress-guide.md](./progress-guide.md)
**示例**: [example/progress-example.py](../example/progress-example.py)

高性能进度条工具，专为 PyTorch 训练优化：
- 时间节流减少 GPU 同步（性能提升 20-30%）
- GPU Tensor 累积（显示运行平均值）
- 可配置更新间隔
- 与 Trainer 集成

**核心类**:
- `Progress` - 进度条类

### 指标跟踪

**模块**: `utils.metrics`
**文档**: [metrics-guide.md](./metrics-guide.md)

两种指标跟踪器：
- `MetricTracker` - GPU 优化，用于训练循环（性能提升 1.3-1.5x）
- `AverageMeter` - CPU 标量，用于非关键路径

**设计理念**: 延迟同步，在循环中累积 GPU Tensor，仅在 epoch 结束时同步一次。

### 数据集加载

**模块**: `utils.data`
**文档**: [data-guide.md](./data-guide.md)

基于注册表模式的数据集加载器：
- 统一的数据集加载接口
- 返回数据集对象和元数据（类别数、图像尺寸、均值/标准差等）
- 支持 CIFAR-10/100、MNIST、FashionMNIST、SVHN、ImageNet 子集等

**核心函数**:
- `load_dataset_info(dataset_name, data_path)`

### 辅助工具

**模块**: `utils.helpers`
**文档**: [helpers-guide.md](./helpers-guide.md)

11 个常用辅助函数，包括：
- 时间工具：`get_current_time()`, `format_time()`
- 随机种子：`set_seed()`
- 设备管理：`get_device()`, `get_device_info()`
- Tensor/模型工具：`move_to_device()`, `count_parameters()`, `get_model_size()`
- 格式化/IO：`format_number()`, `save_json()`, `load_json()`

### 装饰器

**模块**: `utils.decorators`
**文档**: [decorators-guide.md](./decorators-guide.md)

5 个可复用装饰器：
- `@time_it` - 自动计时
- `@no_grad` - PyTorch 无梯度上下文
- `@train_mode` / `@eval_mode` - 自动管理模型训练/评估模式
- `@log_errors` - 异常日志和通知

**设计理念**: 分离业务逻辑和通用功能。

### 检查点管理

**模块**: `utils.checkpoint_manager`
**文档**: [checkpoint_manager-guide.md](./checkpoint_manager-guide.md)

面向对象的检查点管理器：
- 自动保存和加载检查点
- 滚动清理旧检查点（保留最近 N 个）
- 最佳模型保存
- 中断检查点（Ctrl+C 时保存）
- 原子写入（避免损坏）

**核心类**:
- `CheckpointManager`

**设计理念**: 职责分离 - 工具类只负责 I/O，不关心状态内容。

### 早停

**模块**: `utils.early_stopping`
**文档**: [early_stopping-guide.md](./early_stopping-guide.md)

早停逻辑封装：
- 监控验证指标
- 自动判断是否为最佳模型
- Patience 机制
- 支持 min/max 模式
- 状态持久化（可保存到检查点）

**核心类**:
- `EarlyStopper`

### Ntfy 通知

**模块**: `utils.ntfy_notifier`
**文档**: [ntfy_notifier-guide.md](./ntfy_notifier-guide.md)

实时训练状态通知：
- 发送通知到 Ntfy 手机 App
- 不同优先级（开始/成功/失败）
- Markdown 支持
- 重试机制
- 可配置的 Topic 和服务器

**核心类**:
- `NtfyNotifier`

## 快速开始

### 1. 最小化训练脚本

```python
from utils import setup_logging, Trainer
from loguru import logger
import torch
import torch.nn as nn

# 初始化日志
setup_logging(log_dir='./logs')

# 准备模型、优化器、损失函数
model = YourModel().to('cuda')
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# 创建训练器
trainer = Trainer(model, optimizer, criterion, device='cuda')

# 开始训练
trainer.fit(train_loader, val_loader, epochs=100)
```

### 2. 完整功能训练脚本

```python
from utils import (
    setup_config, setup_logging, Trainer,
    log_errors, NtfyNotifier
)

# 配置管理
config = setup_config(DEFAULT_CONFIG, 'config.yaml', vars(args))

# 日志配置
setup_logging(
    log_dir=config.logging.log_dir,
    console_level=config.logging.console_level,
    file_level=config.logging.file_level
)

# 通知器
notifier = NtfyNotifier()

# 训练器（配置驱动模式）
trainer = Trainer.from_config(
    model, optimizer, criterion, device,
    config=config, scheduler=scheduler
)

# 异常处理装饰器
@log_errors(notifier=notifier, re_raise=True)
def main():
    trainer.fit(train_loader, val_loader, epochs=config.training.epochs)

if __name__ == '__main__':
    main()
```

## 设计原则

本工具包的设计遵循以下原则：

### 1. 职责分离
- 工具类只负责"如何做"（How），不关心"做什么"（What）
- 例如：`CheckpointManager` 只负责保存/加载字典，不关心字典内容

### 2. 依赖注入
- 通过构造函数注入已实例化的工具
- 最大化灵活性和可测试性
- 配置驱动模式作为可选的简化方式

### 3. 高性能优先
- 默认集成高性能工具（MetricTracker, Progress）
- 支持 AMP、梯度累积、梯度裁剪
- 避免不必要的 GPU 同步和 I/O

### 4. 可复用性
- 工具类设计保证在不同项目中可直接复用
- 通过继承或重写方法支持定制化
- 提供清晰的扩展点（如 `Trainer._train_step()`）

### 5. 健壮性
- 自动检查点保存和恢复
- 中断处理（Ctrl+C）
- 异常处理和通知
- 完整的日志记录

## 相关链接

- **详细文档**: 参见 `readme/` 文件夹下的各个 `*-guide.md` 文件
- **使用示例**: 参见 `example/` 文件夹下的各个 `*-example.py` 文件
- **项目规范**: 参见项目根目录的 `CLAUDE.md`

## 版本说明

本工具包基于以下技术栈：
- Python 3.13
- PyTorch 2.x (CUDA 12+)
- Loguru (日志)
- PyYAML (配置)
- tqdm (进度条基础)

## 许可证

本项目为研究和学习用途。
