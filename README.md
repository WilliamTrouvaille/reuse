# PyTorch 通用工具箱

（自用）高可复用性、高解耦、高可读性的 PyTorch 深度学习工具类集合。

## 项目特点

- **高可复用性**: 所有工具类设计为独立模块，可按需引入
- **高解耦**: 职责单一，模块间依赖最小化
- **高可读性**: 代码风格统一，注释详尽，符合 PEP8 规范
- **性能优先**: 集成自动混合精度（AMP）、梯度累积、梯度裁剪等优化技术
- **健壮性强**: 完善的异常处理、断言机制、日志系统

## 主要模块

### 核心工具

- **`checkpoint_manager.py`**: 检查点管理（支持最佳模型、滚动保存、中断恢复）
- **`early_stopping.py`**: 早停实现（支持 min/max 模式、最小改善阈值）
- **`train.py`**: 训练循环模板（支持依赖注入和配置驱动两种模式）
- **`metrics.py`**: 指标跟踪（自动计算损失、准确率、Top-5 等）
- **`progress.py`**: 进度条实现（带内存监控和自定义指标显示）
- **`grads.py`**: 梯度工具函数（计算梯度范数、监控梯度爆炸/消失）

### 辅助工具

- **`config.py`**: 配置文件加载（支持 YAML、嵌套配置）
- **`logger_config.py`**: 日志系统配置（基于 loguru）
- **`helpers.py`**: 通用辅助函数（内存管理、随机种子、设备选择、OOM 检测等）
- **`decorators.py`**: 通用装饰器
- **`ntfy_notifier.py`**: 训练通知推送（支持 Ntfy 服务）

### 数据处理

- **`data.py`**: 数据集加载（支持常见数据集的统一接口）

### 优化器

- **`optimizers/`**: 高级优化器（AdamP、Lion、MADGRAD 等）

### 可视化

- **`visualization/`**: 训练可视化工具（收集器、报告器、可视化器)

### 训练回调

- **`callbacks/`**: 训练回调工具（计时器、学习率监控、批大小查找器）

## 快速开始

### 安装依赖

```bash
# 使用 uv 安装依赖
uv sync
```

### 基础使用示例

```python
from utils.config import load_config
from utils.train import Trainer
from utils.helpers import set_random_seed, get_device
import torch.nn as nn
import torch.optim as optim

# 加载配置
config = load_config("config.yaml")

# 设置随机种子
set_random_seed(config.seed)

# 获取设备
device = get_device(config.device)

# 创建模型、优化器、损失函数
model = YourModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
criterion = nn.CrossEntropyLoss()

# 使用配置驱动模式创建 Trainer
trainer = Trainer.from_config(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    config=config
)

# 开始训练
trainer.fit(train_loader, val_loader, epochs=config.training.epochs)
```

## 配置文件示例

参见 `utils/config.yaml.example`。

## 项目结构

```
.
├── utils/                      # 工具类目录
│   ├── checkpoint_manager.py
│   ├── early_stopping.py
│   ├── train.py
│   ├── metrics.py
│   ├── progress.py
│   ├── grads.py               # 梯度工具
│   ├── config.py
│   ├── logger_config.py
│   ├── helpers.py
│   ├── decorators.py
│   ├── ntfy_notifier.py
│   ├── data.py
│   ├── optimizers/
│   ├── visualization/
│   └── callbacks/             # 训练回调工具
│       ├── timer.py           # 训练计时器
│       ├── lr_monitor.py      # 学习率监控
│       └── batch_size_finder.py  # 批大小查找器
├── config.yaml                 # 配置文件
├── main.py                     # 测试/示例文件
├── network.py                  # 网络定义示例
├── LICENSE_APACHE2.txt         # Apache 2.0 许可证
├── THIRD_PARTY_LICENSES.md     # 第三方许可声明
└── README.md                   # 本文件
```

## 依赖管理

本项目使用 [uv](https://github.com/astral-sh/uv) 进行依赖管理。

主要依赖：
- PyTorch 2.x（CUDA 12+）
- loguru（日志系统）
- PyYAML（配置文件解析）
- tqdm（进度条）
- matplotlib（可视化）

## 开发规范

- **代码风格**: 遵循 PEP8，使用蛇形命名（变量/函数）和驼峰命名（类）
- **注释规范**: 采用"大中小注释"结构，详细解释"为什么"和"做什么"
- **日志系统**: 统一使用 loguru，按照重要程度使用不同日志级别
- **错误处理**: 积极使用断言和异常捕获，提供有意义的错误信息

## 第三方代码声明

本项目部分代码源自以下开源项目，特此声明并致谢：

### PyTorch Lightning

- **项目**: PyTorch Lightning
- **仓库**: https://github.com/Lightning-AI/pytorch-lightning
- **许可证**: Apache License 2.0
- **版权所有者**: The Lightning AI team
