# `helpers` 通用辅助函数使用指南

## 📚 目录
1. [快速开始](#快速开始)
2. [功能概览](#功能概览)
3. [时间工具](#时间工具)
4. [随机种子与可复现性](#随机种子与可复现性)
5. [设备管理](#设备管理)
6. [内存管理](#内存管理)
7. [模型与张量工具](#模型与张量工具)
8. [格式化与 IO](#格式化与-io)
9. [常见问题](#常见问题)

---

## 🚀 快速开始

### 一个完整的训练脚本示例

```python
from utils import (
    set_random_seed,
    get_device,
    count_parameters,
    log_memory_usage
)
from loguru import logger

# 1. 设置随机种子
set_random_seed(seed=42)

# 2. 获取计算设备
device = get_device('auto')  # 自动选择 GPU 或 CPU

# 3. 创建模型并移到设备
model = YourModel().to(device)

# 4. 统计模型参数
total_params = count_parameters(model)
trainable_params = count_parameters(model, trainable_only=True)
logger.info(f"总参数: {total_params:,}")
logger.info(f"可训练参数: {trainable_params:,}")

# 5. 记录内存使用
log_memory_usage("模型加载后")

# 6. 开始训练...
```

---

## 📦 功能概览

`helpers.py` 提供了 **11 个实用函数**，分为 5 大类：

| 类别 | 函数 | 用途 |
|------|------|------|
| **时间工具** | `get_time()` | 获取格式化当前时间 |
| | `format_time()` | 格式化秒数为可读字符串 |
| **可复现性** | `set_random_seed()` | 设置全局随机种子 |
| **设备管理** | `get_device()` | 智能设备选择与验证 |
| | `clear_memory()` | 清理 GPU 缓存 |
| | `get_memory_usage()` | 获取 GPU 内存使用情况 |
| | `log_memory_usage()` | 记录内存使用到日志 |
| **张量与模型** | `validate_tensor()` | 检查张量是否包含 NaN/Inf |
| | `count_parameters()` | 统计模型参数数量 |
| **格式化与 IO** | `format_size()` | 格式化字节大小 |
| | `save_dict_to_json()` | 保存字典到 JSON 文件 |
| | `load_dict_from_json()` | 从 JSON 文件加载字典 |

---

## ⏰ 时间工具

### 1. `get_time()` - 获取格式化当前时间

```python
from utils import get_time

# 使用默认格式 "[%Y-%m-%d %H:%M:%S]"
current_time = get_time()
print(current_time)  # [2025-11-04 14:30:22]

# 自定义格式
custom_time = get_time("%Y%m%d_%H%M%S")
print(custom_time)  # 20251104_143022
```

**常见用途**：
- 生成带时间戳的文件名
- 日志记录
- 训练进度显示

**示例：生成带时间戳的检查点文件名**

```python
from utils import get_time

timestamp = get_time("%Y%m%d_%H%M%S")
checkpoint_path = f"checkpoints/model_{timestamp}.pth"
torch.save(model.state_dict(), checkpoint_path)
```

### 2. `format_time()` - 格式化秒数

```python
from utils import format_time

# 小于 60 秒
print(format_time(45.5))      # 45.50s

# 小于 1 小时
print(format_time(135.7))     # 2m 15.7s

# 大于 1 小时
print(format_time(3725.3))    # 1h 2m 5.3s
```

**常见用途**：
- 显示训练总时间
- 显示 Epoch 平均耗时
- 进度条时间估计

**示例：训练结束后显示总耗时**

```python
from utils import format_time
import time

start_time = time.time()

# ... 训练代码 ...

total_time = time.time() - start_time
logger.success(f"训练完成！总耗时: {format_time(total_time)}")
```

---

## 🎲 随机种子与可复现性

### `set_random_seed()` - 设置全局随机种子

**核心功能**：
- 设置 Python、NumPy、PyTorch (CPU & GPU) 的随机种子
- 确保实验可复现
- 支持 cuDNN benchmark 性能优化

```python
from utils import set_random_seed

# 基础用法（保证可复现）
set_random_seed(seed=42)

# 性能优化模式（牺牲部分可复现性）
set_random_seed(seed=42, enable_cudnn_benchmark=True)
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `seed` | int | 42 | 随机种子值 |
| `enable_cudnn_benchmark` | bool | True | 是否启用 cuDNN 自动调优 |

### cuDNN Benchmark 详解

**启用 benchmark (`enable_cudnn_benchmark=True`)**：
- ✅ 在固定输入尺寸时可提升 **20-30% 性能**
- ❌ 会牺牲部分可复现性
- 📌 适用场景：生产部署、大规模训练

**禁用 benchmark (`enable_cudnn_benchmark=False`)**：
- ✅ 保证 **完全可复现**
- ❌ 性能略低
- 📌 适用场景：论文复现、消融实验

### 使用场景

**场景 1: 论文复现（完全可复现）**

```python
from utils import set_random_seed

# 在配置文件中设置
# experiment:
#   seed: 42
#   enable_cudnn_benchmark: false

set_random_seed(seed=42, enable_cudnn_benchmark=False)

# 确保两次运行结果完全一致
```

**场景 2: 生产训练（性能优先）**

```python
from utils import set_random_seed

# 在配置文件中设置
# experiment:
#   seed: 42
#   enable_cudnn_benchmark: true

set_random_seed(seed=42, enable_cudnn_benchmark=True)

# 提升 20-30% 训练速度
```

**场景 3: 与配置文件集成**

```python
from utils import setup_config, set_random_seed

config = setup_config(DEFAULT_CONFIG, 'config.yaml', {})

set_random_seed(
    seed=config.experiment.seed,
    enable_cudnn_benchmark=config.experiment.enable_cudnn_benchmark
)
```

---

## 🖥️ 设备管理

### `get_device()` - 智能设备选择

**核心功能**：
- 自动选择 GPU 或 CPU
- 多 GPU 支持（`cuda:0`, `cuda:1` 等）
- 智能回退（请求的 GPU 不可用时自动回退）

```python
from utils import get_device

# 自动选择（推荐）
device = get_device('auto')  # 有 GPU 用 GPU，没有用 CPU

# 手动指定
device = get_device('cuda')      # 使用 cuda:0
device = get_device('cuda:1')    # 使用第 2 块 GPU
device = get_device('cpu')       # 强制使用 CPU
```

**参数说明**：

| 参数值 | 行为 |
|--------|------|
| `'auto'` | 自动选择 GPU（如果可用），否则 CPU |
| `'cpu'` | 强制使用 CPU |
| `'cuda'` | 使用 `cuda:0`（如果不可用则回退到 CPU） |
| `'cuda:N'` | 使用第 N 块 GPU（如果不可用则回退到 `cuda:0`） |

### 智能回退机制

```python
from utils import get_device

# 场景 1: 没有 GPU
device = get_device('cuda')
# 输出: CUDA 不可用。回退到 CPU。
# 输出: 计算设备已设置为: CPU

# 场景 2: 请求的 GPU 不存在
device = get_device('cuda:3')  # 但只有 2 块 GPU
# 输出: 请求的 GPU cuda:3 不可用 (仅找到 2 块 GPU)。
# 输出: 回退到 'cuda:0'。
# 输出: 计算设备已设置为: cuda:0 (NVIDIA GeForce RTX 5070 Ti)
```

### 使用场景

**场景 1: 单机单卡训练**

```python
from utils import get_device

device = get_device('auto')
model = YourModel().to(device)

for images, labels in train_loader:
    images = images.to(device)
    labels = labels.to(device)
    # ... 训练代码 ...
```

**场景 2: 多 GPU 环境（指定卡号）**

```python
from utils import get_device

# 使用第 2 块 GPU (cuda:1)
device = get_device('cuda:1')
model = YourModel().to(device)
```

**场景 3: 与配置文件集成**

```yaml
# config.yaml
experiment:
  device: "auto"  # 或 "cuda:0", "cpu" 等
```

```python
from utils import setup_config, get_device

config = setup_config(DEFAULT_CONFIG, 'config.yaml', {})
device = get_device(config.experiment.device)
```

---

## 💾 内存管理

### 1. `clear_memory()` - 清理 GPU 缓存

```python
from utils import clear_memory

# 在大量 GPU 操作后清理缓存
clear_memory()
# 输出: 已清理 GPU 缓存 (torch.cuda.empty_cache())。
```

**使用场景**：
- 验证/测试后清理显存
- 模型切换时释放显存
- Out of Memory (OOM) 错误预防

**示例：验证后清理显存**

```python
from utils import clear_memory

# 训练完成后验证
model.eval()
with torch.no_grad():
    # ... 验证代码 ...
    pass

# 清理验证过程中的缓存
clear_memory()
```

### 2. `get_memory_usage()` - 获取内存使用情况

```python
from utils import get_memory_usage

# 使用默认设备（当前设备）
usage = get_memory_usage()
print(usage)
# {'allocated': '2.5 GB', 'reserved': '3.0 GB', 'total': '12.0 GB', 'percent_used': '20.83%'}

# 指定设备
usage = get_memory_usage(device=0)      # 使用 GPU 0
usage = get_memory_usage(device=torch.device('cuda:1'))  # 使用 GPU 1
```

**返回值说明**：

| 键 | 说明 |
|----|------|
| `allocated` | 当前已分配的显存 |
| `reserved` | PyTorch 缓存池中的显存 |
| `total` | GPU 总显存 |
| `percent_used` | 已使用百分比 |

### 3. `log_memory_usage()` - 记录内存使用

```python
from utils import log_memory_usage

# 在关键操作前后记录
log_memory_usage("模型加载前")
model = YourModel().to(device)
log_memory_usage("模型加载后")

# 输出示例：
# 模型加载前: 512.0 MB / 12.0 GB (4.17%)
# 模型加载后: 2.5 GB / 12.0 GB (20.83%)
```

### 完整示例：内存监控

```python
from utils import get_device, log_memory_usage, clear_memory
from loguru import logger

device = get_device('auto')

# 1. 初始状态
log_memory_usage("初始状态")

# 2. 加载模型
model = YourModel().to(device)
log_memory_usage("模型加载后")

# 3. 训练
for epoch in range(100):
    # ... 训练代码 ...

    # 每 10 个 epoch 检查一次内存
    if (epoch + 1) % 10 == 0:
        log_memory_usage(f"Epoch {epoch + 1}")

# 4. 训练结束后清理
del model
clear_memory()
log_memory_usage("清理后")
```

---

## 🧪 模型与张量工具

### 1. `validate_tensor()` - 验证张量有效性

**功能**：检查张量是否包含 NaN 或 Inf 值。

```python
from utils import validate_tensor
import torch

# 正常张量
tensor = torch.randn(3, 3)
is_valid = validate_tensor(tensor, name="my_tensor")
# 返回: True

# 包含 NaN 的张量
tensor_nan = torch.tensor([1.0, float('nan'), 3.0])
is_valid = validate_tensor(tensor_nan, name="nan_tensor")
# 输出: 张量 nan_tensor 包含 NaN 值！
# 返回: False

# 包含 Inf 的张量
tensor_inf = torch.tensor([1.0, float('inf'), 3.0])
is_valid = validate_tensor(tensor_inf, name="inf_tensor")
# 输出: 张量 inf_tensor 包含 Inf 值！
# 返回: False
```

**使用场景**：

**场景 1: 调试训练不稳定**

```python
from utils import validate_tensor

for epoch in range(100):
    for images, labels in train_loader:
        # 前向传播
        outputs = model(images)

        # 验证输出
        if not validate_tensor(outputs, name="model_outputs"):
            logger.error(f"模型输出异常！Epoch {epoch}")
            break

        loss = criterion(outputs, labels)

        # 验证损失
        if not validate_tensor(loss, name="loss"):
            logger.error(f"损失异常！Epoch {epoch}")
            break

        # 反向传播
        loss.backward()
```

**场景 2: 梯度爆炸检测**

```python
from utils import validate_tensor

optimizer.step()

# 检查模型参数
for name, param in model.named_parameters():
    if param.grad is not None:
        if not validate_tensor(param.grad, name=f"grad_{name}"):
            logger.error(f"梯度爆炸检测！参数: {name}")
            break
```

### 2. `count_parameters()` - 统计模型参数

```python
from utils import count_parameters

model = YourModel()

# 统计所有参数
total_params = count_parameters(model)
print(f"总参数: {total_params:,}")  # 总参数: 1,234,567

# 只统计可训练参数
trainable_params = count_parameters(model, trainable_only=True)
print(f"可训练参数: {trainable_params:,}")  # 可训练参数: 1,234,567
```

**使用场景**：

**场景 1: 模型信息统计**

```python
from utils import count_parameters
from loguru import logger

model = YourModel()

total = count_parameters(model)
trainable = count_parameters(model, trainable_only=True)
frozen = total - trainable

logger.info("=" * 60)
logger.info("模型参数统计".center(60))
logger.info("=" * 60)
logger.info(f"总参数: {total:,}")
logger.info(f"可训练参数: {trainable:,}")
logger.info(f"冻结参数: {frozen:,}")
logger.info("=" * 60)
```

**场景 2: 迁移学习（部分冻结）**

```python
from utils import count_parameters
from loguru import logger

# 加载预训练模型
model = torchvision.models.resnet18(pretrained=True)

# 冻结特征提取层
for param in model.parameters():
    param.requires_grad = False

# 替换分类头
model.fc = nn.Linear(512, num_classes)

# 统计参数
total = count_parameters(model)
trainable = count_parameters(model, trainable_only=True)

logger.info(f"总参数: {total:,}")
logger.info(f"可训练参数: {trainable:,} (仅分类头)")
```

---

## 📝 格式化与 IO

### 1. `format_size()` - 格式化字节大小

```python
from utils import format_size

print(format_size(0))           # 0B
print(format_size(1024))        # 1.0 KB
print(format_size(1048576))     # 1.0 MB
print(format_size(1073741824))  # 1.0 GB
```

**使用场景**：

```python
from utils import format_size
import os

# 检查模型文件大小
model_path = "checkpoints/best_model.pth"
file_size = os.path.getsize(model_path)
logger.info(f"模型文件大小: {format_size(file_size)}")
```

### 2. `save_dict_to_json()` - 保存字典到 JSON

```python
from utils import save_dict_to_json

data = {
    "experiment": "CIFAR10_ResNet18",
    "best_acc": 92.5,
    "epochs": 100,
    "batch_size": 128
}

save_dict_to_json(data, "./results/experiment_config.json")
# 输出: 数据已保存到 JSON 文件: ./results/experiment_config.json
```

**特点**：
- ✅ 自动创建目录
- ✅ UTF-8 编码，支持中文
- ✅ 格式化输出（缩进 4 空格）
- ✅ 保存失败时抛出异常

### 3. `load_dict_from_json()` - 从 JSON 加载字典

```python
from utils import load_dict_from_json

data = load_dict_from_json("./results/experiment_config.json")

if data is not None:
    print(data["experiment"])  # CIFAR10_ResNet18
    print(data["best_acc"])    # 92.5
```

**返回值**：
- 成功：返回字典
- 失败：返回 `None`

### 完整示例：保存训练结果

```python
from utils import save_dict_to_json, format_time
import time

# 训练过程
start_time = time.time()
# ... 训练代码 ...
total_time = time.time() - start_time

# 保存结果
results = {
    "dataset": "CIFAR10",
    "model": "ResNet18",
    "best_train_acc": 95.2,
    "best_val_acc": 92.8,
    "total_epochs": 100,
    "total_time": format_time(total_time),
    "batch_size": 128,
    "learning_rate": 0.01
}

save_dict_to_json(results, "./results/training_results.json")
```

---

## ❓ 常见问题

### Q1: `set_random_seed()` 能保证 100% 可复现吗？

**A**: 几乎可以，但有以下例外：

1. **cuDNN Benchmark 启用时**：
   - 设置 `enable_cudnn_benchmark=False` 可保证完全可复现

2. **多线程/多进程数据加载**：
   - 设置 `DataLoader(num_workers=0)` 可避免

3. **硬件差异**：
   - 不同 GPU 型号可能有微小差异

**推荐配置（完全可复现）**：

```yaml
# config.yaml
experiment:
  seed: 42
  enable_cudnn_benchmark: false

dataloader:
  num_workers: 0
```

### Q2: `get_device()` 返回的设备如何使用？

**A**: 直接传给 `.to(device)`：

```python
from utils import get_device

device = get_device('auto')

# 模型
model = model.to(device)

# 张量
images = images.to(device)
labels = labels.to(device)

# 优化器（自动适配）
optimizer = torch.optim.Adam(model.parameters())
```

### Q3: 为什么 `clear_memory()` 后显存没有完全释放？

**A**: PyTorch 的显存管理机制：

1. **缓存池**：PyTorch 会保留部分显存用于加速后续分配
2. **真正释放**：只有在对象被删除后才会释放

**正确的清理方式**：

```python
from utils import clear_memory

# 1. 删除对象
del model
del optimizer

# 2. 清理缓存
clear_memory()

# 3. (可选) 强制垃圾回收
import gc
gc.collect()
```

### Q4: `validate_tensor()` 什么时候使用？

**A**: 主要用于调试以下问题：

- **梯度爆炸/消失**：检查梯度是否为 NaN/Inf
- **数值不稳定**：检查中间层输出
- **损失异常**：检查损失值

**不推荐在生产环境使用**（性能开销）。

### Q5: `count_parameters()` 的两种模式有什么区别？

**A**:

```python
from utils import count_parameters

model = YourModel()

# 模式 1: 统计所有参数（包括冻结的）
total = count_parameters(model, trainable_only=False)

# 模式 2: 只统计可训练参数
trainable = count_parameters(model, trainable_only=True)

# 关系
frozen = total - trainable
```

**使用场景**：
- **迁移学习**：`trainable_only=True` 查看需要训练的参数
- **模型对比**：`trainable_only=False` 查看模型总大小

### Q6: 如何在多 GPU 训练中使用这些工具？

**A**:

**DataParallel 模式**：

```python
from utils import get_device, count_parameters

# 主设备
device = get_device('cuda:0')

# 模型包装
model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
model = model.to(device)

# 统计参数（使用 model.module）
total_params = count_parameters(model.module)
```

**DistributedDataParallel 模式**：

```python
from utils import get_device

# 每个进程使用不同的 GPU
local_rank = int(os.environ['LOCAL_RANK'])
device = get_device(f'cuda:{local_rank}')

model = model.to(device)
model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
```

### Q7: JSON 保存/加载支持哪些数据类型？

**A**: 标准 JSON 类型：

**支持的类型**：
- `str`, `int`, `float`, `bool`
- `list`, `dict`
- `None`

**不支持的类型**（需要转换）：
- `numpy.ndarray` → 转为 `list`
- `torch.Tensor` → 转为 `list`
- `datetime` → 转为 `str`

**示例**：

```python
from utils import save_dict_to_json
import numpy as np

# 错误示例
data = {"array": np.array([1, 2, 3])}  # 会报错

# 正确示例
data = {"array": np.array([1, 2, 3]).tolist()}  # 转为 list
save_dict_to_json(data, "results.json")
```

---

## 📋 最佳实践

### 1. 标准训练脚本模板

```python
from utils import (
    setup_logging,
    setup_config,
    set_random_seed,
    get_device,
    load_dataset_info,
    count_parameters,
    log_memory_usage,
    format_time
)
from loguru import logger
import time

def main():
    # 1. 配置日志
    setup_logging(log_dir='./logs', console_level='INFO', file_level='DEBUG')

    # 2. 加载配置
    logger.info("加载配置...")
    config = setup_config(DEFAULT_CONFIG, 'config.yaml', {})

    # 3. 设置随机种子
    logger.info(f"设置随机种子: {config.experiment.seed}")
    set_random_seed(
        seed=config.experiment.seed,
        enable_cudnn_benchmark=config.experiment.enable_cudnn_benchmark
    )

    # 4. 获取设备
    device = get_device('auto')

    # 5. 加载数据
    logger.info("加载数据集...")
    dataset_info = load_dataset_info(
        dataset_name=config.dataset.name,
        data_path=config.dataset.data_path
    )

    # 6. 创建模型
    logger.info("创建模型...")
    model = create_model(
        input_channels=dataset_info['channel'],
        num_classes=dataset_info['num_classes']
    ).to(device)

    # 7. 统计参数
    total_params = count_parameters(model)
    logger.info(f"模型参数: {total_params:,}")

    # 8. 记录内存
    log_memory_usage("模型加载后")

    # 9. 训练
    logger.info("开始训练...")
    start_time = time.time()

    trainer.fit(train_loader, val_loader, epochs=config.training.epochs)

    total_time = time.time() - start_time
    logger.success(f"训练完成！总耗时: {format_time(total_time)}")

    # 10. 清理
    log_memory_usage("训练结束")

if __name__ == '__main__':
    main()
```

### 2. 内存监控最佳实践

```python
from utils import log_memory_usage, clear_memory

# 训练前
log_memory_usage("训练前")

# 训练循环
for epoch in range(100):
    # ... 训练代码 ...

    # 验证
    model.eval()
    with torch.no_grad():
        # ... 验证代码 ...
        pass

    # 验证后清理
    clear_memory()

    # 每 10 个 epoch 记录一次
    if (epoch + 1) % 10 == 0:
        log_memory_usage(f"Epoch {epoch + 1}")

# 训练后
log_memory_usage("训练后")
```

### 3. 结果保存最佳实践

```python
from utils import save_dict_to_json, format_time, get_time
import time

# 训练过程
start_time = time.time()
best_val_acc = 0.0

for epoch in range(100):
    # ... 训练代码 ...
    if val_acc > best_val_acc:
        best_val_acc = val_acc

total_time = time.time() - start_time

# 保存结果
results = {
    "experiment_name": config.experiment.name,
    "timestamp": get_time("%Y-%m-%d %H:%M:%S"),
    "dataset": config.dataset.name,
    "model": config.model.name,
    "metrics": {
        "best_val_acc": float(best_val_acc),
        "final_loss": float(final_loss)
    },
    "hyperparameters": {
        "epochs": config.training.epochs,
        "batch_size": config.training.batch_size,
        "learning_rate": config.training.lr,
        "optimizer": config.training.optimizer
    },
    "system": {
        "total_time": format_time(total_time),
        "device": str(device),
        "seed": config.experiment.seed
    }
}

# 保存到文件
timestamp = get_time("%Y%m%d_%H%M%S")
save_dict_to_json(results, f"./results/result_{timestamp}.json")
```

---

## 🎯 总结

`helpers` 模块的核心优势：

1. **时间工具**：优雅的时间格式化，方便日志和文件命名
2. **可复现性**：一键设置全局随机种子，支持性能/可复现性权衡
3. **设备管理**：智能设备选择，自动回退机制
4. **内存监控**：实时监控 GPU 显存，及时清理缓存
5. **调试工具**：张量验证、参数统计，快速定位问题
6. **格式化 IO**：标准化的 JSON 保存/加载，方便结果管理

这些工具函数让 PyTorch 训练脚本更加优雅和健壮！🎉
