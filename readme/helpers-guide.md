# 通用辅助函数使用指南

本文档详细说明 `utils/helpers.py` 模块中的通用辅助函数。这些函数按功能分为五大类。

## 时间工具

### `get_time(format_str)` 函数

#### 功能
获取格式化的当前时间字符串。

#### 参数
- `format_str` (str, 可选): 时间格式字符串，默认为 `"[%Y-%m-%d %H:%M:%S]"`

#### 返回值
格式化的时间字符串

#### 使用示例
```python
from utils import get_time

# 使用默认格式
current_time = get_time()
print(current_time)  # [2025-11-04 14:30:22]

# 自定义格式
timestamp = get_time("%Y%m%d_%H%M%S")
print(timestamp)  # 20251104_143022

# 生成带时间戳的文件名
checkpoint_path = f"checkpoints/model_{get_time('%Y%m%d_%H%M%S')}.pth"
```

### `format_time(seconds)` 函数

#### 功能
将秒数格式化为可读的时间字符串。

#### 参数
- `seconds` (float): 秒数

#### 返回值
格式化的时间字符串（格式：`"Xh Ym Z.Zs"` 或 `"Ym Z.Zs"` 或 `"Z.ZZs"`）

#### 格式规则
- 小于60秒：`"Z.ZZs"`
- 小于1小时：`"Ym Z.Zs"`
- 大于1小时：`"Xh Ym Z.Zs"`

#### 使用示例
```python
from utils import format_time
import time

start_time = time.time()
# ... 训练代码 ...
total_time = time.time() - start_time

logger.success(f"训练完成！总耗时: {format_time(total_time)}")
```

## 随机种子与可复现性

### `set_random_seed(seed, enable_cudnn_benchmark)` 函数

#### 功能
设置所有相关库的随机种子以确保实验的可复现性。

#### 参数
- `seed` (int, 可选): 随机种子值，默认为 `42`
- `enable_cudnn_benchmark` (bool, 可选): 是否启用cuDNN自动调优，默认为 `True`

#### cuDNN Benchmark 说明

| 参数值 | 可复现性 | 性能 | 适用场景 |
|--------|---------|------|---------|
| `False` | 完全可复现 | 标准 | 论文复现、消融实验 |
| `True` | 部分牺牲 | 提升20-30% | 生产部署、大规模训练 |

#### 设置的随机种子
- Python内置 `random`
- NumPy `np.random`
- PyTorch CPU `torch.manual_seed`
- PyTorch CUDA（所有GPU）`torch.cuda.manual_seed_all`
- cuDNN确定性算法 `torch.backends.cudnn.deterministic = True`

#### 使用示例
```python
from utils import set_random_seed

# 完全可复现（论文复现）
set_random_seed(seed=42, enable_cudnn_benchmark=False)

# 性能优先（生产训练）
set_random_seed(seed=42, enable_cudnn_benchmark=True)

# 与配置文件集成
from utils import setup_config, set_random_seed

config = setup_config(DEFAULT_CONFIG, 'config.yaml', {})
set_random_seed(
    seed=config.experiment.seed,
    enable_cudnn_benchmark=config.experiment.enable_cudnn_benchmark
)
```

## 设备管理

### `get_device(requested_device)` 函数

#### 功能
获取并验证PyTorch计算设备，支持智能回退。

#### 参数
- `requested_device` (str, 可选): 请求的设备，默认为 `'auto'`

#### 支持的设备字符串

| 字符串 | 行为 |
|--------|------|
| `'auto'` | 自动选择GPU（如果可用），否则CPU |
| `'cpu'` | 强制使用CPU |
| `'cuda'` | 使用 `cuda:0`（如果不可用则回退到CPU） |
| `'cuda:N'` | 使用第N块GPU（如果不可用则回退到 `cuda:0`） |

#### 智能回退机制
- 请求的GPU不存在 → 回退到 `cuda:0`
- CUDA不可用 → 回退到 `cpu`

#### 返回值
验证后的 `torch.device` 对象

#### 使用示例
```python
from utils import get_device

# 自动选择（推荐）
device = get_device('auto')

# 手动指定
device = get_device('cuda:0')  # 使用第1块GPU
device = get_device('cuda:1')  # 使用第2块GPU
device = get_device('cpu')     # 强制使用CPU

# 与配置文件集成
config = setup_config(DEFAULT_CONFIG, 'config.yaml', {})
device = get_device(config.experiment.device)

# 使用设备
model = YourModel().to(device)
images = images.to(device)
labels = labels.to(device)
```

### `clear_memory()` 函数

#### 功能
清理GPU缓存（调用 `torch.cuda.empty_cache()`）。

#### 使用场景
- 验证/测试后清理显存
- 模型切换时释放显存
- 预防OOM（Out of Memory）错误

#### 注意事项
- 仅清理缓存，不会释放被Python对象引用的显存
- 需要先删除对象（`del model`），然后调用此函数

#### 使用示例
```python
from utils import clear_memory

# 训练后验证
model.eval()
with torch.no_grad():
    # ... 验证代码 ...
    pass

# 清理验证过程中的缓存
clear_memory()

# 正确的清理方式
del model
del optimizer
clear_memory()
import gc
gc.collect()
```

### `get_memory_usage(device)` 函数

#### 功能
获取指定CUDA设备的内存使用情况。

#### 参数
- `device` (int | torch.device | None, 可选): 目标设备
  - `None`: 使用当前设备（默认）
  - `int`: 设备编号（如 `0`, `1`）
  - `torch.device`: PyTorch设备对象

#### 返回值
包含以下键的字典（如果不是CUDA设备则返回 `None`）：

| 键 | 说明 |
|----|------|
| `allocated` | 当前已分配的显存 |
| `reserved` | PyTorch缓存池中的显存 |
| `total` | GPU总显存 |
| `percent_used` | 已使用百分比 |

#### 使用示例
```python
from utils import get_memory_usage

# 使用默认设备
usage = get_memory_usage()
print(usage)
# {'allocated': '2.5 GB', 'reserved': '3.0 GB', 'total': '12.0 GB', 'percent_used': '20.83%'}

# 指定设备
usage = get_memory_usage(device=0)      # GPU 0
usage = get_memory_usage(device=torch.device('cuda:1'))  # GPU 1
```

### `log_memory_usage(description)` 函数

#### 功能
记录内存使用情况到日志（DEBUG级别）。

#### 参数
- `description` (str, 可选): 日志描述，默认为 `"当前内存使用"`

#### 使用示例
```python
from utils import log_memory_usage

# 在关键操作前后记录
log_memory_usage("模型加载前")
model = YourModel().to(device)
log_memory_usage("模型加载后")

# 训练循环中定期检查
for epoch in range(100):
    # ... 训练代码 ...

    if (epoch + 1) % 10 == 0:
        log_memory_usage(f"Epoch {epoch + 1}")
```

## 张量与模型工具

### `validate_tensor(tensor, name)` 函数

#### 功能
验证张量是否包含NaN或Inf值，用于调试训练不稳定问题。

#### 参数
- `tensor` (torch.Tensor): 要验证的张量
- `name` (str, 可选): 张量名称（用于日志），默认为 `"tensor"`

#### 返回值
- `True`: 张量有效（不包含NaN或Inf）
- `False`: 张量无效（包含NaN或Inf）

#### 使用场景
- 调试梯度爆炸/消失
- 检查中间层输出
- 检查损失值

#### 注意事项
不推荐在生产环境使用（有性能开销）

#### 使用示例
```python
from utils import validate_tensor

# 检查模型输出
outputs = model(images)
if not validate_tensor(outputs, name="model_outputs"):
    logger.error("模型输出异常！")
    break

# 检查损失
loss = criterion(outputs, labels)
if not validate_tensor(loss, name="loss"):
    logger.error("损失异常！")
    break

# 检查梯度
optimizer.step()
for name, param in model.named_parameters():
    if param.grad is not None:
        if not validate_tensor(param.grad, name=f"grad_{name}"):
            logger.error(f"梯度爆炸！参数: {name}")
            break
```

### `count_parameters(model, trainable_only)` 函数

#### 功能
统计模型参数数量。

#### 参数
- `model` (torch.nn.Module): 模型
- `trainable_only` (bool, 可选): 是否只统计可训练参数，默认为 `False`

#### 返回值
参数数量（int）

#### 使用示例
```python
from utils import count_parameters
from loguru import logger

model = YourModel()

# 统计所有参数
total = count_parameters(model)
logger.info(f"总参数: {total:,}")

# 只统计可训练参数
trainable = count_parameters(model, trainable_only=True)
logger.info(f"可训练参数: {trainable:,}")

# 计算冻结参数
frozen = total - trainable
logger.info(f"冻结参数: {frozen:,}")

# 迁移学习场景
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

## 格式化与IO

### `format_size(size_bytes)` 函数

#### 功能
将字节大小格式化为可读格式（KB, MB, GB等）。

#### 参数
- `size_bytes` (int): 字节数

#### 返回值
格式化的大小字符串

#### 使用示例
```python
from utils import format_size
import os

# 检查文件大小
model_path = "checkpoints/best_model.pth"
file_size = os.path.getsize(model_path)
logger.info(f"模型文件大小: {format_size(file_size)}")

# 示例输出
print(format_size(0))           # 0B
print(format_size(1024))        # 1.0 KB
print(format_size(1048576))     # 1.0 MB
print(format_size(1073741824))  # 1.0 GB
```

### `save_dict_to_json(data, file_path)` 函数

#### 功能
将字典保存到JSON文件。

#### 参数
- `data` (dict): 要保存的字典
- `file_path` (str): 目标文件路径

#### 特性
- 自动创建目录
- UTF-8编码（支持中文）
- 格式化输出（缩进4空格）
- 保存失败时抛出异常

#### 使用示例
```python
from utils import save_dict_to_json

# 保存训练结果
results = {
    "dataset": "CIFAR10",
    "model": "ResNet18",
    "best_train_acc": 95.2,
    "best_val_acc": 92.8,
    "total_epochs": 100,
    "batch_size": 128,
    "learning_rate": 0.01
}

save_dict_to_json(results, "./results/training_results.json")
```

### `load_dict_from_json(file_path)` 函数

#### 功能
从JSON文件加载字典。

#### 参数
- `file_path` (str): JSON文件路径

#### 返回值
- 成功：字典
- 失败：`None`

#### 使用示例
```python
from utils import load_dict_from_json

# 加载配置
data = load_dict_from_json("./results/training_results.json")

if data is not None:
    print(data["dataset"])     # CIFAR10
    print(data["best_val_acc"]) # 92.8
```

## 使用示例：标准训练脚本模板

```python
from utils import (
    setup_logging,
    setup_config,
    set_random_seed,
    get_device,
    load_dataset_info,
    count_parameters,
    log_memory_usage,
    format_time,
    save_dict_to_json,
    get_time
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

    # 10. 保存结果
    results = {
        "experiment_name": config.experiment.name,
        "timestamp": get_time("%Y-%m-%d %H:%M:%S"),
        "total_time": format_time(total_time),
        "best_val_acc": float(best_val_acc),
        # ... 其他结果 ...
    }

    timestamp = get_time("%Y%m%d_%H%M%S")
    save_dict_to_json(results, f"./results/result_{timestamp}.json")

    # 11. 清理
    log_memory_usage("训练结束")

if __name__ == '__main__':
    main()
```
