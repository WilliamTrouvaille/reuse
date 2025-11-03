# `data` 数据加载模块使用指南

## 📚 目录
1. [快速开始](#快速开始)
2. [支持的数据集](#支持的数据集)
3. [配置选项](#配置选项)
4. [常见场景](#常见场景)
5. [返回值说明](#返回值说明)
6. [高级用法](#高级用法)
7. [常见问题](#常见问题)

---

## 🚀 快速开始

### 最小化示例（2 行代码）

```python
from utils import load_dataset_info

# 加载 CIFAR10 数据集
dataset_info = load_dataset_info(dataset_name='CIFAR10', data_path='./data')

# 现在可以直接使用
train_dataset = dataset_info['dst_train']
test_dataset = dataset_info['dst_test']
num_classes = dataset_info['num_classes']  # 10
```

就这么简单！`load_dataset_info` 会自动：
- ✅ 下载数据集（如果本地不存在）
- ✅ 应用标准归一化变换
- ✅ 返回训练集和测试集
- ✅ 提供完整的数据集元数据

---

## 📦 支持的数据集

### 当前支持的数据集

| 数据集 | 图像尺寸 | 通道 | 类别数 | 用途 |
|--------|---------|------|--------|------|
| `MNIST` | 28×28 | 1 (灰度) | 10 | 手写数字识别 |
| `FashionMNIST` | 28×28 | 1 (灰度) | 10 | 时尚物品分类 |
| `CIFAR10` | 32×32 | 3 (RGB) | 10 | 通用物体识别 |
| `CIFAR100` | 32×32 | 3 (RGB) | 100 | 细粒度物体识别 |

### 数据集特点

**MNIST**：
- 经典的手写数字数据集（0-9）
- 训练集：60,000 张图像
- 测试集：10,000 张图像
- 适合快速原型验证

**FashionMNIST**：
- MNIST 的时尚物品版本
- 类别：T恤、裤子、套衫、裙子等
- 比 MNIST 更具挑战性
- 训练集：60,000 张图像
- 测试集：10,000 张图像

**CIFAR10**：
- 彩色图像，10 个类别
- 类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车
- 训练集：50,000 张图像
- 测试集：10,000 张图像
- 适合中等规模实验

**CIFAR100**：
- CIFAR10 的细粒度版本
- 100 个类别（20 个大类，每个大类 5 个小类）
- 训练集：50,000 张图像
- 测试集：10,000 张图像
- 适合多分类任务

---

## ⚙️ 配置选项

### 基础用法

```python
from utils import load_dataset_info

dataset_info = load_dataset_info(
    dataset_name='CIFAR10',  # 数据集名称
    data_path='./data'       # 数据存储路径
)
```

**参数说明**：
- `dataset_name` (str): 数据集名称，必须是支持的数据集之一
- `data_path` (str): 数据集下载和存储的根路径

### 在配置文件中使用

推荐在 `config.yaml` 中配置数据集参数：

```yaml
# config.yaml
dataset:
  name: "CIFAR10"
  data_path: "./data"

dataloader:
  num_workers: 4
  pin_memory: true
  batch_size: 128
```

然后在代码中使用：

```python
from utils import setup_config, load_dataset_info

# 加载配置
config = setup_config(DEFAULT_CONFIG, 'config.yaml', {})

# 从配置加载数据集
dataset_info = load_dataset_info(
    dataset_name=config.dataset.name,
    data_path=config.dataset.data_path
)
```

---

## 📖 常见场景

### 场景 1: 在训练脚本中使用

```python
from torch.utils.data import DataLoader
from utils import load_dataset_info, setup_config
from loguru import logger

def main():
    # 1. 加载配置
    config = setup_config(DEFAULT_CONFIG, 'config.yaml', {})

    # 2. 加载数据集
    dataset_info = load_dataset_info(
        dataset_name=config.dataset.name,
        data_path=config.dataset.data_path
    )

    # 3. 创建 DataLoader
    train_loader = DataLoader(
        dataset_info['dst_train'],
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory
    )

    val_loader = DataLoader(
        dataset_info['dst_test'],
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory
    )

    # 4. 使用数据集元数据
    logger.info(f"数据集: {config.dataset.name}")
    logger.info(f"类别数: {dataset_info['num_classes']}")
    logger.info(f"图像尺寸: {dataset_info['im_size']}")
    logger.info(f"训练集大小: {len(dataset_info['dst_train'])}")
    logger.info(f"测试集大小: {len(dataset_info['dst_test'])}")

    # 5. 开始训练...
    trainer.fit(train_loader, val_loader, epochs=100)
```

### 场景 2: 快速实验（不使用配置文件）

```python
from torch.utils.data import DataLoader
from utils import load_dataset_info

# 直接加载数据集
dataset_info = load_dataset_info('MNIST', './data')

# 快速创建 DataLoader
train_loader = DataLoader(
    dataset_info['dst_train'],
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    dataset_info['dst_test'],
    batch_size=64,
    shuffle=False
)

# 开始训练...
for images, labels in train_loader:
    # ... 训练代码 ...
    pass
```

### 场景 3: 数据探索

```python
from utils import load_dataset_info
import matplotlib.pyplot as plt

# 加载数据集
dataset_info = load_dataset_info('CIFAR10', './data')

# 查看数据集信息
print(f"类别名称: {dataset_info['class_names']}")
print(f"均值: {dataset_info['mean']}")
print(f"标准差: {dataset_info['std']}")

# 可视化样本
train_dataset = dataset_info['dst_train']
image, label = train_dataset[0]

plt.imshow(image.permute(1, 2, 0))  # 转换为 (H, W, C)
plt.title(f"Label: {dataset_info['class_names'][label]}")
plt.show()
```

### 场景 4: 切换数据集（快速对比实验）

```python
from utils import load_dataset_info

# 只需修改一个参数即可切换数据集
datasets_to_test = ['MNIST', 'FashionMNIST', 'CIFAR10']

for dataset_name in datasets_to_test:
    print(f"\n测试数据集: {dataset_name}")

    dataset_info = load_dataset_info(dataset_name, './data')

    # 统一的训练流程
    train_loader = create_dataloader(dataset_info['dst_train'])
    test_loader = create_dataloader(dataset_info['dst_test'])

    # 使用数据集元数据配置模型
    model = create_model(
        input_channels=dataset_info['channel'],
        num_classes=dataset_info['num_classes']
    )

    # 训练...
    train(model, train_loader, test_loader)
```

---

## 📊 返回值说明

### 完整返回字典

`load_dataset_info()` 返回一个字典，包含以下键值：

```python
{
    'dst_train': Dataset,      # PyTorch Dataset 对象（训练集）
    'dst_test': Dataset,       # PyTorch Dataset 对象（测试集）
    'im_size': tuple,          # 图像尺寸 (H, W)，例如 (32, 32)
    'channel': int,            # 通道数，1 (灰度) 或 3 (RGB)
    'num_classes': int,        # 类别数
    'class_names': list,       # 类别名称列表
    'mean': list,              # 归一化均值（每个通道）
    'std': list                # 归一化标准差（每个通道）
}
```

### 字段详解

**核心数据集对象**：
- `dst_train`: 训练集，可直接传给 `DataLoader`
- `dst_test`: 测试集（或验证集），可直接传给 `DataLoader`

**元数据字段**：
- `im_size`: 图像尺寸（高度, 宽度），用于模型输入配置
- `channel`: 通道数，用于模型第一层卷积配置
- `num_classes`: 类别数，用于模型最后一层全连接配置
- `class_names`: 类别名称，用于可视化和报告生成

**归一化参数**（已自动应用）：
- `mean`: 每个通道的均值
- `std`: 每个通道的标准差

---

## 🔧 高级用法

### 1. 扩展支持的数据集

如果需要添加新的数据集，只需在 `utils/data.py` 中修改注册表：

```python
# utils/data.py

_DATASET_REGISTRY = {
    # ... 现有数据集 ...

    'YourDataset': {
        'torchvision_class': datasets.YourDataset,  # torchvision 类
        'im_size': (224, 224),                      # 图像尺寸
        'channel': 3,                               # 通道数
        'num_classes': 1000,                        # 类别数
        'mean': [0.485, 0.456, 0.406],             # ImageNet 均值
        'std': [0.229, 0.224, 0.225],              # ImageNet 标准差
    }
}
```

然后就可以直接使用：

```python
dataset_info = load_dataset_info('YourDataset', './data')
```

### 2. 自定义数据增强

如果需要自定义数据增强，可以在获取数据集后重新设置 `transform`：

```python
from torchvision import transforms
from utils import load_dataset_info

# 先加载数据集（使用默认归一化）
dataset_info = load_dataset_info('CIFAR10', './data')

# 自定义训练集增强
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # 随机裁剪
    transforms.RandomHorizontalFlip(),         # 随机水平翻转
    transforms.ToTensor(),
    transforms.Normalize(
        mean=dataset_info['mean'],
        std=dataset_info['std']
    )
])

# 重新设置变换
dataset_info['dst_train'].transform = train_transform

# 测试集保持默认（已经是归一化）
# dataset_info['dst_test'] 无需修改
```

### 3. 使用自定义数据集

如果使用完全自定义的数据集（不在 torchvision 中），可以：

**方式 1**：直接使用，不调用 `load_dataset_info`

```python
from torch.utils.data import Dataset, DataLoader

class MyCustomDataset(Dataset):
    def __init__(self, ...):
        # 自定义初始化
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 直接创建 DataLoader
train_dataset = MyCustomDataset(...)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

**方式 2**：扩展 `_DATASET_REGISTRY`（推荐，统一接口）

```python
# utils/data.py

from your_module import CustomDataset

_DATASET_REGISTRY['CustomDataset'] = {
    'torchvision_class': CustomDataset,  # 你的自定义类
    'im_size': (256, 256),
    'channel': 3,
    'num_classes': 50,
    'mean': [0.5, 0.5, 0.5],
    'std': [0.5, 0.5, 0.5],
}
```

### 4. 获取数据集统计信息

```python
from utils import load_dataset_info
from loguru import logger

dataset_info = load_dataset_info('CIFAR10', './data')

# 打印完整的数据集信息
logger.info("=" * 60)
logger.info("数据集信息".center(60))
logger.info("=" * 60)
logger.info(f"名称: CIFAR10")
logger.info(f"训练集大小: {len(dataset_info['dst_train'])}")
logger.info(f"测试集大小: {len(dataset_info['dst_test'])}")
logger.info(f"图像尺寸: {dataset_info['im_size'][0]}×{dataset_info['im_size'][1]}")
logger.info(f"通道数: {dataset_info['channel']}")
logger.info(f"类别数: {dataset_info['num_classes']}")
logger.info(f"类别名称: {', '.join(dataset_info['class_names'])}")
logger.info(f"归一化均值: {dataset_info['mean']}")
logger.info(f"归一化标准差: {dataset_info['std']}")
logger.info("=" * 60)
```

---

## ❓ 常见问题

### Q1: 数据集会自动下载吗？

**A**: 是的。如果本地 `data_path` 目录下不存在数据集，`load_dataset_info` 会自动从官方源下载。

```python
# 第一次运行时会下载（可能需要几分钟）
dataset_info = load_dataset_info('CIFAR10', './data')

# 第二次运行时会直接加载本地文件（很快）
dataset_info = load_dataset_info('CIFAR10', './data')
```

### Q2: 下载失败怎么办？

**A**: 可能的原因和解决方案：

1. **网络问题**：
   - 检查网络连接
   - 尝试使用代理或 VPN

2. **磁盘空间不足**：
   - 检查 `data_path` 所在磁盘是否有足够空间
   - CIFAR10 约 170 MB，CIFAR100 约 170 MB

3. **手动下载**：
   - 从镜像站下载数据集
   - 解压到 `data_path` 目录
   - 确保目录结构正确

### Q3: 如何查看支持哪些数据集？

**A**: 有两种方法：

**方法 1**：查看错误提示

```python
from utils import load_dataset_info

try:
    dataset_info = load_dataset_info('UnknownDataset', './data')
except ValueError as e:
    print(e)
    # 输出: 未知的数据集: UnknownDataset。支持的数据集: ['MNIST', 'FashionMNIST', 'CIFAR10', 'CIFAR100']
```

**方法 2**：查看源代码

打开 `utils/data.py`，查看 `_DATASET_REGISTRY` 字典的键。

### Q4: 数据是如何归一化的？

**A**: 使用每个数据集预计算的均值和标准差：

```python
transform = transforms.Compose([
    transforms.ToTensor(),                    # 转换为 Tensor，范围 [0, 1]
    transforms.Normalize(mean=..., std=...)  # 归一化为 N(0, 1)
])
```

**归一化公式**：
```
normalized_value = (original_value - mean) / std
```

**各数据集的归一化参数**：
- **MNIST**: mean=[0.1307], std=[0.3081]
- **FashionMNIST**: mean=[0.2861], std=[0.3530]
- **CIFAR10**: mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
- **CIFAR100**: mean=[0.5071, 0.4866, 0.4409], std=[0.2673, 0.2564, 0.2762]

### Q5: 为什么没有验证集（Validation Set）？

**A**: 默认只返回训练集和测试集。如果需要验证集，可以手动划分：

```python
from torch.utils.data import random_split
from utils import load_dataset_info

dataset_info = load_dataset_info('CIFAR10', './data')

# 将训练集划分为训练集 + 验证集 (80% / 20%)
train_size = int(0.8 * len(dataset_info['dst_train']))
val_size = len(dataset_info['dst_train']) - train_size

train_dataset, val_dataset = random_split(
    dataset_info['dst_train'],
    [train_size, val_size]
)

# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset_info['dst_test'], batch_size=64, shuffle=False)
```

### Q6: 如何获取类别名称？

**A**: 直接从返回字典中获取：

```python
dataset_info = load_dataset_info('CIFAR10', './data')

class_names = dataset_info['class_names']
# ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 使用类别名称
for i, name in enumerate(class_names):
    print(f"类别 {i}: {name}")
```

### Q7: DataLoader 的 num_workers 应该设置为多少？

**A**: 根据平台和 CPU 核心数：

- **Windows**: 建议设置为 `0`（避免多进程问题）
- **Linux/Mac**: 设置为 `2-8`（根据 CPU 核心数）

```yaml
# config.yaml
dataloader:
  num_workers: 0  # Windows
  # num_workers: 4  # Linux/Mac (4-8 核 CPU)
```

**性能提示**：
- `num_workers > 0` 可以并行加载数据，提升训练速度
- 但在 Windows 上可能导致多进程启动问题
- 建议在 Linux 服务器上训练时启用

---

## 📋 最佳实践

### 1. 标准训练流程

```python
from utils import setup_config, load_dataset_info, setup_logging
from torch.utils.data import DataLoader
from loguru import logger

def main():
    # 1. 配置日志
    setup_logging(log_dir='./logs', console_level='INFO', file_level='DEBUG')

    # 2. 加载配置
    logger.info("加载配置...")
    config = setup_config(DEFAULT_CONFIG, 'config.yaml', {})

    # 3. 加载数据集
    logger.info("加载数据集...")
    dataset_info = load_dataset_info(
        dataset_name=config.dataset.name,
        data_path=config.dataset.data_path
    )

    # 4. 创建 DataLoader
    train_loader = DataLoader(
        dataset_info['dst_train'],
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory
    )

    val_loader = DataLoader(
        dataset_info['dst_test'],
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.dataloader.num_workers,
        pin_memory=config.dataloader.pin_memory
    )

    # 5. 创建模型（使用数据集元数据）
    logger.info("创建模型...")
    model = create_model(
        input_channels=dataset_info['channel'],
        num_classes=dataset_info['num_classes']
    )

    # 6. 训练
    logger.info("开始训练...")
    trainer.fit(train_loader, val_loader, epochs=config.training.epochs)

    logger.success("训练完成！")

if __name__ == '__main__':
    main()
```

### 2. 数据集切换最佳实践

将数据集配置放在 YAML 文件中，方便快速切换：

```yaml
# experiments/mnist.yaml
dataset:
  name: "MNIST"
  data_path: "./data"

# experiments/cifar10.yaml
dataset:
  name: "CIFAR10"
  data_path: "./data"
```

运行时指定不同的配置文件：

```bash
# 训练 MNIST
python main.py --config experiments/mnist.yaml

# 训练 CIFAR10
python main.py --config experiments/cifar10.yaml
```

### 3. 记录数据集信息

```python
from utils import load_dataset_info
from loguru import logger

dataset_info = load_dataset_info('CIFAR10', './data')

# 在训练开始时记录关键信息
logger.info("=" * 60)
logger.info("数据集配置".center(60))
logger.info("=" * 60)
logger.info(f"数据集: CIFAR10")
logger.info(f"训练样本数: {len(dataset_info['dst_train'])}")
logger.info(f"测试样本数: {len(dataset_info['dst_test'])}")
logger.info(f"类别数: {dataset_info['num_classes']}")
logger.info("=" * 60)
```

---

## 🎯 总结

`load_dataset_info` 的核心优势：

1. **简单易用**：一行代码加载数据集 + 元数据
2. **自动下载**：无需手动下载和解压
3. **标准归一化**：自动应用最佳归一化参数
4. **元数据丰富**：提供类别名称、图像尺寸等
5. **统一接口**：所有数据集使用相同的 API
6. **易于扩展**：通过注册表模式添加新数据集

配合 `DataLoader` 和配置文件，让数据加载变得轻松愉快！🎉
