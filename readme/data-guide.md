# 数据加载模块使用指南

本文档详细说明 `utils/data.py` 模块中的数据加载工具。

## 核心组件

### 数据集注册表 `_DATASET_REGISTRY`

#### 功能
存储支持的数据集元数据的字典，采用注册表模式分离配置和加载逻辑。

#### 当前支持的数据集

| 数据集 | 图像尺寸 | 通道 | 类别数 | 归一化均值 | 归一化标准差 |
|--------|---------|------|--------|-----------|------------|
| MNIST | 28×28 | 1 | 10 | [0.1307] | [0.3081] |
| FashionMNIST | 28×28 | 1 | 10 | [0.2861] | [0.3530] |
| CIFAR10 | 32×32 | 3 | 10 | [0.4914, 0.4822, 0.4465] | [0.2023, 0.1994, 0.2010] |
| CIFAR100 | 32×32 | 3 | 100 | [0.5071, 0.4866, 0.4409] | [0.2673, 0.2564, 0.2762] |

#### 扩展新数据集

在注册表中添加新的数据集配置：

```python
_DATASET_REGISTRY['YourDataset'] = {
    'torchvision_class': datasets.YourDataset,  # torchvision数据集类
    'im_size': (224, 224),                      # 图像尺寸 (H, W)
    'channel': 3,                               # 通道数
    'num_classes': 1000,                        # 类别数
    'mean': [0.485, 0.456, 0.406],             # 归一化均值（每个通道）
    'std': [0.229, 0.224, 0.225],              # 归一化标准差（每个通道）
}
```

### `load_dataset_info(dataset_name, data_path)` 函数

#### 功能
加载指定数据集并返回包含数据集对象和元数据的字典。

#### 参数
- `dataset_name` (str): 数据集名称，必须在 `_DATASET_REGISTRY` 中
- `data_path` (str): 数据集存储路径

#### 返回值
包含以下键的字典：

| 键 | 类型 | 说明 |
|----|------|------|
| `dst_train` | Dataset | 训练集Dataset对象 |
| `dst_test` | Dataset | 测试集Dataset对象 |
| `im_size` | tuple | 图像尺寸 (H, W) |
| `channel` | int | 通道数 (1: 灰度, 3: RGB) |
| `num_classes` | int | 类别数 |
| `class_names` | list | 类别名称列表 |
| `mean` | list | 归一化均值（每个通道） |
| `std` | list | 归一化标准差（每个通道） |

#### 异常
- `ValueError`: 当 `dataset_name` 不在注册表中时抛出

#### 数据变换
自动应用以下变换：
```python
transforms.Compose([
    transforms.ToTensor(),                    # 转换为Tensor，范围[0, 1]
    transforms.Normalize(mean=..., std=...)  # 归一化为N(0, 1)
])
```

#### 自动下载
如果本地 `data_path` 目录下不存在数据集，会自动从官方源下载。

## 使用示例

### 基础用法

```python
from utils import load_dataset_info
from torch.utils.data import DataLoader

# 加载数据集
dataset_info = load_dataset_info(
    dataset_name='CIFAR10',
    data_path='./data'
)

# 创建DataLoader
train_loader = DataLoader(
    dataset_info['dst_train'],
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    dataset_info['dst_test'],
    batch_size=256,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# 使用元数据
print(f"类别数: {dataset_info['num_classes']}")
print(f"图像尺寸: {dataset_info['im_size']}")
print(f"类别名称: {dataset_info['class_names']}")
```

### 在训练脚本中使用

```python
from utils import load_dataset_info, setup_config
from torch.utils.data import DataLoader

# 从配置加载
config = setup_config(DEFAULT_CONFIG, 'config.yaml', {})

# 加载数据集
dataset_info = load_dataset_info(
    dataset_name=config.dataset.name,
    data_path=config.dataset.data_path
)

# 创建模型（使用数据集元数据）
model = create_model(
    input_channels=dataset_info['channel'],
    num_classes=dataset_info['num_classes']
)

# 创建DataLoader
train_loader = DataLoader(
    dataset_info['dst_train'],
    batch_size=config.training.batch_size,
    shuffle=True,
    num_workers=config.dataloader.num_workers,
    pin_memory=config.dataloader.pin_memory
)
```

### 自定义数据增强

如需自定义数据增强，可在获取数据集后重新设置 `transform`：

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

# 测试集保持默认（已经包含归一化）
# dataset_info['dst_test'] 无需修改
```

## 设计原则

### 职责分离
- `load_dataset_info` 只负责加载数据集和提供元数据
- **不**创建 `DataLoader`，由调用方控制批次大小、workers等参数
- **不**应用数据增强，只应用基础的归一化

### 可扩展性
- 使用注册表模式，添加新数据集只需修改 `_DATASET_REGISTRY`
- 无需修改加载逻辑代码

### 统一接口
- 所有数据集使用相同的API
- 返回值结构统一，便于编写通用的训练代码
