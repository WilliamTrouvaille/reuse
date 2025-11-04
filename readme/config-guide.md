# 配置管理模块使用指南

本文档详细说明 `utils/config.py` 模块中的配置管理工具类和函数。

## 核心类与函数

### `ConfigNamespace` 类

#### 功能
将字典转换为可通过属性访问的对象。

#### 方法

**`__init__(config_dict: dict)`**
- 使用字典初始化ConfigNamespace对象
- 支持嵌套字典的递归转换

**`.to_dict() -> dict`**
- 将ConfigNamespace对象递归转换回字典
- 用于保存配置或序列化

**`.get(key: str, default=None) -> Any`**
- 安全地获取属性值
- 类似字典的`.get()`方法
- 当属性不存在时返回default值

**`.update(new_config_dict: dict)`**
- 使用新字典递归更新现有ConfigNamespace实例
- 支持部分更新，仅修改指定的键值对

### `setup_config(default_config, yaml_config_path, cmd_args)` 函数

#### 功能
编排配置加载流程，按优先级合并配置源。

#### 配置优先级
```
命令行参数 > YAML 文件 > 默认配置
```

#### 参数
- `default_config` (dict): 项目代码中定义的默认配置字典
- `yaml_config_path` (str): YAML配置文件的路径
- `cmd_args` (dict): `argparse`解析后的参数字典 (`vars(args)`)

#### 返回值
`ConfigNamespace` 实例

### `load_config_from_yaml(config_path)` 函数

#### 功能
从YAML文件加载配置。

#### 参数
- `config_path` (str): YAML文件的路径

#### 返回值
- 成功：包含配置的字典
- 失败：空字典 `{}`

### `save_config_to_yaml(config, config_path)` 函数

#### 功能
将配置保存到YAML文件。

#### 参数
- `config` (dict | ConfigNamespace): 要保存的配置对象
- `config_path` (str): 目标YAML文件的路径

#### 注意事项
- 自动处理ConfigNamespace到dict的转换
- 自动创建必要的目录

### `print_config(config, title)` 函数

#### 功能
以美观格式打印配置到日志。

#### 参数
- `config` (dict | ConfigNamespace): 要打印的配置对象
- `title` (str, 可选): 打印输出的标题

#### 输出级别
INFO级别（通过loguru）

## 核心配置项说明

### 实验配置 (experiment)

```yaml
experiment:
  name: "mnist_baseline"  # 实验名称，用于日志文件名和通知
  seed: 42                # 随机种子，确保可复现
  description: "..."      # 实验描述（可选）
  tags: ["tag1", "tag2"]  # 标签，便于管理（可选）
```

**使用场景**
- `name`: 会出现在日志文件名中
- `seed`: 设置Python/NumPy/PyTorch的全局随机种子
- `tags`: 用于实验管理工具（如MLflow, Weights & Biases）

### 数据集配置 (dataset)

```yaml
dataset:
  name: "MNIST"           # 支持: MNIST, FashionMNIST, CIFAR10, CIFAR100
  data_path: "./data"     # 数据存储路径
```

**添加自定义数据集**

在 `utils/data.py` 的 `_DATASET_REGISTRY` 中注册：

```python
_DATASET_REGISTRY['MyDataset'] = {
    'torchvision_class': datasets.MyDataset,
    'im_size': (224, 224),
    'channel': 3,
    'num_classes': 100,
    'mean': [0.5, 0.5, 0.5],
    'std': [0.5, 0.5, 0.5],
}
```

### 数据加载器配置 (dataloader)

```yaml
dataloader:
  batch_size: 128          # 训练批次大小
  eval_batch_size: 256     # 评估批次大小（通常是训练的2倍）
  num_workers: 4           # 数据加载线程数
  pin_memory: true         # 是否固定内存（GPU训练必开）
  persistent_workers: true # 保持工作进程常驻
```

**性能调优建议**

| 硬件配置 | num_workers | batch_size | pin_memory |
|---------|-------------|------------|------------|
| RTX 3060 (12GB) | 4 | 128 | true |
| RTX 4090 (24GB) | 8 | 256 | true |
| CPU Only | 0 | 64 | false |
| Windows 系统 | 0 | 128 | true |

### 训练配置 (training)

#### 基础配置

```yaml
training:
  epochs: 100              # 训练轮数
  lr: 0.001               # 初始学习率
  optimizer: "AdamW"      # 优化器类型
  criterion: "CrossEntropyLoss"  # 损失函数
```

**常用优化器配置**

```yaml
# AdamW (推荐)
optimizer: "AdamW"
optimizer_params:
  weight_decay: 0.01
  betas: [0.9, 0.999]

# SGD with Momentum
optimizer: "SGD"
optimizer_params:
  momentum: 0.9
  nesterov: true
  weight_decay: 0.0001
```

#### 性能优化

```yaml
training:
  use_amp: true           # 自动混合精度（推荐GPU训练开启）
  grad_accum_steps: 2     # 梯度累积（模拟更大batch）
  max_grad_norm: 1.0      # 梯度裁剪（RNN/Transformer必需）
```

**显存不足解决方案**

| 方法 | 配置 | 显存节省 | 性能影响 |
|------|------|---------|---------|
| 启用 AMP | `use_amp: true` | ~30% | +10% 速度 |
| 梯度累积 | `grad_accum_steps: 4` | ~75% | 轻微减速 |
| 减小 batch | `batch_size: 64` | 50% | 可能降低精度 |
| 混合使用 | 以上三者结合 | ~80% | 整体加速 |

#### 学习率调度

```yaml
training:
  scheduler: "CosineAnnealingLR"
  scheduler_params:
    T_max: 100            # 周期长度（通常设为总epochs）
    eta_min: 0.00001      # 最小学习率
```

**常用调度器配置**

```yaml
# 1. 余弦退火 (推荐)
scheduler: "CosineAnnealingLR"
scheduler_params:
  T_max: 100
  eta_min: 0.00001

# 2. 阶梯式下降
scheduler: "StepLR"
scheduler_params:
  step_size: 30          # 每30个epoch降低一次
  gamma: 0.1             # 学习率变为原来的0.1倍

# 3. 基于指标的自适应调整
scheduler: "ReduceLROnPlateau"
scheduler_params:
  mode: 'max'            # 跟踪指标的模式
  factor: 0.1            # 降低因子
  patience: 10           # 耐心值
  threshold: 0.0001      # 最小改善阈值

# 4. OneCycleLR (快速收敛)
scheduler: "OneCycleLR"
scheduler_params:
  max_lr: 0.01
  total_steps: 50000     # 总训练步数
  pct_start: 0.3         # 上升阶段占比
```

#### 早停配置

```yaml
training:
  patience: 10           # 连续10个epoch不改善就停止
  min_delta: 0.001       # 最小改善阈值
  metric_to_track: "acc" # 跟踪的指标
  metric_mode: "max"     # 'max'(准确率) 或 'min'(损失)
```

**早停策略建议**

| 数据集规模 | patience | min_delta |
|-----------|----------|-----------|
| 小型 (<10k) | 5-10 | 0.01 |
| 中型 (10k-100k) | 10-20 | 0.001 |
| 大型 (>100k) | 20-50 | 0.0001 |

### 检查点配置 (checkpoint)

```yaml
checkpoint:
  save_dir: "./checkpoints"  # 存储目录
  max_to_keep: 3            # 保留最近3个epoch检查点
  auto_resume: true         # 自动恢复训练
```

**检查点文件说明**

```
checkpoints/
├── best_model.pth              # 最佳模型（验证指标最好）
├── checkpoint_epoch_97.pth     # 第97个epoch的检查点
├── checkpoint_epoch_98.pth     # 第98个epoch的检查点
├── checkpoint_epoch_99.pth     # 第99个epoch的检查点
└── interrupt_checkpoint.pth    # 中断检查点（Ctrl+C后保存）
```

### Ntfy 通知配置 (ntfy)

```yaml
ntfy:
  enabled: true                          # 是否启用通知
  server_url: "https://ntfy.sh"          # 服务器地址
  topic: "your_unique_topic_name_here"   # 通知主题（需修改）
```

**设置步骤**
1. 手机安装 Ntfy App（iOS/Android）
2. 修改 `topic` 为你的唯一名称（例如: `trouvaille_ml_project_xyz123`）
3. 在 App 中订阅相同的 topic
4. 开始训练，你会收到通知

**通知示例**
- 训练开始: "训练开始 Epochs: 1 → 100"
- 训练成功: "训练已正常完成 总轮数: 100 最佳指标: 0.9845"
- 训练失败: "训练失败: RuntimeError ..."

## 常见使用场景

### 场景 1: 快速实验（使用默认配置）

```yaml
experiment:
  name: "quick_test"

training:
  epochs: 10
  lr: 0.001

checkpoint:
  save_dir: "./test_checkpoints"

ntfy:
  enabled: false
```

### 场景 2: 高性能训练（大batch + AMP）

```yaml
dataloader:
  batch_size: 256
  num_workers: 8
  pin_memory: true

training:
  use_amp: true
  grad_accum_steps: 1
  optimizer: "AdamW"
  scheduler: "OneCycleLR"
  scheduler_params:
    max_lr: 0.01
    total_steps: 50000

advanced:
  cudnn_benchmark: true
```

### 场景 3: 显存受限（小batch + 梯度累积）

```yaml
dataloader:
  batch_size: 32

training:
  use_amp: true
  grad_accum_steps: 8  # 有效batch = 32 × 8 = 256
  max_grad_norm: 1.0

advanced:
  gradient_checkpointing: true
```

### 场景 4: 调试模式

```yaml
training:
  epochs: 5
  log_interval: 1

logging:
  console_level: "DEBUG"
  file_level: "DEBUG"

advanced:
  anomaly_detection: true
  deterministic: true

ntfy:
  enabled: false
```

### 场景 5: 分布式训练（多GPU）

```yaml
device:
  type: "cuda"
  multi_gpu:
    enabled: true
    device_ids: [0, 1, 2, 3]

dataloader:
  batch_size: 64  # 每个GPU的batch size
  num_workers: 4  # 每个GPU的worker数

training:
  use_amp: true
```

## 命令行参数覆盖

### 基础用法

```bash
# 覆盖单个参数
python main.py --training.epochs 200

# 覆盖多个参数
python main.py \
  --training.epochs 200 \
  --training.lr 0.01 \
  --dataloader.batch_size 256

# 指定配置文件
python main.py --config my_experiment.yaml

# 组合使用
python main.py \
  --config baseline.yaml \
  --training.epochs 50 \
  --experiment.name "baseline_v2"
```

### 高级用法

```bash
# 禁用某些功能
python main.py \
  --training.use_amp false \
  --ntfy.enabled false

# 切换优化器
python main.py \
  --training.optimizer SGD \
  --training.optimizer_params.momentum 0.9

# 修改嵌套配置
python main.py \
  --training.scheduler_params.T_max 50 \
  --training.scheduler_params.eta_min 0.00001
```
