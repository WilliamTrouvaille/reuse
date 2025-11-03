# 配置文件使用指南

本文档详细说明如何使用和定制 `config.yaml` 配置文件。

---

## 快速开始

### 1. 创建配置文件

~~~bash
# 复制模板文件
cp config.yaml.example config.yaml

# 根据需求修改配置
vim config.yaml  # 或使用任何文本编辑器
~~~

### 2. 在代码中使用

~~~python
from utils import setup_config

# 方式1: 使用默认路径
config = setup_config(
    default_config=get_project_defaults(),
    yaml_config_path='config.yaml',
    cmd_args={}
)

# 方式2: 通过命令行指定
# python main.py --config my_config.yaml
config = setup_config(
    default_config=get_project_defaults(),
    yaml_config_path=cmd_args['config'],
    cmd_args=cmd_args
)

# 访问配置
print(config.training.epochs)  # 100
print(config.dataset.name)     # 'MNIST'
~~~

---

## 配置优先级

配置的加载遵循以下优先级（从高到低）：

```
命令行参数 > config.yaml > 代码默认值
```

### 示例

**config.yaml:**
~~~yaml
training:
  epochs: 100
  lr: 0.001
~~~

**命令行:**
~~~bash
python main.py --training.epochs 200 --training.lr 0.01
~~~

**最终结果:**
- `config.training.epochs = 200`  # 命令行覆盖
- `config.training.lr = 0.01`     # 命令行覆盖
- `config.training.batch_size = 128`  # 使用代码默认值

---

## 核心配置项详解

### 1. 实验配置 (experiment)

~~~yaml
experiment:
  name: "mnist_baseline"  # 实验名称，用于日志文件名和通知
  seed: 42                # 随机种子，确保可复现
  description: "..."      # 实验描述（可选）
  tags: ["tag1", "tag2"]  # 标签，便于管理（可选）
~~~

**使用场景:**
- `name`: 会出现在日志文件名中（`log_mnist_baseline_20251103.log`）
- `seed`: 设置 Python/NumPy/PyTorch 的全局随机种子
- `tags`: 用于实验管理工具（如 MLflow, Weights & Biases）

### 2. 数据集配置 (dataset)

~~~yaml
dataset:
  name: "MNIST"           # 支持: MNIST, FashionMNIST, CIFAR10, CIFAR100
  data_path: "./data"     # 数据存储路径
~~~

**添加自定义数据集:**

在 `utils/data.py` 的 `_DATASET_REGISTRY` 中注册：

~~~python
_DATASET_REGISTRY['MyDataset'] = {
    'torchvision_class': datasets.MyDataset,
    'im_size': (224, 224),
    'channel': 3,
    'num_classes': 100,
    'mean': [0.5, 0.5, 0.5],
    'std': [0.5, 0.5, 0.5],
}
~~~

### 3. 数据加载器配置 (dataloader)

~~~yaml
dataloader:
  batch_size: 128          # 训练批次大小
  eval_batch_size: 256     # 评估批次大小（通常是训练的2倍）
  num_workers: 4           # 数据加载线程数
  pin_memory: true         # 是否固定内存（GPU训练必开）
  persistent_workers: true # 保持工作进程常驻
~~~

**性能调优建议:**

| 硬件配置 | num_workers | batch_size | pin_memory |
|---------|-------------|------------|------------|
| RTX 3060 (12GB) | 4 | 128 | true |
| RTX 4090 (24GB) | 8 | 256 | true |
| CPU Only | 0 | 64 | false |
| Windows 系统 | 0 | 128 | true |

### 4. 训练配置 (training)

#### 4.1 基础配置

~~~yaml
training:
  epochs: 100              # 训练轮数
  lr: 0.001               # 初始学习率
  optimizer: "AdamW"      # 优化器类型
  criterion: "CrossEntropyLoss"  # 损失函数
~~~

**常用优化器配置:**

~~~yaml
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
~~~

#### 4.2 性能优化

~~~yaml
training:
  use_amp: true           # 自动混合精度（推荐GPU训练开启）
  grad_accum_steps: 2     # 梯度累积（模拟更大batch）
  max_grad_norm: 1.0      # 梯度裁剪（RNN/Transformer必需）
~~~

**显存不足解决方案:**

| 方法 | 配置 | 显存节省 | 性能影响 |
|------|------|---------|---------|
| 启用 AMP | `use_amp: true` | ~30% | +10% 速度 |
| 梯度累积 | `grad_accum_steps: 4` | ~75% | 轻微减速 |
| 减小 batch | `batch_size: 64` | 50% | 可能降低精度 |
| 混合使用 | 以上三者结合 | ~80% | 整体加速 |

#### 4.3 学习率调度

~~~yaml
training:
  scheduler: "CosineAnnealingLR"
  scheduler_params:
    T_max: 100            # 周期长度（通常设为总epochs）
    eta_min: 0.00001      # 最小学习率
~~~

**常用调度器配置:**

~~~yaml
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
~~~

#### 4.4 早停配置

~~~yaml
training:
  patience: 10           # 连续10个epoch不改善就停止
  min_delta: 0.001       # 最小改善阈值
  metric_to_track: "acc" # 跟踪的指标
  metric_mode: "max"     # 'max'(准确率) 或 'min'(损失)
~~~

**早停策略建议:**

| 数据集规模 | patience | min_delta |
|-----------|----------|-----------|
| 小型 (<10k) | 5-10 | 0.01 |
| 中型 (10k-100k) | 10-20 | 0.001 |
| 大型 (>100k) | 20-50 | 0.0001 |

### 5. 检查点配置 (checkpoint)

~~~yaml
checkpoint:
  save_dir: "./checkpoints"  # 存储目录
  max_to_keep: 3            # 保留最近3个epoch检查点
  auto_resume: true         # 自动恢复训练
~~~

**检查点文件说明:**

```
checkpoints/
├── best_model.pth              # 最佳模型（验证指标最好）
├── checkpoint_epoch_97.pth     # 第97个epoch的检查点
├── checkpoint_epoch_98.pth     # 第98个epoch的检查点
├── checkpoint_epoch_99.pth     # 第99个epoch的检查点
└── interrupt_checkpoint.pth    # 中断检查点（Ctrl+C后保存）
```

### 6. Ntfy 通知配置 (ntfy)

~~~yaml
ntfy:
  enabled: true                          # 是否启用通知
  server_url: "https://ntfy.sh"          # 服务器地址
  topic: "your_unique_topic_name_here"   # 通知主题（需修改）
~~~

**设置步骤:**

1. 手机安装 Ntfy App（iOS/Android）
2. 修改 `topic` 为你的唯一名称（例如: `trouvaille_ml_project_xyz123`）
3. 在 App 中订阅相同的 topic
4. 开始训练，你会收到通知！

**通知示例:**

- 🏃 **训练开始**: "训练开始 Epochs: 1 → 100"
- ✅ **训练成功**: "训练已正常完成 总轮数: 100 最佳指标: 0.9845"
- ❌ **训练失败**: "训练失败: RuntimeError ..."

---

## 常见使用场景

### 场景 1: 快速实验（使用默认配置）

**config.yaml:**
~~~yaml
experiment:
  name: "quick_test"

training:
  epochs: 10
  lr: 0.001

checkpoint:
  save_dir: "./test_checkpoints"

ntfy:
  enabled: false
~~~

### 场景 2: 高性能训练（大batch + AMP）

**config.yaml:**
~~~yaml
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
~~~

### 场景 3: 显存受限（小batch + 梯度累积）

**config.yaml:**
~~~yaml
dataloader:
  batch_size: 32

training:
  use_amp: true
  grad_accum_steps: 8  # 有效batch = 32 × 8 = 256
  max_grad_norm: 1.0

advanced:
  gradient_checkpointing: true
~~~

### 场景 4: 调试模式

**config.yaml:**
~~~yaml
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
~~~

### 场景 5: 分布式训练（多GPU）

**config.yaml:**
~~~yaml
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
~~~

---

## 命令行参数覆盖

### 基础用法

~~~bash
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
~~~

### 高级用法

~~~bash
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
~~~

---

## 配置验证与调试

### 1. 打印当前配置

~~~python
from utils import print_config

config = setup_config(...)
print_config(config, title="当前使用的配置")
~~~

**输出示例:**
```
============================================================
当前使用的配置
============================================================
experiment:
  name: mnist_baseline
  seed: 42
training:
  epochs: 100
  lr: 0.001
  ...
============================================================
```

### 2. 保存运行时配置

~~~python
from utils import save_config_to_yaml

# 在训练开始前保存完整配置
save_config_to_yaml(config, './logs/run_config.yaml')
~~~

这样可以确保每次运行都有配置的完整记录。

### 3. 配置验证

~~~python
def validate_config(config):
    """验证配置的合法性"""
    
    # 检查必需字段
    assert hasattr(config, 'training'), "缺少 training 配置"
    assert hasattr(config, 'dataset'), "缺少 dataset 配置"
    
    # 检查参数范围
    assert config.training.epochs > 0, "epochs 必须大于 0"
    assert config.training.lr > 0, "lr 必须大于 0"
    assert config.training.patience > 0, "patience 必须大于 0"
    
    # 检查文件路径
    import os
    os.makedirs(config.checkpoint.save_dir, exist_ok=True)
    os.makedirs(config.logging.log_dir, exist_ok=True)
    
    logger.info("配置验证通过")

# 在 main.py 中使用
config = setup_config(...)
validate_config(config)
~~~

---

## 常见问题

### Q1: 如何添加自定义配置项？

**A:** 直接在 `config.yaml` 中添加，然后在代码中访问：

~~~yaml
# config.yaml
my_custom_config:
  param1: 100
  param2: "hello"
~~~

~~~python
# main.py
print(config.my_custom_config.param1)  # 100
print(config.my_custom_config.param2)  # "hello"
~~~

### Q2: 配置文件太长，如何组织？

**A:** 使用 YAML 的锚点和引用功能：

~~~yaml
# 定义公共配置
common_training: &common_training
  use_amp: true
  grad_accum_steps: 2

# 实验1
experiment1:
  training:
    <<: *common_training
    epochs: 100
    lr: 0.001

# 实验2
experiment2:
  training:
    <<: *common_training
    epochs: 200
    lr: 0.01
~~~

### Q3: 如何管理多个实验的配置？

**A:** 为每个实验创建单独的配置文件：

```
configs/
├── baseline.yaml
├── experiment1_high_lr.yaml
├── experiment2_large_batch.yaml
└── experiment3_augmentation.yaml
```

运行时指定：
~~~bash
python main.py --config configs/experiment1_high_lr.yaml
~~~

### Q4: 命令行参数不生效？

**A:** 检查以下几点：

1. 参数名是否正确（使用点分隔）
2. 是否在 `parse_arguments()` 中定义
3. 配置优先级是否正确

~~~python
# argparse 定义
parser.add_argument('--training.epochs', type=int)

# 命令行使用
python main.py --training.epochs 100  # ✅ 正确
python main.py --epochs 100           # ❌ 错误
~~~

---

## 最佳实践

1. **版本控制**: 将 `config.yaml.example` 提交到 git，`.gitignore` 中排除 `config.yaml`

2. **文档化**: 在配置文件中添加详细注释，说明每个参数的作用

3. **模块化**: 将不同类型的配置分组，保持结构清晰

4. **默认值**: 在代码中提供合理的默认值，配置文件只覆盖需要修改的部分

5. **验证**: 在训练开始前验证配置的合法性

6. **记录**: 每次训练都保存完整的配置快照

7. **实验管理**: 使用有意义的 `experiment.name` 和 `tags`

---

## 总结

配置文件是实验的"蓝图"，合理使用可以：

- ✅ 提高实验的可复现性
- ✅ 简化参数调优流程
- ✅ 便于团队协作和知识共享
- ✅ 支持快速切换实验配置

记住配置优先级：**命令行 > YAML > 代码默认值**