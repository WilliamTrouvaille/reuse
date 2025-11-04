# 指标跟踪模块使用指南

本文档详细说明 `utils/metrics.py` 模块中的指标跟踪工具类。

## 核心类

### `MetricTracker` 类

#### 功能
高性能指标跟踪器，在GPU/设备上累积指标以避免频繁同步，适用于训练和评估循环。

#### 核心优势
- 在训练循环中，所有指标累积都在GPU上进行（非阻塞）
- 只在调用 `compute()` 时才执行一次GPU→CPU同步
- 内存占用O(1)，避免了`torch.cat`导致的OOM风险

#### 初始化

```python
MetricTracker(device, compute_top5=False)
```

**参数**
- `device` (str | torch.device): 累加器所在的设备
- `compute_top5` (bool, 可选): 是否计算Top-5准确率，默认为`False`

#### 方法

**`reset()`**
- 功能：重置所有累加器（通常在每个epoch开始时调用）
- 参数：无
- 返回值：无

**`update(loss, outputs, targets)`**
- 功能：更新累加器，所有操作在GPU上进行（非阻塞）
- 参数：
  - `loss` (Tensor): 当前batch的平均损失（标量）
  - `outputs` (Tensor): 模型输出的logits (shape: [B, C])
  - `targets` (Tensor): 真实标签 (shape: [B])
- 返回值：无
- 注意：此方法在循环内调用，是廉价操作（非阻塞）

**`compute()`**
- 功能：计算最终的平均指标（唯一的同步点）
- 参数：无
- 返回值：包含 `'loss'`, `'acc'`, 和（可选）`'top5'` 的字典
- 注意：此方法在循环后调用，是昂贵操作（阻塞）

#### 使用示例

**基础用法**

```python
from utils import MetricTracker

# 1. 在epoch开始前初始化
tracker = MetricTracker(device=device, compute_top5=True)

# 2. 在循环中更新（廉价操作）
for inputs, labels in loader:
    logits = model(inputs)
    loss = criterion(logits, labels)

    # 更新指标（在GPU上累积）
    tracker.update(loss, logits, labels)

    # 反向传播...
    loss.backward()
    optimizer.step()

# 3. 在epoch结束后计算（昂贵操作）
final_metrics = tracker.compute()
# final_metrics = {'loss': 0.123, 'acc': 95.4, 'top5': 99.8}

# 4. 重置以备下一个epoch
tracker.reset()
```

**在Trainer中使用**

```python
class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # 创建指标跟踪器
        self.metric_tracker = MetricTracker(device=device, compute_top5=False)

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        self.metric_tracker.reset()  # 重置跟踪器

        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 更新指标
            self.metric_tracker.update(loss, outputs, labels)

            # 反向传播
            loss.backward()
            self.optimizer.step()

        # 计算epoch的平均指标
        train_metrics = self.metric_tracker.compute()
        return train_metrics
```

**与Progress结合使用**

```python
from utils import MetricTracker, Progress

# 初始化跟踪器
tracker = MetricTracker(device=device, compute_top5=True)

# 使用Progress显示实时进度
with Progress(train_loader, description="Training") as pbar:
    for images, labels in pbar:
        # 训练代码...
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 更新MetricTracker
        tracker.update(loss, outputs, labels)

        # 更新Progress（显示瞬时值）
        pbar.update({'loss': loss, 'acc': accuracy})

# 获取最终的平均指标
final_metrics = tracker.compute()
logger.info(f"Epoch平均 - Loss: {final_metrics['loss']:.4f}, Acc: {final_metrics['acc']:.2f}%")
```

#### 性能对比

| 方法 | GPU同步次数 | 相对速度 |
|------|-----------|---------|
| 每个batch调用`.item()` | 每个batch | 1.0× |
| `MetricTracker` | 每个epoch一次 | 1.3-1.5× |

### `AverageMeter` 类

#### 功能
轻量级的平均值计算器，用于跟踪CPU标量（例如学习率、数据加载时间）。

#### 注意事项
- 每次`update()`都会同步（`.item()`）
- 不要在GPU循环的热路径中使用此类
- 仅用于非Tensor值的跟踪

#### 初始化

```python
AverageMeter()
```

无参数。

#### 属性
- `val`: 最新的值
- `avg`: 平均值
- `sum`: 累积和
- `count`: 样本数量

#### 方法

**`reset()`**
- 功能：重置所有统计量
- 参数：无
- 返回值：无

**`update(val, n=1)`**
- 功能：更新统计量
- 参数：
  - `val` (float | Tensor): 要添加的值（如果是Tensor，会自动`.item()`）
  - `n` (int, 可选): 样本数量（用于加权平均），默认为`1`
- 返回值：无

#### 使用示例

**跟踪学习率**

```python
from utils import AverageMeter

lr_meter = AverageMeter()

for epoch in range(100):
    for batch_idx, (images, labels) in enumerate(train_loader):
        # 训练代码...

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        lr_meter.update(current_lr)

    logger.info(f"Epoch {epoch} 平均学习率: {lr_meter.avg:.6f}")
    lr_meter.reset()  # 下一个epoch前重置
```

**跟踪数据加载时间**

```python
import time
from utils import AverageMeter

data_time = AverageMeter()
batch_time = AverageMeter()

end = time.time()

for images, labels in train_loader:
    # 测量数据加载时间
    data_time.update(time.time() - end)

    # 训练代码...
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # 测量批次总时间
    batch_time.update(time.time() - end)
    end = time.time()

logger.info(f"平均数据加载时间: {data_time.avg:.3f}s")
logger.info(f"平均批次时间: {batch_time.avg:.3f}s")
```

**加权平均**

```python
from utils import AverageMeter

# 跟踪加权损失
weighted_loss = AverageMeter()

for images, labels in train_loader:
    batch_size = images.size(0)
    outputs = model(images)
    loss = criterion(outputs, labels)

    # 使用batch_size作为权重
    weighted_loss.update(loss.item(), n=batch_size)

logger.info(f"加权平均损失: {weighted_loss.avg:.4f}")
```

## 使用建议

### 何时使用 MetricTracker
- 在训练/评估循环中跟踪Tensor指标（loss, accuracy等）
- 需要最大化GPU性能
- 需要整个epoch的准确平均值

### 何时使用 AverageMeter
- 跟踪非Tensor标量（学习率、时间等）
- 需要简单的平均值计算
- 性能不是关键因素

## 完整示例：训练循环

```python
from utils import MetricTracker, AverageMeter
from loguru import logger
import time

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()

    # 使用MetricTracker跟踪Tensor指标
    metric_tracker = MetricTracker(device=device, compute_top5=False)
    metric_tracker.reset()

    # 使用AverageMeter跟踪标量
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        # 数据加载时间
        data_time.update(time.time() - end)

        images, labels = images.to(device), labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 更新MetricTracker（Tensor指标）
        metric_tracker.update(loss, outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 批次时间
        batch_time.update(time.time() - end)
        end = time.time()

    # 计算epoch的平均指标
    train_metrics = metric_tracker.compute()

    # 记录日志
    logger.info(
        f"Epoch {epoch} | "
        f"Loss: {train_metrics['loss']:.4f} | "
        f"Acc: {train_metrics['acc']:.2f}% | "
        f"Batch Time: {batch_time.avg:.3f}s | "
        f"Data Time: {data_time.avg:.3f}s"
    )

    return train_metrics

def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()

    # 使用MetricTracker
    metric_tracker = MetricTracker(device=device, compute_top5=True)
    metric_tracker.reset()

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            metric_tracker.update(loss, outputs, labels)

    val_metrics = metric_tracker.compute()

    logger.info(
        f"Validation | "
        f"Loss: {val_metrics['loss']:.4f} | "
        f"Acc: {val_metrics['acc']:.2f}% | "
        f"Top5: {val_metrics['top5']:.2f}%"
    )

    return val_metrics
```

## 设计原则

### MetricTracker
- 在GPU上累积，最小化同步
- 内存高效（O(1)）
- 适合高性能训练循环

### AverageMeter
- 简单实用，适合CPU标量
- 不适合GPU循环的热路径
- 提供加权平均功能
