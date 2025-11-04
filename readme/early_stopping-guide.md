# 早停模块使用指南

本文档详细说明 `utils/early_stopping.py` 模块中的早停工具类。

## 核心类

### `EarlyStopper` 类

#### 功能
封装早停（Early Stopping）逻辑，在训练期间监控验证指标，当指标连续N个Epoch未改善时自动停止训练，防止过拟合。

#### 核心特性
- **双模式支持**：支持最大化指标（accuracy）和最小化指标（loss）
- **阈值判断**：支持 `min_delta` 阈值，避免微小波动被误判为改善
- **状态持久化**：支持保存/加载状态，恢复训练时保持早停器状态一致
- **智能日志**：自动记录指标改善/恶化情况和早停触发信息

#### 初始化

```python
EarlyStopper(patience, mode='max', min_delta=0.0, verbose=True)
```

**参数**
- `patience` (int): 触发早停前允许指标不改善的最大Epoch数
- `mode` (Literal['min', 'max'], 可选): 监控模式，默认为 `'max'`
  - `'max'`: 监控需要最大化的指标（如准确率），当指标停止增加时触发
  - `'min'`: 监控需要最小化的指标（如损失），当指标停止减少时触发
- `min_delta` (float, 可选): 被视为"显著改善"的最小变化量，默认为 `0.0`
- `verbose` (bool, 可选): 是否打印详细日志，默认为 `True`

**特殊行为**
- 当 `patience <= 0` 时，早停功能被禁用
- `mode` 参数仅接受 `'min'` 或 `'max'`，否则抛出 `ValueError`

#### 属性

**`should_stop`** (只读)
- 类型：`bool`
- 功能：检查是否应触发早停
- 返回值：
  - `True`: 应该停止训练（连续未改善的Epoch数达到 `patience`）
  - `False`: 继续训练
- 特性：首次触发时自动记录 CRITICAL 级别日志

#### 方法

**`step(current_metric)`**
- 功能：在每个验证Epoch后调用，更新早停状态
- 参数：
  - `current_metric` (float): 最新的验证指标值（如 `val_acc` 或 `val_loss`）
- 返回值：`bool` - 是否为新的最佳模型
  - `True`: 当前Epoch是新的最佳模型（指标显著改善）
  - `False`: 不是最佳模型（指标未改善或改善不显著）
- 副作用：
  - 更新内部的 `best_metric` 和 `patience_counter`
  - 记录指标改善/恶化的日志

**`state_dict()`**
- 功能：返回早停器的状态字典，用于保存到检查点
- 参数：无
- 返回值：包含以下键的字典
  - `patience_counter`: 当前连续未改善的Epoch数
  - `best_metric`: 历史最佳指标值
  - `should_stop`: 是否已触发早停
  - `patience`: 耐心值（超参数）
  - `mode`: 模式（超参数）
  - `min_delta`: 最小变化量（超参数）

**`load_state_dict(state_dict)`**
- 功能：从检查点加载早停器的状态
- 参数：
  - `state_dict` (dict): 状态字典（通常由 `state_dict()` 生成）
- 返回值：无
- 特性：同时恢复内部状态和超参数

## 使用示例

### 基础用法

```python
from utils import EarlyStopper
from loguru import logger

# 初始化早停器
stopper = EarlyStopper(
    patience=10,        # 连续10个epoch未改善则停止
    mode='max',         # 监控准确率（最大化）
    min_delta=0.001     # 至少提升0.1%才算改善
)

best_acc = 0.0

for epoch in range(100):
    # 训练
    train_loss = train_one_epoch(model, train_loader)

    # 验证
    val_loss, val_acc = validate(model, val_loader)

    # 更新早停器
    is_best = stopper.step(val_acc)

    # 如果是最佳模型，保存
    if is_best:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        logger.success(f"新的最佳模型！验证准确率: {val_acc:.2f}%")

    # 检查是否应停止
    if stopper.should_stop:
        logger.info(f"早停触发！最佳验证准确率: {best_acc:.2f}%")
        break
```

### 监控损失（min模式）

```python
from utils import EarlyStopper

# 监控验证损失（最小化）
stopper = EarlyStopper(
    patience=15,
    mode='min',         # 监控损失（最小化）
    min_delta=0.0001    # 至少减少0.0001才算改善
)

for epoch in range(100):
    train_loss = train_one_epoch(model, train_loader)
    val_loss = validate(model, val_loader)

    # 传入验证损失
    is_best = stopper.step(val_loss)

    if is_best:
        torch.save(model.state_dict(), 'best_model.pth')

    if stopper.should_stop:
        logger.info("早停触发！")
        break
```

### 与CheckpointManager集成

```python
from utils import EarlyStopper, CheckpointManager

stopper = EarlyStopper(patience=10, mode='max')
ckpt_mgr = CheckpointManager(save_dir='./checkpoints', device='cuda')

for epoch in range(100):
    # 训练和验证
    train_metrics = train_one_epoch(model, train_loader)
    val_metrics = validate(model, val_loader)

    # 早停器判断是否为最佳
    is_best = stopper.step(val_metrics['acc'])

    # 构建状态字典
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_acc': val_metrics['acc']
    }

    # 保存Epoch检查点
    ckpt_mgr.save_epoch_checkpoint(state, epoch)

    # 如果是最佳模型，保存最佳模型
    if is_best:
        ckpt_mgr.save_best_model(state, metric=val_metrics['acc'])

    # 检查早停
    if stopper.should_stop:
        logger.info("早停触发！")
        break
```

### 状态持久化（恢复训练）

```python
from utils import EarlyStopper, CheckpointManager

# 初始化
stopper = EarlyStopper(patience=10, mode='max')
ckpt_mgr = CheckpointManager(save_dir='./checkpoints', device='cuda')

start_epoch = 0

# 尝试恢复训练
checkpoint = ckpt_mgr.load_latest_checkpoint()
if checkpoint:
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1

    # 恢复早停器状态
    if 'early_stopper' in checkpoint:
        stopper.load_state_dict(checkpoint['early_stopper'])
        logger.info("早停器状态已恢复")

# 训练循环
for epoch in range(start_epoch, 100):
    train_metrics = train_one_epoch(model, train_loader)
    val_metrics = validate(model, val_loader)

    is_best = stopper.step(val_metrics['acc'])

    # 保存检查点（包含早停器状态）
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_acc': val_metrics['acc'],
        'early_stopper': stopper.state_dict()  # 保存早停器状态
    }

    ckpt_mgr.save_epoch_checkpoint(state, epoch)

    if is_best:
        ckpt_mgr.save_best_model(state, metric=val_metrics['acc'])

    if stopper.should_stop:
        logger.info("早停触发！")
        break
```

### 在Trainer类中使用

```python
from utils import EarlyStopper, CheckpointManager

class Trainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

        # 初始化早停器
        self.stopper = EarlyStopper(
            patience=15,
            mode='max',
            min_delta=0.001,
            verbose=True
        )

        # 初始化检查点管理器
        self.ckpt_mgr = CheckpointManager(
            save_dir='./checkpoints',
            device=device,
            max_to_keep=5
        )

        self.start_epoch = 0

    def resume_training(self):
        """恢复训练"""
        checkpoint = self.ckpt_mgr.load_latest_checkpoint()

        if checkpoint:
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch'] + 1

            # 恢复早停器状态
            if 'early_stopper' in checkpoint:
                self.stopper.load_state_dict(checkpoint['early_stopper'])

            logger.info(f"训练已恢复，从 Epoch {self.start_epoch} 开始")

    def save_checkpoint(self, epoch, val_metrics):
        """保存检查点"""
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'val_acc': val_metrics['acc'],
            'early_stopper': self.stopper.state_dict()  # 持久化早停器
        }

        # 保存Epoch检查点
        self.ckpt_mgr.save_epoch_checkpoint(state, epoch)

        # 判断是否为最佳模型
        is_best = self.stopper.step(val_metrics['acc'])
        if is_best:
            self.ckpt_mgr.save_best_model(state, metric=val_metrics['acc'])

        return is_best

    def fit(self, train_loader, val_loader, epochs):
        """训练主函数"""
        # 尝试恢复
        self.resume_training()

        for epoch in range(self.start_epoch, epochs):
            # 训练和验证
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            # 保存检查点
            is_best = self.save_checkpoint(epoch, val_metrics)

            # 检查早停
            if self.stopper.should_stop:
                logger.critical(f"早停触发！在 Epoch {epoch} 停止训练。")
                break

        logger.success("训练完成！")
```

### 禁用早停

```python
from utils import EarlyStopper

# 设置 patience <= 0 禁用早停
stopper = EarlyStopper(patience=0, mode='max')

for epoch in range(100):
    val_acc = validate(model, val_loader)

    # step() 始终返回 False
    is_best = stopper.step(val_acc)

    # should_stop 始终返回 False
    if stopper.should_stop:  # 永远不会执行
        break
```

## min_delta 参数说明

`min_delta` 参数用于定义"显著改善"的阈值，避免微小波动被误判。

### 不使用 min_delta（默认）

```python
stopper = EarlyStopper(patience=5, mode='max', min_delta=0.0)

# Epoch 1: val_acc = 90.0% → 最佳模型
# Epoch 2: val_acc = 90.01% → 最佳模型（改善了0.01%）
# Epoch 3: val_acc = 90.02% → 最佳模型（改善了0.01%）
```

### 使用 min_delta 过滤噪声

```python
stopper = EarlyStopper(patience=5, mode='max', min_delta=0.001)  # 至少提升0.1%

# Epoch 1: val_acc = 90.0% → 最佳模型
# Epoch 2: val_acc = 90.05% → 未改善（0.05% < 0.1%），patience = 1
# Epoch 3: val_acc = 90.15% → 最佳模型（改善了0.15% > 0.1%），patience = 0
```

### 推荐设置

| 指标类型 | 推荐 min_delta | 原因 |
|---------|---------------|------|
| 准确率（%）| 0.001 - 0.01 | 过滤微小波动（0.1% - 1%） |
| 损失值 | 0.0001 - 0.001 | 根据损失量级调整 |
| F1-Score | 0.001 - 0.01 | 类似准确率 |
| AUC | 0.001 - 0.005 | AUC变化通常较小 |

## 工作流程

### 指标改善时

```
Epoch 10: val_acc = 92.5%
↓
stopper.step(92.5)
↓
检测到改善（92.5 > 92.0 + min_delta）
↓
best_metric = 92.5
patience_counter = 0
返回 is_best = True
↓
调用方保存最佳模型
```

### 指标未改善时

```
Epoch 11: val_acc = 92.3%
↓
stopper.step(92.3)
↓
未改善（92.3 < 92.5 + min_delta）
↓
patience_counter += 1  (现在为 1)
返回 is_best = False
↓
记录警告日志: "Patience: 1/10"
```

### 触发早停时

```
Epoch 20: val_acc = 92.1%
↓
stopper.step(92.1)
↓
未改善（连续第10次）
↓
patience_counter = 10 (达到patience)
返回 is_best = False
↓
stopper.should_stop 返回 True
↓
记录CRITICAL日志: "早停触发！"
↓
训练循环检测到 should_stop，执行 break
```

## 设计原则

### 简单易用
- 只需调用 `step()` 传入指标，无需手动判断
- 通过返回值 `is_best` 直接指示是否应保存模型

### 灵活配置
- 支持最大化和最小化两种模式
- 支持 `min_delta` 阈值过滤噪声
- 支持禁用早停（`patience <= 0`）

### 状态一致性
- 支持保存/加载状态，恢复训练时保持一致
- 避免恢复训练后早停器状态重置的问题

### 可观测性
- 详细的日志记录（指标改善/恶化、早停触发）
- 不同级别的日志（SUCCESS/WARNING/CRITICAL）
