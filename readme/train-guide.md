# 训练器模块使用指南

本文档详细说明 `utils/train.py` 模块中的训练器类。

## 核心类

### `Trainer` 类

#### 功能
可复用的 PyTorch 训练协调器（Coordinator），负责标准训练流程的编排，整合了性能优化、检查点管理、早停和通知等功能。

#### 设计理念

**职责分离**
- Trainer 负责"如何训练"（How）
- main.py 负责"训练什么"（What）

**依赖注入**
- 通过 `__init__` 接收所有已实例化的工具
- 最大化灵活性和可测试性

**模板方法**
- 训练逻辑可通过重写 `_train_step()` 等方法定制
- 支持多任务学习、对比学习等复杂场景

**性能优先**
- 默认集成高性能工具（MetricTracker, Progress, AMP）

#### 初始化

```python
Trainer(
    model, optimizer, criterion, device,
    checkpoint_manager=None,
    early_stopper=None,
    notifier=None,
    scheduler=None,
    use_amp=False,
    grad_accum_steps=1,
    max_grad_norm=None,
    metric_to_track='acc',
    metric_mode='max',
    compute_top5=False,
    log_interval=1,
    val_interval=1,
    show_progress=True,
    progress_update_interval=1.5
)
```

**核心参数（必需）**

| 参数 | 类型 | 说明 |
|-----|------|------|
| `model` | `nn.Module` | PyTorch 模型（应已 `.to(device)`） |
| `optimizer` | `Optimizer` | PyTorch 优化器 |
| `criterion` | `nn.Module` | 损失函数 |
| `device` | `str` \| `torch.device` | 计算设备 |

**可选工具（依赖注入）**

| 参数 | 类型 | 说明 |
|-----|------|------|
| `checkpoint_manager` | `CheckpointManager` \| `None` | 检查点管理器 |
| `early_stopper` | `EarlyStopper` \| `None` | 早停器 |
| `notifier` | `NtfyNotifier` \| `None` | Ntfy 通知器 |
| `scheduler` | `_LRScheduler` \| `None` | 学习率调度器 |

**性能优化配置**

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `use_amp` | `bool` | `False` | 是否使用自动混合精度 |
| `grad_accum_steps` | `int` | `1` | 梯度累积步数 |
| `max_grad_norm` | `float` \| `None` | `None` | 梯度裁剪的最大范数 |

**指标与日志配置**

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `metric_to_track` | `str` | `'acc'` | 早停/最佳模型跟踪的指标键 |
| `metric_mode` | `Literal['min', 'max']` | `'max'` | 指标优化方向 |
| `compute_top5` | `bool` | `False` | 是否计算 Top-5 准确率 |
| `log_interval` | `int` | `1` | 详细日志记录间隔（epoch） |
| `val_interval` | `int` | `1` | 验证间隔（epoch） |

**进度条配置**

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `show_progress` | `bool` | `True` | 是否显示进度条 |
| `progress_update_interval` | `float` | `1.5` | 进度条更新间隔（秒） |

#### 类方法

**`from_config(model, optimizer, criterion, device, config, scheduler=None)`**
- 功能：配置驱动模式，从 config 对象自动实例化所有工具
- 参数：
  - `model`, `optimizer`, `criterion`, `device`, `scheduler`: 同 `__init__`
  - `config` (Any): 完整配置对象，必须包含子配置
- 返回值：配置好的 `Trainer` 实例
- 说明：自动从 `config.training`, `config.checkpoint`, `config.ntfy` 创建工具实例

**config 对象结构要求**

```python
config.training:
    use_amp: bool
    grad_accum_steps: int
    max_grad_norm: float | None
    metric_to_track: str
    metric_mode: str
    compute_top5: bool
    log_interval: int
    val_interval: int
    patience: int  # >0 启用早停
    show_progress: bool
    progress_update_interval: float

config.checkpoint:
    enabled: bool
    save_dir: str
    max_to_keep: int

config.ntfy:
    enabled: bool
```

#### 公共方法

**`fit(train_loader, val_loader=None, epochs=100)`**
- 功能：主训练循环（唯一的公共方法）
- 参数：
  - `train_loader` (DataLoader): 训练数据加载器
  - `val_loader` (DataLoader, 可选): 验证数据加载器
  - `epochs` (int, 可选): 总训练轮数，默认为 `100`
- 返回值：包含训练历史和最佳指标的字典
  - `'history'`: 每个 epoch 的指标列表
  - `'best_metric'`: 最佳验证指标值
- 异常处理：
  - `KeyboardInterrupt`: 自动保存中断检查点
  - `Exception`: 记录错误并发送通知
- 副作用：
  - 自动发送开始/成功/失败通知（如果启用）
  - 自动保存检查点（如果启用）
  - 自动触发早停（如果启用）

**`get_current_lr()`**
- 功能：获取当前学习率
- 返回值：`float`

**`get_training_history()`**
- 功能：获取训练历史
- 返回值：包含每个 epoch 指标的列表

#### 可重写的保护方法

这些方法供子类定制训练逻辑，支持多任务学习、对比学习等复杂场景。

**`_train_step(batch)`**
- 功能：单个训练步骤的模板方法
- 参数：
  - `batch`: 来自 DataLoader 的一个批次（通常是 `(inputs, targets)`）
- 返回值：必须包含以下键的字典
  - `'loss'` (Tensor): 当前 batch 的损失（标量）
  - `'outputs'` (Tensor): 模型输出的 logits (shape: [batch, num_classes])
  - `'targets'` (Tensor): 真实标签 (shape: [batch])
- 默认实现：标准分类任务
  - 输入：`(images, labels)`
  - 输出：`logits`
  - 损失：`criterion(logits, labels)`

**`_eval_step(batch)`**
- 功能：单个评估步骤的模板方法
- 参数：同 `_train_step()`
- 返回值：同 `_train_step()`
- 默认实现：与 `_train_step()` 相同（但在 eval 模式和 no_grad 下）

**`_on_train_epoch_end(epoch, train_metrics)`**
- 功能：训练 epoch 结束时的钩子
- 参数：
  - `epoch` (int): 当前 epoch 编号
  - `train_metrics` (dict): 训练指标
- 返回值：无
- 默认实现：空（什么都不做）
- 用途：记录额外信息（如权重直方图）、更新可视化等

**`_on_eval_epoch_end(epoch, val_metrics)`**
- 功能：评估 epoch 结束时的钩子
- 参数：
  - `epoch` (int): 当前 epoch 编号
  - `val_metrics` (dict): 验证指标
- 返回值：无
- 默认实现：空（什么都不做）

#### 内部机制

##### 训练循环流程

```
fit() 主循环
├── _main_training_loop()
│   ├── _train_epoch()
│   │   ├── _train_epoch_inner_loop()
│   │   │   ├── _train_step() (可重写)
│   │   │   ├── backward()
│   │   │   ├── optimizer.step()
│   │   │   └── metric_tracker.update()
│   │   └── metric_tracker.compute()
│   ├── _on_train_epoch_end() (钩子)
│   ├── _eval_epoch()
│   │   ├── _eval_epoch_inner_loop()
│   │   │   ├── _eval_step() (可重写)
│   │   │   └── metric_tracker.update()
│   │   └── metric_tracker.compute()
│   ├── _on_eval_epoch_end() (钩子)
│   ├── _step_scheduler()
│   ├── _log_epoch_metrics()
│   └── _save_and_check_stop()
│       ├── _build_checkpoint_state()
│       ├── checkpoint_manager.save_epoch_checkpoint()
│       ├── early_stopper.step()
│       └── checkpoint_manager.save_best_model()
└── _cleanup()
```

##### 梯度累积机制

```python
for step, batch in enumerate(loader):
    # 1. 前向传播
    step_result = self._train_step(batch)
    loss = step_result['loss']

    # 2. 缩放损失
    scaled_loss = loss / self.grad_accum_steps

    # 3. 反向传播（累积梯度）
    if self.scaler:
        self.scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    # 4. 仅在累积步数达到时更新参数
    if (step + 1) % self.grad_accum_steps == 0:
        # 梯度裁剪
        if self.max_grad_norm is not None:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )

        # 优化器步骤
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        # 清空梯度
        self.optimizer.zero_grad(set_to_none=True)
```

##### AMP（自动混合精度）支持

```python
# 初始化时
self.scaler = GradScaler() if (use_amp and device.type == 'cuda') else None

# 前向传播时
with autocast('cuda', enabled=(self.scaler is not None)):
    outputs = self.model(inputs)
    loss = self.criterion(outputs, targets)

# 反向传播时
if self.scaler:
    self.scaler.scale(loss).backward()
    self.scaler.step(self.optimizer)
    self.scaler.update()
else:
    loss.backward()
    self.optimizer.step()
```

##### 学习率调度器兼容性

```python
def _step_scheduler(self, val_metrics):
    if isinstance(self.scheduler, ReduceLROnPlateau):
        # ReduceLROnPlateau 需要传入指标
        metric_value = val_metrics.get(self.metric_to_track)
        self.scheduler.step(metric_value)
    else:
        # 其他 scheduler（StepLR, CosineAnnealingLR 等）
        self.scheduler.step()
```

##### 检查点自动恢复

```python
def _load_checkpoint(self):
    checkpoint = self.checkpoint_manager.load_latest_checkpoint()

    if checkpoint:
        # 恢复模型和优化器
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.start_epoch = checkpoint['epoch'] + 1

        # 恢复可选组件
        if self.scheduler and 'scheduler_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])

        if self.early_stopper and 'early_stopper_state' in checkpoint:
            self.early_stopper.load_state_dict(checkpoint['early_stopper_state'])

        if self.scaler and 'scaler_state' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state'])

        if 'best_metric' in checkpoint:
            self.best_metric = checkpoint['best_metric']

        if 'history' in checkpoint:
            self.training_history = checkpoint['history']
```

##### 中断处理

```python
try:
    self._main_training_loop(...)
except KeyboardInterrupt:
    logger.critical("检测到键盘中断 (Ctrl+C)")
    self._handle_interrupt()  # 保存中断检查点
    if self.notifier:
        self.notifier.notify_error("训练被用户中断")
```

#### 集成的工具

**MetricTracker**
- 在 GPU 上累积指标（非阻塞）
- 仅在 epoch 结束时同步一次
- 性能提升 1.3-1.5x

**Progress**
- 显示实时训练进度
- 支持自定义更新间隔
- 可通过 `show_progress=False` 禁用

**CheckpointManager**
- 自动保存 Epoch 检查点
- 滚动清理旧检查点
- 支持最佳模型保存和中断恢复

**EarlyStopper**
- 监控验证指标
- 自动判断是否为最佳模型
- 触发早停时返回 `should_stop=True`

**NtfyNotifier**
- 训练开始时发送低优先级通知
- 训练成功时发送高优先级通知
- 训练失败时发送最高优先级通知（包含错误堆栈）

## 两种初始化模式

### 依赖注入模式（推荐）

**特点**
- 完全控制所有组件的创建和配置
- 易于测试（可注入 mock 对象）
- 适合复杂的研究项目

**适用场景**
- 研究代码
- 需要灵活配置
- 需要单元测试

### 配置驱动模式

**特点**
- 配置与代码分离
- 自动实例化所有工具
- 简化代码结构

**适用场景**
- 生产环境
- 标准训练流程
- 多实验管理

## 自定义扩展

### 支持的场景

**多任务学习**
- 重写 `_train_step()` 和 `_eval_step()`
- 返回多个损失的加权和
- 指定用于指标计算的主输出

**对比学习**
- 重写 `_train_step()`
- 处理增强视图
- 计算对比损失

**GAN 训练**
- 创建两个 Trainer 实例（生成器和判别器）
- 或重写 `_train_epoch()` 实现交替训练

**自定义日志**
- 重写 `_on_train_epoch_end()` 和 `_on_eval_epoch_end()`
- 集成 TensorBoard, Weights & Biases 等

## 设计原则

### 职责分离
- Trainer 只负责训练流程编排
- 不负责数据加载、模型定义、优化器创建

### 依赖注入优于配置
- 优先使用 `__init__` 注入已实例化的工具
- 配置驱动模式作为可选的简化方式

### 模板方法支持定制
- 核心流程不可变（`fit()` 是 final 方法）
- 关键步骤可重写（`_train_step()`, `_eval_step()` 等）

### 性能优先
- 默认使用高性能工具（MetricTracker, Progress）
- 支持 AMP, 梯度累积, 梯度裁剪

### 健壮性保障
- 自动检查点保存和恢复
- 中断处理（Ctrl+C）
- 异常处理和通知
