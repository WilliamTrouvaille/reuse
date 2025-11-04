# 检查点管理模块使用指南

本文档详细说明 `utils/checkpoint_manager.py` 模块中的检查点管理工具类。

## 核心类

### `CheckpointManager` 类

#### 功能
通用的检查点管理器，负责保存和加载训练状态的字典，支持最佳模型保存、滚动Epoch检查点和中断恢复。

#### 核心特性
- **原子性保存**：使用临时文件和重命名机制，防止保存中断导致文件损坏
- **自动滚动清理**：保留最新的N个Epoch检查点，自动删除旧检查点
- **智能恢复**：支持优先级加载（中断点 > Epoch点）
- **设备映射**：加载时自动映射到指定设备

#### 初始化

```python
CheckpointManager(save_dir='checkpoint', device='cpu', max_to_keep=5)
```

**参数**
- `save_dir` (str, 可选): 检查点保存的根目录，默认为 `'checkpoint'`
- `device` (str, 可选): 加载检查点时映射到的设备，默认为 `'cpu'`
- `max_to_keep` (int, 可选): 滚动保存的最大Epoch检查点数量，默认为 `5`

**类常量**

| 常量 | 值 | 说明 |
|------|----|----|
| `BEST_MODEL_NAME` | `"best_model.pth"` | 最佳模型文件名 |
| `EPOCH_CKPT_PREFIX` | `"checkpoint_epoch_"` | Epoch检查点文件名前缀 |
| `INTERRUPT_CKPT_NAME` | `"interrupt_checkpoint.pth"` | 中断检查点文件名 |

#### 方法

##### 最佳模型相关

**`save_best_model(state, metric)`**
- 功能：保存最佳模型检查点（由调用方判断是否为最佳）
- 参数：
  - `state` (dict): 要保存的状态字典（包含 `model.state_dict()`, `optimizer.state_dict()` 等）
  - `metric` (float): 当前最佳指标值（用于日志记录）
- 返回值：无
- 日志级别：SUCCESS

**`load_best_model()`**
- 功能：加载最佳模型检查点
- 参数：无
- 返回值：`dict | None`（成功返回状态字典，失败返回 `None`）
- 日志级别：INFO

##### Epoch检查点相关

**`save_epoch_checkpoint(state, epoch)`**
- 功能：保存特定Epoch的检查点，并自动触发滚动清理
- 参数：
  - `state` (dict): 要保存的状态字典
  - `epoch` (int): 当前Epoch编号
- 返回值：无
- 特性：保存后自动调用 `_cleanup_epoch_checkpoints()` 清理旧文件
- 文件命名：`checkpoint_epoch_{epoch}.pth`

**`load_latest_checkpoint()`**
- 功能：加载最新的检查点以恢复训练
- 参数：无
- 返回值：`dict | None`
- 加载优先级：
  1. `interrupt_checkpoint.pth`（如果存在）
  2. 最新的 `checkpoint_epoch_X.pth`（如果存在）
- 特殊行为：加载中断点后会自动删除该文件，防止重复加载

##### 中断恢复相关

**`save_interrupt_checkpoint(state)`**
- 功能：在训练中断（Ctrl+C）时保存快照
- 参数：
  - `state` (dict): 要保存的状态字典
- 返回值：无
- 日志级别：WARNING + CRITICAL
- 使用场景：在 `KeyboardInterrupt` 捕获块中调用

#### 内部机制

##### 原子性保存机制

```python
def _save(self, state, filename):
    # 1. 先保存到临时文件
    tmp_filepath = f"{filepath}.tmp"
    torch.save(state, tmp_filepath)

    # 2. 原子性重命名（os.replace）
    os.replace(tmp_filepath, filepath)  # 防止保存中断导致文件损坏
```

##### 滚动清理机制

```python
def _cleanup_epoch_checkpoints(self):
    # 1. 查找所有 checkpoint_epoch_*.pth 文件
    # 2. 解析文件名中的 epoch 编号
    # 3. 按 epoch 编号降序排序（最新的在前）
    # 4. 删除超出 max_to_keep 的旧文件
```

**清理时机**：每次调用 `save_epoch_checkpoint()` 后自动执行

## 使用示例

### 基础用法

```python
from utils import CheckpointManager

# 初始化管理器
ckpt_mgr = CheckpointManager(
    save_dir='./checkpoints',
    device='cuda',
    max_to_keep=5
)

# 训练循环
best_acc = 0.0

for epoch in range(100):
    # ... 训练代码 ...

    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_loss, val_acc = validate(model, val_loader)

    # 1. 保存 Epoch 检查点（每个 epoch）
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_acc': best_acc,
        'train_loss': train_loss,
        'val_acc': val_acc
    }
    ckpt_mgr.save_epoch_checkpoint(state, epoch)

    # 2. 保存最佳模型（条件触发）
    if val_acc > best_acc:
        best_acc = val_acc
        ckpt_mgr.save_best_model(state, metric=best_acc)
```

### 中断恢复

```python
from utils import CheckpointManager
from loguru import logger

def train():
    ckpt_mgr = CheckpointManager(save_dir='./checkpoints', device='cuda')

    model = create_model()
    optimizer = create_optimizer(model)

    start_epoch = 0
    best_acc = 0.0

    # 尝试恢复训练
    checkpoint = ckpt_mgr.load_latest_checkpoint()
    if checkpoint:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        logger.info(f"从 Epoch {checkpoint['epoch']} 恢复训练，最佳准确率: {best_acc:.2f}%")

    # 训练循环
    try:
        for epoch in range(start_epoch, 100):
            # ... 训练代码 ...

            # 保存检查点
            state = {'epoch': epoch, 'model': model.state_dict(), ...}
            ckpt_mgr.save_epoch_checkpoint(state, epoch)

    except KeyboardInterrupt:
        logger.warning("检测到中断信号（Ctrl+C），正在保存当前状态...")

        # 保存中断检查点
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc
        }
        ckpt_mgr.save_interrupt_checkpoint(state)

        logger.critical("中断检查点已保存，可以安全退出。")

if __name__ == '__main__':
    train()
```

### 加载最佳模型进行测试

```python
from utils import CheckpointManager
from loguru import logger

# 初始化管理器
ckpt_mgr = CheckpointManager(save_dir='./checkpoints', device='cuda')

# 加载最佳模型
checkpoint = ckpt_mgr.load_best_model()

if checkpoint is None:
    logger.error("未找到最佳模型检查点！")
    exit(1)

# 恢复模型
model = create_model()
model.load_state_dict(checkpoint['model'])
model.eval()

# 在测试集上评估
test_acc = evaluate(model, test_loader)
logger.info(f"最佳模型测试集准确率: {test_acc:.2f}%")
```

### 自定义状态字典

```python
from utils import CheckpointManager

ckpt_mgr = CheckpointManager(save_dir='./checkpoints', device='cuda')

# 可以保存任意状态
state = {
    # 模型相关
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),

    # 训练状态
    'epoch': epoch,
    'global_step': global_step,

    # 指标
    'best_acc': best_acc,
    'train_loss': train_loss,
    'val_acc': val_acc,

    # 其他元信息
    'config': config,
    'experiment_name': 'resnet18_cifar10',
    'timestamp': datetime.now().isoformat()
}

ckpt_mgr.save_epoch_checkpoint(state, epoch)
```

### 在Trainer类中使用

```python
from utils import CheckpointManager

class Trainer:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # 初始化检查点管理器
        self.ckpt_mgr = CheckpointManager(
            save_dir='./checkpoints',
            device=device,
            max_to_keep=5
        )

        self.start_epoch = 0
        self.best_metric = 0.0

    def resume_training(self):
        """恢复训练"""
        checkpoint = self.ckpt_mgr.load_latest_checkpoint()

        if checkpoint:
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_metric = checkpoint['best_metric']
            logger.info(f"训练已恢复，从 Epoch {self.start_epoch} 开始")
        else:
            logger.info("从头开始训练")

    def save_checkpoint(self, epoch, metrics):
        """保存检查点"""
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'metrics': metrics
        }

        # 保存 Epoch 检查点
        self.ckpt_mgr.save_epoch_checkpoint(state, epoch)

        # 如果是最佳模型，也保存最佳模型
        if metrics['val_acc'] > self.best_metric:
            self.best_metric = metrics['val_acc']
            self.ckpt_mgr.save_best_model(state, metric=self.best_metric)

    def fit(self, train_loader, val_loader, epochs):
        """训练主函数"""
        # 尝试恢复
        self.resume_training()

        try:
            for epoch in range(self.start_epoch, epochs):
                # 训练和验证
                train_metrics = self.train_epoch(train_loader)
                val_metrics = self.validate(val_loader)

                metrics = {**train_metrics, **val_metrics}

                # 保存检查点
                self.save_checkpoint(epoch, metrics)

        except KeyboardInterrupt:
            logger.warning("训练被中断，正在保存...")
            state = {
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'best_metric': self.best_metric
            }
            self.ckpt_mgr.save_interrupt_checkpoint(state)
```

## 文件组织结构

典型的检查点目录结构：

```
checkpoints/
├── best_model.pth                  # 最佳模型（始终存在）
├── checkpoint_epoch_96.pth         # 最新的 5 个 Epoch 检查点
├── checkpoint_epoch_97.pth
├── checkpoint_epoch_98.pth
├── checkpoint_epoch_99.pth
└── checkpoint_epoch_100.pth
```

中断后的目录结构：

```
checkpoints/
├── best_model.pth
├── interrupt_checkpoint.pth        # 中断点（优先级最高）
├── checkpoint_epoch_45.pth
├── checkpoint_epoch_46.pth
└── checkpoint_epoch_47.pth         # 被中断时的最新 Epoch
```

## 设计原则

### 通用性
- 不关心状态字典的具体内容，由调用方决定保存什么
- 适用于任何PyTorch模型和训练流程

### 可靠性
- 原子性保存机制防止文件损坏
- 智能恢复机制确保训练可以从最近的状态继续

### 自动化
- 自动滚动清理旧检查点，节省磁盘空间
- 自动优先级判断，无需手动选择加载哪个文件

### 健壮性
- 完善的异常处理和日志记录
- 临时文件清理机制防止垃圾文件残留
