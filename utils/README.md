# utils 工具包使用说明

本 `README.md` 旨在说明 `utils/__init__.py` 文件中导出的核心工具函数和类，以便快速查阅和使用。

## 日志配置 (`logger_config.py`)

### `setup_logging(log_dir, console_level, file_level)`

**作用**: (程序入口调用) 初始化全局 `loguru` 日志记录器。

* **`log_dir` (str)**: 日志文件的存储目录 (例如: `"./logs"`)。
* **`console_level` (str)**: 控制台输出的最低级别 (例如: `"INFO"`)。
* **`file_level` (str)**: 文件输出的最低级别 (例如: `"DEBUG"`)。
* **注意**: 此函数具有幂等性，会先移除所有旧的 handlers 再添加新的。

---

## 配置管理 (`config.py`)

### `ConfigNamespace` (类)

**作用**: 将字典 (dict) 转换为可通过属性访问的对象 (例如 `config.dataset.name`)。

* **`__init__(config_dict)`**: 使用字典初始化。
* **`.to_dict()` (方法)**: 将 `ConfigNamespace` 对象递归转换回字典。
* **`.get(key, default)` (方法)**: 安全地获取属性，类似字典的 `.get()`。
* **`.update(new_config_dict)` (方法)**: 使用新字典递归更新 `ConfigNamespace` 实例。

### `setup_config(default_config, yaml_config_path, cmd_args)`

**作用**: (推荐使用) 编排配置加载，按优先级合并配置。

* **优先级**: 命令行参数 > YAML 文件 > 默认配置。
* **`default_config` (dict)**: 项目代码中定义的默认配置字典。
* **`yaml_config_path` (str)**: YAML 配置文件的路径 (例如: `"config.yaml"`)。
* **`cmd_args` (dict)**: `argparse` 解析后的参数字典 (`vars(args)`)。
* **返回**: 一个 `ConfigNamespace` 实例。

### `load_config_from_yaml(config_path)`

**作用**: 仅从 YAML 文件加载配置。

* **`config_path` (str)**: YAML 文件的路径。
* **返回**: 包含配置的字典 (dict)。如果文件不存在或解析失败，返回空字典 `{}`。

### `save_config_to_yaml(config, config_path)`

**作用**: 将配置字典或 `ConfigNamespace` 对象保存回 YAML 文件。

* **`config` (dict | ConfigNamespace)**: 要保存的配置对象。
* **`config_path` (str)**: 目标 YAML 文件的路径。

### `print_config(config, title)`

**作用**: 将配置字典或 `ConfigNamespace` 以美观的格式打印到 `loguru` (INFO 级别)。

* **`config` (dict | ConfigNamespace)**: 要打印的配置对象。
* **`title` (str)**: (可选) 打印输出的标题。

## NTFY 通知 (`ntfy_notifier.py`)

### `NtfyNotifier` (类)

**作用**: (在服务器上推荐使用) 发送实时训练状态通知到 ntfy 手机 App。

* **`__init__(server_url)`**: 初始化。`server_url` 默认为 `"https://ntfy.sh"`。
* **`.notify_start(message)`**: (低优先级) 发送 "训练开始" 通知。
* **`.notify_success(message)`**: (高优先级) 发送 "训练成功" 通知。
* **`.notify_error(message, error_details)`**: (最高优先级) 发送 "训练失败" 通知。`error_details` (可选) 可传入 traceback
  字符串，将使用 Markdown 代码块格式化。
* **`.send(message, title, priority, tags)`**: (高级) 发送自定义通知。

## 检查点管理 (`checkpoint_manager.py`)

### `CheckpointManager` (类)

**作用**: (推荐使用) 提供一个面向对象的管理器，专门处理训练检查点的**保存**、**加载**和**滚动清理**。

---

### 💡 核心设计理念：职责分离 (Separation of Concerns)

本工具类的设计严格遵守“职责分离”原则，这对保持其“可复用性”至关重要：

1. **`CheckpointManager` (工具类) 的职责**:
    * **只负责 I/O**：它只关心如何将一个 `dict` 对象保存到文件，以及如何从文件加载回 `dict`。
    * **只负责文件管理**：它负责文件的命名（`best_model.pth`, `checkpoint_epoch_X.pth`）、滚动清理（保留最新的 `max_to_keep`
      个）和加载优先级。
    * **它不知道内容**：它*不*知道也不*关心* `dict` 中存的是 `model_state`、`optimizer_state` 还是 `image_syn`。

2. **`main.py` (调用方) 的职责**:
    * **只负责内容**：`main.py` 负责在 `state` 字典中*构建*所有需要保存的状态（如 `epoch`, `model_state`,
      `optimizer_state`, `best_metric`, `lr_scheduler_state` 等）。

**这种设计的优势**:
如果未来您的训练需要额外保存 `lr_scheduler` 的状态，您**不需要修改** `CheckpointManager` 的任何代码。您只需在 `main.py` 中将
`lr_scheduler.state_dict()` 添加到 `state` 字典中即可。

```python
# --- main.py 中的使用示例 ---

# 1. 调用方 (main.py) 负责构建 state
state = {
    'epoch': current_epoch,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'lr_scheduler_state': scheduler.state_dict(),
    'best_metric': best_metric
}

# 2. 管理器 (utils) 负责保存
# 管理器并不知道 state 里面有什么，它只负责保存
ckpt_manager.save_epoch_checkpoint(state, current_epoch)
```

---

### 公共 API

#### `__init__(save_dir, device, max_to_keep)`

**作用**: 初始化管理器。

* **`save_dir` (str)**: 检查点保存的根目录 (例如: `"./checkpoints"`)。
* **`device` (str)**: (可选) 加载检查点时映射到的设备 (例如: `"cpu"`, `"cuda"`)。
* **`max_to_keep` (int)**: (可选, 默认 3) 滚动保存 `checkpoint_epoch_*.pth` 文件的最大数量。

#### `save_best_model(state, metric)`

**作用**: 将 `state` 字典保存为 `best_model.pth`。

* **`state` (dict)**: 由 `main.py` 构建的包含所有状态的字典。
* **`metric` (float)**: `main.py` 传入的当前最佳指标值（仅用于日志记录）。

#### `load_best_model()`

**作用**: 加载 `best_model.pth`。

* **返回**: 包含所有状态的 `dict`，或在失败时返回 `None`。
* **注意**: 此方法通常用于**推理或评估**，而不是恢复训练。

#### `save_epoch_checkpoint(state, epoch)`

**作用**: 将 `state` 字典保存为 `checkpoint_epoch_X.pth` 并自动触发清理。

* **`state` (dict)**: 要保存的状态字典。
* **`epoch` (int)**: 当前的 epoch 编号，用于文件名。
* **功能**: 保存后，会自动删除*最旧*的 epoch 检查点，确保只保留 `max_to_keep` 个。

#### `load_latest_checkpoint()`

**作用**: (核心) 加载最新的检查点以**恢复训练**。

* **加载优先级**:
    1. **`interrupt_checkpoint.pth`** (最高优先级): 如果此文件存在，意味着上次训练被非正常中断。
    2. **最新的 `checkpoint_epoch_X.pth`** (第二优先级): 如果中断文件不存在，则加载 epoch 编号最大的文件。
* **返回**: 包含所有状态的 `dict`，或在没有找到任何检查点时返回 `None`。
* **注意**: 如果成功加载了 `interrupt_checkpoint.pth`，该文件会被**自动删除**，以防陷入无限恢复循环。

#### `save_interrupt_checkpoint(state)`

**作用**: (在 `try...except KeyboardInterrupt` 中调用) 保存训练中断时的快照。

* **`state` (dict)**: `Ctrl+C` 被按下时，由 `main.py` 构建的当前状态字典。
* **功能**: 将 `state` 保存为 `interrupt_checkpoint.pth`。

## 数据处理 (`data.py`)

### `load_dataset_info(dataset_name, data_path)`

**作用**: 加载 `torch.utils.data.Dataset` 对象并返回相关的元数据。

* **`dataset_name` (str)**: 要加载的数据集名称 (例如: `"CIFAR10"`, `"MNIST"`).
* **`data_path` (str)**: 数据存储的根目录 (例如: `"../data"`).
* **返回** (dict):
  一个包含数据集和元数据的字典。
    ```python
    info = {
        'dst_train': <Dataset object>,
        'dst_test': <Dataset object>,
        'im_size': (32, 32),
        'channel': 3,
        'num_classes': 10,
        'class_names': ['airplane', ...],
        'mean': [0.4914, ...],
        'std': [0.2023, ...]
    }
    ```
* **注意**: 此函数**不会**创建 `DataLoader`。创建 `DataLoader` 是调用方 (`main.py`) 的责任，这允许调用方完全控制
  `batch_size`, `num_workers`, `pin_memory` 等性能参数。

## 进度条 (`progress_tracker.py`)

### `ProgressTracker` (类)

**作用**: (重要) 替换 `tqdm`，用于在 PyTorch 训练/评估循环中显示进度，同时**避免 I/O 和 `.item()` 导致的性能瓶颈**。

**核心原理**:
[cite_start]您不应该在 `for` 循环的每一步都调用 `.item()` 或 `tqdm.set_postfix()`，因为它们是缓慢的同步 I/O 操作 。
[cite_start]`ProgressTracker` 通过**时间节流 (Time-Based Throttling)** 解决了这个问题 ：

1. 它接收 `torch.Tensor` (例如 `loss`)。
2. [cite_start]它在 GPU 上对这些 Tensor 进行累加（非阻塞）。
3. [cite_start]它只在固定的时间间隔（例如每 0.5 秒）触发**一次** `.item()` 同步和 `set_postfix` I/O 。
4. 它显示的指标 (例如 `loss=0.1234`) **始终是整个 epoch 到目前为止的运行平均值**。

**用法**:

```python
from utils import ProgressTracker
import torch

# 1. (模拟) 环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_loader = [(torch.randn(1, device=device), torch.randn(1, device=device)) for _ in range(1000)]
TOTAL_EPOCHS = 5

# 2. (外循环)
for epoch in range(TOTAL_EPOCHS):

    # 3. (内循环) 包装 data_loader
    #    `leave=False` 意味着它在结束后会消失
    #    (支持 'with' 语句自动 .close())
    with ProgressTracker(
            data_loader,
            description=f"Epoch {epoch + 1}",
            leave=False,
            device=device  # 传入 device
    ) as tracker:

        for images, labels in tracker:
            # --- 模拟训练 ---
            time.sleep(0.001)  # 模拟高速 GPU 工作
            # 必须在 device 上的 Tensors
            sim_loss = (torch.randn(1, device=device) + 0.5)
            sim_acc = (torch.randn(1, device=device) + 0.8)
            current_lr = 1e-4
            # ------------------

            # 4. (关键) 更新指标
            #    传入 Tensors 和 floats
            #    这在 99% 的情况下是非阻塞的
            tracker.update({
                'loss': sim_loss,
                'acc': sim_acc,
                'lr': current_lr
            })

    # 5. 'with' 语句结束，tracker 自动 .close()

    # 6. (可选) 获取该 epoch 的最终平均值
    final_metrics = tracker.get_final_metrics()
    logger.info(f"Epoch {epoch + 1} Final Avg Loss: {final_metrics.get('loss', 0):.4f}")
```

## 装饰器 (`decorators.py`)

本模块提供可复用的装饰器，用于分离业务逻辑和通用功能（如日志、计时、错误处理）。

### `@time_it`

**作用**: 自动测量并 `INFO` 级别记录函数的执行时间。

**用法**:
```python
from utils import time_it

@time_it
def load_my_data():
    # ... 耗时操作 ...
    pass

# (日志输出): 函数 'load_my_data' 执行完毕，耗时: 1m 15.3s
```

### `@no_grad`

**作用**: (PyTorch) 在 `torch.no_grad()` 上下文中自动执行函数。

**用法**:
```python
from utils import no_grad

@no_grad
def evaluate_model(model, test_loader):
    # 这里的代码自动在 no_grad() 块中
    ...
```

### `@train_mode(model_attr='model')`
### `@eval_mode(model_attr='model')`

**作用**: (PyTorch) 自动管理 `model.train()` 和 `model.eval()` 状态。

* **前提**: 必须用于**类方法** (第一个参数是 `self`)。
* **`model_attr` (str)**: `self` 拥有的 `nn.Module` 属性的名称 (默认为 `'model'`)。
* **特性**: 它会在函数执行后，自动将模型**恢复到其原始状态**。

**用法**:
```python
from utils import train_mode, eval_mode, no_grad

class MyTrainer:
    def __init__(self, net):
        self.net = net # 注意：属性名叫 'net'

    @train_mode(model_attr='net') # 告诉装饰器属性名叫 'net'
    def train_step(self, data):
        # self.net 在这里自动是 .train() 模式
        ...

    @eval_mode(model_attr='net')
    @no_grad # 装饰器可以堆叠
    def evaluate(self):
        # self.net 在这里自动是 .eval() 模式
        ...
```

### `@log_errors(notifier=None, re_raise=True)`

**作用**: (工厂) 自动 `try...except` 包裹函数，记录**完整的堆栈跟踪**，并（可选）发送 Ntfy 通知。

* **`notifier` (NtfyNotifier, optional)**: 传入一个 `NtfyNotifier` 实例，在出错时发送通知。
* **`re_raise` (bool)**:
    * `True` (默认): 记录错误后，重新抛出异常（程序崩溃）。
    * `False`: 记录错误后，“吞掉”异常（程序继续运行）。

**用法**:
```python
from utils import log_errors, NtfyNotifier

# 在 main.py 中
my_notifier = NtfyNotifier()

# 装饰整个 main 函数，在崩溃时通知我
@log_errors(notifier=my_notifier, re_raise=True)
def main():
    ...
    if something_bad:
        raise ValueError("模拟崩溃")
    ...
```

## 指标跟踪 (`metrics.py`)

本模块提供两类指标跟踪器。

### `MetricTracker` (类)

**作用**: (高性能, 推荐用于训练/评估循环)
在 GPU/设备 上高效累积指标，**避免在循环中调用 `.item()`** 导致的 GPU 同步瓶颈。

**核心原理**:
* `update(loss, outputs, targets)`: (在循环内调用) 这是一个廉价的操作。它在 GPU 上执行 `sum` (非阻塞)，内存占用 O(1)。
* `compute()`: (在循环后调用) 这是一个昂贵的操作。它只在最后执行一次 `.item()` 来获取总和，并计算最终的 `loss`, `acc`, `top5`。

**用法**:
```python
from utils import MetricTracker

# 1. 在 epoch 开始前初始化
tracker = MetricTracker(device=device, compute_top5=True)

# 2. 在循环中 (例如 ProgressTracker 内部)
for inputs, labels in loader:
    logits = model(inputs)
    loss = criterion(logits, labels)
    
    # 3. (廉价) 在每一步调用 update
    tracker.update(loss, logits, labels)

# 4. (昂贵) 在 epoch 结束后调用 compute
final_epoch_metrics = tracker.compute()
# final_epoch_metrics = {'loss': 0.123, 'acc': 95.4, 'top5': 99.8}

# 5. 重置以备下一个 epoch
tracker.reset()
```

### `AverageMeter` (类)

**作用**: (轻量级) 简单的平均值计算器，用于 CPU 标量。
**不**要在 GPU 循环的热路径 (hot loop) 中使用它，因为它**每次 update 都会同步**。

**用法**: (例如跟踪学习率)
```python
from utils import AverageMeter
lr_meter = AverageMeter()

for ... in ...:
    lr = optimizer.param_groups[0]['lr']
    lr_meter.update(lr)

logger.info(f"平均学习率: {lr_meter.avg}")
```

## 早停 (`early_stopping.py`)

### `EarlyStopper` (类)

**作用**: 封装早停逻辑，在 `Trainer` 中使用。

**核心原理**:
* `step(metric)`: (在 `eval_epoch` 后调用) 传入最新的验证指标。**返回 `bool` (is_best)**。
* `is_best_so_far` (属性): `step` 方法会自动设置此标志。`Trainer` 检查此标志以决定是否调用 `CheckpointManager.save_best_model()`。
* `should_stop` (属性): `step` 方法会自动更新内部计数器。`Trainer` 检查此标志以决定是否中断训练循环。
* `state_dict()` / `load_state_dict(dict)`: (重要) 用于在检查点中保存和恢复早停的状态（`counter` 和 `best_metric`）。

**用法**:
```python
from utils import EarlyStopper

# 1. 在训练开始前初始化
#    (Patience=10, 监控 'acc' (越高越好), 至少提升 0.01 才算数)
stopper = EarlyStopper(patience=10, mode='max', min_delta=0.01)

# --- (在 Trainer 内部循环中) ---
for epoch in ...:
    val_metrics = evaluate(...)
    
    # 2. 传入最新的指标，并获取 is_best
    is_best = stopper.step(val_metrics['acc'])
    
    # 3. 检查是否应保存
    if is_best:
        save_best_model(...)
        
    # 4. 检查是否应停止
    if stopper.should_stop:
        break
```