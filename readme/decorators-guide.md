# 通用装饰器使用指南

本文档详细说明 `utils/decorators.py` 模块中的通用装饰器。这些装饰器用于分离业务逻辑和通用功能（日志、计时、状态管理、错误处理）。

## 核心装饰器

### `@time_it` 装饰器

#### 功能
自动测量并记录函数的执行时间（INFO级别日志）。

#### 使用场景
- 测量耗时操作（数据加载、模型初始化等）
- 性能分析和优化

#### 使用示例

```python
from utils import time_it

@time_it
def load_large_dataset():
    """加载大型数据集"""
    # ... 数据加载代码 ...
    pass

# 调用函数
load_large_dataset()

# 日志输出示例：
# 函数 'load_large_dataset' 执行完毕，耗时: 1m 15.3s
```

**在类方法中使用**

```python
from utils import time_it

class DataProcessor:
    @time_it
    def preprocess_data(self, data):
        """预处理数据"""
        # ... 预处理代码 ...
        return processed_data

    @time_it
    def augment_data(self, data):
        """数据增强"""
        # ... 增强代码 ...
        return augmented_data
```

### `@no_grad` 装饰器

#### 功能
在 `torch.no_grad()` 上下文中自动执行函数，用于评估、验证或推理。

#### 优势
- 自动禁用梯度计算
- 节省显存
- 提升推理速度

#### 使用示例

```python
from utils import no_grad

@no_grad
def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()

    total_correct = 0
    total_samples = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # 自动在no_grad()上下文中执行
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = total_correct / total_samples * 100
    return accuracy
```

**与其他装饰器组合**

```python
from utils import no_grad, time_it

@time_it
@no_grad
def run_inference(model, data_loader):
    """运行推理"""
    results = []
    for batch in data_loader:
        outputs = model(batch)
        results.append(outputs)
    return results
```

### `@train_mode` 和 `@eval_mode` 装饰器

#### 功能
自动管理模型的训练/评估状态，并在函数执行后恢复原始状态。

#### 参数
- `model_attr` (str, 可选): `self` 拥有的 `nn.Module` 属性的名称，默认为 `'model'`

#### 特性
- 自动设置 `model.train()` 或 `model.eval()`
- 函数执行后自动恢复原始模式
- 防止状态混乱

#### 注意事项
- 必须用于类方法（第一个参数是`self`）
- 需要指定正确的`model_attr`参数

#### 使用示例

**基础用法**

```python
from utils import train_mode, eval_mode, no_grad

class Trainer:
    def __init__(self, model):
        self.model = model  # 属性名为 'model'

    @train_mode()  # 使用默认的 model_attr='model'
    def train_step(self, images, labels):
        """训练步骤"""
        # self.model 自动处于 .train() 模式
        outputs = self.model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        return loss

    @eval_mode()
    @no_grad  # 装饰器可以堆叠
    def evaluate(self, test_loader):
        """评估"""
        # self.model 自动处于 .eval() 模式
        total_correct = 0
        for images, labels in test_loader:
            outputs = self.model(images)
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
        return total_correct
```

**自定义模型属性名**

```python
from utils import train_mode, eval_mode

class MyTrainer:
    def __init__(self, network):
        self.network = network  # 属性名为 'network'

    @train_mode(model_attr='network')  # 指定属性名
    def train_step(self, data):
        """训练步骤"""
        # self.network 自动处于 .train() 模式
        outputs = self.network(data)
        return outputs

    @eval_mode(model_attr='network')
    @no_grad
    def validate(self, data):
        """验证"""
        # self.network 自动处于 .eval() 模式
        outputs = self.network(data)
        return outputs
```

**状态自动恢复**

```python
from utils import train_mode, eval_mode

class Trainer:
    def __init__(self, model):
        self.model = model

    @eval_mode()
    def compute_validation_loss(self, val_loader):
        """计算验证损失"""
        # 进入函数时：model 切换到 .eval() 模式
        # ... 验证代码 ...
        # 退出函数时：model 自动恢复到原始模式
        pass

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()  # 设置为训练模式

        for batch in train_loader:
            # ... 训练代码 ...

            # 每10个batch验证一次
            if batch % 10 == 0:
                val_loss = self.compute_validation_loss(val_loader)
                # compute_validation_loss 退出后，model 自动恢复为 .train() 模式
                # 无需手动调用 self.model.train()

            # ... 继续训练 ...
```

### `@log_errors` 装饰器（工厂函数）

#### 功能
自动捕获异常，记录完整的堆栈跟踪，并（可选）通过Ntfy发送通知。

#### 参数
- `notifier` (NtfyNotifier, 可选): Ntfy通知器实例，默认为`None`
- `re_raise` (bool, 可选): 是否重新抛出异常，默认为`True`

#### 行为
- `re_raise=True`（默认）：记录错误后重新抛出异常（程序崩溃）
- `re_raise=False`：记录错误后"吞掉"异常（程序继续运行），返回`None`

#### 使用示例

**基础用法（记录并重新抛出）**

```python
from utils import log_errors

@log_errors(re_raise=True)
def risky_function():
    """可能抛出异常的函数"""
    # ... 可能出错的代码 ...
    raise ValueError("模拟错误")

# 调用函数
try:
    risky_function()
except ValueError:
    print("异常被捕获")

# 日志会自动记录完整的堆栈跟踪
```

**记录但不重新抛出（优雅降级）**

```python
from utils import log_errors

@log_errors(re_raise=False)
def optional_operation():
    """可选操作，失败也无妨"""
    # ... 可能出错的代码 ...
    raise RuntimeError("操作失败")

# 调用函数
result = optional_operation()  # 返回 None，不会抛出异常
print(f"结果: {result}")  # 结果: None
```

**与Ntfy通知集成**

```python
from utils import log_errors, NtfyNotifier

# 创建通知器
notifier = NtfyNotifier()

@log_errors(notifier=notifier, re_raise=True)
def main():
    """主训练函数"""
    # ... 训练代码 ...
    if some_error:
        raise RuntimeError("训练失败")
    # ...

# 运行主函数
if __name__ == '__main__':
    main()

# 如果发生错误：
# 1. 记录完整的堆栈跟踪到日志
# 2. 发送Ntfy通知到手机
# 3. 重新抛出异常（程序崩溃）
```

**装饰整个训练脚本**

```python
from utils import log_errors, NtfyNotifier, setup_logging

setup_logging()
notifier = NtfyNotifier()

@log_errors(notifier=notifier, re_raise=True)
def main():
    """主函数"""
    # 加载配置
    config = setup_config(...)

    # 创建模型
    model = create_model(...)

    # 训练
    for epoch in range(100):
        train_epoch(model, ...)

        if some_condition:
            raise RuntimeError("训练中断")

if __name__ == '__main__':
    main()
```

## 装饰器组合使用

### 推荐的组合模式

**评估函数**

```python
from utils import eval_mode, no_grad, time_it

class Trainer:
    def __init__(self, model):
        self.model = model

    @time_it
    @eval_mode()
    @no_grad
    def evaluate(self, test_loader):
        """评估（测量时间 + 评估模式 + 禁用梯度）"""
        # self.model 处于 .eval() 模式
        # 梯度计算被禁用
        # 执行时间会被记录
        pass
```

**训练步骤**

```python
from utils import train_mode, log_errors

class Trainer:
    def __init__(self, model, notifier):
        self.model = model
        self.notifier = notifier

    @log_errors(notifier=self.notifier, re_raise=True)
    @train_mode()
    def train_step(self, batch):
        """训练步骤（自动错误处理 + 训练模式）"""
        # self.model 处于 .train() 模式
        # 错误会被自动记录和通知
        pass
```

### 装饰器堆叠顺序

装饰器从下到上执行：

```python
@decorator_A
@decorator_B
@decorator_C
def my_function():
    pass

# 执行顺序：C → B → A → my_function → A → B → C
```

**推荐顺序**（从下到上）：
1. `@no_grad` - 最内层，控制梯度
2. `@eval_mode` 或 `@train_mode` - 控制模型状态
3. `@time_it` - 测量时间
4. `@log_errors` - 最外层，捕获所有错误

```python
@log_errors(notifier=notifier)
@time_it
@eval_mode()
@no_grad
def evaluate(self, test_loader):
    """完整的评估函数"""
    pass
```

## 使用示例：完整的Trainer类

```python
from utils import (
    train_mode,
    eval_mode,
    no_grad,
    time_it,
    log_errors,
    NtfyNotifier
)

class Trainer:
    def __init__(self, model, optimizer, criterion, device, notifier=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.notifier = notifier or NtfyNotifier()

    @train_mode()
    def train_step(self, images, labels):
        """单个训练步骤"""
        # self.model 自动处于 .train() 模式
        images, labels = images.to(self.device), labels.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @time_it
    @eval_mode()
    @no_grad
    def evaluate(self, test_loader):
        """评估模型"""
        # self.model 自动处于 .eval() 模式
        # 梯度计算被禁用
        # 执行时间会被记录

        total_correct = 0
        total_samples = 0

        for images, labels in test_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            outputs = self.model(images)
            predictions = outputs.argmax(dim=1)

            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        accuracy = total_correct / total_samples * 100
        return accuracy

    @log_errors(notifier=notifier, re_raise=True)
    def fit(self, train_loader, test_loader, epochs):
        """训练主函数"""
        # 错误会被自动记录和通知

        for epoch in range(epochs):
            # 训练
            for images, labels in train_loader:
                loss = self.train_step(images, labels)

            # 评估
            accuracy = self.evaluate(test_loader)
            print(f"Epoch {epoch}: Acc = {accuracy:.2f}%")
```

## 设计原则

### 关注点分离
- 装饰器处理横切关注点（日志、计时、状态管理）
- 业务逻辑保持简洁

### 可复用性
- 装饰器可以应用于任何函数/方法
- 通过组合实现复杂功能

### 声明式编程
- 使用装饰器声明函数的行为
- 代码更易读、更易维护
