# Ntfy通知模块使用指南

本文档详细说明 `utils/ntfy_notifier.py` 模块中的通知工具类。

## 核心类

### `NtfyNotifier` 类

#### 功能
封装 [ntfy.sh](https://ntfy.sh) HTTP API 的通知器，用于发送实时推送通知到手机/桌面客户端，支持训练进度、成功和错误通知。

#### 核心特性
- **自动重试机制**：使用 `tenacity` 库实现智能重试（最多3次，指数退避）
- **连接复用**：使用 `requests.Session` 提高性能
- **Markdown支持**：消息内容支持完整的Markdown格式
- **优先级控制**：不同事件使用不同的通知优先级
- **错误详情**：自动格式化错误堆栈跟踪

#### 初始化

```python
NtfyNotifier(server_url="https://ntfy.sh", topic=TOPIC)
```

**参数**
- `server_url` (str, 可选): ntfy 服务器的 URL，默认为 `"https://ntfy.sh"`（公共服务器）
- `topic` (str, 可选): ntfy 频道名称，默认为类常量 `TOPIC`

**类常量**
- `TOPIC`: 默认频道名称（硬编码）

**注意事项**
- 初始化时会创建 `requests.Session` 并设置全局 Markdown 头
- 建议使用自定义频道名称以避免与他人冲突

#### 核心方法

**`send(message, title, priority, tags=None)`**
- 功能：发送通知的核心方法（带自动重试）
- 参数：
  - `message` (str): 消息主体（支持 Markdown 格式）
  - `title` (str): 消息标题
  - `priority` (str): 优先级（`"low"`, `"default"`, `"high"`, `"max"` 或 `1-5`）
  - `tags` (list[str], 可选): ntfy 标签列表（用于显示表情符号等）
- 返回值：`bool` - 发送成功返回 `True`，失败返回 `False`
- 重试机制：
  - 最多重试 3 次
  - 指数退避：2s, 4s, 8s
  - 只对 `RequestException` 重试
  - 自动记录重试日志

#### 预定义方法

**`notify_start(message="训练已开始。")`**
- 功能：发送训练开始通知
- 参数：
  - `message` (str, 可选): 自定义消息，默认为 `"训练已开始。"`
- 优先级：`low`（2级）
- 标签：`["runner"]`（🏃图标）
- 标题：`"🏃 训练开始"`

**`notify_success(message="训练已成功完成。")`**
- 功能：发送训练成功通知
- 参数：
  - `message` (str, 可选): 自定义消息，默认为 `"训练已成功完成。"`
- 优先级：`high`（4级）
- 标签：`["white_check_mark"]`（✅图标）
- 标题：`"✅ 训练成功"`

**`notify_error(message, error_details=None)`**
- 功能：发送训练错误或中断通知
- 参数：
  - `message` (str): 简短的错误摘要（如 `"训练在 Epoch 50 失败"`）
  - `error_details` (str, 可选): 详细的错误信息（如异常堆栈跟踪）
- 优先级：`max`（5级，最高）
- 标签：`["x"]`（❌图标）
- 标题：`"❌ 训练失败"`
- 特性：
  - 自动使用 Markdown 代码块格式化错误详情
  - 错误详情超过 3000 字符时自动截断

## 使用示例

### 基础用法

```python
from utils import NtfyNotifier

# 初始化通知器（使用默认频道）
notifier = NtfyNotifier()

# 发送自定义通知
notifier.send(
    message="这是一条测试消息，支持 **Markdown** 格式！",
    title="测试通知",
    priority="default",
    tags=["bell"]
)
```

### 训练生命周期通知

```python
from utils import NtfyNotifier, setup_logging
from loguru import logger

setup_logging()
notifier = NtfyNotifier()

# 1. 训练开始时通知
notifier.notify_start(message="ResNet18 在 CIFAR10 上的训练已开始")

try:
    # 2. 执行训练
    for epoch in range(100):
        train_loss = train_one_epoch(model, train_loader)
        val_acc = validate(model, val_loader)

        logger.info(f"Epoch {epoch}: Loss={train_loss:.4f}, Acc={val_acc:.2f}%")

    # 3. 训练成功时通知
    notifier.notify_success(
        message=f"训练完成！\n\n最终验证准确率: {val_acc:.2f}%\n总Epoch数: {epoch+1}"
    )

except Exception as e:
    # 4. 训练失败时通知
    import traceback
    error_traceback = traceback.format_exc()

    notifier.notify_error(
        message=f"训练在 Epoch {epoch} 时失败",
        error_details=error_traceback
    )

    logger.error(f"训练失败: {e}")
    raise
```

### 与装饰器集成

```python
from utils import NtfyNotifier, log_errors

notifier = NtfyNotifier()

@log_errors(notifier=notifier, re_raise=True)
def main():
    """主训练函数"""
    logger.info("开始训练...")

    # 发送开始通知
    notifier.notify_start(message="实验开始：ResNet18-CIFAR10-Baseline")

    # 训练代码
    config = setup_config(...)
    model = create_model(...)
    trainer = Trainer(model, ...)

    trainer.fit(train_loader, val_loader, epochs=100)

    # 发送成功通知
    notifier.notify_success(
        message=f"训练完成！\n\n最佳验证准确率: {trainer.best_acc:.2f}%"
    )

if __name__ == '__main__':
    main()
    # 如果发生异常，@log_errors 会自动调用 notifier 发送错误通知
```

### 自定义频道

```python
from utils import NtfyNotifier

# 使用自定义频道（推荐）
notifier = NtfyNotifier(
    server_url="https://ntfy.sh",
    topic="my_custom_channel_xyz123"
)

# 使用自建服务器
notifier = NtfyNotifier(
    server_url="https://ntfy.myserver.com",
    topic="ml_training"
)
```

### 发送带格式的通知

```python
from utils import NtfyNotifier

notifier = NtfyNotifier()

# Markdown 格式
message = """
## 训练报告

### 模型信息
- **模型**: ResNet18
- **数据集**: CIFAR10
- **Batch Size**: 128

### 训练结果
| Metric | Value |
|--------|-------|
| Train Acc | 95.2% |
| Val Acc | 92.8% |
| Test Acc | 92.5% |

训练用时: **2小时 30分钟**
"""

notifier.send(
    message=message,
    title="训练报告",
    priority="high",
    tags=["chart_with_upwards_trend"]
)
```

### 在Trainer类中集成

```python
from utils import NtfyNotifier

class Trainer:
    def __init__(self, model, optimizer, device, enable_notifications=True):
        self.model = model
        self.optimizer = optimizer
        self.device = device

        # 初始化通知器
        self.notifier = NtfyNotifier() if enable_notifications else None

    def _send_notification(self, method_name, *args, **kwargs):
        """安全地发送通知（如果启用）"""
        if self.notifier:
            method = getattr(self.notifier, method_name)
            try:
                method(*args, **kwargs)
            except Exception as e:
                logger.warning(f"发送通知失败，但训练继续: {e}")

    def fit(self, train_loader, val_loader, epochs):
        """训练主函数"""
        # 发送开始通知
        self._send_notification(
            'notify_start',
            message=f"开始训练：{epochs} 个 Epoch"
        )

        try:
            best_acc = 0.0

            for epoch in range(epochs):
                train_metrics = self.train_epoch(train_loader)
                val_metrics = self.validate(val_loader)

                if val_metrics['acc'] > best_acc:
                    best_acc = val_metrics['acc']

            # 发送成功通知
            self._send_notification(
                'notify_success',
                message=f"训练完成！\n\n最佳验证准确率: {best_acc:.2f}%"
            )

        except KeyboardInterrupt:
            # 用户中断
            self._send_notification(
                'notify_error',
                message="训练被用户中断（Ctrl+C）",
                error_details=f"在 Epoch {epoch} 时中断"
            )
            raise

        except Exception as e:
            # 训练失败
            import traceback
            self._send_notification(
                'notify_error',
                message=f"训练失败：{str(e)}",
                error_details=traceback.format_exc()
            )
            raise
```

### 发送阶段性进度通知

```python
from utils import NtfyNotifier

notifier = NtfyNotifier()

# 训练开始
notifier.notify_start(message="训练开始")

best_acc = 0.0

for epoch in range(100):
    train_metrics = train_one_epoch(model, train_loader)
    val_metrics = validate(model, val_loader)

    # 每 20 个 epoch 发送一次进度通知
    if (epoch + 1) % 20 == 0:
        notifier.send(
            message=f"**进度更新** (Epoch {epoch+1}/100)\n\n"
                    f"训练损失: {train_metrics['loss']:.4f}\n"
                    f"验证准确率: {val_metrics['acc']:.2f}%\n"
                    f"最佳准确率: {best_acc:.2f}%",
            title="训练进度",
            priority="default",
            tags=["hourglass"]
        )

    if val_metrics['acc'] > best_acc:
        best_acc = val_metrics['acc']

# 训练完成
notifier.notify_success(
    message=f"训练完成！\n\n最佳验证准确率: {best_acc:.2f}%"
)
```

## 重试机制

### 重试配置

```python
@retry(
    stop=stop_after_attempt(3),                    # 最多重试 3 次
    wait=wait_exponential(multiplier=1, min=2, max=10),  # 指数退避: 2s, 4s, 8s
    retry=retry_if_exception_type(RequestException),     # 只重试网络异常
    before_sleep=before_sleep_log(logger, 'WARNING'),    # 重试前记录日志
    reraise=True                                    # 重试失败后重新抛出异常
)
```

### 重试流程

```
第1次尝试失败（网络错误）
↓
等待 2 秒
↓
第2次尝试失败（超时）
↓
等待 4 秒
↓
第3次尝试失败（服务器错误）
↓
等待 8 秒
↓
第4次尝试失败（达到最大重试次数）
↓
抛出 RetryError，记录错误日志
↓
send() 返回 False
```

## ntfy 优先级说明

| 优先级值 | 别名 | 说明 | 通知行为 |
|---------|------|------|---------|
| 1 | `min` | 最低优先级 | 无声通知 |
| 2 | `low` | 低优先级 | 无声通知 |
| 3 | `default` | 默认优先级 | 正常通知 |
| 4 | `high` | 高优先级 | 高优先级通知 |
| 5 | `max` / `urgent` | 最高优先级 | 紧急通知（绕过勿扰） |

### 本工具的优先级策略

| 方法 | 优先级 | 原因 |
|-----|-------|------|
| `notify_start()` | `low` (2) | 训练开始是常规事件，不需要立即关注 |
| `notify_success()` | `high` (4) | 成功完成需要及时知晓，但不紧急 |
| `notify_error()` | `max` (5) | 错误需要立即处理，绕过勿扰模式 |

## ntfy 标签说明

ntfy 使用标签（tags）来显示表情符号和图标。

### 常用标签

| 标签名 | 显示图标 | 适用场景 |
|-------|---------|---------|
| `runner` | 🏃 | 训练开始 |
| `white_check_mark` | ✅ | 训练成功 |
| `x` | ❌ | 训练失败 |
| `warning` | ⚠️ | 警告信息 |
| `chart_with_upwards_trend` | 📈 | 进度更新 |
| `hourglass` | ⏳ | 等待/进行中 |
| `bell` | 🔔 | 一般通知 |
| `fire` | 🔥 | 重要事件 |

### 使用自定义标签

```python
notifier.send(
    message="模型在验证集上达到新高！",
    title="新记录",
    priority="high",
    tags=["trophy", "fire", "chart_with_upwards_trend"]  # 🏆🔥📈
)
```

## 客户端配置

### 1. 手机客户端

**Android / iOS**
1. 下载 ntfy 应用：[Google Play](https://play.google.com/store/apps/details?id=io.heckel.ntfy) / [App Store](https://apps.apple.com/app/ntfy/id1625396347)
2. 打开应用，点击 "Subscribe to topic"
3. 输入你的频道名称（如 `my_custom_channel_xyz123`）
4. 订阅成功后即可接收通知

**设置建议**
- 启用 "高优先级消息绕过勿扰"
- 启用 "后台运行"
- 为不同优先级设置不同的通知声音

### 2. 桌面客户端

**Windows / macOS / Linux**
- 使用 Web 界面：https://ntfy.sh/your_topic_name
- 或安装桌面应用：https://github.com/binwiederhier/ntfy-desktop

## 错误处理

### 网络错误自动重试

```python
notifier = NtfyNotifier()

# 网络临时中断时会自动重试
result = notifier.send(
    message="测试消息",
    title="测试",
    priority="default"
)

if result:
    logger.info("通知发送成功")
else:
    logger.warning("通知发送失败（已重试3次）")
    # 程序继续运行，不会因为通知失败而中断
```

### 优雅降级

```python
def safe_notify(notifier, method_name, *args, **kwargs):
    """
    安全发送通知的辅助函数。
    即使通知失败，也不会影响主程序运行。
    """
    if notifier is None:
        return

    try:
        method = getattr(notifier, method_name)
        method(*args, **kwargs)
    except Exception as e:
        logger.warning(f"通知发送失败，但程序继续: {e}")

# 使用
safe_notify(notifier, 'notify_start', message="训练开始")
```

## 设计原则

### 非阻塞
- 通知发送失败不应中断主程序
- 使用自动重试机制，但不会无限重试

### 智能重试
- 只对网络相关异常重试
- 使用指数退避避免服务器压力
- 自动记录重试过程

### 信息丰富
- 支持 Markdown 格式化
- 错误通知包含完整堆栈跟踪
- 不同优先级区分不同事件

### 易于集成
- 预定义常用通知类型
- 支持与装饰器和 Trainer 类集成
- 支持自定义频道和服务器
