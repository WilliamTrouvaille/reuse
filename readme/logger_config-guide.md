# 日志配置模块使用指南

本文档详细说明 `utils/logger_config.py` 模块中的日志配置工具。

## 核心函数

### `setup_logging(log_dir="logs", console_level="INFO", file_level="DEBUG")` 函数

#### 功能
配置 Loguru 日志记录器，设置控制台和文件输出。此函数是幂等的，可以多次调用而不会导致日志重复。

#### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `log_dir` | `str` | `"logs"` | 保存日志文件的目录 |
| `console_level` | `str` | `"INFO"` | 控制台输出的最低日志级别 |
| `file_level` | `str` | `"DEBUG"` | 文件输出的最低日志级别 |

#### 返回值
无

#### 副作用
- 移除所有已存在的日志处理器
- 添加控制台输出（stderr）
- 添加文件输出（轮转日志）
- 自动创建日志目录（如果不存在）

#### 特性

**幂等性**
- 自动移除旧的处理器
- 多次调用不会导致日志重复
- 最后一次配置生效

**控制台输出**
- 输出到 `sys.stderr`
- 支持彩色输出（ANSI颜色代码）
- 异步处理（不阻塞主线程）

**文件输出**
- 自动按日期命名：`log_YYYYMMDD.log`
- 自动轮转：文件达到 10 MB 时创建新文件
- 自动清理：只保留最近 10 天的日志
- UTF-8 编码
- 异步写入
- 完整堆栈跟踪（`backtrace=True`）
- 详细错误诊断（`diagnose=True`）

**日志格式**

```
<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> |
<level>{level: <8}</level> |
<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> -
<level>{message}</level>
```

**输出示例**

```
2025-11-04 14:30:22.123 | INFO     | main:train:45 - 开始训练...
2025-11-04 14:30:23.456 | SUCCESS  | trainer:fit:102 - Epoch 1 完成
2025-11-04 14:30:24.789 | ERROR    | data:load:23 - 文件未找到
```

#### 工作机制

**初始化流程**

```python
def setup_logging(...):
    # 1. 移除所有已存在的处理器（确保幂等性）
    try:
        logger.remove()
    except ValueError:
        pass  # 没有处理器时忽略

    # 2. 定义统一的日志格式
    log_format = "..."

    # 3. 添加控制台 Sink
    logger.add(
        sys.stderr,
        level=console_level.upper(),
        format=log_format,
        colorize=True,
        enqueue=True  # 异步
    )

    # 4. 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    # 5. 添加文件 Sink
    logger.add(
        log_file_path,
        level=file_level.upper(),
        format=log_format,
        rotation="10 MB",
        retention="10 days",
        encoding="utf-8",
        enqueue=True,
        backtrace=True,
        diagnose=True
    )

    # 6. 记录配置完成信息
    logger.info("Loguru 日志记录器配置完成。")
```

**日志轮转规则**

```
rotation="10 MB"     # 文件大小达到 10 MB 时自动轮转
retention="10 days"  # 只保留最近 10 天的日志
```

**文件命名规则**

```
logs/
├── log_20251103.log      # 2025年11月3日的日志
├── log_20251104.log      # 2025年11月4日的日志
└── log_20251104.log.1    # 轮转后的文件
```

#### 异常处理

**日志目录创建失败**
- 捕获 `OSError`
- 记录错误到控制台（控制台 Sink 已添加）
- 提前返回（只保留控制台输出）

```python
try:
    os.makedirs(log_dir, exist_ok=True)
except OSError as e:
    logger.error(f"无法创建日志目录: {log_dir}。错误: {e}")
    return  # 只使用控制台输出
```

## 日志级别

### Loguru 支持的级别

| 级别 | 数值 | 用途 |
|------|-----|------|
| `TRACE` | 5 | 最详细的追踪信息 |
| `DEBUG` | 10 | 详细调试信息 |
| `INFO` | 20 | 一般信息 |
| `SUCCESS` | 25 | 成功消息（Loguru 特有） |
| `WARNING` | 30 | 警告信息 |
| `ERROR` | 40 | 错误信息 |
| `CRITICAL` | 50 | 严重错误 |

### 使用方式

```python
from loguru import logger

# DEBUG: 详细的调试信息
logger.debug(f"输入形状: {inputs.shape}, 设备: {device}")

# INFO: 程序流程信息
logger.info("开始训练...")
logger.info(f"训练集大小: {len(train_dataset)}")

# SUCCESS: 成功消息
logger.success("训练完成！")
logger.success(f"最佳准确率: {best_acc:.2f}%")

# WARNING: 潜在问题
logger.warning("GPU 内存使用率超过 90%")

# ERROR: 错误但可继续
logger.error("保存检查点失败")

# CRITICAL: 严重错误
logger.critical("CUDA 不可用，无法继续训练")
```

### 级别过滤机制

**控制台级别 vs 文件级别**

```python
setup_logging(
    console_level='INFO',   # 控制台只显示 INFO 及以上
    file_level='DEBUG'      # 文件记录 DEBUG 及以上
)

logger.debug("这条日志")  # 只会出现在文件中
logger.info("这条日志")   # 同时出现在控制台和文件中
```

**推荐配置**

| 场景 | 控制台级别 | 文件级别 | 原因 |
|------|-----------|---------|------|
| 开发调试 | `DEBUG` | `DEBUG` | 查看所有信息 |
| 训练脚本 | `INFO` | `DEBUG` | 控制台简洁，文件详细 |
| 生产环境 | `WARNING` | `INFO` | 减少输出，保留关键信息 |
| 快速测试 | `INFO` | `CRITICAL` | 基本禁用文件日志 |

## 高级特性

### 异步日志记录

```python
# enqueue=True 启用异步处理
logger.add(..., enqueue=True)
```

**优势**
- 日志 I/O 不阻塞主线程
- 提高程序性能
- 适合高频日志场景

### 完整堆栈跟踪

```python
# backtrace=True, diagnose=True
logger.add(..., backtrace=True, diagnose=True)
```

**功能**
- `backtrace=True`: 显示完整的函数调用链
- `diagnose=True`: 提供详细的变量值和上下文

**示例**

```python
from loguru import logger

def inner():
    x = 10
    raise ValueError("错误")

def outer():
    inner()

try:
    outer()
except Exception:
    logger.exception("发生异常")
```

**输出**（带完整堆栈和变量值）

```
2025-11-04 14:30:25.123 | ERROR | main:outer:15 - 发生异常
Traceback (most recent call last):
  File "main.py", line 13, in <module>
    outer()
    └ <function outer at 0x...>
  File "main.py", line 11, in outer
    inner()
    └ <function inner at 0x...>
  File "main.py", line 7, in inner
    raise ValueError("错误")
ValueError: 错误
```

### 结构化日志

```python
from loguru import logger

# 添加额外上下文
logger.bind(epoch=10, lr=0.001).info("训练开始")

# 输出: ... | epoch=10 lr=0.001 | 训练开始
```

### 禁用特定模块的日志

```python
from loguru import logger

# 禁用第三方库的日志
logger.disable("transformers")
logger.disable("urllib3")

# 重新启用
logger.enable("transformers")
```

## 使用场景

### 场景 1: 开发环境

```python
from utils import setup_logging

# 控制台和文件都显示详细日志
setup_logging(
    log_dir='./logs',
    console_level='DEBUG',
    file_level='DEBUG'
)
```

**特点**
- 查看所有调试信息
- 方便排查问题
- 适合交互式开发

### 场景 2: 生产环境

```python
from utils import setup_logging

# 控制台只显示重要信息，文件记录详细日志
setup_logging(
    log_dir='./production_logs',
    console_level='WARNING',
    file_level='INFO'
)
```

**特点**
- 减少控制台输出
- 避免日志刷屏
- 保留文件日志便于回溯

### 场景 3: 训练脚本

```python
from utils import setup_logging

# 平衡配置
setup_logging(
    log_dir='./training_logs',
    console_level='INFO',
    file_level='DEBUG'
)
```

**特点**
- 控制台显示训练进度
- 文件记录详细调试信息
- 推荐配置

### 场景 4: 仅控制台输出

```python
from utils import setup_logging

# 禁用文件日志
setup_logging(
    log_dir='/tmp/throw_away',
    console_level='INFO',
    file_level='CRITICAL'  # 几乎不会记录
)
```

**特点**
- 快速测试
- 不产生日志文件
- 适合临时脚本

## 与其他工具集成

### 与配置模块集成

```python
from utils import setup_config, setup_logging

config = setup_config(DEFAULT_CONFIG, 'config.yaml', {})

setup_logging(
    log_dir=config.logging.log_dir,
    console_level=config.logging.console_level,
    file_level=config.logging.file_level
)
```

### 与异常装饰器集成

```python
from utils import setup_logging, log_errors, NtfyNotifier
from loguru import logger

setup_logging()
notifier = NtfyNotifier()

@log_errors(notifier=notifier, re_raise=True)
def main():
    logger.info("程序开始")
    # ... 训练代码 ...
    logger.success("程序完成")

if __name__ == '__main__':
    main()
```

### 与训练循环集成

```python
from utils import setup_logging, Trainer
from loguru import logger

setup_logging(
    log_dir='./logs',
    console_level='INFO',
    file_level='DEBUG'
)

# Trainer 内部会自动使用配置好的 logger
trainer = Trainer(model, optimizer, criterion, device)
trainer.fit(train_loader, val_loader, epochs=100)
```

## 设计原则

### 简单易用
- 一行代码完成配置
- 默认参数适合大多数场景
- 无需手动管理处理器

### 自动管理
- 自动轮转和清理
- 自动编码处理（UTF-8）
- 自动异步写入

### 灵活配置
- 分别控制控制台和文件级别
- 支持自定义日志目录
- 支持多次重新配置

### 生产就绪
- 异步日志不阻塞主线程
- 自动限制日志文件大小
- 完整的异常堆栈跟踪
