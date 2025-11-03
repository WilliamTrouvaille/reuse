# `logger_config` 日志配置模块使用指南

## 📚 目录
1. [快速开始](#快速开始)
2. [配置选项](#配置选项)
3. [常见场景](#常见场景)
4. [日志级别说明](#日志级别说明)
5. [高级用法](#高级用法)
6. [常见问题](#常见问题)

---

## 🚀 快速开始

### 最小化示例（1 行代码）

```python
from utils import setup_logging

# 使用默认配置
setup_logging()

# 现在可以使用 logger
from loguru import logger

logger.debug("这是调试信息")
logger.info("这是普通信息")
logger.warning("这是警告")
logger.error("这是错误")
logger.success("这是成功消息")
```

就这么简单！`setup_logging` 会自动配置：
- ✅ 彩色控制台输出
- ✅ 自动按日期轮转的日志文件
- ✅ 异步日志记录（不阻塞主线程）
- ✅ 完整的堆栈跟踪

---

## ⚙️ 配置选项

### 基础配置

```python
from utils import setup_logging

setup_logging(
    log_dir='./logs',           # 日志文件保存目录
    console_level='INFO',        # 控制台输出级别
    file_level='DEBUG'          # 文件输出级别
)
```

**参数说明**：
- `log_dir` (str): 日志文件保存目录，默认 `"logs"`
- `console_level` (str): 控制台最低日志级别，默认 `"INFO"`
- `file_level` (str): 文件最低日志级别，默认 `"DEBUG"`

**日志文件命名**：
- 格式：`log_YYYYMMDD.log`
- 示例：`log_20251104.log`
- 特点：每天自动创建新文件

**日志轮转规则**：
- ✅ 文件大小达到 10 MB 时自动轮转
- ✅ 保留最近 10 天的日志
- ✅ 使用 UTF-8 编码

---

## 📖 常见场景

### 场景 1: 开发环境（详细日志）

```python
from utils import setup_logging

# 控制台和文件都显示详细日志
setup_logging(
    log_dir='./logs',
    console_level='DEBUG',  # 控制台显示所有信息
    file_level='DEBUG'      # 文件记录所有信息
)
```

**适用于**：
- 🔧 开发调试
- 🐛 Bug 排查
- 🧪 功能测试

### 场景 2: 生产环境（精简控制台）

```python
from utils import setup_logging

# 控制台只显示重要信息，文件记录详细日志
setup_logging(
    log_dir='./production_logs',
    console_level='WARNING',  # 控制台只显示警告和错误
    file_level='INFO'         # 文件记录 INFO 及以上
)
```

**适用于**：
- 🚀 生产部署
- 📊 长期运行任务
- 🖥️ 服务器应用

### 场景 3: 训练脚本（平衡日志）

```python
from utils import setup_logging

# 控制台显示关键信息，文件记录详细日志
setup_logging(
    log_dir='./training_logs',
    console_level='INFO',   # 控制台显示 INFO 及以上
    file_level='DEBUG'      # 文件记录所有信息
)

# 在训练中使用
from loguru import logger

logger.info("开始训练...")
logger.debug(f"批次大小: {batch_size}, 学习率: {lr}")
logger.success(f"Epoch 1 完成: 准确率 {acc:.2f}%")
logger.warning("GPU 内存使用率超过 90%")
logger.error("训练失败: CUDA out of memory")
```

**推荐配置**：
- ✅ 控制台：`INFO`（不太吵闹）
- ✅ 文件：`DEBUG`（方便回溯）

### 场景 4: 禁用文件日志（仅控制台）

```python
from utils import setup_logging

# 只配置控制台
setup_logging(
    log_dir='/tmp/throw_away',  # 临时目录
    console_level='INFO',
    file_level='CRITICAL'  # 设置为最高级别，基本不会记录
)
```

**适用于**：
- 🧪 快速测试
- 💻 交互式开发

---

## 📊 日志级别说明

| 级别 | 数值 | 用途 | 示例 |
|------|-----|------|------|
| `DEBUG` | 10 | 详细调试信息 | 变量值、函数调用 |
| `INFO` | 20 | 一般信息 | 程序进度、状态更新 |
| `SUCCESS` | 25 | 成功消息 | 操作完成、验证通过 |
| `WARNING` | 30 | 警告信息 | 资源不足、配置问题 |
| `ERROR` | 40 | 错误信息 | 异常捕获、操作失败 |
| `CRITICAL` | 50 | 严重错误 | 系统崩溃、数据损坏 |

### 使用建议

```python
from loguru import logger

# DEBUG: 详细的调试信息
logger.debug(f"收到输入: shape={inputs.shape}, dtype={inputs.dtype}")

# INFO: 程序流程信息
logger.info("开始加载数据集...")
logger.info(f"训练集大小: {len(train_dataset)}")

# SUCCESS: 成功消息（loguru 特有）
logger.success("模型训练完成！")
logger.success(f"最佳准确率: {best_acc:.2f}%")

# WARNING: 潜在问题
logger.warning("GPU 内存使用率: 95%")
logger.warning("学习率可能过大，考虑降低")

# ERROR: 错误但可继续
logger.error("保存检查点失败，将在下个 epoch 重试")

# CRITICAL: 严重错误，需要立即处理
logger.critical("CUDA 不可用，无法继续训练")
```

---

## 🔧 高级用法

### 1. 在配置文件中使用

```yaml
# config.yaml
logging:
  log_dir: "./logs"
  console_level: "INFO"
  file_level: "DEBUG"
```

```python
from utils import setup_config, setup_logging

config = setup_config(DEFAULT_CONFIG, 'config.yaml', {})

# 从配置初始化日志
setup_logging(
    log_dir=config.logging.log_dir,
    console_level=config.logging.console_level,
    file_level=config.logging.file_level
)
```

### 2. 异常日志（自动堆栈跟踪）

```python
from loguru import logger

try:
    result = risky_operation()
except Exception as e:
    # 使用 logger.exception 自动记录堆栈
    logger.exception("操作失败")
    # 等价于：
    # logger.error(f"操作失败: {e}")
    # traceback.print_exc()
```

### 3. 结构化日志（使用上下文）

```python
from loguru import logger

# 添加额外上下文
logger.bind(epoch=10, lr=0.001).info("训练开始")
# 输出: ... | epoch=10 lr=0.001 | 训练开始

# 在函数中使用
def train_epoch(epoch, lr):
    log = logger.bind(epoch=epoch, lr=lr)
    log.info("Epoch 开始")
    log.success("Epoch 完成")
```

### 4. 自定义日志格式（不推荐修改）

如果确实需要自定义格式，可以修改 `logger_config.py` 中的 `log_format` 变量：

```python
# 默认格式
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
```

**输出示例**：
```
2025-11-04 14:30:22.123 | INFO     | main:train:45 - 开始训练...
2025-11-04 14:30:23.456 | SUCCESS  | trainer:fit:102 - Epoch 1 完成
```

### 5. 多次调用 setup_logging（幂等性）

```python
from utils import setup_logging

# 第一次调用
setup_logging(log_dir='./logs1', console_level='INFO')

# 第二次调用（会覆盖之前的配置）
setup_logging(log_dir='./logs2', console_level='DEBUG')

# 最终生效的是最后一次配置
```

**注意**：
- ✅ `setup_logging` 是幂等的
- ✅ 自动移除旧的处理器
- ✅ 不会导致日志重复

---

## ❓ 常见问题

### Q1: 日志文件在哪里？

**A**: 默认在 `./logs/` 目录下，文件名格式为 `log_YYYYMMDD.log`。

```bash
logs/
├── log_20251103.log
├── log_20251104.log
└── log_20251104.log.1  # 轮转文件
```

### Q2: 如何只输出到控制台？

**A**: 将 `file_level` 设置为 `CRITICAL`：

```python
setup_logging(console_level='INFO', file_level='CRITICAL')
```

### Q3: 日志文件太大怎么办？

**A**: 已自动配置轮转：
- 文件大小达到 10 MB 时自动创建新文件
- 只保留最近 10 天的日志

如需调整，修改 `logger_config.py`：
```python
rotation="10 MB",    # 改为 "5 MB" 或 "50 MB"
retention="10 days"  # 改为 "7 days" 或 "30 days"
```

### Q4: 如何在多模块项目中使用？

**A**: 只需在主入口调用一次 `setup_logging`：

```python
# main.py
from utils import setup_logging

setup_logging()

# 然后在任何模块中直接使用
# module_a.py
from loguru import logger
logger.info("来自模块 A 的日志")

# module_b.py
from loguru import logger
logger.info("来自模块 B 的日志")
```

### Q5: 日志颜色显示不正常？

**A**: 检查终端是否支持 ANSI 颜色：
- ✅ 支持：Linux/Mac 终端、Windows Terminal、VSCode 终端
- ❌ 不支持：Windows CMD（部分版本）

如果不支持，颜色代码会显示为原始字符。

### Q6: 如何禁用某个模块的日志？

**A**: 使用 logger 的过滤功能：

```python
from loguru import logger

# 禁用特定模块
logger.disable("transformers")  # 禁用 transformers 库的日志
logger.disable("urllib3")       # 禁用 urllib3 的日志

# 重新启用
logger.enable("transformers")
```

---

## 📋 最佳实践

### 1. 在训练脚本中使用

```python
from utils import setup_logging, setup_config
from loguru import logger

def main():
    # 1. 首先配置日志
    setup_logging(log_dir='./logs', console_level='INFO', file_level='DEBUG')

    # 2. 加载配置
    logger.info("加载配置...")
    config = setup_config(DEFAULT_CONFIG, 'config.yaml', {})
    logger.success(f"配置加载完成: {config.experiment.name}")

    # 3. 设置随机种子
    logger.info(f"设置随机种子: {config.experiment.seed}")
    set_random_seed(config.experiment.seed)

    # 4. 训练
    logger.info("开始训练...")
    try:
        trainer.fit(train_loader, val_loader, epochs=100)
        logger.success("训练完成！")
    except KeyboardInterrupt:
        logger.warning("训练被用户中断")
    except Exception as e:
        logger.exception("训练失败")
        raise

if __name__ == '__main__':
    main()
```

### 2. 记录关键操作

```python
from loguru import logger

# 开始/结束
logger.info("=" * 60)
logger.info("开始数据预处理...")
# ... 处理代码 ...
logger.success("数据预处理完成")

# 资源使用
from utils import log_memory_usage
log_memory_usage("训练前")

# 检查点保存
logger.info(f"保存检查点: epoch_{epoch}.pth")
checkpoint_manager.save_epoch_checkpoint(state, epoch)
logger.success("检查点保存成功")
```

### 3. 错误处理

```python
from loguru import logger

def load_model(path):
    try:
        logger.info(f"加载模型: {path}")
        model = torch.load(path)
        logger.success("模型加载成功")
        return model
    except FileNotFoundError:
        logger.error(f"模型文件不存在: {path}")
        return None
    except Exception as e:
        logger.exception("加载模型失败")
        raise
```

---

## 🎯 总结

`setup_logging` 的核心优势：
1. **简单易用**：一行代码完成配置
2. **自动管理**：日志轮转、编码、异步
3. **灵活配置**：分别控制控制台和文件级别
4. **开发友好**：彩色输出、完整堆栈
5. **生产就绪**：异步记录、自动清理

配合 `loguru` 的强大功能，让日志记录变得轻松愉快！🎉
