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
* **`.notify_error(message, error_details)`**: (最高优先级) 发送 "训练失败" 通知。`error_details` (可选) 可传入 traceback 字符串，将使用 Markdown 代码块格式化。
* **`.send(message, title, priority, tags)`**: (高级) 发送自定义通知。