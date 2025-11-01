#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/1 15:12
@version : 1.0.0
@author  : William_Trouvaille
@function: 主文件，用于测试
"""


import time
import traceback
from loguru import logger
from utils import setup_logging, NtfyNotifier

# --- 模拟设置 ---
# 将此项更改为 True 来测试错误通知
SIMULATE_ERROR = True
SIMULATION_DURATION_SECONDS = 3
# ------------------

def main():
    """
    主执行函数
    """

    # 1. (必需) 配置日志记录器
    setup_logging(
        log_dir="logs",
        console_level="DEBUG",
        file_level="DEBUG"
    )

    logger.info("NtfyNotifier 测试脚本启动。")

    # 2. 初始化 Notifier
    try:
        notifier = NtfyNotifier()
    except Exception as e:
        logger.error(f"初始化 NtfyNotifier 失败: {e}")
        logger.error("请检查 uv.lock 中的 'requests' 依赖是否已安装 (uv add requests)。")
        return

    # 3. 使用 try...except...else 结构来捕获所有状态
    try:
        # --- 3a. 发送开始通知 ---
        logger.info("发送 '训练开始' 通知...")
        start_message = f"测试任务已启动。\n" \
                        f"模式: {'模拟错误' if SIMULATE_ERROR else '模拟成功'}\n" \
                        f"预计持续时间: {SIMULATION_DURATION_SECONDS} 秒"
        notifier.notify_start(start_message)

        # --- 3b. 模拟长时间运行的任务 ---
        logger.info(f"开始模拟工作，持续 {SIMULATION_DURATION_SECONDS} 秒...")
        for i in range(SIMULATION_DURATION_SECONDS):
            logger.debug(f"模拟工作... {i + 1}/{SIMULATION_DURATION_SECONDS}")
            time.sleep(1)

        # --- 3c. 模拟错误 (如果已配置) ---
        if SIMULATE_ERROR:
            logger.warning("正在模拟一个运行时错误 (ValueError)...")
            # 这将触发下面的 `except Exception` 块
            raise ValueError("这是一个用于测试 Ntfy 错误通知的模拟异常。")

        logger.info("模拟工作完成，未发生错误。")

    except KeyboardInterrupt:
        # --- 4. 处理用户中断 (Ctrl+C) ---
        logger.warning("检测到用户中断 (KeyboardInterrupt)！")

        # 按照要求，发送最高优先级的错误通知
        notifier.notify_error(
            message="任务被用户手动中断 (Ctrl+C)。",
            error_details="KeyboardInterrupt"
        )

    except Exception as e:
        # --- 5. 处理所有其他异常 ---
        logger.error(f"捕获到未处理的异常: {e}")

        # 获取完整的堆栈跟踪信息
        error_details = traceback.format_exc()
        logger.debug(f"堆栈跟踪:\n{error_details}")

        # 按照要求，发送最高优先级的错误通知
        notifier.notify_error(
            message=f"任务因运行时错误而失败: {type(e).__name__}",
            error_details=error_details  # 传递完整的堆栈跟踪
        )

    else:
        # --- 6. 处理成功 (仅当 try 块未发生异常时) ---
        logger.success("任务成功完成。")

        # 按照要求，发送次高优先级的成功通知
        success_message = f"测试任务已成功完成。\n" \
                          f"总运行时长: {SIMULATION_DURATION_SECONDS} 秒。"
        notifier.notify_success(success_message)

    finally:
        # --- 7. 清理 (无论如何都会执行) ---
        logger.info("NtfyNotifier 测试脚本执行完毕。")


if __name__ == "__main__":
    main()
