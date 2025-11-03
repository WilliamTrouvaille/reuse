#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/1 16:18
@author  : William_Trouvaille
@function: NTFY 通知工具
"""

import requests
from loguru import logger
from requests.exceptions import RequestException
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError
)


class NtfyNotifier:
    """
    一个封装了 ntfy.sh HTTP API 的通知器。

    Attributes:
        topic (str): (硬编码) 目标 ntfy 频道。
        server_url (str): ntfy 服务器地址。
        session (requests.Session): 用于 HTTP 请求的会话对象。
    """

    # 频道默认硬编码
    TOPIC = "trouvaille_william_yK5aEPt72KfT6m9z"

    def __init__(self, server_url: str = "https://ntfy.sh", topic: str = TOPIC):
        """
        初始化通知器。

        参数:
            server_url (str): ntfy 服务器的 URL。默认为公共服务器。
        """
        self.server_url = server_url
        self.topic = topic
        self.topic_url = f"{self.server_url}/{self.topic}"

        # 1. 初始化 requests.Session 以复用连接
        self.session = requests.Session()

        # 2. 按照要求，为所有请求启用 Markdown
        self.session.headers.update({"Markdown": "yes"})

        logger.info(f"NtfyNotifier 初始化完毕。")
        logger.debug(f"Ntfy 主题 URL: {self.topic_url}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(RequestException),
        before_sleep=before_sleep_log(logger, 'WARNING'),
        reraise=True
    )
    def _send_with_retry(self, message: str, title: str, headers: dict) -> bool:
        """
        (私有) 带重试机制的实际发送方法。

        使用 tenacity 库实现自动重试：
        - 最多重试 3 次
        - 指数退避策略：初始等待 2 秒，最长 10 秒
        - 只对 RequestException 进行重试
        - 自动记录重试日志

        参数:
            message (str): 编码后的消息主体
            title (str): 消息标题（用于日志）
            headers (dict): HTTP 请求头

        返回:
            bool: 发送成功返回 True

        异常:
            RequestException: 重试次数用尽后抛出
        """
        logger.debug(f"准备发送 Ntfy 通知...")

        response = self.session.post(
            self.topic_url,
            data=message.encode('utf-8'),
            headers=headers,
            timeout=10
        )

        # 检查 HTTP 错误 (例如 4xx, 5xx)
        response.raise_for_status()

        logger.success(f"Ntfy 通知已发送: '{title}'")
        return True

    def send(self, message: str, title: str, priority: str, tags: list[str] = None) -> bool:
        """
        发送通知的核心方法（带自动重试机制）。

        使用 tenacity 库实现智能重试：
        - 自动重试最多 3 次
        - 指数退避策略（2s, 4s, 8s）
        - 只对网络相关异常重试
        - 自动记录重试日志

        参数:
            message (str): 消息主体 (支持 Markdown)。
            title (str): 消息标题。
            priority (str): 优先级 ("low", "default", "high", "max" 或 1-5)。
            tags (list[str], optional): ntfy 标签 (例如用于表情符号)。

        返回:
            bool: 消息是否发送成功。
        """
        if tags is None:
            tags = []

        # ntfy 的 Header 是区分大小写的
        headers = {
            "Title": title.encode('utf-8'),
            "Priority": priority,
            "Tags": ",".join(tags).encode('utf-8')
        }

        try:
            return self._send_with_retry(message, title, headers)
        except RetryError as e:
            # 重试次数用尽
            original_exception = e.last_attempt.exception()
            logger.error(f"发送 Ntfy 通知失败（已重试 3 次）。错误: {original_exception}")

            # 尝试获取服务器响应
            if hasattr(original_exception, 'response') and original_exception.response is not None:
                try:
                    error_text = original_exception.response.text
                except UnicodeDecodeError:
                    error_text = original_exception.response.content
                logger.error(f"Ntfy 服务器响应: {error_text}")

            return False
        except RequestException as e:
            # 单次请求失败（不应到达这里，因为 tenacity 会捕获）
            logger.error(f"发送 Ntfy 通知失败。错误: {e}")
            return False

    # --- 预定义的消息类型 ---

    def notify_start(self, message: str = "训练已开始。"):
        """
        (低优先级) 发送训练开始通知。

        参数:
            message (str): 要发送的具体消息。
        """
        logger.info("发送 '训练开始' 通知...")
        self.send(
            message=message,
            title="🏃 训练开始",
            priority="low",  # 2 (low)
            tags=["runner"]
        )

    def notify_success(self, message: str = "训练已成功完成。"):
        """
        (次高优先级) 发送训练成功通知。

        参数:
            message (str): 要发送的具体消息。
        """
        logger.info("发送 '训练成功' 通知...")
        self.send(
            message=message,
            title="✅ 训练成功",
            priority="high",  # 4 (high)
            tags=["white_check_mark"]
        )

    def notify_error(self, message: str, error_details: str = None):
        """
        (最高优先级) 发送训练报错或中断通知。

        参数:
            message (str): 简短的错误摘要 (例如: "训练在 Epoch 50 失败")。
            error_details (str, optional): 详细的错误信息，例如 traceback。
                                           将使用 Markdown 代码块格式化。
        """
        logger.warning("发送 '训练报错' 通知...")

        # 使用 Markdown 格式化错误详情
        full_message = f"**错误摘要:**\n{message}\n\n"

        if error_details:
            # 使用 Markdown (```) 来格式化代码/traceback
            # 限制详细信息的长度，以防消息体过大
            if len(error_details) > 3000:
                error_details = error_details[:3000] + "\n... (错误信息已截断)"

            full_message += f"**详细信息:**\n```\n{error_details}\n```"

        self.send(
            message=full_message.strip(),
            title="❌ 训练失败",
            priority="max",  # 5 (urgent/max)
            tags=["x"]
        )
