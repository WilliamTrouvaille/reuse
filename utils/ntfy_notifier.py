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


class NtfyNotifier:
    """
    一个封装了 ntfy.sh HTTP API 的通知器。

    Attributes:
        topic (str): (硬编码) 目标 ntfy 频道。
        server_url (str): ntfy 服务器地址。
        session (requests.Session): 用于 HTTP 请求的会话对象。
    """

    # 将频道硬编码
    TOPIC = "trouvaille_william_yK5aEPt72KfT6m9z"

    def __init__(self, server_url: str = "https://ntfy.sh"):
        """
        初始化通知器。

        参数:
            server_url (str): ntfy 服务器的 URL。默认为公共服务器。
        """
        self.server_url = server_url
        self.topic_url = f"{self.server_url}/{self.TOPIC}"

        # 1. 初始化 requests.Session 以复用连接
        self.session = requests.Session()

        # 2. 按照要求，为所有请求启用 Markdown
        self.session.headers.update({"Markdown": "yes"})

        logger.info(f"NtfyNotifier 初始化完毕。")
        logger.debug(f"Ntfy 主题 URL: {self.topic_url}")

    def send(self, message: str, title: str, priority: str, tags: list[str] = None):
        """
        发送通知的核心方法。

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
            "Priority": priority,  # Priority 总是 ASCII，str 格式是安全的
            "Tags": ",".join(tags).encode('utf-8')
        }

        try:
            logger.debug(f"准备发送 Ntfy 通知 (Priority: {priority})...")

            response = self.session.post(
                self.topic_url,
                data=message.encode('utf-8'),  # 推荐使用 UTF-8 编码发送
                headers=headers,
                timeout=10
            )

            # 检查 HTTP 错误 (例如 4xx, 5xx)
            response.raise_for_status()

            logger.success(f"Ntfy 通知已发送: '{title}'")
            return True

        except RequestException as e:
            # 捕获所有 requests 相关的异常 (连接、超时、HTTP错误等)
            logger.error(f"发送 Ntfy 通知失败。错误: {e}")
            if hasattr(e, 'response') and e.response is not None:
                # 尝试解码响应，如果失败则显示原始字节
                try:
                    error_text = e.response.text
                except UnicodeDecodeError:
                    error_text = e.response.content
                logger.error(f"Ntfy 服务器响应: {error_text}")
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
