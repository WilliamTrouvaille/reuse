#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/1 18:08
@author: William_Trouvaille
@function: 检查点管理
@detail: 这个管理器是通用的，它只负责保存和加载字典，而不关心字典中的具体内容。
"""

import glob
import os
import re
from typing import Dict, Any, Optional

import torch
from loguru import logger


class CheckpointManager:
    # --- 核心文件名 ---
    BEST_MODEL_NAME = "best_model.pth"
    EPOCH_CKPT_PREFIX = "checkpoint_epoch_"
    INTERRUPT_CKPT_NAME = "interrupt_checkpoint.pth"

    def __init__(self, save_dir: str = 'checkpoint', device: str = 'cpu', max_to_keep: int = 5):
        """
        初始化检查点管理器。

        参数:
            save_dir (str): 所有检查点保存的根目录。
            device (str): 加载检查点时要映射到的设备 (例如 'cuda' 或 'cpu')。
            max_to_keep (int): 滚动保存 epoch 检查点的最大数量。
        """
        self.save_dir = os.path.abspath(save_dir)
        self.device = device
        self.max_to_keep = max(1, max_to_keep)  # 至少保留1个

        # 确保保存目录存在
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            logger.info(f"CheckpointManager 初始化，保存目录: {self.save_dir}")
            logger.info(f"将滚动保留最新的 {self.max_to_keep} 个 epoch 检查点。")
        except OSError as e:
            logger.error(f"无法创建检查点目录: {self.save_dir}。错误: {e}")
            raise

    def _save(self, state: Dict[str, Any], filename: str) -> str:
        """
        私有的核心保存函数。

        参数:
            state (dict): 要保存的状态字典 (例如 model.state_dict(), optim...)
            filename (str): 要保存的文件名 (例如 'best_model.pth')

        返回:
            str: 完整的文件路径。
        """
        # 确保目录存在（以防被外部删除）
        os.makedirs(self.save_dir, exist_ok=True)

        filepath = os.path.join(self.save_dir, filename)
        tmp_filepath = f"{filepath}.tmp"

        try:
            # 使用临时文件和重命名来确保原子性，防止在保存大文件时被中断导致文件损坏
            torch.save(state, tmp_filepath)
            os.replace(tmp_filepath, filepath)  # 原子操作

            logger.debug(f"检查点已保存至: {filepath}")
        except Exception as e:
            logger.error(f"保存检查点失败: {filepath}。错误: {e}")

            # 清理临时文件
            if os.path.exists(tmp_filepath):
                try:
                    os.remove(tmp_filepath)
                    logger.debug(f"已清理临时文件: {tmp_filepath}")
                except OSError as cleanup_error:
                    logger.warning(f"清理临时文件失败: {cleanup_error}")

            # 可以在这里触发 ntfy 通知
        return filepath

    def _load(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        私有的核心加载函数。
        """
        filepath = os.path.join(self.save_dir, filename)
        if not os.path.exists(filepath):
            logger.debug(f"检查点文件未找到: {filepath}")
            return None

        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            logger.info(f"成功加载检查点: {filepath} (映射到 {self.device})")
            return checkpoint
        except Exception as e:
            logger.error(f"加载检查点失败: {filepath}。文件可能已损坏。错误: {e}")
            return None

    def _cleanup_epoch_checkpoints(self):
        """
        (私有) 清理旧的 epoch 检查点，只保留最新的 `max_to_keep` 个。
        """
        logger.debug(f"开始清理旧的 epoch 检查点，保留 {self.max_to_keep} 个...")

        # 1. 查找所有匹配的 epoch 检查点
        pattern = os.path.join(self.save_dir, f"{self.EPOCH_CKPT_PREFIX}*.pth")
        epoch_files = glob.glob(pattern)

        if len(epoch_files) <= self.max_to_keep:
            logger.debug("检查点数量未达上限，无需清理。")
            return

        # 2. 从文件名中解析 epoch 编号
        parsed_files = []
        for f in epoch_files:
            match = re.search(rf"{self.EPOCH_CKPT_PREFIX}(\d+).pth", os.path.basename(f))
            if match:
                parsed_files.append((int(match.group(1)), f))

        # 3. 按 epoch 编号降序排序 (最新的在前)
        parsed_files.sort(key=lambda x: x[0], reverse=True)

        # 4. 确定要删除的文件
        files_to_delete = [f for _, f in parsed_files[self.max_to_keep:]]

        # 5. 执行删除
        for f in files_to_delete:
            try:
                os.remove(f)
                logger.info(f"已删除旧检查点: {os.path.basename(f)}")
            except OSError as e:
                logger.error(f"删除检查点失败: {f}。错误: {e}")

    def _get_latest_epoch_file(self) -> Optional[str]:
        """(私有) 查找最新的 epoch 检查点文件"""
        pattern = os.path.join(self.save_dir, f"{self.EPOCH_CKPT_PREFIX}*.pth")
        epoch_files = glob.glob(pattern)

        if not epoch_files:
            return None

        parsed_files = []
        for f in epoch_files:
            match = re.search(rf"{self.EPOCH_CKPT_PREFIX}(\d+).pth", os.path.basename(f))
            if match:
                parsed_files.append((int(match.group(1)), f))

        if not parsed_files:
            return None

        parsed_files.sort(key=lambda x: x[0], reverse=True)
        return os.path.basename(parsed_files[0][1])  # 返回文件名

    # --- 公共 API ---

    def save_best_model(self, state: Dict[str, Any], metric: float):
        """
        保存 "最佳模型" 检查点。
        (由调用方逻辑判断是否为最佳)

        参数:
            state (dict): 要保存的状态字典。
            metric (float): 用于日志记录的当前最佳指标值。
        """
        logger.success(f"发现新的最佳模型！指标: {metric:.6f}。正在保存...")
        self._save(state, self.BEST_MODEL_NAME)

    def load_best_model(self) -> Optional[Dict[str, Any]]:
        """
        加载 "最佳模型" 检查点。
        """
        logger.info(f"正在加载最佳模型 ({self.BEST_MODEL_NAME})...")
        return self._load(self.BEST_MODEL_NAME)

    def save_epoch_checkpoint(self, state: Dict[str, Any], epoch: int):
        """
        保存特定 epoch 的检查点，并自动触发滚动清理。

        参数:
            state (dict): 要保存的状态字典。
            epoch (int): 当前的 epoch 编号。
        """
        filename = f"{self.EPOCH_CKPT_PREFIX}{epoch}.pth"
        logger.info(f"正在保存 Epoch {epoch} 检查点...")
        self._save(state, filename)

        # 立即执行清理
        self._cleanup_epoch_checkpoints()

    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        加载最新的检查点以恢复训练。

        加载优先级:
        1. `interrupt_checkpoint.pth` (如果存在)
        2. 最新的 `checkpoint_epoch_X.pth` (如果存在)

        返回:
            dict 或 None: 最新的检查点状态，如果不存在则返回 None。
        """
        logger.info("正在查找最新的 epoch 检查点以恢复训练...")

        # 1. (最高优先级) 检查中断文件
        interrupt_file = os.path.join(self.save_dir, self.INTERRUPT_CKPT_NAME)
        if os.path.exists(interrupt_file):
            logger.warning(f"检测到中断检查点 ({self.INTERRUPT_CKPT_NAME})，将从此中断点恢复。")

            checkpoint = self._load(self.INTERRUPT_CKPT_NAME)

            # (重要) 加载成功后，删除该文件，防止下次重复加载
            if checkpoint:
                try:
                    os.remove(interrupt_file)
                    logger.info(f"已删除中断检查点: {self.INTERRUPT_CKPT_NAME} (防止重复加载)")
                except OSError as e:
                    logger.error(f"删除中断检查点失败: {e}")
            return checkpoint

        # 2. (第二优先级) 检查最新的 Epoch 文件
        latest_epoch_filename = self._get_latest_epoch_file()
        if latest_epoch_filename:
            logger.info(f"找到最新的 Epoch 检查点: {latest_epoch_filename}")
            return self._load(latest_epoch_filename)

        # 3. (无) 未找到任何检查点
        logger.warning("未找到可恢复的检查点 (既没有中断点，也没有 epoch 存档)。")
        return None

    def save_interrupt_checkpoint(self, state: Dict[str, Any]):
        """
        (在 Ctrl+C 时调用) 保存训练中断时的快照。
        """
        logger.warning(f"正在保存中断检查点 ({self.INTERRUPT_CKPT_NAME})...")
        self._save(state, self.INTERRUPT_CKPT_NAME)
        logger.critical("中断检查点已保存。程序即将退出。")
