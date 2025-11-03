#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/1 18:22
@author  : William_Trouvaille
@function: CheckpointManager 测试脚本
"""

import sys
import time
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from utils import setup_logging, CheckpointManager

# --- 模拟设置 ---
CHECKPOINT_DIR = "./checkpoints"
TOTAL_EPOCHS = 10
SIMULATE_IMPROVEMENT = True # 模拟指标提升
SIMULATE_SLEEP_S = 1.5 # 模拟训练
# ------------------

class SimpleModel(nn.Module):
    """一个用于测试的简单模型"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

def main():
    """
    主执行函数
    """

    # 1. (必需) 配置日志记录器
    setup_logging(console_level="INFO", file_level="DEBUG")

    logger.info("CheckpointManager 测试脚本启动。")

    # --- 2. 清理旧的测试目录 (可选) ---
    # if os.path.exists(CHECKPOINT_DIR):
    #     logger.warning(f"正在删除旧的测试目录: {CHECKPOINT_DIR}")
    #     shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)

    # --- 3. 初始化 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_to_keep = 3 # 满足需求：滚动保存最近3次

    model = SimpleModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 初始化 CheckpointManager
    ckpt_manager = CheckpointManager(
        save_dir=CHECKPOINT_DIR,
        device=device,
        max_to_keep=max_to_keep
    )

    # --- 4. 尝试恢复训练 (测试 load_latest_checkpoint) ---
    start_epoch = 0
    best_metric = 0.0

    logger.info("尝试从最新检查点恢复...")
    latest_ckpt = ckpt_manager.load_latest_checkpoint()

    if latest_ckpt:
        try:
            model.load_state_dict(latest_ckpt['model_state'])
            optimizer.load_state_dict(latest_ckpt['optimizer_state'])
            start_epoch = latest_ckpt['epoch'] + 1
            best_metric = latest_ckpt['metric']
            logger.success(f"成功从 Epoch {start_epoch - 1} 恢复训练。")
        except Exception as e:
            logger.error(f"加载检查点失败: {e}。将从头开始。")
            start_epoch = 0
            best_metric = 0.0
    else:
        logger.info("未找到检查点，从头开始训练 (Epoch 0)。")

    # --- 5. 模拟训练循环 ---
    logger.info(f"将从 Epoch {start_epoch} 训练到 Epoch {TOTAL_EPOCHS - 1}...")

    try:
        for epoch in range(start_epoch, TOTAL_EPOCHS):
            logger.info(f"--- 模拟 Epoch {epoch} ---")

            # 5a. 模拟训练
            # (空训练，只休眠)
            time.sleep(SIMULATE_SLEEP_S)

            # 5b. 模拟验证
            # 模拟一个不断提升的指标
            current_metric = best_metric + (0.1 if SIMULATE_IMPROVEMENT else -0.1)
            logger.info(f"Epoch {epoch} 完成。指标: {current_metric:.4f}")

            # 5c. 准备 state 字典 (由 main.py 负责)
            state = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'metric': current_metric
            }

            # 5d. 保存滚动检查点 (测试 save_epoch_checkpoint 和 cleanup)
            ckpt_manager.save_epoch_checkpoint(state, epoch)

            # 5e. 保存最佳模型 (测试 save_best_model)
            if current_metric > best_metric:
                best_metric = current_metric
                ckpt_manager.save_best_model(state, best_metric)

    except KeyboardInterrupt:
        # --- 6. (测试) 处理 Ctrl+C ---
        logger.critical("检测到用户中断 (KeyboardInterrupt)！")

        # 准备最后的状态
        # (注意：此时 epoch 变量是中断时的 epoch)
        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'metric': current_metric
        }

        # 测试 save_interrupt_checkpoint
        ckpt_manager.save_interrupt_checkpoint(state)
        sys.exit(0) # 退出

    except Exception as e:
        logger.error(f"模拟训练中发生未捕获的异常: {e}")
        logger.exception("详细堆栈信息如下：")

    # --- 7. 结束检查 ---
    logger.success("模拟训练完成。")
    logger.info("=" * 40)
    logger.info(f"最终检查点目录内容: {CHECKPOINT_DIR}".center(40))

    final_files = os.listdir(CHECKPOINT_DIR)
    for f in sorted(final_files):
        logger.info(f"-> {f}")

    logger.info("=" * 40)
    logger.info(f"预期应只保留 {max_to_keep} 个 epoch 检查点 和 1 个 best_model.pth。")
    logger.info(f"例如: checkpoint_epoch_{TOTAL_EPOCHS-3}.pth, "
                f"checkpoint_epoch_{TOTAL_EPOCHS-2}.pth, "
                f"checkpoint_epoch_{TOTAL_EPOCHS-1}.pth")


if __name__ == '__main__':
    main()
