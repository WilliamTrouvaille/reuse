#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/4 00:15
@author  : William_Trouvaille
@function: 训练指标可视化器
@description: 提供 MetricsVisualizer 类，用于将训练指标绘制成各类图表，
              包括训练曲线、混淆矩阵、ROC 曲线、分类报告热图和耗时分布。
              完全解耦，不依赖任何机器学习框架，只接受标准数据格式。
"""
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    classification_report,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize


class MetricsVisualizer:
    """
    训练指标可视化器，负责将指标数据绘制成各类图表。

    核心功能:
        - 绘制训练曲线（Loss 曲线和准确率曲线）
        - 绘制混淆矩阵热图
        - 绘制 ROC 曲线（支持多分类）
        - 绘制分类报告热图
        - 绘制 epoch 耗时分布

    设计理念:
        - 完全解耦：所有方法接受标准的 NumPy 数组或 Python 列表
        - 统一样式：所有图表使用一致的风格和配置
        - 异常处理：单个图表失败不影响其他图表生成
    """

    # 统一的绘图样式配置
    DEFAULT_STYLE = {
        'figure.dpi': 100,
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
    }

    def __init__(
            self,
            style: str = 'seaborn-v0_8-darkgrid',
            dpi: int = 100,
            figure_size: tuple = (10, 6)
    ):
        """
        初始化可视化器。

        参数:
            style (str): matplotlib 样式名称。
            dpi (int): 图表分辨率（DPI）。
            figure_size (tuple): 默认图表尺寸 (width, height)。
        """
        self.style = style
        self.dpi = dpi
        self.figure_size = figure_size

        # --- 配置中文字体显示 ---
        # 设置中文字体（按优先级尝试，Windows系统字体优先）
        chinese_fonts = [
            'Microsoft YaHei',      # 微软雅黑（Windows默认，最常用）
            'Microsoft YaHei UI',   # 微软雅黑UI
            'SimHei',               # 黑体
            'SimSun',               # 宋体
            'KaiTi',                # 楷体
            'FangSong',             # 仿宋
        ]

        # 获取系统可用字体列表用于验证
        from matplotlib.font_manager import FontManager
        fm = FontManager()
        available_fonts = {f.name for f in fm.ttflist}

        selected_font = None
        font_set = False
        for font in chinese_fonts:
            if font in available_fonts:
                selected_font = font
                plt.rcParams['font.sans-serif'] = [font]
                font_set = True
                logger.debug(f"成功设置中文字体: {font}")
                break

        if not font_set:
            # 如果所有预定义字体都不可用，尝试找一个包含CJK的字体
            logger.warning("未找到预定义的中文字体，尝试查找其他CJK字体...")
            cjk_fonts = [f.name for f in fm.ttflist if 'CJK' in f.name or 'Chinese' in f.name]
            if cjk_fonts:
                selected_font = cjk_fonts[0]
                plt.rcParams['font.sans-serif'] = [selected_font]
                logger.info(f"使用备用字体: {selected_font}")
            else:
                logger.error("系统中未找到任何中文字体，中文将无法正常显示！")
                selected_font = 'sans-serif'  # 使用默认字体

        # 设置负号正常显示（避免显示为方块）
        plt.rcParams['axes.unicode_minus'] = False

        # 更新默认样式（关键：将字体配置也加入plot_config）
        self.plot_config = self.DEFAULT_STYLE.copy()
        self.plot_config['figure.dpi'] = dpi
        self.plot_config['font.sans-serif'] = [selected_font]  # 添加字体配置
        self.plot_config['axes.unicode_minus'] = False  # 添加负号配置

        # 尝试应用样式（如果不存在则使用默认）
        try:
            plt.style.use(style)
        except Exception as e:
            logger.warning(f"无法应用样式 '{style}'，使用默认样式。错误: {e}")

        logger.debug(
            f"MetricsVisualizer 已初始化: style={style}, dpi={dpi}, "
            f"figure_size={figure_size}"
        )

    def plot_training_curves(
            self,
            train_losses: List[float],
            val_losses: List[float],
            train_accs: List[float],
            val_accs: List[float],
            save_path: Union[str, Path],
            title: str = '训练过程曲线'
    ):
        """
        绘制训练曲线（Loss 和准确率）。

        生成 2x1 子图布局：上方为 Loss 曲线，下方为准确率曲线。

        参数:
            train_losses (list): 训练损失历史。
            val_losses (list): 验证损失历史。
            train_accs (list): 训练准确率历史。
            val_accs (list): 验证准确率历史。
            save_path (str | Path): 图表保存路径。
            title (str): 图表总标题。
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with plt.rc_context(self.plot_config):
                # 创建 2x1 子图
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figure_size[0], self.figure_size[1] * 1.2))
                epochs = range(1, len(train_losses) + 1)

                # 子图 1: Loss 曲线
                ax1.plot(epochs, train_losses, 'o-', label='训练损失', linewidth=2, markersize=4)
                if val_losses:
                    ax1.plot(epochs, val_losses, 's-', label='验证损失', linewidth=2, markersize=4)
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.set_title('损失曲线')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # 子图 2: 准确率曲线
                ax2.plot(epochs, train_accs, 'o-', label='训练准确率', linewidth=2, markersize=4)
                if val_accs:
                    ax2.plot(epochs, val_accs, 's-', label='验证准确率', linewidth=2, markersize=4)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('准确率 (%)')
                ax2.set_title('准确率曲线')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                # 设置总标题
                fig.suptitle(title, fontsize=16, fontweight='bold')
                plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间

                # 保存图表
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                plt.close(fig)

                logger.success(f"训练曲线已保存: {save_path}")

        except Exception as e:
            logger.error(f"绘制训练曲线失败: {e}")
            raise

    def plot_confusion_matrix(
            self,
            y_true: Union[np.ndarray, List],
            y_pred: Union[np.ndarray, List],
            class_names: List[str],
            save_path: Union[str, Path],
            normalize: bool = True,
            cmap: str = 'Blues',
            title: str = '混淆矩阵'
    ):
        """
        绘制混淆矩阵热图。

        参数:
            y_true (ndarray | list): 真实标签。
            y_pred (ndarray | list): 预测标签。
            class_names (list): 类别名称列表。
            save_path (str | Path): 图表保存路径。
            normalize (bool): 是否归一化显示（百分比）。
            cmap (str): 颜色映射方案。
            title (str): 图表标题。
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # 计算混淆矩阵
            cm = confusion_matrix(y_true, y_pred)

            # 归一化处理（可选）
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
                fmt = '.2%'  # 百分比格式
            else:
                fmt = 'd'  # 整数格式

            with plt.rc_context(self.plot_config):
                fig, ax = plt.subplots(figsize=self.figure_size)

                # 使用 seaborn 绘制热图（heatmap 自动处理颜色映射和标注）
                sns.heatmap(
                    cm,
                    annot=True,  # 显示数值
                    fmt=fmt,
                    cmap=cmap,
                    xticklabels=class_names,
                    yticklabels=class_names,
                    cbar_kws={'label': '比例' if normalize else '数量'},
                    ax=ax
                )

                ax.set_xlabel('预测标签')
                ax.set_ylabel('真实标签')
                ax.set_title(title)

                plt.tight_layout()
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                plt.close(fig)

                logger.success(f"混淆矩阵已保存: {save_path}")

        except Exception as e:
            logger.error(f"绘制混淆矩阵失败: {e}")
            raise

    def plot_roc_curve(
            self,
            y_true: Union[np.ndarray, List],
            y_proba: Union[np.ndarray, List],
            class_names: List[str],
            save_path: Union[str, Path],
            multi_class: str = 'ovr',
            title: str = 'ROC 曲线'
    ):
        """
        绘制 ROC 曲线（支持多分类）。

        对于多分类问题，使用 One-vs-Rest 或 One-vs-One 策略绘制每个类别的 ROC 曲线。

        参数:
            y_true (ndarray | list): 真实标签（整数编码）。
            y_proba (ndarray | list): 预测概率，形状 [N, C]（C 为类别数）。
            class_names (list): 类别名称列表。
            save_path (str | Path): 图表保存路径。
            multi_class (str): 多分类策略，'ovr' 或 'ovo'。
            title (str): 图表标题。
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            y_true = np.asarray(y_true)
            y_proba = np.asarray(y_proba)

            # 确定类别数量
            n_classes = len(class_names)

            # 处理二分类和多分类
            if n_classes == 2:
                # 二分类：只需使用正类的概率
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)

                with plt.rc_context(self.plot_config):
                    fig, ax = plt.subplots(figsize=self.figure_size)
                    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}', linewidth=2)
                    ax.plot([0, 1], [0, 1], 'k--', label='随机猜测', linewidth=1)
                    ax.set_xlabel('假正率 (FPR)')
                    ax.set_ylabel('真正率 (TPR)')
                    ax.set_title(title)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                    plt.close(fig)

            else:
                # 多分类：One-vs-Rest 策略
                # 将标签二值化（binarize labels for multi-class ROC）
                y_true_bin = label_binarize(y_true, classes=range(n_classes))

                with plt.rc_context(self.plot_config):
                    fig, ax = plt.subplots(figsize=self.figure_size)

                    # 为每个类别绘制 ROC 曲线
                    for i in range(n_classes):
                        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(
                            fpr, tpr,
                            label=f'{class_names[i]} (AUC = {roc_auc:.3f})',
                            linewidth=2
                        )

                    # 绘制随机猜测基准线
                    ax.plot([0, 1], [0, 1], 'k--', label='随机猜测', linewidth=1)

                    ax.set_xlabel('假正率 (FPR)')
                    ax.set_ylabel('真正率 (TPR)')
                    ax.set_title(title)
                    ax.legend(loc='lower right')
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                    plt.close(fig)

            logger.success(f"ROC 曲线已保存: {save_path}")

        except Exception as e:
            logger.error(f"绘制 ROC 曲线失败: {e}")
            raise

    def plot_classification_report(
            self,
            y_true: Union[np.ndarray, List],
            y_pred: Union[np.ndarray, List],
            class_names: List[str],
            save_path: Union[str, Path],
            title: str = '分类报告'
    ):
        """
        绘制分类报告热图（Precision、Recall、F1-Score）。

        参数:
            y_true (ndarray | list): 真实标签。
            y_pred (ndarray | list): 预测标签。
            class_names (list): 类别名称列表。
            save_path (str | Path): 图表保存路径。
            title (str): 图表标题。
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # 生成分类报告（字典格式）
            report = classification_report(
                y_true, y_pred,
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )

            # 提取每个类别的指标（Precision、Recall、F1-Score）
            metrics = ['precision', 'recall', 'f1-score']
            data = []
            for class_name in class_names:
                data.append([report[class_name][metric] for metric in metrics])

            data = np.array(data)

            with plt.rc_context(self.plot_config):
                fig, ax = plt.subplots(figsize=(8, len(class_names) * 0.5 + 2))

                # 绘制热图
                sns.heatmap(
                    data,
                    annot=True,
                    fmt='.3f',
                    cmap='YlGnBu',
                    xticklabels=['Precision', 'Recall', 'F1-Score'],
                    yticklabels=class_names,
                    cbar_kws={'label': '分数'},
                    vmin=0,
                    vmax=1,
                    ax=ax
                )

                ax.set_title(title)
                ax.set_xlabel('指标')
                ax.set_ylabel('类别')

                plt.tight_layout()
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                plt.close(fig)

                logger.success(f"分类报告热图已保存: {save_path}")

        except Exception as e:
            logger.error(f"绘制分类报告失败: {e}")
            raise

    def plot_epoch_time_distribution(
            self,
            epoch_times: List[float],
            save_path: Union[str, Path],
            title: str = 'Epoch 耗时分布'
    ):
        """
        绘制每个 epoch 的耗时分布（柱状图）。

        参数:
            epoch_times (list): 每个 epoch 的耗时列表（秒）。
            save_path (str | Path): 图表保存路径。
            title (str): 图表标题。
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with plt.rc_context(self.plot_config):
                fig, ax = plt.subplots(figsize=self.figure_size)
                epochs = range(1, len(epoch_times) + 1)

                # 绘制柱状图
                ax.bar(epochs, epoch_times, color='skyblue', edgecolor='navy', alpha=0.7)

                # 添加平均线
                avg_time = np.mean(epoch_times)
                ax.axhline(
                    avg_time, color='red', linestyle='--',
                    label=f'平均耗时: {avg_time:.2f}s',
                    linewidth=2
                )

                ax.set_xlabel('Epoch')
                ax.set_ylabel('耗时 (秒)')
                ax.set_title(title)
                ax.legend()
                ax.grid(True, axis='y', alpha=0.3)

                plt.tight_layout()
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                plt.close(fig)

                logger.success(f"耗时分布图已保存: {save_path}")

        except Exception as e:
            logger.error(f"绘制耗时分布图失败: {e}")
            raise
