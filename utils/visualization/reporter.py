#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Created on 2025/11/4 00:20
@author  : William_Trouvaille
@function: 训练报告生成器
@description: 提供 TrainingReporter 类，负责组织和协调整个报告生成流程，
              包括调用可视化器生成图表、生成文本摘要，并统一管理保存路径。
"""
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Any, Protocol

from loguru import logger

from .collector import MetricsCollector
from .visualizer import MetricsVisualizer
from ..helpers import format_time


class ReporterConfig(Protocol):
    """
    Reporter 配置对象的协议定义（类型注解）。

    使用 Protocol 定义配置接口，提供类型检查和 IDE 自动补全支持。
    """
    save_dir: str
    use_timestamp: bool

    # 嵌套配置对象（可选）
    plots: Optional[Any]
    enable_plots: Optional[Any]
    roc: Optional[Any]
    confusion_matrix: Optional[Any]


class TrainingReporter:
    """
    训练报告生成器，负责协调和组织完整的训练成果报告。

    核心功能:
        - 管理报告保存路径（支持时间戳子目录）
        - 从 MetricsCollector 获取数据
        - 调用 MetricsVisualizer 生成所有图表
        - 生成 Markdown 格式的文本摘要报告
        - 提供一键生成完整报告的接口

    设计理念:
        - 一键生成：一个方法调用完成所有工作
        - 灵活配置：通过配置对象控制生成内容
        - 错误容忍：单个图表生成失败不影响其他内容
    """

    def __init__(self, config: ReporterConfig):
        """
        初始化报告生成器。

        参数:
            config (ReporterConfig): 报告配置对象，包含以下字段：
                - save_dir (str): 报告保存根目录
                - use_timestamp (bool): 是否使用时间戳子目录
                - plots (object): 图表配置（dpi、figure_size、style）
                - enable_plots (object): 图表开关配置
                - roc (object): ROC 曲线配置
                - confusion_matrix (object): 混淆矩阵配置
        """
        self.config = config

        # 解析保存路径
        self.save_dir = Path(config.save_dir)
        self.use_timestamp = getattr(config, 'use_timestamp', True)

        # 如果启用时间戳，创建带时间戳的子目录
        if self.use_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.report_dir = self.save_dir / f"report_{timestamp}"
        else:
            self.report_dir = self.save_dir

        # 创建保存目录
        self.report_dir.mkdir(parents=True, exist_ok=True)

        # 初始化可视化器
        plots_config = getattr(config, 'plots', None)
        if plots_config:
            self.visualizer = MetricsVisualizer(
                style=getattr(plots_config, 'style', 'seaborn-v0_8-darkgrid'),
                dpi=getattr(plots_config, 'dpi', 100),
                figure_size=tuple(getattr(plots_config, 'figure_size', [10, 6]))
            )
        else:
            self.visualizer = MetricsVisualizer()

        # 获取图表开关配置（默认全部启用）
        self.enable_plots = getattr(config, 'enable_plots', None)

        logger.info(f"TrainingReporter 已初始化，报告将保存至: {self.report_dir}")

    def _is_plot_enabled(self, plot_name: str) -> bool:
        """
        检查指定的图表是否启用。

        参数:
            plot_name (str): 图表名称（如 'training_curves'）。

        返回:
            bool: 是否启用。
        """
        if self.enable_plots is None:
            return True  # 如果未配置，默认启用
        return getattr(self.enable_plots, plot_name, True)

    def generate_text_summary(self, collector: MetricsCollector) -> str:
        """
        生成文本摘要（Markdown 格式）。

        参数:
            collector (MetricsCollector): 指标收集器实例。

        返回:
            str: Markdown 格式的摘要文本。
        """
        summary = collector.get_summary()

        # 构建 Markdown 文本
        md_text = "# 训练成果摘要\n\n"
        md_text += f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        md_text += "## 训练统计\n\n"
        md_text += f"- **总训练轮数**: {summary['total_epochs']} epochs\n"
        md_text += f"- **总训练时间**: {format_time(summary['total_time'])}\n"
        md_text += f"- **平均每轮耗时**: {format_time(summary['avg_epoch_time'])}\n\n"

        md_text += "## 性能指标\n\n"
        md_text += f"- **最佳训练准确率**: {summary['best_train_acc']:.2f}%\n"
        md_text += f"- **最佳验证准确率**: {summary['best_val_acc']:.2f}%\n"
        md_text += f"- **最终训练损失**: {summary['final_train_loss']:.4f}\n"
        md_text += f"- **最终验证损失**: {summary['final_val_loss']:.4f}\n\n"

        md_text += "## 生成的图表\n\n"
        md_text += "- `training_curves.png` - 训练曲线（Loss & 准确率）\n"
        md_text += "- `confusion_matrix.png` - 混淆矩阵\n"
        md_text += "- `roc_curve.png` - ROC 曲线\n"
        md_text += "- `classification_report.png` - 分类报告热图\n"
        md_text += "- `epoch_time_distribution.png` - Epoch 耗时分布\n\n"

        md_text += "---\n\n"
        md_text += "*该报告由 TrainingReporter 自动生成*\n"

        return md_text

    def save_summary_markdown(self, summary: str, filename: str = 'summary.md'):
        """
        保存 Markdown 格式的摘要报告。

        参数:
            summary (str): Markdown 文本。
            filename (str): 文件名。
        """
        save_path = self.report_dir / filename

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            logger.success(f"摘要报告已保存: {save_path}")
        except Exception as e:
            logger.error(f"保存摘要报告失败: {e}")
            raise

    def generate_full_report(
            self,
            collector: MetricsCollector,
            class_names: Optional[List[str]] = None
    ):
        """
        一键生成完整的训练报告（所有图表 + 文本摘要）。

        参数:
            collector (MetricsCollector): 指标收集器实例，包含所有训练数据。
            class_names (list, optional): 类别名称列表（用于混淆矩阵、ROC 等）。
        """
        logger.info("=" * 60)
        logger.info("开始生成训练报告...".center(60))
        logger.info("=" * 60)

        # 1. 生成训练曲线
        if self._is_plot_enabled('training_curves'):
            try:
                self.visualizer.plot_training_curves(
                    train_losses=collector.train_losses,
                    val_losses=collector.val_losses,
                    train_accs=collector.train_accs,
                    val_accs=collector.val_accs,
                    save_path=self.report_dir / 'training_curves.png'
                )
            except Exception as e:
                logger.warning(f"训练曲线生成失败（已跳过）: {e}")

        # 2. 生成混淆矩阵（需要预测结果）
        if (self._is_plot_enabled('confusion_matrix') and
                collector.predictions is not None and
                class_names is not None):
            try:
                cm_config = getattr(self.config, 'confusion_matrix', None)
                normalize = getattr(cm_config, 'normalize', True) if cm_config else True
                cmap = getattr(cm_config, 'cmap', 'Blues') if cm_config else 'Blues'

                self.visualizer.plot_confusion_matrix(
                    y_true=collector.targets,
                    y_pred=collector.predictions,
                    class_names=class_names,
                    save_path=self.report_dir / 'confusion_matrix.png',
                    normalize=normalize,
                    cmap=cmap
                )
            except Exception as e:
                logger.warning(f"混淆矩阵生成失败（已跳过）: {e}")

        # 3. 生成 ROC 曲线（需要预测概率）
        if (self._is_plot_enabled('roc_curve') and
                collector.probabilities is not None and
                class_names is not None):
            try:
                roc_config = getattr(self.config, 'roc', None)
                multi_class = getattr(roc_config, 'multi_class', 'ovr') if roc_config else 'ovr'

                self.visualizer.plot_roc_curve(
                    y_true=collector.targets,
                    y_proba=collector.probabilities,
                    class_names=class_names,
                    save_path=self.report_dir / 'roc_curve.png',
                    multi_class=multi_class
                )
            except Exception as e:
                logger.warning(f"ROC 曲线生成失败（已跳过）: {e}")

        # 4. 生成分类报告热图（需要预测结果）
        if (self._is_plot_enabled('classification_report') and
                collector.predictions is not None and
                class_names is not None):
            try:
                self.visualizer.plot_classification_report(
                    y_true=collector.targets,
                    y_pred=collector.predictions,
                    class_names=class_names,
                    save_path=self.report_dir / 'classification_report.png'
                )
            except Exception as e:
                logger.warning(f"分类报告热图生成失败（已跳过）: {e}")

        # 5. 生成 Epoch 耗时分布
        if self._is_plot_enabled('epoch_time_distribution') and collector.epoch_times:
            try:
                self.visualizer.plot_epoch_time_distribution(
                    epoch_times=collector.epoch_times,
                    save_path=self.report_dir / 'epoch_time_distribution.png'
                )
            except Exception as e:
                logger.warning(f"耗时分布图生成失败（已跳过）: {e}")

        # 6. 生成文本摘要
        try:
            summary_text = self.generate_text_summary(collector)
            self.save_summary_markdown(summary_text)
        except Exception as e:
            logger.warning(f"文本摘要生成失败（已跳过）: {e}")

        # 7. 保存原始指标数据（JSON 格式）
        try:
            collector.save_to_json(self.report_dir / 'metrics_data.json')
        except Exception as e:
            logger.warning(f"指标数据保存失败（已跳过）: {e}")

        # 完成
        logger.info("=" * 60)
        logger.success(f"训练报告生成完成！保存位置: {self.report_dir}".center(60))
        logger.info("=" * 60)

    def __repr__(self) -> str:
        """返回对象的字符串表示，方便调试。"""
        return f"TrainingReporter(report_dir={self.report_dir})"
