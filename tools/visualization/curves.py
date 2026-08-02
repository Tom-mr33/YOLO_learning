"""训练曲线可视化工具

绘制训练过程中的 loss、mAP 等指标曲线
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.utils.logger import LoggerMixin
from tools.utils.helpers import ensure_dir


class CurvePlotter(LoggerMixin):
    """训练曲线绘制器

    绘制训练过程中的各种指标曲线，支持多实验对比。

    Attributes:
        config: 绘图配置字典
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化曲线绘制器

        Args:
            config: 绘图配置字典
        """
        self.config = config or {}
        self._setup_matplotlib()

    def _setup_matplotlib(self) -> None:
        """设置 matplotlib"""
        try:
            import matplotlib
            matplotlib.use("Agg")  # 非交互式后端
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 设置样式
            sns.set_style("whitegrid")
            plt.rcParams["figure.figsize"] = (12, 8)
            plt.rcParams["font.size"] = 12
            plt.rcParams["axes.unicode_minus"] = False

            self.plt = plt
            self.sns = sns
        except ImportError:
            self.logger.warning("matplotlib 或 seaborn 未安装，可视化功能不可用")
            self.plt = None
            self.sns = None

    def plot_training_curves(
        self,
        results_csv: Union[str, Path],
        output_dir: Union[str, Path],
        experiment_name: str = "experiment",
    ) -> None:
        """绘制训练曲线

        Args:
            results_csv: 训练结果 CSV 文件路径（Ultralytics 生成）
            output_dir: 输出目录
            experiment_name: 实验名称
        """
        if self.plt is None:
            self.logger.error("matplotlib 未安装，无法绘制曲线")
            return

        import pandas as pd

        results_csv = Path(results_csv)
        output_dir = Path(output_dir)
        ensure_dir(output_dir)

        if not results_csv.exists():
            self.logger.error(f"结果文件不存在: {results_csv}")
            return

        # 读取结果
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()  # 去除列名空格

        self.logger.info(f"绘制训练曲线: {experiment_name}")

        # 创建子图
        fig, axes = self.plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Training Curves - {experiment_name}", fontsize=16)

        # 1. Box Loss
        if "train/box_loss" in df.columns:
            axes[0, 0].plot(df["epoch"], df["train/box_loss"], label="train", linewidth=2)
            if "val/box_loss" in df.columns:
                axes[0, 0].plot(df["epoch"], df["val/box_loss"], label="val", linewidth=2)
            axes[0, 0].set_title("Box Loss")
            axes[0, 0].set_xlabel("Epoch")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].legend()
            axes[0, 0].grid(True)

        # 2. Class Loss
        if "train/cls_loss" in df.columns:
            axes[0, 1].plot(df["epoch"], df["train/cls_loss"], label="train", linewidth=2)
            if "val/cls_loss" in df.columns:
                axes[0, 1].plot(df["epoch"], df["val/cls_loss"], label="val", linewidth=2)
            axes[0, 1].set_title("Class Loss")
            axes[0, 1].set_xlabel("Epoch")
            axes[0, 1].set_ylabel("Loss")
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        # 3. DFL Loss
        if "train/dfl_loss" in df.columns:
            axes[0, 2].plot(df["epoch"], df["train/dfl_loss"], label="train", linewidth=2)
            if "val/dfl_loss" in df.columns:
                axes[0, 2].plot(df["epoch"], df["val/dfl_loss"], label="val", linewidth=2)
            axes[0, 2].set_title("DFL Loss")
            axes[0, 2].set_xlabel("Epoch")
            axes[0, 2].set_ylabel("Loss")
            axes[0, 2].legend()
            axes[0, 2].grid(True)

        # 4. Precision
        if "metrics/precision(B)" in df.columns:
            axes[1, 0].plot(df["epoch"], df["metrics/precision(B)"], linewidth=2, color="green")
            axes[1, 0].set_title("Precision")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Precision")
            axes[1, 0].grid(True)

        # 5. Recall
        if "metrics/recall(B)" in df.columns:
            axes[1, 1].plot(df["epoch"], df["metrics/recall(B)"], linewidth=2, color="orange")
            axes[1, 1].set_title("Recall")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("Recall")
            axes[1, 1].grid(True)

        # 6. mAP
        if "metrics/mAP50(B)" in df.columns:
            axes[1, 2].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5", linewidth=2)
            if "metrics/mAP50-95(B)" in df.columns:
                axes[1, 2].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", linewidth=2)
            axes[1, 2].set_title("mAP")
            axes[1, 2].set_xlabel("Epoch")
            axes[1, 2].set_ylabel("mAP")
            axes[1, 2].legend()
            axes[1, 2].grid(True)

        # 保存图片
        self.plt.tight_layout()
        output_file = output_dir / f"{experiment_name}_curves.png"
        self.plt.savefig(output_file, dpi=300, bbox_inches="tight")
        self.plt.close()

        self.logger.info(f"训练曲线已保存到: {output_file}")

    def plot_comparison(
        self,
        results_dict: Dict[str, Union[str, Path]],
        output_dir: Union[str, Path],
        metric: str = "metrics/mAP50(B)",
    ) -> None:
        """绘制多实验对比曲线

        Args:
            results_dict: 实验名称到结果 CSV 文件路径的映射
            output_dir: 输出目录
            metric: 要对比的指标
        """
        if self.plt is None:
            self.logger.error("matplotlib 未安装，无法绘制曲线")
            return

        import pandas as pd

        output_dir = Path(output_dir)
        ensure_dir(output_dir)

        self.logger.info(f"绘制多实验对比曲线: {metric}")

        # 创建图形
        self.plt.figure(figsize=(12, 8))

        # 绘制每个实验的曲线
        for exp_name, csv_path in results_dict.items():
            csv_path = Path(csv_path)
            if not csv_path.exists():
                self.logger.warning(f"结果文件不存在: {csv_path}")
                continue

            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()

            if metric in df.columns:
                self.plt.plot(df["epoch"], df[metric], label=exp_name, linewidth=2)

        self.plt.title(f"Comparison - {metric}")
        self.plt.xlabel("Epoch")
        self.plt.ylabel(metric)
        self.plt.legend()
        self.plt.grid(True)

        # 保存图片
        output_file = output_dir / f"comparison_{metric.replace('/', '_')}.png"
        self.plt.savefig(output_file, dpi=300, bbox_inches="tight")
        self.plt.close()

        self.logger.info(f"对比曲线已保存到: {output_file}")

    def plot_learning_rate(
        self,
        results_csv: Union[str, Path],
        output_dir: Union[str, Path],
        experiment_name: str = "experiment",
    ) -> None:
        """绘制学习率曲线

        Args:
            results_csv: 训练结果 CSV 文件路径
            output_dir: 输出目录
            experiment_name: 实验名称
        """
        if self.plt is None:
            self.logger.error("matplotlib 未安装，无法绘制曲线")
            return

        import pandas as pd

        results_csv = Path(results_csv)
        output_dir = Path(output_dir)
        ensure_dir(output_dir)

        if not results_csv.exists():
            self.logger.error(f"结果文件不存在: {results_csv}")
            return

        # 读取结果
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()

        # 查找学习率列
        lr_columns = [col for col in df.columns if "lr" in col.lower()]

        if not lr_columns:
            self.logger.warning("未找到学习率数据")
            return

        # 创建图形
        self.plt.figure(figsize=(10, 6))

        for col in lr_columns:
            self.plt.plot(df["epoch"], df[col], label=col, linewidth=2)

        self.plt.title(f"Learning Rate - {experiment_name}")
        self.plt.xlabel("Epoch")
        self.plt.ylabel("Learning Rate")
        self.plt.legend()
        self.plt.grid(True)
        self.plt.yscale("log")

        # 保存图片
        output_file = output_dir / f"{experiment_name}_lr.png"
        self.plt.savefig(output_file, dpi=300, bbox_inches="tight")
        self.plt.close()

        self.logger.info(f"学习率曲线已保存到: {output_file}")


def main():
    """主函数，用于命令行调用"""
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 训练曲线绘制工具")
    parser.add_argument("--results", type=str, required=True, help="训练结果 CSV 文件路径")
    parser.add_argument("--output", type=str, default="runs/curves", help="输出目录")
    parser.add_argument("--name", type=str, default="experiment", help="实验名称")
    args = parser.parse_args()

    # 创建绘制器
    plotter = CurvePlotter()

    # 绘制曲线
    plotter.plot_training_curves(args.results, args.output, args.name)
    plotter.plot_learning_rate(args.results, args.output, args.name)


if __name__ == "__main__":
    main()
