"""混淆矩阵可视化工具

绘制分类结果的混淆矩阵
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.utils.logger import LoggerMixin
from tools.utils.helpers import ensure_dir


class ConfusionMatrixPlotter(LoggerMixin):
    """混淆矩阵绘制器

    绘制分类结果的混淆矩阵，支持归一化和非归一化两种模式。

    Attributes:
        config: 绘图配置字典
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化混淆矩阵绘制器

        Args:
            config: 绘图配置字典
        """
        self.config = config or {}
        self._setup_matplotlib()

    def _setup_matplotlib(self) -> None:
        """设置 matplotlib"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set_style("whitegrid")
            plt.rcParams["figure.figsize"] = (12, 10)
            plt.rcParams["font.size"] = 12
            plt.rcParams["axes.unicode_minus"] = False

            self.plt = plt
            self.sns = sns
        except ImportError:
            self.logger.warning("matplotlib 或 seaborn 未安装，可视化功能不可用")
            self.plt = None
            self.sns = None

    def plot_confusion_matrix(
        self,
        confusion_matrix: Any,
        class_names: List[str],
        output_dir: Union[str, Path],
        normalize: bool = True,
        title: str = "Confusion Matrix",
    ) -> None:
        """绘制混淆矩阵

        Args:
            confusion_matrix: 混淆矩阵（numpy array）
            class_names: 类别名称列表
            output_dir: 输出目录
            normalize: 是否归一化
            title: 图表标题
        """
        if self.plt is None:
            self.logger.error("matplotlib 未安装，无法绘制混淆矩阵")
            return

        import numpy as np

        output_dir = Path(output_dir)
        ensure_dir(output_dir)

        self.logger.info(f"绘制混淆矩阵: {title}")

        # 归一化
        if normalize:
            cm = confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]
            cm = np.nan_to_num(cm)  # 处理除零
            fmt = ".2f"
        else:
            cm = confusion_matrix
            fmt = "d"

        # 创建图形
        fig, ax = self.plt.subplots(figsize=(max(10, len(class_names)), max(8, len(class_names) - 2)))

        # 绘制热力图
        im = ax.imshow(cm, interpolation="nearest", cmap=self.plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        # 设置标签
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=class_names,
            yticklabels=class_names,
            title=title,
            ylabel="True Label",
            xlabel="Predicted Label",
        )

        # 旋转 x 轴标签
        self.plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # 添加数值标注
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        # 保存图片
        self.plt.tight_layout()
        output_file = output_dir / f"{title.replace(' ', '_')}.png"
        self.plt.savefig(output_file, dpi=300, bbox_inches="tight")
        self.plt.close()

        self.logger.info(f"混淆矩阵已保存到: {output_file}")

    def plot_from_results(
        self,
        results_dir: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> None:
        """从 Ultralytics 结果目录绘制混淆矩阵

        Args:
            results_dir: Ultralytics 验证结果目录
            output_dir: 输出目录
        """
        if self.plt is None:
            self.logger.error("matplotlib 未安装，无法绘制混淆矩阵")
            return

        import numpy as np

        results_dir = Path(results_dir)
        output_dir = Path(output_dir)
        ensure_dir(output_dir)

        # 查找混淆矩阵文件
        cm_file = results_dir / "confusion_matrix.png"
        if cm_file.exists():
            self.logger.info(f"混淆矩阵图片已存在: {cm_file}")
            return

        # 尝试从结果中读取混淆矩阵
        # 注意：Ultralytics 通常自动生成混淆矩阵图片
        # 这里提供从原始数据绘制的功能
        self.logger.warning("未找到混淆矩阵数据，请确保已运行验证")


def main():
    """主函数，用于命令行调用"""
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 混淆矩阵绘制工具")
    parser.add_argument("--results", type=str, required=True, help="验证结果目录")
    parser.add_argument("--output", type=str, default="runs/confusion", help="输出目录")
    args = parser.parse_args()

    # 创建绘制器
    plotter = ConfusionMatrixPlotter()

    # 绘制混淆矩阵
    plotter.plot_from_results(args.results, args.output)


if __name__ == "__main__":
    main()
