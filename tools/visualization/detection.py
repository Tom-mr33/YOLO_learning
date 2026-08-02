"""检测结果可视化工具

绘制边界框、批量展示检测结果
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tools.utils.logger import LoggerMixin
from tools.utils.helpers import ensure_dir


class DetectionVisualizer(LoggerMixin):
    """检测结果可视化器

    绘制检测结果的边界框、标签、置信度等信息。

    Attributes:
        config: 可视化配置字典
    """

    # 默认颜色映射（BGR 格式）
    DEFAULT_COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化检测结果可视化器

        Args:
            config: 可视化配置字典
        """
        self.config = config or {}
        self.colors = self.config.get("colors", self.DEFAULT_COLORS)

    def draw_detections(
        self,
        image: Any,
        boxes: List[List[float]],
        scores: List[float],
        class_ids: List[int],
        class_names: Optional[List[str]] = None,
        thickness: int = 2,
        font_scale: float = 0.5,
    ) -> Any:
        """在图片上绘制检测结果

        Args:
            image: 输入图片 (numpy array, BGR 格式)
            boxes: 边界框列表，格式 [x1, y1, x2, y2]
            scores: 置信度列表
            class_ids: 类别 ID 列表
            class_names: 类别名称列表（可选）
            thickness: 边界框线宽
            font_scale: 字体大小

        Returns:
            绘制后的图片
        """
        import cv2
        import numpy as np

        image = image.copy()

        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            # 获取颜色
            color = self.colors[class_id % len(self.colors)]

            # 绘制边界框
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # 准备标签文本
            if class_names and class_id < len(class_names):
                label = f"{class_names[class_id]}: {score:.2f}"
            else:
                label = f"Class {class_id}: {score:.2f}"

            # 计算文本大小
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # 绘制标签背景
            cv2.rectangle(
                image,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1,
            )

            # 绘制标签文本
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
            )

        return image

    def visualize_results(
        self,
        results: List[Any],
        output_dir: Union[str, Path],
        class_names: Optional[List[str]] = None,
        max_images: int = 20,
    ) -> None:
        """可视化检测结果

        Args:
            results: Ultralytics 检测结果列表
            output_dir: 输出目录
            class_names: 类别名称列表
            max_images: 最大处理图片数
        """
        import cv2

        output_dir = Path(output_dir)
        ensure_dir(output_dir)

        self.logger.info(f"可视化检测结果，最大处理 {max_images} 张图片")

        for i, result in enumerate(results[:max_images]):
            # 获取原始图片
            image = result.orig_img

            # 获取检测框
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)

                # 绘制检测框
                image = self.draw_detections(
                    image, boxes, scores, class_ids, class_names
                )

            # 保存图片
            output_file = output_dir / f"detection_{i:04d}.jpg"
            cv2.imwrite(str(output_file), image)

        self.logger.info(f"检测结果已保存到: {output_dir}")

    def create_grid(
        self,
        images: List[Any],
        grid_size: Tuple[int, int] = (4, 5),
        output_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """创建图片网格

        Args:
            images: 图片列表
            grid_size: 网格大小 (rows, cols)
            output_path: 输出路径（可选）

        Returns:
            网格图片
        """
        import cv2
        import numpy as np

        if not images:
            return None

        rows, cols = grid_size
        n_images = min(len(images), rows * cols)

        # 获取图片尺寸
        h, w = images[0].shape[:2]

        # 创建空白网格
        grid = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)

        # 填充图片
        for i in range(n_images):
            row = i // cols
            col = i % cols
            grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = images[i]

        # 保存图片
        if output_path:
            output_path = Path(output_path)
            ensure_dir(output_path.parent)
            cv2.imwrite(str(output_path), grid)

        return grid

    def plot_detection_statistics(
        self,
        results: List[Any],
        output_dir: Union[str, Path],
        class_names: Optional[List[str]] = None,
    ) -> None:
        """绘制检测统计图表

        Args:
            results: Ultralytics 检测结果列表
            output_dir: 输出目录
            class_names: 类别名称列表
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            self.logger.error("matplotlib 未安装，无法绘制统计图表")
            return

        output_dir = Path(output_dir)
        ensure_dir(output_dir)

        # 统计每个类别的检测数量
        class_counts = {}
        confidence_scores = []

        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                scores = result.boxes.conf.cpu().numpy()

                for class_id, score in zip(class_ids, scores):
                    class_counts[class_id] = class_counts.get(class_id, 0) + 1
                    confidence_scores.append(score)

        # 创建子图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 1. 类别分布
        if class_counts:
            class_ids = sorted(class_counts.keys())
            counts = [class_counts[cid] for cid in class_ids]
            labels = [
                class_names[cid] if class_names and cid < len(class_names) else f"Class {cid}"
                for cid in class_ids
            ]

            axes[0].bar(range(len(counts)), counts, color="steelblue")
            axes[0].set_xticks(range(len(counts)))
            axes[0].set_xticklabels(labels, rotation=45, ha="right")
            axes[0].set_title("Detection Count by Class")
            axes[0].set_xlabel("Class")
            axes[0].set_ylabel("Count")

        # 2. 置信度分布
        if confidence_scores:
            axes[1].hist(confidence_scores, bins=50, color="green", alpha=0.7)
            axes[1].set_title("Confidence Score Distribution")
            axes[1].set_xlabel("Confidence Score")
            axes[1].set_ylabel("Frequency")

        # 保存图片
        plt.tight_layout()
        output_file = output_dir / "detection_statistics.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"检测统计图表已保存到: {output_file}")


def main():
    """主函数，用于命令行调用"""
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 检测结果可视化工具")
    parser.add_argument("--results", type=str, required=True, help="检测结果目录")
    parser.add_argument("--output", type=str, default="runs/visualization", help="输出目录")
    args = parser.parse_args()

    # 创建可视化器
    visualizer = DetectionVisualizer()

    # 可视化结果
    # 注意：这里需要从结果目录加载结果，实际使用时需要调整
    print("请使用 Python API 调用可视化功能")


if __name__ == "__main__":
    main()
