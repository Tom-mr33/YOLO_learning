"""YOLOv8 评估器

详细评估报告，混淆矩阵，性能分析
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.utils.logger import LoggerMixin, setup_logger
from tools.utils.helpers import load_yaml, ensure_dir, timer


class YOLOEvaluator(LoggerMixin):
    """YOLOv8 评估器

    提供详细的模型评估功能，包括混淆矩阵、PR 曲线、性能分析等。

    Attributes:
        config: 评估配置字典
        model: YOLO 模型实例
        results: 评估结果
    """

    def __init__(self, config: Union[str, Path, Dict[str, Any]]):
        """初始化评估器

        Args:
            config: 配置字典或配置文件路径
        """
        # 加载配置
        if isinstance(config, (str, Path)):
            self.config = load_yaml(config)
        else:
            self.config = config

        self.model = None
        self.results = None

        # 设置日志
        self._setup_logging()

    def _setup_logging(self) -> None:
        """设置日志"""
        log_dir = self.config.get("val", {}).get("project", "runs/val")
        name = self.config.get("val", {}).get("name", "exp")
        ensure_dir(log_dir)
        setup_logger(
            name="evaluator",
            level="info",
            log_dir=str(Path(log_dir) / name),
        )

    def _build_model(self) -> None:
        """构建模型"""
        from ultralytics import YOLO

        model_config = self.config.get("model", {})
        weights = model_config.get("weights", "yolov8n.pt")

        self.logger.info(f"加载模型权重: {weights}")
        self.model = YOLO(weights)

    def evaluate(self, data: Optional[str] = None) -> Dict[str, Any]:
        """执行完整评估

        Args:
            data: 数据集配置文件路径（可选）

        Returns:
            评估结果字典
        """
        if self.model is None:
            self._build_model()

        # 获取验证参数
        val_config = self.config.get("val", {})
        data_config = self.config.get("data", {})

        # 构建数据配置文件路径
        data_yaml = data or data_config.get("name", "coco128.yaml")
        if not Path(data_yaml).is_absolute():
            config_path = project_root / "configs" / "data" / data_yaml
            if config_path.exists():
                data_yaml = str(config_path)

        args = {
            "data": data_yaml,
            "batch": val_config.get("batch", 32),
            "imgsz": val_config.get("imgsz", 640),
            "conf": val_config.get("conf", 0.001),
            "iou": val_config.get("iou", 0.6),
            "device": val_config.get("device", 0),
            "workers": val_config.get("workers", 8),
            "project": val_config.get("project", "runs/val"),
            "name": val_config.get("name", "exp"),
            "plots": True,  # 始终生成图表
            "save_json": True,  # 保存 JSON 结果
            "verbose": val_config.get("verbose", True),
        }

        self.logger.info("开始评估...")

        with timer("模型评估"):
            self.results = self.model.val(**args)

        self.logger.info("评估完成！")

        return self.get_full_report()

    def get_full_report(self) -> Dict[str, Any]:
        """获取完整评估报告

        Returns:
            完整评估报告字典
        """
        if self.results is None:
            return {}

        report = {
            "summary": self._get_summary(),
            "per_class": self._get_per_class_metrics(),
            "curves": self._get_curve_data(),
            "confusion_matrix": self._get_confusion_matrix(),
        }

        return report

    def _get_summary(self) -> Dict[str, Any]:
        """获取摘要指标"""
        box = getattr(self.results, "box", None)
        if box is None:
            return {}

        return {
            "map50": getattr(box, "map50", None),
            "map50_95": getattr(box, "map", None),
            "map75": getattr(box, "map75", None),
            "precision": getattr(box, "mp", None),
            "recall": getattr(box, "mr", None),
            "f1": getattr(box, "f1", None),
            "fitness": getattr(box, "fitness", None),
        }

    def _get_per_class_metrics(self) -> List[Dict[str, Any]]:
        """获取每个类别的指标"""
        box = getattr(self.results, "box", None)
        if box is None:
            return []

        per_class = []
        names = getattr(self.results, "names", {})

        # 获取每个类别的 AP
        if hasattr(box, "maps"):
            maps = box.maps
            for i, ap in enumerate(maps):
                class_name = names.get(i, f"class_{i}")
                per_class.append({
                    "class_id": i,
                    "class_name": class_name,
                    "ap50": float(ap),
                })

        return per_class

    def _get_curve_data(self) -> Dict[str, Any]:
        """获取曲线数据"""
        curves = {}

        # PR 曲线
        if hasattr(self.results, "box") and hasattr(self.results.box, "pr_curve"):
            curves["pr_curve"] = self.results.box.pr_curve

        # F1 曲线
        if hasattr(self.results, "box") and hasattr(self.results.box, "f1_curve"):
            curves["f1_curve"] = self.results.box.f1_curve

        return curves

    def _get_confusion_matrix(self) -> Optional[Any]:
        """获取混淆矩阵"""
        if hasattr(self.results, "confusion_matrix"):
            return self.results.confusion_matrix
        return None

    def print_report(self) -> None:
        """打印评估报告"""
        report = self.get_full_report()
        summary = report.get("summary", {})

        print("\n" + "=" * 60)
        print("评估报告")
        print("=" * 60)

        # 总体指标
        print("\n[总体指标]")

        # 辅助函数：安全地格式化数值
        def safe_format(value, format_str="{:.4f}"):
            if value is None:
                return "N/A"
            if isinstance(value, (list, tuple)) and len(value) > 0:
                value = value[0]  # 取第一个元素
            try:
                return format_str.format(float(value))
            except (ValueError, TypeError):
                return "N/A"

        print(f"  mAP@0.5:      {safe_format(summary.get('map50'))}")
        print(f"  mAP@0.5:0.95: {safe_format(summary.get('map50_95'))}")
        print(f"  mAP@0.75:     {safe_format(summary.get('map75'))}")
        print(f"  精确率:        {safe_format(summary.get('precision'))}")
        print(f"  召回率:        {safe_format(summary.get('recall'))}")
        print(f"  F1 分数:       {safe_format(summary.get('f1'))}")

        # 每个类别的指标
        per_class = report.get("per_class", [])
        if per_class:
            print("\n[各类别 AP@0.5]")
            # 按 AP 排序
            per_class_sorted = sorted(per_class, key=lambda x: x["ap50"], reverse=True)
            for item in per_class_sorted[:10]:  # 只显示前 10 个
                print(f"  {item['class_name']:20s}: {item['ap50']:.4f}")
            if len(per_class_sorted) > 10:
                print(f"  ... 还有 {len(per_class_sorted) - 10} 个类别")

        print("\n" + "=" * 60)

    def save_report(self, output_path: Union[str, Path]) -> None:
        """保存评估报告到文件

        Args:
            output_path: 输出文件路径
        """
        import json

        report = self.get_full_report()
        output_path = Path(output_path)
        ensure_dir(output_path.parent)

        # 转换 numpy 类型为 Python 原生类型
        def convert_numpy(obj):
            import numpy as np
            import inspect

            # 跳过方法和函数
            if inspect.ismethod(obj) or inspect.isfunction(obj):
                return str(obj)
            # 处理 ConfusionMatrix 等特殊对象
            elif hasattr(obj, '__dict__') and not isinstance(obj, (dict, list, tuple, str, int, float, bool, type(None))):
                # 尝试转换为字典，如果失败则返回字符串表示
                try:
                    if hasattr(obj, 'matrix'):  # ConfusionMatrix 对象
                        return {"matrix": obj.matrix.tolist() if hasattr(obj.matrix, 'tolist') else str(obj.matrix)}
                    return str(obj)
                except:
                    return str(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(item) for item in obj]
            return obj

        report = convert_numpy(report)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.logger.info(f"评估报告已保存到: {output_path}")


def main():
    """主函数，用于命令行调用"""
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 评估器")
    parser.add_argument("--config", type=str, default="configs/val.yaml", help="配置文件路径")
    parser.add_argument("--weights", type=str, default=None, help="模型权重路径")
    parser.add_argument("--data", type=str, default=None, help="数据集配置路径")
    parser.add_argument("--save", type=str, default=None, help="保存报告路径")
    args = parser.parse_args()

    # 创建评估器
    evaluator = YOLOEvaluator(args.config)

    # 如果指定了权重，覆盖配置
    if args.weights:
        evaluator.config["model"]["weights"] = args.weights

    # 开始评估
    report = evaluator.evaluate(data=args.data)

    # 打印报告
    evaluator.print_report()

    # 保存报告
    if args.save:
        evaluator.save_report(args.save)


if __name__ == "__main__":
    main()
