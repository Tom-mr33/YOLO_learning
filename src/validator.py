"""YOLOv8 验证器

模型验证，计算 mAP 等指标
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.utils.logger import LoggerMixin, setup_logger
from tools.utils.helpers import load_yaml, ensure_dir, timer


class YOLOValidator(LoggerMixin):
    """YOLOv8 验证器

    封装 Ultralytics YOLOv8 的验证流程，计算 mAP、精确率、召回率等指标。

    Attributes:
        config: 验证配置字典
        model: YOLO 模型实例
        results: 验证结果
    """

    def __init__(self, config: Union[str, Path, Dict[str, Any]]):
        """初始化验证器

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
            name="validator",
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

    def _get_val_args(self) -> Dict[str, Any]:
        """获取验证参数"""
        val_config = self.config.get("val", {})
        data_config = self.config.get("data", {})

        # 构建数据配置文件路径
        data_yaml = data_config.get("name", "coco128.yaml")
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
            "max_det": val_config.get("max_det", 300),
            "device": val_config.get("device", 0),
            "workers": val_config.get("workers", 8),
            "project": val_config.get("project", "runs/val"),
            "name": val_config.get("name", "exp"),
            "exist_ok": val_config.get("exist_ok", False),
            "half": val_config.get("half", False),
            "dnn": val_config.get("dnn", False),
            "plots": val_config.get("plots", True),
            "save_json": val_config.get("save_json", False),
            "save_hybrid": val_config.get("save_hybrid", False),
            "verbose": val_config.get("verbose", True),
            "save_txt": val_config.get("save_txt", False),
            "save_conf": val_config.get("save_conf", False),
            "save_crop": val_config.get("save_crop", False),
            "show_labels": val_config.get("show_labels", True),
            "show_conf": val_config.get("show_conf", True),
            "vid_stride": val_config.get("vid_stride", 1),
            "line_width": val_config.get("line_width", None),
            "visualize": val_config.get("visualize", False),
            "augment": val_config.get("augment", False),
            "agnostic_nms": val_config.get("agnostic_nms", False),
            "classes": val_config.get("classes", None),
            "retina_masks": val_config.get("retina_masks", False),
            "boxes": val_config.get("boxes", True),
            "split": val_config.get("split", "val"),
        }

        return args

    def validate(self) -> Dict[str, Any]:
        """执行验证

        Returns:
            验证结果字典
        """
        if self.model is None:
            self._build_model()

        args = self._get_val_args()

        self.logger.info("开始验证...")
        self.logger.info(f"验证参数: batch={args['batch']}, imgsz={args['imgsz']}, conf={args['conf']}")

        with timer("模型验证"):
            self.results = self.model.val(**args)

        self.logger.info("验证完成！")

        return self.get_metrics()

    def get_metrics(self) -> Dict[str, Any]:
        """获取验证指标

        Returns:
            验证指标字典
        """
        if self.results is None:
            return {}

        box = getattr(self.results, "box", None)
        if box is None:
            return {}

        metrics = {
            "map50": getattr(box, "map50", None),
            "map50_95": getattr(box, "map", None),
            "precision": getattr(box, "mp", None),
            "recall": getattr(box, "mr", None),
        }

        # 添加每个类别的指标
        if hasattr(box, "maps"):
            metrics["per_class_map"] = box.maps.tolist()

        # 添加结果目录
        if hasattr(self.results, "save_dir"):
            metrics["save_dir"] = str(self.results.save_dir)

        return metrics

    def print_metrics(self) -> None:
        """打印验证指标"""
        metrics = self.get_metrics()

        print("\n" + "=" * 50)
        print("验证结果")
        print("=" * 50)
        print(f"mAP@0.5:      {metrics.get('map50', 'N/A'):.4f}" if metrics.get('map50') else "mAP@0.5:      N/A")
        print(f"mAP@0.5:0.95: {metrics.get('map50_95', 'N/A'):.4f}" if metrics.get('map50_95') else "mAP@0.5:0.95: N/A")
        print(f"精确率:        {metrics.get('precision', 'N/A'):.4f}" if metrics.get('precision') else "精确率:        N/A")
        print(f"召回率:        {metrics.get('recall', 'N/A'):.4f}" if metrics.get('recall') else "召回率:        N/A")
        print("=" * 50)


def main():
    """主函数，用于命令行调用"""
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 验证器")
    parser.add_argument("--config", type=str, default="configs/val.yaml", help="配置文件路径")
    parser.add_argument("--weights", type=str, default=None, help="模型权重路径")
    args = parser.parse_args()

    # 创建验证器
    validator = YOLOValidator(args.config)

    # 如果指定了权重，覆盖配置
    if args.weights:
        validator.config["model"]["weights"] = args.weights

    # 开始验证
    metrics = validator.validate()

    # 打印结果
    validator.print_metrics()


if __name__ == "__main__":
    main()
