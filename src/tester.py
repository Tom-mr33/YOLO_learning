"""YOLOv8 测试器

批量测试，生成检测结果
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.utils.logger import LoggerMixin, setup_logger
from tools.utils.helpers import load_yaml, ensure_dir, timer


class YOLOTester(LoggerMixin):
    """YOLOv8 测试器

    封装 Ultralytics YOLOv8 的推理流程，支持图片、视频、摄像头等多种输入源。

    Attributes:
        config: 测试配置字典
        model: YOLO 模型实例
        results: 测试结果
    """

    def __init__(self, config: Union[str, Path, Dict[str, Any]]):
        """初始化测试器

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
        log_dir = self.config.get("test", {}).get("project", "runs/test")
        name = self.config.get("test", {}).get("name", "exp")
        ensure_dir(log_dir)
        setup_logger(
            name="tester",
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

    def _get_test_args(self) -> Dict[str, Any]:
        """获取测试参数"""
        test_config = self.config.get("test", {})
        data_config = self.config.get("data", {})

        args = {
            "source": data_config.get("source", "data/images"),
            "imgsz": test_config.get("imgsz", 640),
            "conf": test_config.get("conf", 0.25),
            "iou": test_config.get("iou", 0.45),
            "max_det": test_config.get("max_det", 1000),
            "device": test_config.get("device", 0),
            "save_txt": test_config.get("save_txt", False),
            "save_conf": test_config.get("save_conf", False),
            "save_crop": test_config.get("save_crop", False),
            "save": test_config.get("save", True),
            "project": test_config.get("project", "runs/test"),
            "name": test_config.get("name", "exp"),
            "exist_ok": test_config.get("exist_ok", False),
            "half": test_config.get("half", False),
            "show_labels": test_config.get("show_labels", True),
            "show_conf": test_config.get("show_conf", True),
            "show_boxes": test_config.get("show_boxes", True),
            "vid_stride": test_config.get("vid_stride", 1),
            "line_width": test_config.get("line_width", None),
            "visualize": test_config.get("visualize", False),
            "augment": test_config.get("augment", False),
            "agnostic_nms": test_config.get("agnostic_nms", False),
            "classes": test_config.get("classes", None),
            "retina_masks": test_config.get("retina_masks", False),
            "stream": test_config.get("stream", False),
            "verbose": test_config.get("verbose", True),
        }

        return args

    def test(self, source: Optional[str] = None) -> List[Any]:
        """执行测试

        Args:
            source: 测试源（可选，覆盖配置文件）

        Returns:
            测试结果列表
        """
        if self.model is None:
            self._build_model()

        args = self._get_test_args()
        if source:
            args["source"] = source

        self.logger.info("开始测试...")
        self.logger.info(f"测试源: {args['source']}")
        self.logger.info(f"测试参数: imgsz={args['imgsz']}, conf={args['conf']}, iou={args['iou']}")

        with timer("模型测试"):
            self.results = self.model.predict(**args)

        self.logger.info("测试完成！")
        self.logger.info(f"结果保存在: {self.model.predictor.save_dir}")

        return self.results

    def test_image(self, image_path: str, **kwargs) -> Any:
        """测试单张图片

        Args:
            image_path: 图片路径
            **kwargs: 额外参数

        Returns:
            测试结果
        """
        if self.model is None:
            self._build_model()

        args = self._get_test_args()
        args.update(kwargs)
        args["source"] = image_path

        self.logger.info(f"测试图片: {image_path}")
        results = self.model.predict(**args)

        return results

    def test_video(self, video_path: str, **kwargs) -> Any:
        """测试视频

        Args:
            video_path: 视频路径
            **kwargs: 额外参数

        Returns:
            测试结果
        """
        if self.model is None:
            self._build_model()

        args = self._get_test_args()
        args.update(kwargs)
        args["source"] = video_path

        self.logger.info(f"测试视频: {video_path}")
        results = self.model.predict(**args)

        return results

    def test_camera(self, camera_id: int = 0, **kwargs) -> Any:
        """测试摄像头

        Args:
            camera_id: 摄像头编号
            **kwargs: 额外参数

        Returns:
            测试结果
        """
        if self.model is None:
            self._build_model()

        args = self._get_test_args()
        args.update(kwargs)
        args["source"] = camera_id

        self.logger.info(f"测试摄像头: {camera_id}")
        results = self.model.predict(**args)

        return results

    def get_summary(self) -> Dict[str, Any]:
        """获取测试摘要

        Returns:
            测试摘要字典
        """
        if self.results is None:
            return {}

        summary = {
            "total_images": len(self.results),
            "save_dir": str(self.model.predictor.save_dir) if hasattr(self.model, "predictor") else None,
        }

        # 统计检测到的目标数量
        total_detections = 0
        for result in self.results:
            if hasattr(result, "boxes") and result.boxes is not None:
                total_detections += len(result.boxes)
        summary["total_detections"] = total_detections

        return summary


def main():
    """主函数，用于命令行调用"""
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 测试器")
    parser.add_argument("--config", type=str, default="configs/test.yaml", help="配置文件路径")
    parser.add_argument("--weights", type=str, default=None, help="模型权重路径")
    parser.add_argument("--source", type=str, default=None, help="测试源路径")
    args = parser.parse_args()

    # 创建测试器
    tester = YOLOTester(args.config)

    # 如果指定了权重，覆盖配置
    if args.weights:
        tester.config["model"]["weights"] = args.weights

    # 开始测试
    results = tester.test(source=args.source)

    # 打印摘要
    summary = tester.get_summary()
    print("\n测试摘要:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
