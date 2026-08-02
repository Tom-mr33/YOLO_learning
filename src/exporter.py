"""YOLOv8 导出器

ONNX、TensorRT 等格式模型导出
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.utils.logger import LoggerMixin, setup_logger
from tools.utils.helpers import load_yaml, ensure_dir, timer


class YOLOExporter(LoggerMixin):
    """YOLOv8 导出器

    封装 Ultralytics YOLOv8 的模型导出功能，支持 ONNX、TensorRT、OpenVINO 等格式。

    Attributes:
        config: 导出配置字典
        model: YOLO 模型实例
        export_paths: 导出文件路径列表
    """

    # 支持的导出格式
    SUPPORTED_FORMATS = [
        "onnx",
        "engine",      # TensorRT
        "openvino",
        "torchscript",
        "tflite",
        "pb",
        "mlmodel",
        "ncnn",
    ]

    def __init__(self, config: Union[str, Path, Dict[str, Any]]):
        """初始化导出器

        Args:
            config: 配置字典或配置文件路径
        """
        # 加载配置
        if isinstance(config, (str, Path)):
            self.config = load_yaml(config)
        else:
            self.config = config

        self.model = None
        self.export_paths = []

        # 设置日志
        self._setup_logging()

    def _setup_logging(self) -> None:
        """设置日志"""
        log_dir = self.config.get("export", {}).get("project", "runs/export")
        name = self.config.get("export", {}).get("name", "exp")
        ensure_dir(log_dir)
        setup_logger(
            name="exporter",
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

    def export(
        self,
        format: str = "onnx",
        imgsz: int = 640,
        half: bool = False,
        dynamic: bool = False,
        simplify: bool = True,
        opset: Optional[int] = None,
        workspace: int = 4,
        nms: bool = False,
        **kwargs
    ) -> str:
        """导出模型

        Args:
            format: 导出格式 (onnx, engine, openvino 等)
            imgsz: 输入图片尺寸
            half: 是否使用半精度
            dynamic: 是否使用动态轴
            simplify: 是否简化 ONNX 模型
            opset: ONNX opset 版本
            workspace: TensorRT 工作空间大小 (GB)
            nms: 是否包含 NMS
            **kwargs: 额外参数

        Returns:
            导出文件路径
        """
        if self.model is None:
            self._build_model()

        if format not in self.SUPPORTED_FORMATS:
            raise ValueError(f"不支持的导出格式: {format}，支持: {self.SUPPORTED_FORMATS}")

        self.logger.info(f"导出模型为 {format} 格式...")

        export_args = {
            "format": format,
            "imgsz": imgsz,
            "half": half,
            "dynamic": dynamic,
            "simplify": simplify,
            "nms": nms,
            **kwargs,
        }

        # ONNX 特定参数
        if format == "onnx" and opset:
            export_args["opset"] = opset

        # TensorRT 特定参数
        if format == "engine":
            export_args["workspace"] = workspace

        with timer(f"模型导出 ({format})"):
            export_path = self.model.export(**export_args)

        self.export_paths.append(export_path)
        self.logger.info(f"模型已导出到: {export_path}")

        return export_path

    def export_onnx(
        self,
        imgsz: int = 640,
        half: bool = False,
        dynamic: bool = True,
        simplify: bool = True,
        opset: int = 12,
        **kwargs
    ) -> str:
        """导出 ONNX 模型

        Args:
            imgsz: 输入图片尺寸
            half: 是否使用半精度
            dynamic: 是否使用动态轴
            simplify: 是否简化模型
            opset: ONNX opset 版本
            **kwargs: 额外参数

        Returns:
            导出文件路径
        """
        return self.export(
            format="onnx",
            imgsz=imgsz,
            half=half,
            dynamic=dynamic,
            simplify=simplify,
            opset=opset,
            **kwargs
        )

    def export_tensorrt(
        self,
        imgsz: int = 640,
        half: bool = True,
        dynamic: bool = False,
        workspace: int = 4,
        **kwargs
    ) -> str:
        """导出 TensorRT 模型

        Args:
            imgsz: 输入图片尺寸
            half: 是否使用半精度（推荐）
            dynamic: 是否使用动态轴
            workspace: 工作空间大小 (GB)
            **kwargs: 额外参数

        Returns:
            导出文件路径
        """
        return self.export(
            format="engine",
            imgsz=imgsz,
            half=half,
            dynamic=dynamic,
            workspace=workspace,
            **kwargs
        )

    def export_openvino(
        self,
        imgsz: int = 640,
        half: bool = False,
        **kwargs
    ) -> str:
        """导出 OpenVINO 模型

        Args:
            imgsz: 输入图片尺寸
            half: 是否使用半精度
            **kwargs: 额外参数

        Returns:
            导出文件路径
        """
        return self.export(
            format="openvino",
            imgsz=imgsz,
            half=half,
            **kwargs
        )

    def export_all(
        self,
        formats: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, str]:
        """导出所有支持的格式

        Args:
            formats: 要导出的格式列表（默认为所有支持的格式）
            **kwargs: 导出参数

        Returns:
            格式到导出路径的映射字典
        """
        if formats is None:
            formats = ["onnx", "engine", "openvino"]

        results = {}
        for fmt in formats:
            try:
                path = self.export(format=fmt, **kwargs)
                results[fmt] = path
            except Exception as e:
                self.logger.error(f"导出 {fmt} 失败: {e}")
                results[fmt] = None

        return results

    def validate_export(self, export_path: str) -> bool:
        """验证导出模型

        Args:
            export_path: 导出文件路径

        Returns:
            验证是否成功
        """
        path = Path(export_path)
        if not path.exists():
            self.logger.error(f"导出文件不存在: {export_path}")
            return False

        file_size = path.stat().st_size
        self.logger.info(f"导出文件大小: {file_size / 1024**2:.2f} MB")

        # 根据格式验证
        suffix = path.suffix.lower()
        if suffix == ".onnx":
            return self._validate_onnx(path)
        elif suffix == ".engine":
            return self._validate_tensorrt(path)
        else:
            self.logger.info(f"跳过 {suffix} 格式验证")
            return True

    def _validate_onnx(self, path: Path) -> bool:
        """验证 ONNX 模型"""
        try:
            import onnx
            model = onnx.load(str(path))
            onnx.checker.check_model(model)
            self.logger.info("ONNX 模型验证通过")
            return True
        except Exception as e:
            self.logger.error(f"ONNX 模型验证失败: {e}")
            return False

    def _validate_tensorrt(self, path: Path) -> bool:
        """验证 TensorRT 模型"""
        try:
            # 简单检查文件是否存在且大小合理
            file_size = path.stat().st_size
            if file_size > 0:
                self.logger.info("TensorRT 模型验证通过")
                return True
            return False
        except Exception as e:
            self.logger.error(f"TensorRT 模型验证失败: {e}")
            return False


def main():
    """主函数，用于命令行调用"""
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 导出器")
    parser.add_argument("--config", type=str, default="configs/export.yaml", help="配置文件路径")
    parser.add_argument("--weights", type=str, default=None, help="模型权重路径")
    parser.add_argument("--format", type=str, default="onnx", help="导出格式")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图片尺寸")
    parser.add_argument("--half", action="store_true", help="使用半精度")
    parser.add_argument("--dynamic", action="store_true", help="使用动态轴")
    parser.add_argument("--all", action="store_true", help="导出所有格式")
    args = parser.parse_args()

    # 创建导出器
    exporter = YOLOExporter(args.config)

    # 如果指定了权重，覆盖配置
    if args.weights:
        exporter.config["model"]["weights"] = args.weights

    # 导出模型
    if args.all:
        results = exporter.export_all(imgsz=args.imgsz, half=args.half)
        print("\n导出结果:")
        for fmt, path in results.items():
            status = "成功" if path else "失败"
            print(f"  {fmt}: {status}")
    else:
        path = exporter.export(
            format=args.format,
            imgsz=args.imgsz,
            half=args.half,
            dynamic=args.dynamic,
        )
        print(f"\n模型已导出到: {path}")


if __name__ == "__main__":
    main()
