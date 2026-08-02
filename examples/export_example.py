"""导出示例

演示如何使用 YOLOExporter 导出模型
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.exporter import YOLOExporter


def example_onnx_export():
    """ONNX 导出示例"""
    print("=" * 60)
    print("示例 1: ONNX 导出")
    print("=" * 60)

    config = {
        "model": {"weights": "yolov8n.pt"},
        "export": {
            "project": "runs/export",
            "name": "onnx_export",
        },
    }

    exporter = YOLOExporter(config)
    export_path = exporter.export_onnx(
        imgsz=640,
        half=False,
        dynamic=True,
        simplify=True,
        opset=12,
    )

    print(f"\nONNX 模型已导出到: {export_path}")


def example_tensorrt_export():
    """TensorRT 导出示例"""
    print("\n" + "=" * 60)
    print("示例 2: TensorRT 导出")
    print("=" * 60)

    config = {
        "model": {"weights": "yolov8n.pt"},
        "export": {
            "project": "runs/export",
            "name": "tensorrt_export",
        },
    }

    exporter = YOLOExporter(config)

    # 注意：TensorRT 导出需要安装 TensorRT
    try:
        export_path = exporter.export_tensorrt(
            imgsz=640,
            half=True,
            dynamic=False,
            workspace=4,
        )
        print(f"\nTensorRT 模型已导出到: {export_path}")
    except Exception as e:
        print(f"TensorRT 导出失败: {e}")
        print("请确保已安装 TensorRT")


def example_openvino_export():
    """OpenVINO 导出示例"""
    print("\n" + "=" * 60)
    print("示例 3: OpenVINO 导出")
    print("=" * 60)

    config = {
        "model": {"weights": "yolov8n.pt"},
        "export": {
            "project": "runs/export",
            "name": "openvino_export",
        },
    }

    exporter = YOLOExporter(config)

    try:
        export_path = exporter.export_openvino(
            imgsz=640,
            half=False,
        )
        print(f"\nOpenVINO 模型已导出到: {export_path}")
    except Exception as e:
        print(f"OpenVINO 导出失败: {e}")
        print("请确保已安装 OpenVINO")


def example_all_formats():
    """导出所有格式示例"""
    print("\n" + "=" * 60)
    print("示例 4: 导出所有格式")
    print("=" * 60)

    config = {
        "model": {"weights": "yolov8n.pt"},
        "export": {
            "project": "runs/export",
            "name": "all_formats",
        },
    }

    exporter = YOLOExporter(config)

    # 导出所有支持的格式
    results = exporter.export_all(
        formats=["onnx", "engine", "openvino"],
        imgsz=640,
        half=True,
    )

    print("\n导出结果:")
    for fmt, path in results.items():
        status = "成功" if path else "失败"
        print(f"  {fmt}: {status}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 导出示例")
    parser.add_argument("--example", type=int, default=1, choices=[1, 2, 3, 4], help="示例编号")
    args = parser.parse_args()

    examples = {
        1: example_onnx_export,
        2: example_tensorrt_export,
        3: example_openvino_export,
        4: example_all_formats,
    }

    example_func = examples.get(args.example)
    if example_func:
        example_func()
    else:
        print(f"未知示例: {args.example}")


if __name__ == "__main__":
    main()
