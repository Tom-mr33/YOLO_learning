"""验证示例

演示如何使用 YOLOValidator 进行模型验证
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.validator import YOLOValidator


def example_basic_validation():
    """基础验证示例"""
    print("=" * 60)
    print("示例 1: 基础验证")
    print("=" * 60)

    # 使用配置文件验证
    config_path = project_root / "configs" / "val.yaml"
    validator = YOLOValidator(config_path)

    # 开始验证
    metrics = validator.validate()
    validator.print_metrics()


def example_custom_weights():
    """自定义权重验证示例"""
    print("\n" + "=" * 60)
    print("示例 2: 自定义权重验证")
    print("=" * 60)

    # 使用字典配置
    config = {
        "model": {
            "weights": "yolov8n.pt",  # 可以替换为自己的权重路径
        },
        "data": {
            "name": "coco128.yaml",
        },
        "val": {
            "batch": 32,
            "imgsz": 640,
            "conf": 0.001,
            "iou": 0.6,
            "device": 0,
            "project": "runs/val",
            "name": "custom_val",
        },
    }

    validator = YOLOValidator(config)
    metrics = validator.validate()
    validator.print_metrics()


def example_batch_validation():
    """批量验证示例"""
    print("\n" + "=" * 60)
    print("示例 3: 批量验证（对比不同模型）")
    print("=" * 60)

    models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]

    results = {}
    for model_name in models:
        print(f"\n验证模型: {model_name}")

        config = {
            "model": {"weights": model_name},
            "data": {"name": "coco128.yaml"},
            "val": {
                "batch": 32,
                "imgsz": 640,
                "device": 0,
                "project": "runs/val",
                "name": f"batch_{model_name.split('.')[0]}",
            },
        }

        validator = YOLOValidator(config)
        metrics = validator.validate()
        results[model_name] = metrics

    # 打印对比结果
    print("\n" + "=" * 60)
    print("模型对比结果")
    print("=" * 60)
    print(f"{'模型':<15} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15} {'精确率':<12} {'召回率':<12}")
    print("-" * 60)
    for model_name, metrics in results.items():
        map50 = metrics.get("map50", 0)
        map50_95 = metrics.get("map50_95", 0)
        precision = metrics.get("precision", 0)
        recall = metrics.get("recall", 0)
        print(f"{model_name:<15} {map50:<12.4f} {map50_95:<15.4f} {precision:<12.4f} {recall:<12.4f}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 验证示例")
    parser.add_argument("--example", type=int, default=1, choices=[1, 2, 3], help="示例编号")
    args = parser.parse_args()

    examples = {
        1: example_basic_validation,
        2: example_custom_weights,
        3: example_batch_validation,
    }

    example_func = examples.get(args.example)
    if example_func:
        example_func()
    else:
        print(f"未知示例: {args.example}")


if __name__ == "__main__":
    main()
