"""训练示例

演示如何使用 YOLOTrainer 进行模型训练
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trainer import YOLOTrainer


def example_basic_training():
    """基础训练示例"""
    print("=" * 60)
    print("示例 1: 基础训练")
    print("=" * 60)

    # 使用配置文件训练
    config_path = project_root / "configs" / "train.yaml"
    trainer = YOLOTrainer(config_path)

    # 开始训练
    metrics = trainer.train()

    print("\n训练结果:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def example_custom_config():
    """自定义配置训练示例"""
    print("\n" + "=" * 60)
    print("示例 2: 自定义配置训练")
    print("=" * 60)

    # 使用字典配置
    config = {
        "model": {
            "name": "yolov8n.pt",
        },
        "data": {
            "name": "coco128.yaml",
        },
        "train": {
            "epochs": 10,
            "batch": 16,
            "imgsz": 640,
            "device": 0,
            "project": "train",
            "name": "custom_exp",
        },
    }

    trainer = YOLOTrainer(config)
    metrics = trainer.train()

    print("\n训练结果:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def example_resume_training():
    """断点续训示例"""
    print("\n" + "=" * 60)
    print("示例 3: 断点续训")
    print("=" * 60)

    config_path = project_root / "configs" / "train.yaml"
    trainer = YOLOTrainer(config_path)

    # 断点续训
    metrics = trainer.train(resume=True)

    print("\n训练结果:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def example_model_comparison():
    """模型对比示例"""
    print("\n" + "=" * 60)
    print("示例 4: 模型对比（n/s/m）")
    print("=" * 60)

    models = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]

    for model_name in models:
        print(f"\n训练模型: {model_name}")

        config = {
            "model": {"name": model_name},
            "data": {"name": "coco128.yaml"},
            "train": {
                "epochs": 5,
                "batch": 16,
                "imgsz": 640,
                "device": 0,
                "project": "train",
                "name": f"compare_{model_name.split('.')[0]}",
            },
        }

        trainer = YOLOTrainer(config)
        metrics = trainer.train()

        print(f"最佳 mAP: {metrics.get('best_fitness', 'N/A')}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 训练示例")
    parser.add_argument("--example", type=int, default=1, choices=[1, 2, 3, 4], help="示例编号")
    args = parser.parse_args()

    examples = {
        1: example_basic_training,
        2: example_custom_config,
        3: example_resume_training,
        4: example_model_comparison,
    }

    example_func = examples.get(args.example)
    if example_func:
        example_func()
    else:
        print(f"未知示例: {args.example}")


if __name__ == "__main__":
    main()
