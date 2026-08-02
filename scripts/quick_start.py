"""快速开始脚本

一键运行完整的 YOLOv8 工作流程：训练 → 验证 → 测试 → 评估
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trainer import YOLOTrainer
from src.validator import YOLOValidator
from src.tester import YOLOTester
from src.evaluator import YOLOEvaluator
from tools.utils.env_check import check_environment, print_environment_report
from tools.utils.helpers import timer, ensure_dir


def check_env():
    """检查环境"""
    print("=" * 60)
    print("步骤 0: 环境检查")
    print("=" * 60)
    env_info = check_environment()
    print_environment_report(env_info)

    if not env_info["python"]["version_ok"]:
        print("\n[错误] Python 版本不满足要求，请升级到 3.8+")
        return False

    if not env_info["cuda"]["available"]:
        print("\n[警告] 未检测到 CUDA，将使用 CPU 训练（速度较慢）")
        response = input("是否继续? (y/n): ")
        if response.lower() != "y":
            return False

    return True


def train_model():
    """训练模型"""
    print("\n" + "=" * 60)
    print("步骤 1: 训练模型")
    print("=" * 60)

    config_path = project_root / "configs" / "train.yaml"
    if not config_path.exists():
        print(f"[错误] 配置文件不存在: {config_path}")
        return None

    trainer = YOLOTrainer(config_path)
    metrics = trainer.train()

    print("\n训练完成！")
    print(f"最佳模型: {metrics.get('best_model', 'N/A')}")

    return metrics


def validate_model(weights_path=None):
    """验证模型"""
    print("\n" + "=" * 60)
    print("步骤 2: 验证模型")
    print("=" * 60)

    config_path = project_root / "configs" / "val.yaml"
    if not config_path.exists():
        print(f"[错误] 配置文件不存在: {config_path}")
        return None

    validator = YOLOValidator(config_path)

    # 如果提供了权重路径，覆盖配置
    if weights_path:
        validator.config["model"]["weights"] = weights_path

    metrics = validator.validate()
    validator.print_metrics()

    return metrics


def test_model(weights_path=None, source=None):
    """测试模型"""
    print("\n" + "=" * 60)
    print("步骤 3: 测试模型")
    print("=" * 60)

    config_path = project_root / "configs" / "test.yaml"
    if not config_path.exists():
        print(f"[错误] 配置文件不存在: {config_path}")
        return None

    tester = YOLOTester(config_path)

    # 如果提供了权重路径，覆盖配置
    if weights_path:
        tester.config["model"]["weights"] = weights_path

    # 如果提供了测试源，覆盖配置
    if source:
        tester.config["data"]["source"] = source

    results = tester.test()
    summary = tester.get_summary()

    print("\n测试完成！")
    print(f"处理图片数: {summary.get('total_images', 0)}")
    print(f"检测目标数: {summary.get('total_detections', 0)}")

    return results


def evaluate_model(weights_path=None):
    """评估模型"""
    print("\n" + "=" * 60)
    print("步骤 4: 评估模型")
    print("=" * 60)

    config_path = project_root / "configs" / "val.yaml"
    if not config_path.exists():
        print(f"[错误] 配置文件不存在: {config_path}")
        return None

    evaluator = YOLOEvaluator(config_path)

    # 如果提供了权重路径，覆盖配置
    if weights_path:
        evaluator.config["model"]["weights"] = weights_path

    report = evaluator.evaluate()
    evaluator.print_report()

    # 保存报告
    report_path = project_root / "runs" / "evaluation_report.json"
    ensure_dir(report_path.parent)
    evaluator.save_report(report_path)

    return report


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 快速开始脚本")
    parser.add_argument("--skip-env-check", action="store_true", help="跳过环境检查")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练")
    parser.add_argument("--skip-val", action="store_true", help="跳过验证")
    parser.add_argument("--skip-test", action="store_true", help="跳过测试")
    parser.add_argument("--skip-eval", action="store_true", help="跳过评估")
    parser.add_argument("--weights", type=str, default=None, help="模型权重路径")
    parser.add_argument("--source", type=str, default=None, help="测试源路径")
    args = parser.parse_args()

    print("=" * 60)
    print("YOLOv8 阶段一快速开始")
    print("=" * 60)

    # 步骤 0: 环境检查
    if not args.skip_env_check:
        if not check_env():
            print("\n[错误] 环境检查未通过，退出")
            return

    # 步骤 1: 训练模型
    best_weights = args.weights
    if not args.skip_train:
        metrics = train_model()
        if metrics:
            best_weights = metrics.get("best_model")
    else:
        print("\n[跳过] 训练步骤")

    # 步骤 2: 验证模型
    if not args.skip_val:
        validate_model(best_weights)
    else:
        print("\n[跳过] 验证步骤")

    # 步骤 3: 测试模型
    if not args.skip_test:
        test_model(best_weights, args.source)
    else:
        print("\n[跳过] 测试步骤")

    # 步骤 4: 评估模型
    if not args.skip_eval:
        evaluate_model(best_weights)
    else:
        print("\n[跳过] 评估步骤")

    print("\n" + "=" * 60)
    print("快速开始完成！")
    print("=" * 60)
    print("\n输出目录:")
    print(f"  训练结果: runs/train/")
    print(f"  验证结果: runs/val/")
    print(f"  测试结果: runs/test/")
    print(f"  评估报告: runs/evaluation_report.json")


if __name__ == "__main__":
    main()
