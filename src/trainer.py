"""YOLOv8 训练器

封装训练流程，支持断点续训、多实验配置管理
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.utils.logger import LoggerMixin, setup_logger
from tools.utils.helpers import load_yaml, ensure_dir, timer, get_gpu_info


class YOLOTrainer(LoggerMixin):
    """YOLOv8 训练器

    封装 Ultralytics YOLOv8 的训练流程，提供配置管理、断点续训等功能。

    Attributes:
        config: 训练配置字典
        model: YOLO 模型实例
        results: 训练结果
    """

    def __init__(self, config: Union[str, Path, Dict[str, Any]]):
        """初始化训练器

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

        # 验证配置
        self._validate_config()

    def _setup_logging(self) -> None:
        """设置日志"""
        project = self.config.get("train", {}).get("project", "train")
        name = self.config.get("train", {}).get("name", "exp")
        log_dir = Path("runs") / project / name
        ensure_dir(log_dir)
        self._logger = setup_logger(
            name="trainer",
            level="info",
            log_dir=str(log_dir),
        )

    def _validate_config(self) -> None:
        """验证配置"""
        required_keys = {
            "model": ["name"],
            "data": ["name"],
            "train": ["epochs", "batch", "imgsz"],
        }
        for key, subkeys in required_keys.items():
            if key not in self.config:
                raise ValueError(f"配置缺少必需字段: {key}")
            for subkey in subkeys:
                if subkey not in self.config[key]:
                    raise ValueError(f"配置缺少必需字段: {key}.{subkey}")

        # 检查 GPU
        gpu_info = get_gpu_info()
        if gpu_info["available"]:
            self.logger.info(f"检测到 {gpu_info['count']} 个 GPU")
            for device in gpu_info["devices"]:
                self.logger.info(f"  GPU {device['index']}: {device['name']} ({device['total_memory_gb']})")
        else:
            self.logger.warning("未检测到 GPU，将使用 CPU 训练（速度较慢）")

    def _build_model(self) -> None:
        """构建模型"""
        from ultralytics import YOLO

        model_config = self.config.get("model", {})
        model_name = model_config.get("name", "yolov8n.pt")

        self.logger.info(f"加载模型: {model_name}")
        self.model = YOLO(model_name)

    def _get_train_args(self) -> Dict[str, Any]:
        """获取训练参数"""
        train_config = self.config.get("train", {})
        data_config = self.config.get("data", {})
        augment_config = self.config.get("augment", {})

        # 构建数据配置文件路径
        data_yaml = data_config.get("name", "coco128.yaml")
        if not Path(data_yaml).is_absolute():
            # 尝试从 configs/data 目录查找
            config_path = project_root / "configs" / "data" / data_yaml
            if config_path.exists():
                data_yaml = str(config_path)

        # 合并所有参数
        args = {
            "data": data_yaml,
            "epochs": train_config.get("epochs", 50),
            "batch": train_config.get("batch", 16),
            "imgsz": train_config.get("imgsz", 640),
            "lr0": train_config.get("lr0", 0.01),
            "lrf": train_config.get("lrf", 0.01),
            "momentum": train_config.get("momentum", 0.937),
            "weight_decay": train_config.get("weight_decay", 0.0005),
            "warmup_epochs": train_config.get("warmup_epochs", 3.0),
            "warmup_momentum": train_config.get("warmup_momentum", 0.8),
            "box": train_config.get("box", 7.5),
            "cls": train_config.get("cls", 0.5),
            "dfl": train_config.get("dfl", 1.5),
            "label_smoothing": train_config.get("label_smoothing", 0.0),
            "nbs": train_config.get("nbs", 64),
            "overlap_mask": train_config.get("overlap_mask", True),
            "val": train_config.get("val", True),
            "save": train_config.get("save", True),
            "save_period": train_config.get("save_period", -1),
            "cache": train_config.get("cache", False),
            "device": train_config.get("device", 0),
            "workers": train_config.get("workers", 8),
            "project": train_config.get("project", "train"),
            "name": train_config.get("name", "exp"),
            "exist_ok": train_config.get("exist_ok", False),
            "pretrained": train_config.get("pretrained", True),
            "optimizer": train_config.get("optimizer", "SGD"),
            "verbose": train_config.get("verbose", True),
            "seed": train_config.get("seed", 0),
            "deterministic": train_config.get("deterministic", True),
            "single_cls": train_config.get("single_cls", False),
            "rect": train_config.get("rect", False),
            "cos_lr": train_config.get("cos_lr", False),
            "close_mosaic": train_config.get("close_mosaic", 10),
            "resume": train_config.get("resume", False),
            "amp": train_config.get("amp", True),
            "fraction": train_config.get("fraction", 1.0),
            "profile": train_config.get("profile", False),
            "freeze": train_config.get("freeze", None),
            "multi_scale": train_config.get("multi_scale", False),
            "copy_paste": train_config.get("copy_paste", 0.0),
            "auto_augment": train_config.get("auto_augment", "randaugment"),
            "erasing": train_config.get("erasing", 0.4),
            "crop_fraction": train_config.get("crop_fraction", 1.0),
            # 数据增强参数
            "hsv_h": augment_config.get("hsv_h", 0.015),
            "hsv_s": augment_config.get("hsv_s", 0.7),
            "hsv_v": augment_config.get("hsv_v", 0.4),
            "degrees": augment_config.get("degrees", 0.0),
            "translate": augment_config.get("translate", 0.1),
            "scale": augment_config.get("scale", 0.5),
            "shear": augment_config.get("shear", 0.0),
            "perspective": augment_config.get("perspective", 0.0),
            "flipud": augment_config.get("flipud", 0.0),
            "fliplr": augment_config.get("fliplr", 0.5),
            "mosaic": augment_config.get("mosaic", 1.0),
            "mixup": augment_config.get("mixup", 0.0),
        }

        return args

    def train(self, resume: bool = False) -> Dict[str, Any]:
        """执行训练

        Args:
            resume: 是否断点续训

        Returns:
            训练结果字典
        """
        if self.model is None:
            self._build_model()

        # 获取训练参数
        args = self._get_train_args()
        if resume:
            args["resume"] = True

        self.logger.info("开始训练...")
        self.logger.info(f"训练参数: epochs={args['epochs']}, batch={args['batch']}, imgsz={args['imgsz']}")

        with timer("模型训练"):
            self.results = self.model.train(**args)

        self.logger.info("训练完成！")

        # 获取最佳模型路径
        save_dir = Path(self.results.save_dir) if hasattr(self.results, "save_dir") else None
        if save_dir:
            best_model = save_dir / "weights" / "best.pt"
            if best_model.exists():
                self.logger.info(f"最佳模型保存在: {best_model}")
            else:
                self.logger.info(f"训练结果保存在: {save_dir}")

        return self.get_metrics()

    def get_metrics(self) -> Dict[str, Any]:
        """获取训练指标

        Returns:
            训练指标字典
        """
        if self.results is None:
            return {}

        # 新版 ultralytics 不再把 best_fitness/best_epoch 挂在 train() 的返回值上：
        # best_fitness 记录在 trainer 上，best_epoch 需从 results.csv 反查。
        trainer = getattr(self.model, "trainer", None)
        metrics = {
            "best_fitness": getattr(trainer, "best_fitness", None),
            "best_epoch": None,
        }

        # 添加结果目录和模型路径
        if hasattr(self.results, "save_dir"):
            save_dir = Path(self.results.save_dir)
            metrics["save_dir"] = str(save_dir)
            metrics["best_epoch"] = self._get_best_epoch(save_dir)

            # 构建最佳模型和最后模型的完整路径
            best_model = save_dir / "weights" / "best.pt"
            last_model = save_dir / "weights" / "last.pt"

            if best_model.exists():
                metrics["best_model"] = str(best_model)
            if last_model.exists():
                metrics["last_model"] = str(last_model)

        return metrics

    @staticmethod
    def _get_best_epoch(save_dir: Path) -> Optional[int]:
        """从 results.csv 反查最佳 epoch（fitness = 0.1*mAP50 + 0.9*mAP50-95）。"""
        import csv

        results_csv = save_dir / "results.csv"
        if not results_csv.exists():
            return None

        best_epoch, best_fitness = None, -float("inf")
        with open(results_csv, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    epoch = int(float(row["epoch"]))
                    fitness = 0.1 * float(row["metrics/mAP50(B)"]) + 0.9 * float(row["metrics/mAP50-95(B)"])
                except (KeyError, ValueError):
                    continue
                if fitness > best_fitness:
                    best_epoch, best_fitness = epoch, fitness
        return best_epoch

    def export(self, format: str = "onnx", **kwargs) -> str:
        """导出模型

        Args:
            format: 导出格式 (onnx, engine, openvino 等)
            **kwargs: 导出参数

        Returns:
            导出文件路径
        """
        if self.model is None:
            self._build_model()  # 自动加载模型

        self.logger.info(f"导出模型为 {format} 格式...")
        export_path = self.model.export(format=format, **kwargs)
        self.logger.info(f"模型已导出到: {export_path}")

        return export_path


def main():
    """主函数，用于命令行调用"""
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 训练器")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="配置文件路径")
    parser.add_argument("--resume", action="store_true", help="断点续训")
    args = parser.parse_args()

    # 创建训练器
    trainer = YOLOTrainer(args.config)

    # 开始训练
    metrics = trainer.train(resume=args.resume)

    # 打印结果
    print("\n训练结果:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
