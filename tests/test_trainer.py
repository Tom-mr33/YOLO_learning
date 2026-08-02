"""训练器单元测试"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.trainer import YOLOTrainer


class TestYOLOTrainer:
    """YOLOTrainer 测试类"""

    def test_init_with_config_file(self):
        """测试使用配置文件初始化"""
        config_path = project_root / "configs" / "train.yaml"
        if config_path.exists():
            trainer = YOLOTrainer(config_path)
            assert trainer.config is not None
            assert "model" in trainer.config
            assert "data" in trainer.config
            assert "train" in trainer.config

    def test_init_with_dict(self):
        """测试使用字典配置初始化"""
        config = {
            "model": {"name": "yolov8n.pt"},
            "data": {"name": "coco128.yaml"},
            "train": {"epochs": 1, "batch": 1},
        }
        trainer = YOLOTrainer(config)
        assert trainer.config == config

    def test_validate_config(self):
        """测试配置验证"""
        config = {
            "model": {"name": "yolov8n.pt"},
            "data": {"name": "coco128.yaml"},
            "train": {"epochs": 1},
        }
        trainer = YOLOTrainer(config)
        # 如果配置无效，_validate_config 应该抛出异常
        trainer._validate_config()

    def test_get_train_args(self):
        """测试获取训练参数"""
        config = {
            "model": {"name": "yolov8n.pt"},
            "data": {"name": "coco128.yaml"},
            "train": {
                "epochs": 10,
                "batch": 16,
                "imgsz": 640,
            },
        }
        trainer = YOLOTrainer(config)
        args = trainer._get_train_args()

        assert args["epochs"] == 10
        assert args["batch"] == 16
        assert args["imgsz"] == 640


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
