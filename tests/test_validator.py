"""验证器单元测试"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from src.validator import YOLOValidator


class TestYOLOValidator:
    """YOLOValidator 测试类"""

    def test_init_with_config_file(self):
        """测试使用配置文件初始化"""
        config_path = project_root / "configs" / "val.yaml"
        if config_path.exists():
            validator = YOLOValidator(config_path)
            assert validator.config is not None
            assert "model" in validator.config
            assert "data" in validator.config
            assert "val" in validator.config

    def test_init_with_dict(self):
        """测试使用字典配置初始化"""
        config = {
            "model": {"weights": "yolov8n.pt"},
            "data": {"name": "coco128.yaml"},
            "val": {"batch": 32, "imgsz": 640},
        }
        validator = YOLOValidator(config)
        assert validator.config == config

    def test_get_val_args(self):
        """测试获取验证参数"""
        config = {
            "model": {"weights": "yolov8n.pt"},
            "data": {"name": "coco128.yaml"},
            "val": {
                "batch": 32,
                "imgsz": 640,
                "conf": 0.001,
            },
        }
        validator = YOLOValidator(config)
        args = validator._get_val_args()

        assert args["batch"] == 32
        assert args["imgsz"] == 640
        assert args["conf"] == 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
