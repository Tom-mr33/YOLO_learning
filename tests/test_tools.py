"""工具函数单元测试"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
from tools.utils.helpers import (
    get_project_root,
    ensure_dir,
    format_time,
    load_yaml,
    save_yaml,
)


class TestHelpers:
    """辅助函数测试类"""

    def test_get_project_root(self):
        """测试获取项目根目录"""
        root = get_project_root()
        assert root.exists()
        assert (root / "setup.py").exists() or (root / ".git").exists()

    def test_ensure_dir(self, tmp_path):
        """测试确保目录存在"""
        test_dir = tmp_path / "test_dir"
        ensure_dir(test_dir)
        assert test_dir.exists()

    def test_format_time(self):
        """测试时间格式化"""
        assert format_time(30) == "30.00 秒"
        assert format_time(90) == "1.50 分钟"
        assert format_time(3661) == "1.02 小时"

    def test_load_save_yaml(self, tmp_path):
        """测试 YAML 加载和保存"""
        config = {
            "model": {"name": "yolov8n.pt"},
            "train": {"epochs": 50},
        }

        # 保存
        yaml_path = tmp_path / "test.yaml"
        save_yaml(config, yaml_path)
        assert yaml_path.exists()

        # 加载
        loaded_config = load_yaml(yaml_path)
        assert loaded_config == config


class TestEnvCheck:
    """环境检查测试类"""

    def test_check_environment(self):
        """测试环境检查"""
        from tools.utils.env_check import check_environment

        env_info = check_environment()
        assert "python" in env_info
        assert "os" in env_info
        assert "cuda" in env_info
        assert "gpu" in env_info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
