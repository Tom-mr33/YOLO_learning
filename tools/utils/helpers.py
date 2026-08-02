"""辅助函数工具

提供常用的辅助函数
"""

import os
import time
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager


def get_project_root() -> Path:
    """获取项目根目录

    Returns:
        项目根目录路径
    """
    # 从当前文件向上查找，直到找到包含 setup.py 或 .git 的目录
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "setup.py").exists() or (parent / ".git").exists():
            return parent
    # 如果没找到，返回当前文件的上三级目录
    return current.parent.parent.parent


def ensure_dir(path: Union[str, Path]) -> Path:
    """确保目录存在，如果不存在则创建

    Args:
        path: 目录路径

    Returns:
        目录路径对象
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def timer(name: str = "操作"):
    """计时器上下文管理器

    Args:
        name: 操作名称

    Example:
        with timer("训练"):
            # 执行训练代码
            pass
    """
    start_time = time.time()
    print(f"[{name}] 开始...")
    yield
    elapsed_time = time.time() - start_time
    print(f"[{name}] 完成，耗时: {format_time(elapsed_time)}")


def format_time(seconds: float) -> str:
    """格式化时间

    Args:
        seconds: 秒数

    Returns:
        格式化后的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f} 秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} 分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} 小时"


def get_gpu_info() -> Dict[str, Any]:
    """获取 GPU 信息

    Returns:
        GPU 信息字典
    """
    gpu_info = {
        "available": False,
        "count": 0,
        "devices": [],
    }

    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["count"] = torch.cuda.device_count()
            for i in range(gpu_info["count"]):
                device = torch.cuda.get_device_properties(i)
                gpu_info["devices"].append({
                    "index": i,
                    "name": device.name,
                    "total_memory": device.total_memory,
                    "total_memory_gb": f"{device.total_memory / 1024**3:.2f} GB",
                    "major": device.major,
                    "minor": device.minor,
                })
    except ImportError:
        pass

    return gpu_info


def load_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """加载 YAML 配置文件

    Args:
        file_path: YAML 文件路径

    Returns:
        配置字典
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def save_yaml(config: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """保存配置到 YAML 文件

    Args:
        config: 配置字典
        file_path: YAML 文件路径
    """
    file_path = Path(file_path)
    ensure_dir(file_path.parent)

    with open(file_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """合并配置字典

    Args:
        base_config: 基础配置
        override_config: 覆盖配置

    Returns:
        合并后的配置
    """
    merged = base_config.copy()

    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged


def get_file_size(file_path: Union[str, Path]) -> str:
    """获取文件大小

    Args:
        file_path: 文件路径

    Returns:
        格式化后的文件大小字符串
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return "0 B"

    size = file_path.stat().st_size
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def count_files(directory: Union[str, Path], pattern: str = "*") -> int:
    """统计目录中匹配模式的文件数量

    Args:
        directory: 目录路径
        pattern: 文件匹配模式

    Returns:
        文件数量
    """
    directory = Path(directory)
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def get_relative_path(path: Union[str, Path], start: Optional[Union[str, Path]] = None) -> Path:
    """获取相对路径

    Args:
        path: 目标路径
        start: 起始路径（默认为项目根目录）

    Returns:
        相对路径
    """
    if start is None:
        start = get_project_root()

    path = Path(path).resolve()
    start = Path(start).resolve()

    try:
        return path.relative_to(start)
    except ValueError:
        return path


def validate_path(path: Union[str, Path], must_exist: bool = False, create_if_missing: bool = False) -> Path:
    """验证路径

    Args:
        path: 路径
        must_exist: 是否必须存在
        create_if_missing: 如果不存在是否创建

    Returns:
        验证后的路径对象

    Raises:
        FileNotFoundError: 路径不存在且 must_exist 为 True
    """
    path = Path(path)

    if not path.exists():
        if must_exist:
            raise FileNotFoundError(f"路径不存在: {path}")
        elif create_if_missing:
            if path.suffix:
                # 是文件路径，创建父目录
                ensure_dir(path.parent)
            else:
                # 是目录路径，直接创建
                ensure_dir(path)

    return path


def format_dict(d: Dict[str, Any], indent: int = 0) -> str:
    """格式化字典为可读字符串

    Args:
        d: 字典
        indent: 缩进级别

    Returns:
        格式化后的字符串
    """
    lines = []
    for key, value in d.items():
        prefix = "  " * indent
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(format_dict(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for item in value:
                if isinstance(item, dict):
                    lines.append(format_dict(item, indent + 1))
                else:
                    lines.append(f"{prefix}  - {item}")
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines)


if __name__ == "__main__":
    # 测试辅助函数
    print("项目根目录:", get_project_root())
    print("GPU 信息:", get_gpu_info())

    with timer("测试"):
        time.sleep(1)
