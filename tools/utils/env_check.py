"""环境检查工具

检测系统环境、CUDA 配置、依赖安装情况
"""

import sys
import subprocess
import platform
from typing import Dict, Optional, Tuple
from pathlib import Path


def get_python_version() -> str:
    """获取 Python 版本"""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def check_python_version(min_version: Tuple[int, int] = (3, 8)) -> bool:
    """检查 Python 版本是否满足要求"""
    return sys.version_info >= min_version


def get_os_info() -> Dict[str, str]:
    """获取操作系统信息"""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


def check_cuda_available() -> bool:
    """检查 CUDA 是否可用"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_cuda_version() -> Optional[str]:
    """获取 CUDA 版本"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.version.cuda
    except ImportError:
        pass
    return None


def get_gpu_info() -> Dict[str, any]:
    """获取 GPU 信息"""
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
                    "total_memory": f"{device.total_memory / 1024**3:.2f} GB",
                    "major": device.major,
                    "minor": device.minor,
                })
    except ImportError:
        pass

    return gpu_info


def check_nvidia_driver() -> Tuple[bool, Optional[str]]:
    """检查 NVIDIA 驱动是否安装"""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            # 解析 CUDA 版本
            for line in result.stdout.split("\n"):
                if "CUDA Version" in line:
                    cuda_version = line.split("CUDA Version:")[1].strip().split()[0]
                    return True, cuda_version
            return True, None
        return False, None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, None


def check_package_installed(package_name: str) -> Tuple[bool, Optional[str]]:
    """检查包是否已安装"""
    try:
        module = __import__(package_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, None


def check_dependencies() -> Dict[str, Dict[str, any]]:
    """检查关键依赖安装情况"""
    dependencies = {
        "torch": {"required": True, "min_version": "2.0.0"},
        "torchvision": {"required": True, "min_version": "0.15.0"},
        "ultralytics": {"required": True, "min_version": "8.0.0"},
        "cv2": {"required": True, "min_version": "4.8.0"},
        "numpy": {"required": True, "min_version": "1.24.0"},
        "yaml": {"required": True, "min_version": "6.0"},
        "matplotlib": {"required": False, "min_version": "3.7.0"},
        "seaborn": {"required": False, "min_version": "0.12.0"},
        "albumentations": {"required": False, "min_version": "1.3.0"},
        "onnx": {"required": False, "min_version": "1.14.0"},
    }

    results = {}
    for package, info in dependencies.items():
        installed, version = check_package_installed(package)
        results[package] = {
            "installed": installed,
            "version": version,
            "required": info["required"],
            "min_version": info["min_version"],
            "satisfied": installed and version >= info["min_version"] if installed else False,
        }

    return results


def check_environment() -> Dict[str, any]:
    """检查完整环境配置

    Returns:
        包含环境信息的字典
    """
    env_info = {
        "python": {
            "version": get_python_version(),
            "version_ok": check_python_version(),
            "executable": sys.executable,
        },
        "os": get_os_info(),
        "cuda": {
            "available": check_cuda_available(),
            "version": get_cuda_version(),
        },
        "gpu": get_gpu_info(),
        "nvidia_driver": check_nvidia_driver(),
        "dependencies": check_dependencies(),
    }

    return env_info


def print_environment_report(env_info: Optional[Dict] = None) -> None:
    """打印环境检查报告"""
    if env_info is None:
        env_info = check_environment()

    print("=" * 60)
    print("环境检查报告")
    print("=" * 60)

    # Python 信息
    print("\n[Python]")
    print(f"  版本: {env_info['python']['version']}")
    print(f"  路径: {env_info['python']['executable']}")
    status = "✓" if env_info['python']['version_ok'] else "✗"
    print(f"  版本满足要求 (>=3.8): {status}")

    # 操作系统信息
    print("\n[操作系统]")
    print(f"  系统: {env_info['os']['system']} {env_info['os']['release']}")
    print(f"  架构: {env_info['os']['machine']}")

    # CUDA 信息
    print("\n[CUDA]")
    cuda_available = env_info['cuda']['available']
    status = "✓" if cuda_available else "✗"
    print(f"  CUDA 可用: {status}")
    if cuda_available:
        print(f"  CUDA 版本: {env_info['cuda']['version']}")

    # GPU 信息
    print("\n[GPU]")
    gpu_info = env_info['gpu']
    if gpu_info['available']:
        print(f"  GPU 数量: {gpu_info['count']}")
        for device in gpu_info['devices']:
            print(f"  GPU {device['index']}: {device['name']}")
            print(f"    显存: {device['total_memory']}")
    else:
        print("  未检测到可用 GPU")

    # NVIDIA 驱动
    print("\n[NVIDIA 驱动]")
    driver_installed, driver_cuda = env_info['nvidia_driver']
    status = "✓" if driver_installed else "✗"
    print(f"  驱动已安装: {status}")
    if driver_cuda:
        print(f"  驱动 CUDA 版本: {driver_cuda}")

    # 依赖检查
    print("\n[依赖检查]")
    deps = env_info['dependencies']
    all_required_ok = True
    for package, info in deps.items():
        status = "✓" if info['satisfied'] else ("⚠" if info['installed'] else "✗")
        required = "必需" if info['required'] else "可选"
        version_str = f"v{info['version']}" if info['version'] else "未安装"
        print(f"  {status} {package:20s} {version_str:15s} ({required})")
        if info['required'] and not info['satisfied']:
            all_required_ok = False

    # 总结
    print("\n" + "=" * 60)
    if all_required_ok:
        print("✓ 环境检查通过，可以开始使用！")
    else:
        print("✗ 环境检查未通过，请安装缺失的依赖")
        print("  运行: pip install -r requirements.txt")
    print("=" * 60)


def install_dependencies(cuda_version: Optional[str] = None) -> bool:
    """安装项目依赖

    Args:
        cuda_version: CUDA 版本，如 "11.8", "12.1"，None 表示 CPU 版本

    Returns:
        安装是否成功
    """
    try:
        # 升级 pip
        print("升级 pip...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
        )

        # 安装 PyTorch
        print("安装 PyTorch...")
        if cuda_version:
            if cuda_version.startswith("12.1"):
                index_url = "https://download.pytorch.org/whl/cu121"
            elif cuda_version.startswith("11.8"):
                index_url = "https://download.pytorch.org/whl/cu118"
            else:
                index_url = "https://download.pytorch.org/whl/cu118"
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", index_url],
                check=True,
            )
        else:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"],
                check=True,
            )

        # 安装项目依赖
        print("安装项目依赖...")
        requirements_path = Path(__file__).parent.parent.parent / "requirements.txt"
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
            check=True,
        )

        print("依赖安装完成！")
        return True

    except subprocess.CalledProcessError as e:
        print(f"安装失败: {e}")
        return False


def main():
    """主函数，用于命令行调用"""
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 环境检查工具")
    parser.add_argument("--install", action="store_true", help="安装依赖")
    parser.add_argument("--cuda", type=str, default=None, help="CUDA 版本 (如 11.8, 12.1)")
    args = parser.parse_args()

    if args.install:
        success = install_dependencies(args.cuda)
        sys.exit(0 if success else 1)
    else:
        print_environment_report()


if __name__ == "__main__":
    main()
