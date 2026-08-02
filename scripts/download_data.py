"""数据集下载脚本

自动下载 COCO128 等示例数据集
"""

import sys
import zipfile
import urllib.request
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.utils.logger import setup_logger
from tools.utils.helpers import ensure_dir

# 设置日志
logger = setup_logger("download", level="info")


# 数据集下载链接
DATASETS = {
    "coco128": {
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
        "description": "COCO128 示例数据集（128 张图片，80 个类别）",
    },
}


def download_file(url: str, output_path: Path) -> bool:
    """下载文件

    Args:
        url: 下载链接
        output_path: 输出路径

    Returns:
        是否下载成功
    """
    try:
        logger.info(f"下载: {url}")
        urllib.request.urlretrieve(url, output_path)
        logger.info(f"下载完成: {output_path}")
        return True
    except Exception as e:
        logger.error(f"下载失败: {e}")
        return False


def extract_zip(zip_path: Path, output_dir: Path) -> bool:
    """解压 ZIP 文件

    Args:
        zip_path: ZIP 文件路径
        output_dir: 输出目录

    Returns:
        是否解压成功
    """
    try:
        logger.info(f"解压: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        logger.info(f"解压完成: {output_dir}")
        return True
    except Exception as e:
        logger.error(f"解压失败: {e}")
        return False


def download_coco128(data_dir: Path = None) -> bool:
    """下载 COCO128 数据集

    Args:
        data_dir: 数据目录（默认为项目根目录下的 data 目录）

    Returns:
        是否下载成功
    """
    if data_dir is None:
        data_dir = project_root / "data"

    data_dir = ensure_dir(data_dir)
    coco128_dir = data_dir / "coco128"

    # 检查是否已存在
    if coco128_dir.exists():
        logger.info(f"COCO128 数据集已存在: {coco128_dir}")
        return True

    # 下载
    dataset_info = DATASETS["coco128"]
    zip_path = data_dir / "coco128.zip"

    logger.info(f"开始下载 {dataset_info['description']}")

    if not download_file(dataset_info["url"], zip_path):
        return False

    # 解压
    if not extract_zip(zip_path, data_dir):
        return False

    # 删除 ZIP 文件
    zip_path.unlink()
    logger.info("清理临时文件")

    logger.info(f"COCO128 数据集下载完成: {coco128_dir}")
    return True


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 数据集下载工具")
    parser.add_argument("--dataset", type=str, default="coco128", choices=list(DATASETS.keys()), help="数据集名称")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else None

    if args.dataset == "coco128":
        success = download_coco128(output_dir)
    else:
        logger.error(f"未知数据集: {args.dataset}")
        success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
