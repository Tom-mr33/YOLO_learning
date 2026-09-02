"""YOLO 标签转 X-AnyLabeling JSON 格式脚本

把已有的 YOLO txt 标签（class cx cy w h，归一化坐标）转换成
X-AnyLabeling 的内部标注格式（labelme 风格 json），写入图片目录，
这样在 X-AnyLabeling 里打开图片目录就能直接看到已有的框。

用法（在项目根目录执行）:
    python scripts/yolo_to_xanylabeling.py --images data/mask-wearing/train/images --labels data/mask-wearing/train/labels
    python scripts/yolo_to_xanylabeling.py --images data/mask-wearing/valid/images --labels data/mask-wearing/valid/labels
    python scripts/yolo_to_xanylabeling.py --images data/mask-wearing/test/images --labels data/mask-wearing/test/labels
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.utils.logger import setup_logger

# 设置日志
logger = setup_logger("yolo2json", level="info")

# 默认类别配置文件（一行一个类别名，行号即类别 ID）
DEFAULT_CLASSES_FILE = project_root / "data" / "manual-practice" / "classes.txt"

# 与 X-AnyLabeling 4.0.5 导出的 json 字段保持一致
XANY_VERSION = "4.0.5"


def load_classes(classes_file: Path) -> list:
    """读取类别配置文件

    Args:
        classes_file: classes.txt 路径，一行一个类别名

    Returns:
        类别名列表，下标即类别 ID
    """
    classes = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise ValueError(f"类别文件为空: {classes_file}")
    return classes


def parse_yolo_txt(txt_path: Path) -> list:
    """解析单个 YOLO 标签文件

    Args:
        txt_path: 标签文件路径

    Returns:
        [(class_id, cx, cy, w, h), ...]，坐标为归一化 0-1
    """
    boxes = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            logger.warning(f"跳过格式异常的行: {txt_path.name} -> {line!r}")
            continue
        cls, cx, cy, w, h = (float(p) for p in parts)
        boxes.append((int(cls), cx, cy, w, h))
    return boxes


def build_json(image_path: Path, boxes: list, classes: list) -> dict:
    """构造 X-AnyLabeling 格式的标注字典

    Args:
        image_path: 图片路径
        boxes: YOLO 格式框列表
        classes: 类别名列表

    Returns:
        labelme 风格的 dict
    """
    with Image.open(image_path) as img:
        width, height = img.size

    shapes = []
    for cls, cx, cy, w, h in boxes:
        # 归一化中心点坐标 -> 像素左上/右下角坐标
        x1 = (cx - w / 2) * width
        y1 = (cy - h / 2) * height
        x2 = (cx + w / 2) * width
        y2 = (cy + h / 2) * height
        shapes.append({
            "label": classes[cls],
            "score": None,  # 人工标注无置信度
            "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "rectangle",
            "flags": {},
            "attributes": {},
            "kie_linking": [],
        })

    return {
        "version": XANY_VERSION,
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def convert(images_dir: Path, labels_dir: Path, classes_file: Path, overwrite: bool = False) -> tuple:
    """批量转换一个 split 的标签

    Args:
        images_dir: 图片目录（json 也写到这里）
        labels_dir: YOLO txt 标签目录
        classes_file: 类别配置文件
        overwrite: 是否覆盖已存在的 json

    Returns:
        (生成数量, 跳过数量, 无标签图片数量)
    """
    classes = load_classes(classes_file)
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif"}
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in image_exts)

    n_created, n_skipped, n_empty = 0, 0, 0
    for image_path in images:
        json_path = image_path.with_suffix(".json")
        if json_path.exists() and not overwrite:
            n_skipped += 1
            continue

        txt_path = labels_dir / (image_path.stem + ".txt")
        if txt_path.exists():
            boxes = parse_yolo_txt(txt_path)
        else:
            boxes = []  # 无标签图片也生成空 json，软件里会显示为"未标注"
            n_empty += 1

        data = build_json(image_path, boxes, classes)
        json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        n_created += 1

    return n_created, n_skipped, n_empty


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLO 标签转 X-AnyLabeling JSON 格式")
    parser.add_argument("--images", type=str, default="data/mask-wearing/train/images", help="图片目录")
    parser.add_argument("--labels", type=str, default="data/mask-wearing/train/labels", help="YOLO 标签目录")
    parser.add_argument("--classes", type=str, default=str(DEFAULT_CLASSES_FILE), help="类别配置文件（一行一个类别名）")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 json（默认跳过，保护手工标注）")
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    classes_file = Path(args.classes)

    for name, path in [("图片目录", images_dir), ("标签目录", labels_dir), ("类别文件", classes_file)]:
        if not path.exists():
            logger.error(f"{name}不存在: {path}")
            sys.exit(1)

    logger.info(f"开始转换: {images_dir}")
    n_created, n_skipped, n_empty = convert(images_dir, labels_dir, classes_file, args.overwrite)
    logger.info(f"完成: 生成 {n_created} 个 json，跳过已存在 {n_skipped} 个，无标签图片 {n_empty} 张")

    if n_empty:
        logger.info("提示: 无标签图片生成了空 shapes 的 json，在软件中显示为未标注状态")


if __name__ == "__main__":
    main()
