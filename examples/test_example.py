"""测试示例

演示如何使用 YOLOTester 进行模型测试
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tester import YOLOTester


def example_image_test():
    """图片测试示例"""
    print("=" * 60)
    print("示例 1: 图片测试")
    print("=" * 60)

    # 使用配置文件测试
    config_path = project_root / "configs" / "test.yaml"
    tester = YOLOTester(config_path)

    # 测试图片
    # 注意：需要先准备测试图片
    # results = tester.test_image("path/to/image.jpg")

    print("请修改代码中的图片路径后运行")


def example_video_test():
    """视频测试示例"""
    print("\n" + "=" * 60)
    print("示例 2: 视频测试")
    print("=" * 60)

    config = {
        "model": {"weights": "yolov8n.pt"},
        "data": {"source": "path/to/video.mp4"},  # 修改为你的视频路径
        "test": {
            "imgsz": 640,
            "conf": 0.25,
            "iou": 0.45,
            "device": 0,
            "project": "test",
            "name": "video_test",
            "save": True,
        },
    }

    tester = YOLOTester(config)
    # results = tester.test_video("path/to/video.mp4")

    print("请修改代码中的视频路径后运行")


def example_camera_test():
    """摄像头测试示例"""
    print("\n" + "=" * 60)
    print("示例 3: 摄像头测试")
    print("=" * 60)

    config = {
        "model": {"weights": "yolov8n.pt"},
        "test": {
            "imgsz": 640,
            "conf": 0.25,
            "iou": 0.45,
            "device": 0,
            "project": "test",
            "name": "camera_test",
            "view_img": True,  # 显示图片
        },
    }

    tester = YOLOTester(config)
    # results = tester.test_camera(camera_id=0)

    print("请取消注释代码后运行（需要摄像头）")


def example_batch_test():
    """批量测试示例"""
    print("\n" + "=" * 60)
    print("示例 4: 批量测试")
    print("=" * 60)

    config = {
        "model": {"weights": "yolov8n.pt"},
        "data": {"source": "data/images"},  # 修改为你的图片目录
        "test": {
            "imgsz": 640,
            "conf": 0.25,
            "iou": 0.45,
            "device": 0,
            "project": "test",
            "name": "batch_test",
            "save": True,
        },
    }

    tester = YOLOTester(config)
    # results = tester.test()
    # summary = tester.get_summary()

    # print(f"处理图片数: {summary.get('total_images', 0)}")
    # print(f"检测目标数: {summary.get('total_detections', 0)}")

    print("请修改代码中的图片目录后运行")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8 测试示例")
    parser.add_argument("--example", type=int, default=1, choices=[1, 2, 3, 4], help="示例编号")
    args = parser.parse_args()

    examples = {
        1: example_image_test,
        2: example_video_test,
        3: example_camera_test,
        4: example_batch_test,
    }

    example_func = examples.get(args.example)
    if example_func:
        example_func()
    else:
        print(f"未知示例: {args.example}")


if __name__ == "__main__":
    main()
