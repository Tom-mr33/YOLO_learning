# 快速开始

本文档帮助你在 5 分钟内上手 YOLOv8 项目。

## 前提条件

确保已完成 [安装指南](installation.md) 中的环境配置。

## 一键运行（推荐）

运行快速开始脚本，自动完成训练、验证、测试、评估全流程：

```bash
python scripts/quick_start.py
```

## 分步运行

### 1. 下载数据集

```bash
python scripts/download_data.py --dataset coco128
```

### 2. 训练模型

```bash
# 使用默认配置训练
python src/trainer.py --config configs/train.yaml

# 或使用示例脚本
python examples/train_example.py --example 1
```

### 3. 验证模型

```bash
# 使用默认配置验证
python src/validator.py --config configs/val.yaml

# 或使用示例脚本
python examples/val_example.py --example 1
```

### 4. 测试模型

```bash
# 使用默认配置测试
python src/tester.py --config configs/test.yaml

# 或使用示例脚本
python examples/test_example.py --example 1
```

### 5. 评估模型

```bash
# 使用默认配置评估
python src/evaluator.py --config configs/val.yaml

# 保存评估报告
python src/evaluator.py --config configs/val.yaml --save runs/evaluation_report.json
```

## 使用 Python API

### 训练模型

```python
from src.trainer import YOLOTrainer

# 创建训练器
trainer = YOLOTrainer("configs/train.yaml")

# 开始训练
metrics = trainer.train()

print(f"最佳模型: {metrics['best_model']}")
```

### 验证模型

```python
from src.validator import YOLOValidator

# 创建验证器
validator = YOLOValidator("configs/val.yaml")

# 开始验证
metrics = validator.validate()
validator.print_metrics()
```

### 测试模型

```python
from src.tester import YOLOTester

# 创建测试器
tester = YOLOTester("configs/test.yaml")

# 测试图片
results = tester.test_image("path/to/image.jpg")

# 测试视频
results = tester.test_video("path/to/video.mp4")

# 测试摄像头
results = tester.test_camera(camera_id=0)
```

### 导出模型

```python
from src.exporter import YOLOExporter

# 创建导出器
exporter = YOLOExporter("configs/export.yaml")

# 导出 ONNX
exporter.export_onnx(imgsz=640, half=False, dynamic=True)

# 导出 TensorRT
exporter.export_tensorrt(imgsz=640, half=True)
```

## 自定义配置

### 修改训练参数

编辑 `configs/train.yaml`：

```yaml
train:
  epochs: 100        # 训练轮数
  batch: 16          # 批次大小
  imgsz: 640         # 输入图片尺寸
  lr0: 0.01          # 初始学习率
  device: 0          # 训练设备（0 表示 GPU 0）
```

### 使用自定义数据集

1. 准备数据集（YOLO 格式）：

```
data/custom/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

2. 创建数据集配置文件 `configs/data/custom.yaml`：

```yaml
path: ../data/custom
train: images/train
val: images/val
test: images/test

nc: 2  # 类别数量
names:
  0: class1
  1: class2
```

3. 修改训练配置 `configs/train.yaml`：

```yaml
data:
  name: custom.yaml
```

## 可视化结果

### 绘制训练曲线

```python
from tools.visualization.curves import CurvePlotter

plotter = CurvePlotter()
plotter.plot_training_curves(
    results_csv="runs/train/exp/results.csv",
    output_dir="runs/curves",
    experiment_name="my_experiment"
)
```

### 绘制混淆矩阵

```python
from tools.visualization.confusion import ConfusionMatrixPlotter

plotter = ConfusionMatrixPlotter()
plotter.plot_from_results(
    results_dir="runs/val/exp",
    output_dir="runs/confusion"
)
```

## 常见问题

### Q: 训练时提示 CUDA out of memory？

A: 尝试以下方法：
- 减小批次大小（batch size）
- 减小输入图片尺寸（imgsz）
- 使用较小的模型（如 yolov8n）

### Q: 如何使用自己的数据集？

A: 请参考 [数据准备](data_preparation.md) 文档。

### Q: 如何对比多个模型？

A: 运行示例脚本：

```bash
python examples/val_example.py --example 3
```

## 下一步

- 查看 [数据准备](data_preparation.md) 了解如何准备自定义数据集
- 查看 [API 文档](api_reference.md) 了解详细的 API 使用说明
- 查看 [学习路线](../plan/plan.md) 了解完整的学习路径
