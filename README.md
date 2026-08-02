# YOLOv8 阶段一学习项目

> 从环境搭建到基础实践的完整 YOLOv8 学习框架

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.0%2B-green)](https://github.com/ultralytics/ultralytics)

---

## 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [文档](#文档)
- [示例](#示例)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 项目简介

本项目是 YOLOv8 阶段一学习项目，旨在帮助开发者快速搭建 YOLOv8 开发环境，掌握完整的训练工作流程。项目采用模块化设计，提供一键环境配置、训练、验证、测试、评估、导出等完整功能。

**学习目标：**
- 掌握 YOLOv8 环境搭建和配置
- 熟悉训练、验证、测试、评估完整流程
- 学会使用数据预处理工具
- 掌握可视化分析方法
- 了解模型导出和部署

---

## 功能特性

- **一键环境配置**: 自动检测 CUDA、安装依赖、验证环境
- **完整工作流程**: 训练 → 验证 → 测试 → 评估 → 导出
- **数据预处理**: 数据增强、格式转换、数据集划分
- **可视化分析**: 训练曲线、混淆矩阵、检测结果展示
- **模型导出**: ONNX、TensorRT、OpenVINO 等格式
- **配置驱动**: YAML 配置文件，支持多实验管理
- **模块化设计**: 易于扩展和定制

---

## 快速开始

### 1. 环境配置

**Windows:**
```bash
# 双击运行
scripts\setup_env.bat
```

**Linux/macOS:**
```bash
# 添加执行权限
chmod +x scripts/setup_env.sh

# 运行安装脚本
./scripts/setup_env.sh
```

### 2. 下载数据集

```bash
python scripts/download_data.py --dataset coco128
```

### 3. 一键运行

```bash
python scripts/quick_start.py
```

### 4. 分步运行

```bash
# 训练
python src/trainer.py --config configs/train.yaml

# 验证
python src/validator.py --config configs/val.yaml

# 测试
python src/tester.py --config configs/test.yaml

# 评估
python src/evaluator.py --config configs/val.yaml
```

---

## 项目结构

```
YOLO_learning/
├── configs/                    # 配置文件目录
│   ├── train.yaml             # 训练配置
│   ├── val.yaml               # 验证配置
│   ├── test.yaml              # 测试配置
│   └── data/                  # 数据集配置
│       ├── coco128.yaml       # COCO128 数据集配置
│       └── custom.yaml        # 自定义数据集模板
│
├── src/                       # 源代码目录
│   ├── trainer.py             # 训练器
│   ├── validator.py           # 验证器
│   ├── tester.py              # 测试器
│   ├── evaluator.py           # 评估器
│   └── exporter.py            # 导出器
│
├── tools/                     # 工具目录
│   ├── data/                  # 数据处理工具
│   │   ├── augment.py         # 数据增强
│   │   ├── convert.py         # 格式转换
│   │   └── split.py           # 数据集划分
│   ├── visualization/         # 可视化工具
│   │   ├── curves.py          # 训练曲线
│   │   ├── confusion.py       # 混淆矩阵
│   │   └── detection.py       # 检测结果
│   └── utils/                 # 通用工具
│       ├── env_check.py       # 环境检查
│       ├── logger.py          # 日志管理
│       └── helpers.py         # 辅助函数
│
├── scripts/                   # 脚本目录
│   ├── setup_env.sh           # Linux/macOS 环境配置脚本
│   ├── setup_env.bat          # Windows 环境配置脚本
│   ├── quick_start.py         # 快速开始脚本
│   └── download_data.py       # 数据集下载脚本
│
├── examples/                  # 示例目录
│   ├── train_example.py       # 训练示例
│   ├── val_example.py         # 验证示例
│   ├── test_example.py        # 测试示例
│   └── export_example.py      # 导出示例
│
├── docs/                      # 文档目录
│   ├── installation.md        # 安装指南
│   ├── quickstart.md          # 快速开始
│   ├── data_preparation.md    # 数据准备
│   └── api_reference.md       # API 文档
│
├── tests/                     # 测试目录
│   ├── test_trainer.py        # 训练器测试
│   ├── test_validator.py      # 验证器测试
│   └── test_tools.py          # 工具测试
│
├── data/                      # 数据目录（gitignore）
├── runs/                      # 运行输出目录（gitignore）
├── weights/                   # 模型权重目录（gitignore）
│
├── .gitignore                 # Git 忽略文件
├── requirements.txt           # Python 依赖列表
├── setup.py                   # 包安装配置
├── LICENSE                    # MIT 许可证
└── README.md                  # 项目说明文档
```

---

## 文档

- [安装指南](docs/installation.md) - 详细的环境配置说明
- [快速开始](docs/quickstart.md) - 5 分钟上手教程
- [数据准备](docs/data_preparation.md) - 数据集格式和准备方法
- [API 文档](docs/api_reference.md) - 核心模块使用说明
- [学习路线](plan/plan.md) - 完整的 YOLO 学习路径

---

## 示例

### 训练示例

```python
from src.trainer import YOLOTrainer

# 创建训练器
trainer = YOLOTrainer("configs/train.yaml")

# 开始训练
metrics = trainer.train()

print(f"最佳模型: {metrics['best_model']}")
```

### 验证示例

```python
from src.validator import YOLOValidator

# 创建验证器
validator = YOLOValidator("configs/val.yaml")

# 开始验证
metrics = validator.validate()
validator.print_metrics()
```

### 测试示例

```python
from src.tester import YOLOTester

# 创建测试器
tester = YOLOTester("configs/test.yaml")

# 测试图片
results = tester.test_image("path/to/image.jpg")
```

### 导出示例

```python
from src.exporter import YOLOExporter

# 创建导出器
exporter = YOLOExporter("configs/export.yaml")

# 导出 ONNX
exporter.export_onnx(imgsz=640, half=False, dynamic=True)
```

更多示例请查看 [examples/](examples/) 目录。

---

## 贡献指南

欢迎贡献！请随时提交 Issue 或 Pull Request。

1. Fork 本仓库
2. 创建功能分支（`git checkout -b feature/AmazingFeature`）
3. 提交更改（`git commit -m 'Add some AmazingFeature'`）
4. 推送到分支（`git push origin feature/AmazingFeature`）
5. 提交 Pull Request

---

## 许可证

本项目采用 MIT 许可证 —— 详见 [LICENSE](LICENSE) 文件

---

## 致谢

- [Ultralytics](https://github.com/ultralytics/ultralytics) 提供了优秀的 YOLO 框架
- [Roboflow](https://roboflow.com) 提供了数据集托管和标注工具
- 所有数据集提供者和开源贡献者

---

<div align="center">

**如果觉得有帮助，请给本项目点个 Star！**

</div>
