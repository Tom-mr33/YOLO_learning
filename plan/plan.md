# YOLO 学习路线

> 从 YOLO 基础到工程落地的完整学习路线

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

---

## 目录

- [概述](#概述)
- [前置要求](#前置要求)
- [学习路线](#学习路线)
  - [第一阶段：环境搭建与基础实践](#第一阶段环境搭建与基础实践)
  - [第二阶段：完整项目实战](#第二阶段完整项目实战)
  - [第三阶段：进阶专题深入](#第三阶段进阶专题深入)
  - [第四阶段：工程部署与落地](#第四阶段工程部署与落地)
- [数据集推荐](#数据集推荐)
- [学习资源](#学习资源)
- [时间规划](#时间规划)
- [贡献指南](#贡献指南)
- [许可证](#许可证)

---

## 概述

本仓库记录了一条结构化的 YOLO（You Only Look Once）目标检测学习路线，从基础概念到生产环境部署。适合已经理解 YOLO 基本原理和架构，希望通过动手项目深入实践的开发者。

**学习目标：**
- 掌握完整的 YOLO 工作流程：数据准备 → 训练 → 评估 → 推理
- 完成多个端到端项目，覆盖检测、分割、姿态估计和跟踪任务
- 掌握针对复杂场景（小目标、遮挡等）的模型优化技巧
- 将模型部署到边缘设备和生产环境

---

## 前置要求

开始之前，请确保具备以下条件：

- **硬件**：NVIDIA GPU（推荐 RTX 3060 及以上）或云 GPU 资源（Colab、Kaggle、AutoDL）
- **软件**：Python 3.8+、PyTorch 2.0+、CUDA 11.8+
- **知识储备**：CNN 基础、目标检测概念（IoU、NMS、Anchor）、PyTorch 基础

**环境检查：**

```bash
# 验证 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 安装 Ultralytics
pip install ultralytics
```

---

## 学习路线

### 第一阶段：环境搭建与基础实践

**周期**：1-2 周  
**目标**：跑通完整的 YOLO 工作流程，熟悉训练管道

#### 任务清单

- [ ] 安装 Ultralytics 并验证环境
- [ ] 使用预训练模型进行推理（`yolov8n.pt`、`yolo11n.pt`）
- [ ] 在 COCO128 上训练，验证训练流程
- [ ] 实验关键超参数（`imgsz`、`batch`、`lr0`、`epochs`）
- [ ] 对比不同模型大小（n/s/m/l/x）的速度与精度权衡

#### 快速开始

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 图片推理
results = model('path/to/image.jpg')

# 在 COCO128 上训练
model.train(data='coco128.yaml', epochs=50, imgsz=640)
```

#### 推荐数据集

| 数据集 | 图片数 | 类别数 | 用途 |
|--------|--------|--------|------|
| COCO128 | 128 | 80 | 验证训练流程 |

---

### 第二阶段：完整项目实战

**周期**：3-6 周  
**目标**：独立完成从数据标注到模型评估的完整项目

#### 项目 2.1：自定义目标检测

**场景选择**：车牌检测 / 安全帽检测 / 口罩佩戴检测

**工作流程：**

1. **数据收集**：从 Roboflow Universe 或开源数据集下载
2. **数据标注**：使用 LabelImg 或 Roboflow Annotate（YOLO 格式）
3. **数据集划分**：训练集/验证集/测试集 = 70/20/10
4. **模型训练**：对比不同大小的模型
5. **模型评估**：mAP@0.5、mAP@0.5:0.95、精确率、召回率、混淆矩阵
6. **推理应用**：图片、视频、实时摄像头

**推荐数据集：**

| 数据集 | 图片数 | 类别数 | 下载地址 |
|--------|--------|--------|----------|
| 口罩佩戴检测 | 约 800 | 2 | [Roboflow](https://public.roboflow.com/object-detection/mask-wearing) |
| SHWD 安全帽检测 | 7,581 | 2 | [Gitee](https://gitee.com/shixiuyu/Safety-Helmet-Wearing-Dataset) |
| CCPD 车牌检测 | 250,000+ | 1 | [GitHub](https://github.com/detectRecog/CCPD) |

#### 项目 2.2：实例分割

```python
model = YOLO('yolov8n-seg.pt')
model.train(data='seg.yaml', epochs=100)
```

**应用场景**：缺陷检测、医学图像分割、农作物病害分割

#### 项目 2.3：姿态估计

```python
model = YOLO('yolov8n-pose.pt')
model.train(data='coco8-pose.yaml', epochs=100)
```

**应用场景**：人体姿态估计、动作识别、运动分析

#### 项目 2.4：目标跟踪

```python
model = YOLO('yolov8n.pt')
results = model.track(source='video.mp4', tracker='bytetrack.yaml')
```

**应用场景**：车辆计数、行人轨迹分析、多目标跟踪

---

### 第三阶段：进阶专题深入

**周期**：6-10 周  
**目标**：针对特定难点深入研究，掌握模型改进能力

#### 专题 3.1：小目标检测（航拍/无人机场景）

**数据集**：VisDrone（6,471 张训练图，10 个类别）

**改进策略：**

- 增加 P2 检测头，提升高分辨率小目标检测能力
- 引入注意力机制（CBAM、SE、ECA、CA）
- 调整 Anchor 尺寸，增加小 Anchor 比例
- 使用 SAHI（切片辅助超推理）技术

#### 专题 3.2：模型改进与论文复现

**常见改进模块：**

| 类别 | 模块 |
|------|------|
| 注意力机制 | CBAM、SE、ECA、CA（坐标注意力） |
| 特征融合 | BiFPN、GSConv |
| 损失函数 | SIoU、WIoU、MPDIoU |
| 检测头 | 解耦头、ASFF |

**实现示例：**

```yaml
# ultralytics/cfg/models/v8/yolov8.yaml
backbone:
  [[-1, 1, Conv, [64, 3, 2]],
   [-1, 1, Conv, [128, 3, 2]],
   [-1, 1, CBAM, [128]],  # 添加注意力模块
   ...]
```

#### 专题 3.3：实时检测系统开发

**技术栈：**
- 后端：Flask / FastAPI
- 前端：Gradio / Streamlit / HTML+JS
- 视频流：RTSP、WebRTC
- 数据库：SQLite / PostgreSQL 存储检测结果

**项目示例**：智能安防监控系统——实时检测画面中的人员、车辆，触发告警，记录事件到数据库，提供 Web 界面查看

---

### 第四阶段：工程部署与落地

**周期**：4-8 周  
**目标**：将模型部署到真实环境，掌握端侧/边缘部署能力

#### 部署链路

```
PyTorch (.pt) → ONNX → TensorRT / OpenVINO / NCNN / TFLite
```

#### 平台部署方案

| 目标平台 | 模型 | 推理引擎 | 预期帧率 |
|----------|------|----------|----------|
| 服务器 GPU（RTX 3060） | YOLOv8s | TensorRT FP16 | 200+ FPS |
| Jetson Nano | YOLOv8n | TensorRT FP16 | 15-25 FPS |
| 树莓派 4B | YOLOv8n | NCNN | 5-10 FPS |
| Android 手机 | YOLOv8n | TFLite | 10-20 FPS |

#### 模型导出命令

```python
# ONNX 导出（跨平台通用）
model.export(format='onnx', simplify=True, dynamic=True)

# TensorRT 导出（NVIDIA GPU 最优性能）
model.export(format='engine', half=True)

# OpenVINO 导出（Intel CPU 优化）
model.export(format='openvino')
```

---

## 数据集推荐

### 快速参考

| 数据集 | 规模 | 难度 | 应用场景 | 链接 |
|--------|------|------|----------|------|
| COCO128 | 128 张 | 入门 | 流程验证 | 内置 |
| 口罩佩戴检测 | 约 800 张 | 入门 | 第一个项目 | [Roboflow](https://public.roboflow.com/object-detection/mask-wearing) |
| SHWD 安全帽检测 | 7,581 张 | 入门 | 安全帽检测 | [Gitee](https://gitee.com/shixiuyu/Safety-Helmet-Wearing-Dataset) |
| CCPD 车牌检测 | 250,000+ 张 | 中级 | 车牌检测 | [GitHub](https://github.com/detectRecog/CCPD) |
| TT100K 交通标志 | 100,000 张 | 中级 | 交通标志检测 | [Ultralytics](https://docs.ultralytics.com/datasets/detect/tt100k/) |
| VisDrone 无人机航拍 | 8,629 张 | 高级 | 小目标检测 | [GitHub](https://github.com/VisDrone/VisDrone-Dataset) |
| KITTI 自动驾驶 | 15,000 张 | 高级 | 自动驾驶 | [KITTI](http://www.cvlibs.net/datasets/kitti/) |

### 数据集获取平台

- **Roboflow Universe**：[universe.roboflow.com](https://universe.roboflow.com) —— 一键下载 YOLO 格式数据集
- **Kaggle**：[kaggle.com/datasets](https://www.kaggle.com/datasets) —— 社区数据集
- **Ultralytics 官方文档**：[docs.ultralytics.com/datasets](https://docs.ultralytics.com/datasets/) —— 自动下载支持
- **飞桨 AI Studio**：[aistudio.baidu.com](https://aistudio.baidu.com) —— 国内数据集镜像，下载速度快

---

## 学习资源

### 官方文档

- [Ultralytics 官方文档](https://docs.ultralytics.com)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
- [YOLO 系列论文综述](https://arxiv.org/abs/2510.09653)

### 学习材料

- YOLOv1-v12 论文精读（理解每个版本的改进动机）
- CBAM、BiFPN、解耦头等经典模块原始论文
- 目标检测综述类论文

### 学习社区

- CSDN / 知乎：YOLO 改进和部署的中文教程
- B 站：视频教程（搜索"YOLOv8 教程"）
- GitHub：搜索 "YOLOv8 improvement" 获取开源改进项目

---

## 时间规划

```
第 1-2 周：   环境搭建 + 预训练模型推理 + 超参数实验
第 3-6 周：   自定义检测项目（数据标注 → 训练 → 评估）
第 7-8 周：   分割 + 姿态估计 + 跟踪（各 1-2 周）
第 9-14 周：  小目标检测专题（VisDrone）+ 模型改进实验
第 15-18 周： 实时检测系统开发
第 19-24 周： 模型部署到边缘设备
```

**总周期**：约 4-6 个月

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
