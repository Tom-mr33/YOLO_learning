# 数据准备

本文档介绍如何准备 YOLO 格式的数据集。

## YOLO 格式说明

YOLO 格式的数据集包含以下结构：

```
dataset/
├── images/
│   ├── train/          # 训练集图片
│   ├── val/            # 验证集图片
│   └── test/           # 测试集图片（可选）
└── labels/
    ├── train/          # 训练集标注
    ├── val/            # 验证集标注
    └── test/           # 测试集标注（可选）
```

### 标注文件格式

每个图片对应一个 `.txt` 标注文件，每行表示一个目标：

```
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id`: 类别 ID（从 0 开始）
- `x_center`: 边界框中心点 x 坐标（归一化到 0-1）
- `y_center`: 边界框中心点 y 坐标（归一化到 0-1）
- `width`: 边界框宽度（归一化到 0-1）
- `height`: 边界框高度（归一化到 0-1）

**示例：**

```
0 0.5 0.5 0.2 0.3
1 0.3 0.7 0.1 0.15
```

## 数据集划分

使用数据划分工具将数据集划分为训练集、验证集、测试集：

```bash
python tools/data/split.py \
    --data data/raw \
    --output data/split \
    --train 0.7 \
    --val 0.2 \
    --test 0.1
```

**参数说明：**
- `--data`: 原始数据集目录（包含 images 和 labels 子目录）
- `--output`: 输出目录
- `--train`: 训练集比例（默认 0.7）
- `--val`: 验证集比例（默认 0.2）
- `--test`: 测试集比例（默认 0.1）
- `--seed`: 随机种子（默认 42）
- `--move`: 移动文件（默认复制）

## 格式转换

### VOC 转 YOLO

```bash
python tools/data/convert.py \
    --format voc \
    --input data/voc \
    --output data/yolo
```

**VOC 数据集结构：**

```
voc/
├── Annotations/        # XML 标注文件
└── JPEGImages/         # 图片文件
```

### COCO 转 YOLO

```bash
python tools/data/convert.py \
    --format coco \
    --input data/coco/annotations.json \
    --images data/coco/images \
    --output data/yolo
```

**COCO 数据集结构：**

```
coco/
├── annotations.json    # COCO JSON 标注文件
└── images/             # 图片文件
```

## 数据增强

使用数据增强工具对数据集进行增强：

```bash
python tools/data/augment.py \
    --data data/train \
    --output data/train_augmented \
    --num 2
```

**参数说明：**
- `--data`: 数据集目录
- `--output`: 输出目录
- `--num`: 每张图片增强次数（默认 1）

**支持的增强方法：**
- 水平翻转、垂直翻转
- 旋转、平移、缩放、剪切
- HSV 颜色空间变换
- 模糊、噪声
- 随机擦除

## 数据集配置文件

创建数据集配置文件 `configs/data/custom.yaml`：

```yaml
# 数据集根目录
path: ../data/custom

# 训练集、验证集、测试集路径（相对于 path）
train: images/train
val: images/val
test: images/test

# 类别数量
nc: 2

# 类别名称
names:
  0: class1
  1: class2
```

## 数据标注工具

推荐使用以下工具进行数据标注：

### LabelImg

- **下载**: https://github.com/tzutalin/labelImg
- **优点**: 简单易用，支持 YOLO 格式
- **缺点**: 功能较少

### Roboflow Annotate

- **网址**: https://roboflow.com/annotate
- **优点**: 在线标注，支持团队协作，自动数据增强
- **缺点**: 需要网络连接

### CVAT

- **网址**: https://cvat.org/
- **优点**: 功能强大，支持视频标注
- **缺点**: 部署较复杂

## 数据集推荐

### 入门数据集

| 数据集 | 图片数 | 类别数 | 下载地址 |
|--------|--------|--------|----------|
| COCO128 | 128 | 80 | 内置 |
| 口罩佩戴检测 | 约 800 | 2 | [Roboflow](https://public.roboflow.com/object-detection/mask-wearing) |
| SHWD 安全帽检测 | 7,581 | 2 | [Gitee](https://gitee.com/shixiuyu/Safety-Helmet-Wearing-Dataset) |

### 进阶数据集

| 数据集 | 图片数 | 类别数 | 下载地址 |
|--------|--------|--------|----------|
| CCPD 车牌检测 | 250,000+ | 1 | [GitHub](https://github.com/detectRecog/CCPD) |
| TT100K 交通标志 | 100,000 | 221 | [Ultralytics](https://docs.ultralytics.com/datasets/detect/tt100k/) |
| VisDrone 无人机航拍 | 8,629 | 10 | [GitHub](https://github.com/VisDrone/VisDrone-Dataset) |

## 常见问题

### Q: 如何检查标注文件是否正确？

A: 可以使用以下 Python 代码检查：

```python
from pathlib import Path

def check_labels(labels_dir):
    labels_dir = Path(labels_dir)
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"格式错误: {label_file}")
                    continue
                class_id, x, y, w, h = map(float, parts)
                if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    print(f"坐标超出范围: {label_file}")

check_labels("data/labels/train")
```

### Q: 如何可视化标注结果？

A: 可以使用以下 Python 代码可视化：

```python
import cv2
from pathlib import Path

def visualize_label(image_path, label_path, class_names):
    # 读取图片
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]

    # 读取标注
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            class_id, x, y, w_norm, h_norm = map(float, parts)

            # 转换为像素坐标
            x1 = int((x - w_norm / 2) * w)
            y1 = int((y - h_norm / 2) * h)
            x2 = int((x + w_norm / 2) * w)
            y2 = int((y + h_norm / 2) * h)

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, class_names[int(class_id)], (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示图片
    cv2.imshow("Label", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 使用示例
visualize_label(
    "data/images/train/image1.jpg",
    "data/labels/train/image1.txt",
    ["class1", "class2"]
)
```

## 下一步

- 查看 [快速开始](quickstart.md) 了解如何使用准备好的数据集进行训练
- 查看 [API 文档](api_reference.md) 了解数据处理工具的详细使用说明
