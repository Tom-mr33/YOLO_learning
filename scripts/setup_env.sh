#!/bin/bash

# ============================================
# YOLOv8 阶段一环境配置脚本 (Linux/macOS)
# 自动检测 CUDA、安装依赖、验证环境
# ============================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的信息
info() {
    echo -e "${BLUE}[信息]${NC} $1"
}

success() {
    echo -e "${GREEN}[成功]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[警告]${NC} $1"
}

error() {
    echo -e "${RED}[错误]${NC} $1"
}

echo ""
echo "=========================================="
echo "  YOLOv8 阶段一环境配置脚本"
echo "=========================================="
echo ""

# 检查 Python 是否安装
if ! command -v python3 &> /dev/null; then
    error "未检测到 Python3，请先安装 Python 3.8+"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "  macOS: brew install python3"
    exit 1
fi

# 获取 Python 版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
info "检测到 Python 版本: $PYTHON_VERSION"

# 检查 Python 版本是否 >= 3.8
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" &> /dev/null; then
    error "Python 版本需要 >= 3.8，当前版本: $PYTHON_VERSION"
    exit 1
fi

# 检查 pip 是否安装
if ! command -v pip3 &> /dev/null; then
    error "未检测到 pip3，请先安装 pip"
    exit 1
fi

info "pip3 已安装"

# 检测 NVIDIA GPU 和 CUDA
echo ""
info "检测 NVIDIA GPU 和 CUDA..."
if command -v nvidia-smi &> /dev/null; then
    info "检测到 NVIDIA GPU"
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    info "CUDA 版本: $CUDA_VERSION"
    CUDA_AVAILABLE=true
else
    warning "未检测到 NVIDIA GPU 或驱动未安装"
    info "将安装 CPU 版本的 PyTorch"
    CUDA_AVAILABLE=false
    CUDA_VERSION="cpu"
fi

# 创建虚拟环境（可选）
echo ""
read -p "是否创建虚拟环境? (y/n, 默认: n): " CREATE_VENV
if [[ "$CREATE_VENV" =~ ^[Yy]$ ]]; then
    info "创建虚拟环境..."
    python3 -m venv venv
    info "激活虚拟环境..."
    source venv/bin/activate
    success "虚拟环境已激活"
fi

# 升级 pip
echo ""
info "升级 pip..."
python3 -m pip install --upgrade pip

# 安装 PyTorch
echo ""
info "安装 PyTorch..."
if [ "$CUDA_AVAILABLE" = true ]; then
    # 根据 CUDA 版本安装对应的 PyTorch
    # CUDA 13.x 和 12.x 都使用 cu126（向下兼容）
    if [[ "$CUDA_VERSION" == 13* ]]; then
        info "检测到 CUDA 13.x，安装 CUDA 12.6 版本的 PyTorch（向下兼容）..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    elif [[ "$CUDA_VERSION" == 12.6* ]]; then
        info "安装 CUDA 12.6 版本的 PyTorch..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    elif [[ "$CUDA_VERSION" == 12.1* ]]; then
        info "安装 CUDA 12.1 版本的 PyTorch..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_VERSION" == 11.8* ]]; then
        info "安装 CUDA 11.8 版本的 PyTorch..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        warning "未识别的 CUDA 版本: $CUDA_VERSION"
        info "尝试安装 CUDA 12.6 版本的 PyTorch..."
        pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    fi
else
    info "安装 CPU 版本的 PyTorch..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 安装项目依赖
echo ""
info "安装项目依赖..."
pip3 install -r requirements.txt

# 验证安装
echo ""
info "验证环境配置..."
python3 -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
print(f'CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
print(f'GPU 数量: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')
"

if [ $? -ne 0 ]; then
    error "PyTorch 安装验证失败"
    exit 1
fi

# 验证 Ultralytics
if python3 -c "from ultralytics import YOLO; print('Ultralytics 安装成功')" &> /dev/null; then
    success "Ultralytics 安装成功"
else
    error "Ultralytics 安装失败"
    exit 1
fi

# 下载预训练权重
echo ""
read -p "是否下载 YOLOv8n 预训练权重? (y/n, 默认: y): " DOWNLOAD_WEIGHTS
if [[ ! "$DOWNLOAD_WEIGHTS" =~ ^[Nn]$ ]]; then
    info "下载 YOLOv8n 预训练权重..."
    python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
fi

# 创建必要目录
echo ""
info "创建项目目录..."
mkdir -p data
mkdir -p runs/detect/train
mkdir -p runs/detect/val
mkdir -p runs/detect/test
mkdir -p weights/pretrained
mkdir -p weights/trained

# 完成
echo ""
echo "=========================================="
echo "  环境配置完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "  1. 运行快速开始脚本: python3 scripts/quick_start.py"
echo "  2. 查看文档: docs/quickstart.md"
echo "  3. 开始训练: python3 examples/train_example.py"
echo ""
