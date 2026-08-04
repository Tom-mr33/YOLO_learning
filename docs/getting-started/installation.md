# 安装指南

本文档提供详细的环境配置说明，帮助你快速搭建 YOLOv8 开发环境。

## 系统要求

- **操作系统**: Windows 10/11, Linux, macOS
- **Python**: 3.8 或更高版本
- **GPU**: NVIDIA GPU（推荐 RTX 3060 及以上）或 CPU
- **CUDA**: 11.8 或 12.1（如果使用 GPU）

## 一键安装（推荐）

### Windows

1. 双击运行 `scripts/setup_env.bat`
2. 按照提示完成安装

### Linux/macOS

```bash
# 添加执行权限
chmod +x scripts/setup_env.sh

# 运行安装脚本
./scripts/setup_env.sh
```

## 手动安装

### 1. 安装 Python

确保已安装 Python 3.8 或更高版本：

```bash
python --version
```

### 2. 创建虚拟环境（可选但推荐）

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. 安装 PyTorch

根据你的 CUDA 版本选择合适的安装命令：

**CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CPU 版本:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. 安装项目依赖

```bash
pip install -r requirements.txt
```

### 5. 验证安装

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## 常见问题

### Q: 安装 PyTorch 时提示 CUDA 版本不匹配？

A: 请访问 [PyTorch 官网](https://pytorch.org/get-started/locally/) 查看与你的 CUDA 版本匹配的安装命令。

### Q: 如何检查 CUDA 版本？

A: 运行以下命令：

```bash
nvidia-smi
```

在输出中查找 "CUDA Version" 字段。

### Q: 没有 NVIDIA GPU 可以使用吗？

A: 可以，但训练速度会较慢。建议：
- 使用较小的模型（如 yolov8n）
- 减少批次大小（batch size）
- 减少训练轮数（epochs）

### Q: 安装 Ultralytics 失败？

A: 尝试以下方法：

```bash
# 升级 pip
pip install --upgrade pip

# 重新安装
pip install ultralytics --no-cache-dir
```

## 下一步

安装完成后，请查看 [快速开始](quickstart.md) 文档。
