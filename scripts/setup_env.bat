@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

::: ============================================
::: YOLOv8 阶段一环境配置脚本 (Windows)
::: 自动检测 CUDA、安装依赖、验证环境
::: ============================================

echo.
echo ==========================================
echo   YOLOv8 阶段一环境配置脚本
echo ==========================================
echo.

::: 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

::: 获取 Python 版本
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [信息] 检测到 Python 版本: %PYTHON_VERSION%

::: 检查 Python 版本是否 >= 3.8
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [错误] Python 版本需要 >= 3.8，当前版本: %PYTHON_VERSION%
    pause
    exit /b 1
)

::: 检查 pip 是否安装
pip --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 pip，请先安装 pip
    pause
    exit /b 1
)

echo [信息] pip 已安装

::: 检测 NVIDIA GPU 和 CUDA
echo.
echo [信息] 检测 NVIDIA GPU 和 CUDA...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [警告] 未检测到 NVIDIA GPU 或驱动未安装
    echo [信息] 将安装 CPU 版本的 PyTorch
    set CUDA_AVAILABLE=false
    set CUDA_VERSION=cpu
) else (
    echo [信息] 检测到 NVIDIA GPU
    :: 获取 CUDA 版本
    for /f "tokens=9" %%i in ('nvidia-smi ^| findstr "CUDA Version"') do set CUDA_VERSION=%%i
    echo [信息] CUDA 版本: !CUDA_VERSION!
    set CUDA_AVAILABLE=true
)

::: 创建虚拟环境（可选）
echo.
set /p CREATE_VENV="是否创建虚拟环境? (y/n, 默认: n): "
if /i "%CREATE_VENV%"=="y" (
    echo [信息] 创建虚拟环境...
    python -m venv venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败
        pause
        exit /b 1
    )
    echo [信息] 激活虚拟环境...
    call venv\Scripts\activate.bat
    echo [信息] 虚拟环境已激活
)

::: 升级 pip
echo.
echo [信息] 升级 pip...
python -m pip install --upgrade pip

::: 安装 PyTorch
echo.
echo [信息] 安装 PyTorch...
if "%CUDA_AVAILABLE%"=="true" (
    :: 根据 CUDA 版本安装对应的 PyTorch
    :: CUDA 13.x 和 12.x 都使用 cu126（向下兼容）
    if "!CUDA_VERSION:~0,2!"=="13" (
        echo [信息] 检测到 CUDA 13.x，安装 CUDA 12.6 版本的 PyTorch（向下兼容）...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ) else if "!CUDA_VERSION:~0,4!"=="12.6" (
        echo [信息] 安装 CUDA 12.6 版本的 PyTorch...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    ) else if "!CUDA_VERSION:~0,4!"=="12.1" (
        echo [信息] 安装 CUDA 12.1 版本的 PyTorch...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ) else if "!CUDA_VERSION:~0,4!"=="11.8" (
        echo [信息] 安装 CUDA 11.8 版本的 PyTorch...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ) else (
        echo [警告] 未识别的 CUDA 版本: !CUDA_VERSION!
        echo [信息] 尝试安装 CUDA 12.6 版本的 PyTorch...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    )
) else (
    echo [信息] 安装 CPU 版本的 PyTorch...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

::: 安装项目依赖
echo.
echo [信息] 安装项目依赖...
pip install -r requirements.txt

::: 验证安装
echo.
echo [信息] 验证环境配置...
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU 数量: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

if errorlevel 1 (
    echo [错误] PyTorch 安装验证失败
    pause
    exit /b 1
)

::: 验证 Ultralytics
python -c "from ultralytics import YOLO; print('Ultralytics 安装成功')" >nul 2>&1
if errorlevel 1 (
    echo [错误] Ultralytics 安装失败
    pause
    exit /b 1
) else (
    echo [信息] Ultralytics 安装成功
)

::: 下载预训练权重
echo.
set /p DOWNLOAD_WEIGHTS="是否下载 YOLOv8n 预训练权重? (y/n, 默认: y): "
if /i "%DOWNLOAD_WEIGHTS%"=="n" goto :skip_download
echo [信息] 下载 YOLOv8n 预训练权重...
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
:skip_download

::: 创建必要目录
echo.
echo [信息] 创建项目目录...
if not exist "data" mkdir data
if not exist "runs" mkdir runs
if not exist "runs\train" mkdir runs\train
if not exist "runs\val" mkdir runs\val
if not exist "runs\test" mkdir runs\test
if not exist "weights" mkdir weights
if not exist "weights\pretrained" mkdir weights\pretrained
if not exist "weights\trained" mkdir weights\trained

::: 完成
echo.
echo ==========================================
echo   环境配置完成！
echo ==========================================
echo.
echo 下一步:
echo   1. 运行快速开始脚本: python scripts/quick_start.py
echo   2. 查看文档: docs/quickstart.md
echo   3. 开始训练: python examples/train_example.py
echo.

pause
