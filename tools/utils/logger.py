"""日志管理工具

提供统一的日志配置和管理
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime


# 日志格式
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"

# 日志级别映射
LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


def setup_logger(
    name: str = "yolo",
    level: str = "info",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    format_str: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """设置日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别 (debug/info/warning/error/critical)
        log_file: 日志文件路径（可选）
        log_dir: 日志目录（可选，如果提供 log_file 则忽略）
        format_str: 日志格式（可选）
        console: 是否输出到控制台

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(LEVEL_MAP.get(level.lower(), logging.INFO))

    # 清除已有的处理器
    logger.handlers.clear()

    # 设置日志格式
    if format_str is None:
        format_str = DEFAULT_FORMAT
    formatter = logging.Formatter(format_str)

    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(LEVEL_MAP.get(level.lower(), logging.INFO))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件处理器
    if log_file or log_dir:
        if log_dir:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir_path / f"{name}_{timestamp}.log"
        else:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(LEVEL_MAP.get(level.lower(), logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "yolo") -> logging.Logger:
    """获取日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        日志记录器实例
    """
    return logging.getLogger(name)


class LoggerMixin:
    """日志记录器混入类"""

    @property
    def logger(self) -> logging.Logger:
        """获取日志记录器"""
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


def log_function_call(func):
    """装饰器：记录函数调用"""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"调用函数: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"函数 {func.__name__} 执行成功")
            return result
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行失败: {e}")
            raise
    return wrapper


def log_execution_time(func):
    """装饰器：记录函数执行时间"""
    import time

    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.info(f"函数 {func.__name__} 执行时间: {elapsed_time:.2f} 秒")
        return result
    return wrapper


if __name__ == "__main__":
    # 测试日志功能
    logger = setup_logger("test", level="debug", log_dir="logs")
    logger.debug("这是一条调试信息")
    logger.info("这是一条普通信息")
    logger.warning("这是一条警告信息")
    logger.error("这是一条错误信息")
    print(f"日志文件已保存到: logs/")
