"""计算设备检测：CUDA / Apple MPS / CPU 自动选择，含 CPU 回退提示。"""

from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger(__name__)


class Device(str, Enum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


def detect_device(force: str = "auto") -> Device:
    """检测可用计算设备。

    优先级：cuda > mps > cpu。force 可指定 cuda/mps/cpu/auto。
    无 GPU 时回退 CPU 并提示耗时增加。
    """
    if force and force != "auto":
        dev = Device(force)
        if dev == Device.CPU:
            logger.warning("使用 CPU 模式，音色迁移会较慢（一首歌可能数分钟到数十分钟）。")
        return dev

    try:
        import torch
    except ImportError:
        logger.warning("未安装 torch，使用 CPU 模式。安装 ml/rvc 依赖以启用 GPU 加速。")
        return Device.CPU

    if torch.cuda.is_available():
        logger.info("检测到 CUDA GPU: %s", torch.cuda.get_device_name(0))
        return Device.CUDA

    if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("检测到 Apple MPS (Metal)。")
        return Device.MPS

    logger.warning("未检测到可用 GPU，回退 CPU。音色迁移会较慢。")
    return Device.CPU


def torch_device_str(dev: Device) -> str:
    """转为 torch 可用的设备字符串。"""
    return dev.value
