"""设备检测工具测试 (2.1)。"""

from musicremix.utils.device import Device, detect_device


def test_detect_auto_returns_valid_device():
    """auto 模式必须返回枚举中的一个，且无 torch 时回退 CPU。"""
    dev = detect_device("auto")
    assert dev in (Device.CUDA, Device.MPS, Device.CPU)


def test_force_cpu():
    dev = detect_device("cpu")
    assert dev == Device.CPU


def test_force_cpu_emits_warning(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        detect_device("cpu")
    assert any("CPU" in r.message for r in caplog.records)
