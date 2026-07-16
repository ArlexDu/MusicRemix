"""音色迁移单元测试 (5.6)。

ExternalRVCConverter 不依赖重库，可测试构造与错误处理。
"""

from __future__ import annotations

import pytest

from musicremix.conversion import get_converter
from musicremix.conversion.base import ConversionParams, VoiceConverter
from musicremix.conversion.rvc_converter import ExternalRVCConverter


def test_conversion_params_defaults():
    p = ConversionParams()
    # 自然度优先默认值
    assert p.index_rate == 0.75
    assert p.f0_method == "rmvpe"
    assert p.pitch == 0
    assert 0.0 <= p.protect <= 1.0


def test_get_converter_external_is_default():
    c = get_converter(model_name="singerB.pth")
    assert isinstance(c, ExternalRVCConverter)
    assert isinstance(c, VoiceConverter)


def test_get_converter_unknown_raises():
    with pytest.raises(ValueError):
        get_converter("nonexistent", model_name="x.pth")


def test_get_converter_external_requires_model_name():
    """external 后端必填 model_name。"""
    with pytest.raises(ValueError, match="model_name"):
        get_converter("external")


def test_external_converter_missing_rvc_home_raises(tmp_path):
    """未配置 RVC 环境时，convert 应给出清晰错误。"""
    c = ExternalRVCConverter(model_name="singerB.pth", rvc_home=tmp_path / "no_rvc", device="cpu")
    with pytest.raises(FileNotFoundError, match="infer_cli"):
        c.convert("/tmp/none.wav", "/tmp/none.index", tmp_path / "out.wav")


def test_external_converter_resolves_python_bin(tmp_path, monkeypatch):
    """python_bin 可被环境变量覆盖。"""
    monkeypatch.setenv("MUSICREMIX_RVC_PYTHON", "/usr/bin/python3")
    c = ExternalRVCConverter(model_name="singerB.pth", rvc_home=tmp_path, device="cpu")
    assert c.python_bin == "/usr/bin/python3"
