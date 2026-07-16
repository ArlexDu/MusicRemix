"""人声分离单元测试 (3.5)。

依赖 demucs/torch 的测试在缺依赖时自动跳过。
"""

from __future__ import annotations

import pytest

from musicremix.separation import get_separator
from musicremix.separation.base import SourceSeparator

demucs = pytest.importorskip("demucs", reason="demucs 未安装，跳过分离测试")
torch = pytest.importorskip("torch", reason="torch 未安装，跳过分离测试")

# Demucs 分离需加载模型 + CPU 推理，标记为 slow 集成测试，默认不跑
pytestmark = pytest.mark.slow


@pytest.fixture
def separator():
    return get_separator("demucs", device="cpu")


def test_separator_is_source_separator(separator):
    assert isinstance(separator, SourceSeparator)


def test_separate_returns_two_tracks(separator, wav_file):
    p, sr = wav_file
    result = separator.separate(p, output_sr=sr, vocals_only=False)
    assert result.vocals is not None
    assert result.accompaniment is not None
    assert result.sample_rate == sr
    # 时长与原曲一致
    assert abs(result.duration_seconds() - 2.0) < 0.2


def test_vocals_only_mode(separator, wav_file):
    p, sr = wav_file
    result = separator.separate(p, output_sr=sr, vocals_only=True)
    assert result.vocals is not None
    assert result.accompaniment.shape[-1] == 0 or result.accompaniment.sum() == 0
