"""测试共用夹具：生成合成音频。"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def sine_audio() -> tuple[np.ndarray, int]:
    """44100Hz, 2 秒, 440Hz 正弦波, 单声道 (1, samples)。"""
    sr = 44100
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    y = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return y[np.newaxis, :], sr


@pytest.fixture
def sine_audio_stereo(sine_audio) -> tuple[np.ndarray, int]:
    """立体声版本 (2, samples)。"""
    y, sr = sine_audio
    return np.repeat(y, 2, axis=0), sr


@pytest.fixture
def wav_file(tmp_path, sine_audio_stereo):
    """临时 wav 文件。"""
    from musicremix.utils.audio import save_audio

    y, sr = sine_audio_stereo
    p = tmp_path / "test.wav"
    save_audio(y, sr, p)
    return p, sr
