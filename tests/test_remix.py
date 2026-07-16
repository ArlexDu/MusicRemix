"""混音合成单元测试 (6.5)。纯 numpy，可完整运行。"""

from __future__ import annotations

import numpy as np

from musicremix.remix import get_remixer
from musicremix.remix.base import MixParams, Remixer
from musicremix.utils.audio import load_audio, save_audio


def _write_sine(path, sr, dur, freq, channels=2):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    y = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if channels == 2:
        y = np.stack([y, y], axis=0)
    else:
        y = y[np.newaxis, :]
    save_audio(y, sr, path)
    return path


def test_remixer_is_remixer():
    r = get_remixer("default")
    assert isinstance(r, Remixer)


def test_mix_produces_output(tmp_path):
    sr = 44100
    v = _write_sine(tmp_path / "vocal.wav", sr, 2.0, 440)
    a = _write_sine(tmp_path / "acc.wav", sr, 2.0, 220)
    out = tmp_path / "out.wav"
    get_remixer("default").mix(v, a, out)
    assert out.exists()
    y, loaded_sr = load_audio(out, sr=None, mono=False)
    assert loaded_sr == sr
    # 时长约 2 秒
    assert 1.8 <= y.shape[-1] / sr <= 2.2


def test_mix_volume_params(tmp_path):
    sr = 44100
    v = _write_sine(tmp_path / "vocal.wav", sr, 1.0, 440)
    a = _write_sine(tmp_path / "acc.wav", sr, 1.0, 220)
    out_loud = tmp_path / "loud.wav"
    out_quiet = tmp_path / "quiet.wav"
    get_remixer("default").mix(v, a, out_loud, MixParams(vocal_volume=1.0, accompaniment_volume=1.0))
    get_remixer("default").mix(v, a, out_quiet, MixParams(vocal_volume=0.1, accompaniment_volume=0.1))
    y_loud, _ = load_audio(out_loud, sr=None, mono=False)
    y_quiet, _ = load_audio(out_quiet, sr=None, mono=False)
    assert float(np.max(np.abs(y_loud))) > float(np.max(np.abs(y_quiet)))


def test_mix_aligns_mismatched_duration(tmp_path):
    """人声比伴奏短 50ms，应被补零对齐而非报错。"""
    sr = 44100
    v = _write_sine(tmp_path / "vocal.wav", sr, 2.0, 440)
    a = _write_sine(tmp_path / "acc.wav", sr, 2.05, 220)
    out = tmp_path / "out.wav"
    get_remixer("default").mix(v, a, out)
    y, _ = load_audio(out, sr=None, mono=False)
    # 输出时长应跟随伴奏（约 2.05s）
    assert y.shape[-1] / sr >= 2.0


def test_no_clipping_on_hot_signal(tmp_path):
    sr = 44100
    # 极大音量，混合后应被归一化防削波
    v = _write_sine(tmp_path / "vocal.wav", sr, 1.0, 440)
    a = _write_sine(tmp_path / "acc.wav", sr, 1.0, 220)
    out = tmp_path / "out.wav"
    get_remixer("default").mix(v, a, out, MixParams(vocal_volume=5.0, accompaniment_volume=5.0))
    y, _ = load_audio(out, sr=None, mono=False)
    assert float(np.max(np.abs(y))) <= 1.0 + 1e-6
