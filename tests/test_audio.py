"""音频 I/O 工具测试 (2.2)。"""

import numpy as np

from musicremix.utils.audio import get_duration, load_audio, resample, save_audio


def test_save_and_load_roundtrip(tmp_path, sine_audio_stereo):
    y, sr = sine_audio_stereo
    p = tmp_path / "out.wav"
    save_audio(y, sr, p)
    loaded, loaded_sr = load_audio(p, sr=None, mono=False)
    assert loaded_sr == sr
    assert loaded.shape[0] == 2
    # 时长一致
    assert abs(loaded.shape[1] - y.shape[1]) <= 1


def test_resample_changes_length(sine_audio):
    y, sr = sine_audio
    out = resample(y, sr, sr // 2)
    # 采样率减半 -> 样本数约减半
    assert out.shape[-1] < y.shape[-1]


def test_resample_identity(sine_audio):
    y, sr = sine_audio
    out = resample(y, sr, sr)
    assert np.allclose(out, y)


def test_unsupported_format_raises(tmp_path):
    p = tmp_path / "bad.xyz"
    p.write_bytes(b"")
    try:
        load_audio(p)
        assert False, "应抛出 ValueError"
    except ValueError as e:
        assert "不支持" in str(e)


def test_get_duration(wav_file):
    p, sr = wav_file
    dur = get_duration(p)
    assert 1.9 <= dur <= 2.1
