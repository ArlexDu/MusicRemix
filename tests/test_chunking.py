"""分片处理工具测试 (2.4)。"""

import numpy as np

from musicremix.utils.chunking import chunk_audio, merge_chunks, process_in_chunks


def test_short_audio_single_chunk(sine_audio):
    y, sr = sine_audio  # 2 秒
    chunks = chunk_audio(y, sr, chunk_seconds=30.0, overlap=1.0)
    assert len(chunks) == 1
    assert chunks[0][1] == 0


def test_long_audio_multiple_chunks():
    sr = 44100
    t = np.linspace(0, 120.0, int(sr * 120.0), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)[np.newaxis, :]
    chunks = chunk_audio(y, sr, chunk_seconds=30.0, overlap=1.0)
    assert len(chunks) > 1
    # 第一个分片起始为 0
    assert chunks[0][1] == 0


def test_merge_identity_passthrough(sine_audio):
    """处理函数为恒等时，合并结果应近似原音频。"""
    y, sr = sine_audio
    out = process_in_chunks(y, sr, lambda c: c, chunk_seconds=0.5, overlap=0.1)
    # 长度一致，幅值近似（边界淡入淡出会有微小差异）
    assert abs(out.shape[-1] - y.shape[-1]) <= 1
    # 中间区域应几乎相等
    mid = y.shape[-1] // 2
    assert np.allclose(out[0, mid - 100: mid + 100], y[0, mid - 100: mid + 100], atol=1e-3)


def test_chunk_merge_roundtrip_length():
    sr = 44100
    n = sr * 60
    y = np.zeros((2, n), dtype=np.float32)
    chunks = chunk_audio(y, sr, chunk_seconds=30.0, overlap=1.0)
    merged = merge_chunks([(c, s) for c, s in chunks], n, 1.0, sr)
    assert merged.shape == y.shape
