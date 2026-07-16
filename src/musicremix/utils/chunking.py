"""分片处理工具：长音频切片 + 无缝拼接，控制内存/显存峰值。"""

from __future__ import annotations

from typing import Callable, List

import numpy as np


def chunk_audio(
    waveform: np.ndarray, sr: int, chunk_seconds: float, overlap: float
) -> List[tuple[np.ndarray, int]]:
    """把波形切成带重叠的分片。

    返回 [(chunk_waveform, start_sample), ...]
    chunk_waveform 形状与输入一致（(channels, samples) 或 (samples,)）。
    """
    if waveform.ndim == 1:
        n = waveform.shape[0]
    else:
        n = waveform.shape[1]

    chunk_size = int(chunk_seconds * sr)
    hop = chunk_size - int(overlap * sr)
    if hop <= 0:
        hop = chunk_size

    if n <= chunk_size:
        return [(waveform, 0)]

    chunks = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if waveform.ndim == 1:
            chunk = waveform[start:end]
        else:
            chunk = waveform[:, start:end]
        chunks.append((chunk, start))
        if end >= n:
            break
        start += hop
    return chunks


def _fade(length: int) -> np.ndarray:
    """生成线性淡入/淡出窗。"""
    if length <= 0:
        return np.ones(1)
    return np.linspace(0, 1, length)


def merge_chunks(
    chunks: List[tuple[np.ndarray, int]],
    total_samples: int,
    overlap: float,
    sr: int,
) -> np.ndarray:
    """把分片按起始位置 + 交叉淡入淡出合并回完整波形。"""
    if not chunks:
        raise ValueError("无分片可合并")
    ndim = chunks[0][0].ndim
    if ndim == 1:
        out = np.zeros(total_samples, dtype=np.float32)
        weight = np.zeros(total_samples, dtype=np.float32)
    else:
        channels = chunks[0][0].shape[0]
        out = np.zeros((channels, total_samples), dtype=np.float32)
        weight = np.zeros((channels, total_samples), dtype=np.float32)

    fade_len = int(overlap * sr)
    for chunk, start in chunks:
        if ndim == 1:
            n = chunk.shape[0]
        else:
            n = chunk.shape[1]
        end = start + n

        w = np.ones(n, dtype=np.float32)
        # 起始淡入（除非是首个分片且 start==0）
        if start > 0 and fade_len > 0:
            fl = min(fade_len, n)
            w[:fl] *= _fade(fl)
        # 结尾淡出（除非到达末尾）
        if end < total_samples and fade_len > 0:
            fl = min(fade_len, n)
            w[-fl:] *= _fade(fl)[::-1]

        if ndim == 1:
            out[start:end] += chunk * w
            weight[start:end] += w
        else:
            out[:, start:end] += chunk * w[np.newaxis, :]
            weight[:, start:end] += w[np.newaxis, :]

    # 归一化（避免重叠区叠加放大）
    weight = np.where(weight < 1e-8, 1.0, weight)
    return out / weight


def process_in_chunks(
    waveform: np.ndarray,
    sr: int,
    process_fn: Callable[[np.ndarray], np.ndarray],
    chunk_seconds: float = 30.0,
    overlap: float = 1.0,
) -> np.ndarray:
    """对长音频分片应用处理函数后无缝拼接。

    process_fn 接收 (channels, samples) 或 (samples,)，返回同形状。
    """
    is_mono_1d = waveform.ndim == 1
    chunks = chunk_audio(waveform, sr, chunk_seconds, overlap)
    total = waveform.shape[0] if is_mono_1d else waveform.shape[1]

    processed = []
    for chunk, start in chunks:
        out = process_fn(chunk)
        # 处理后长度可能变化，以输出为准记录
        processed.append((out, start))
    # 重算 total 为输出最大末端
    out_total = max(
        (start + (c.shape[0] if c.ndim == 1 else c.shape[1])) for c, start in processed
    )
    return merge_chunks(processed, out_total, overlap, sr)
