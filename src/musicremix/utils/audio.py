"""音频 I/O 工具：wav/mp3/flac 读取、写入、重采样、格式转换。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 支持的输入格式
SUPPORTED_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def load_audio(path: str | Path, sr: int | None = None, mono: bool = True) -> Tuple[np.ndarray, int]:
    """加载音频文件，返回 (waveform[channels, samples], sample_rate)。

    - 支持 wav/mp3/flac 等常见格式
    - sr 指定时重采样到目标采样率
    - mono=True 时转为单声道
    """
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_EXTS:
        raise ValueError(f"不支持的音频格式: {path.suffix}（支持 {sorted(SUPPORTED_EXTS)}）")

    try:
        import librosa
    except ImportError as e:
        raise ImportError("需要 librosa 读取音频，请安装: pip install librosa") from e

    y, orig_sr = librosa.load(str(path), sr=sr, mono=mono)
    # librosa 返回 (samples,) 或 (channels, samples)
    if y.ndim == 1:
        y = y[np.newaxis, :]
    return y.astype(np.float32), orig_sr if sr is None else sr


def save_audio(
    waveform: np.ndarray, sr: int, path: str | Path, format: str | None = None
) -> Path:
    """保存音频文件。

    waveform: (channels, samples) 或 (samples,)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import soundfile as sf
    except ImportError as e:
        raise ImportError("需要 soundfile 写入音频，请安装: pip install soundfile") from e

    if waveform.ndim == 1:
        data = waveform
    else:
        # (channels, samples) -> (samples, channels)
        data = waveform.T

    subtype = "PCM_16" if (format or path.suffix).lower() in ("mp3",) else None
    sf.write(str(path), data, sr, format=format, subtype=subtype)
    return path


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """重采样到目标采样率。"""
    if orig_sr == target_sr:
        return waveform
    try:
        import librosa
    except ImportError as e:
        raise ImportError("需要 librosa 重采样，请安装: pip install librosa") from e

    if waveform.ndim == 1:
        return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
    # 多声道逐通道重采样
    return np.stack(
        [librosa.resample(waveform[i], orig_sr=orig_sr, target_sr=target_sr) for i in range(waveform.shape[0])],
        axis=0,
    )


def get_duration(path: str | Path) -> float:
    """获取音频时长（秒）。"""
    try:
        import librosa
    except ImportError as e:
        raise ImportError("需要 librosa 获取时长") from e
    return float(librosa.get_duration(path=str(path)))


def convert_format(
    src: str | Path, dst: str | Path, sr: int | None = None
) -> Path:
    """格式转换（如 wav->mp3）。"""
    y, orig_sr = load_audio(src, sr=sr)
    save_audio(y, orig_sr if sr is None else sr, dst)
    return Path(dst)
