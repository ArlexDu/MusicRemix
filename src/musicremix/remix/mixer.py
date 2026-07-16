"""简单混音器：时长对齐 + 线性混合 + 格式输出。

时长对齐策略：以伴奏为基准，对人声做重采样时间拉伸对齐（消除轻微错位）。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..utils.audio import load_audio, resample, save_audio
from .base import MixParams, Remixer

logger = logging.getLogger(__name__)


class SimpleRemixer(Remixer):
    """基于时长对齐的线性混音。"""

    def mix(
        self,
        vocal_path: str | Path,
        accompaniment_path: str | Path,
        output_path: str | Path,
        params: MixParams | None = None,
    ) -> Path:
        params = params or MixParams()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 加载人声与伴奏，统一到输出采样率
        vocal, v_sr = load_audio(vocal_path, sr=params.output_sr, mono=False)
        acc, a_sr = load_audio(accompaniment_path, sr=params.output_sr, mono=False)

        # 声道对齐：取二者较小声道数，或把单声道扩展
        vocal, acc = _align_channels(vocal, acc)

        # 时长对齐：以伴奏为基准，对人声做时长拉伸对齐
        if vocal.shape[-1] != acc.shape[-1]:
            vocal = _align_duration(vocal, acc.shape[-1], v_sr)

        # 音量调整 + 线性混合
        mixed = (
            vocal * params.vocal_volume + acc * params.accompaniment_volume
        )
        # 防削波
        peak = float(np.max(np.abs(mixed)))
        if peak > 1.0:
            mixed = mixed / peak
            logger.info("检测到削波，已归一化 (峰值 %.2f)", peak)

        save_audio(mixed, params.output_sr, output_path, format=params.output_format)
        logger.info("混音完成: %s (时长 %.2fs)", output_path, mixed.shape[-1] / params.output_sr)
        return output_path


def _align_channels(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对齐两个波形的声道数（取较大者，单声道复制为立体声）。"""
    ca, cb = a.shape[0], b.shape[0]
    target = max(ca, cb)
    if ca < target:
        a = np.repeat(a, target, axis=0)
    if cb < target:
        b = np.repeat(b, target, axis=0)
    return a, b


def _align_duration(wav: np.ndarray, target_samples: int, sr: int) -> np.ndarray:
    """把波形时长对齐到 target_samples：略短则补零，略长则裁剪，差异大则重采样。"""
    n = wav.shape[-1]
    diff = target_samples - n
    # 差异 < 100ms 直接补零/裁剪（消除轻微错位）
    if abs(diff) <= int(0.1 * sr):
        if diff > 0:
            pad = np.zeros((*wav.shape[:-1], diff), dtype=wav.dtype)
            return np.concatenate([wav, pad], axis=-1)
        else:
            return wav[..., :target_samples]

    # 差异较大：按比例重采样做时间拉伸
    ratio = target_samples / n
    if wav.ndim == 1:
        stretched = _time_stretch(wav, ratio)
    else:
        stretched = np.stack([_time_stretch(wav[i], ratio) for i in range(wav.shape[0])], axis=0)
    # 裁齐到精确长度
    if stretched.shape[-1] > target_samples:
        stretched = stretched[..., :target_samples]
    elif stretched.shape[-1] < target_samples:
        pad = np.zeros((*stretched.shape[:-1], target_samples - stretched.shape[-1]), dtype=stretched.dtype)
        stretched = np.concatenate([stretched, pad], axis=-1)
    return stretched


def _time_stretch(wav: np.ndarray, ratio: float) -> np.ndarray:
    """简单线性重采样做时间拉伸（相位不保真但足够对齐）。"""
    n = wav.shape[-1]
    new_n = max(1, int(round(n * ratio)))
    indices = np.linspace(0, n - 1, new_n)
    return np.interp(indices, np.arange(n), wav).astype(wav.dtype)
