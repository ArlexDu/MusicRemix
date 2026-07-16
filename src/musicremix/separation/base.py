"""SourceSeparator 抽象接口：输入音频路径，输出人声/伴奏轨。"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SeparationResult:
    """分离结果。"""

    vocals: np.ndarray  # (channels, samples) 人声轨
    accompaniment: np.ndarray  # (channels, samples) 伴奏轨
    sample_rate: int

    def duration_seconds(self) -> float:
        n = self.vocals.shape[-1]
        return n / self.sample_rate


class SourceSeparator(abc.ABC):
    """音频源分离抽象基类。"""

    @abc.abstractmethod
    def separate(
        self,
        audio_path: str | Path,
        output_sr: int = 44100,
        vocals_only: bool = False,
    ) -> SeparationResult:
        """分离人声与伴奏。

        Args:
            audio_path: 输入音频文件路径
            output_sr: 输出采样率
            vocals_only: 仅提取人声（伴奏可为空）
        """
        ...


def get_separator(name: str = "demucs", **kwargs) -> SourceSeparator:
    """按名称获取分离后端实例。"""
    if name == "demucs":
        from .demucs import DemucsSeparator

        return DemucsSeparator(**kwargs)
    raise ValueError(f"未知分离后端: {name}（支持: demucs）")
