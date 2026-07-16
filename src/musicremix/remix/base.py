"""Remixer 抽象接口：输入转换人声 + 原曲伴奏，输出合成歌曲。"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MixParams:
    """混音参数。"""

    vocal_volume: float = 1.0
    accompaniment_volume: float = 1.0
    output_sr: int = 44100
    output_format: str = "wav"


class Remixer(abc.ABC):
    """混音合成抽象基类。"""

    @abc.abstractmethod
    def mix(
        self,
        vocal_path: str | Path,
        accompaniment_path: str | Path,
        output_path: str | Path,
        params: MixParams | None = None,
    ) -> Path:
        """将人声与伴奏合成为完整歌曲，并做时长对齐。"""
        ...


def get_remixer(name: str = "default", **kwargs) -> Remixer:
    """按名称获取混音后端。"""
    if name in ("default", "simple"):
        from .mixer import SimpleRemixer

        return SimpleRemixer(**kwargs)
    raise ValueError(f"未知混音后端: {name}（支持: default/simple）")
