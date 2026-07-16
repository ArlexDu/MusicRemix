"""VoiceModeler 抽象接口：输入参考人声素材，输出可持久化音色索引。"""

from __future__ import annotations

import abc
from pathlib import Path


class VoiceModeler(abc.ABC):
    """目标音色建模抽象基类。"""

    @abc.abstractmethod
    def build_index(
        self,
        reference_audio_paths: list[str | Path],
        target_id: str,
        workdir: str | Path | None = None,
    ) -> Path:
        """从参考歌曲构建目标音色特征索引。

        Args:
            reference_audio_paths: 目标歌手的参考歌曲路径列表
            target_id: 目标歌手标识（用于缓存）
            workdir: 中间产物目录

        Returns:
            生成的索引文件路径
        """
        ...

    @abc.abstractmethod
    def get_index(self, target_id: str) -> Path | None:
        """获取已缓存的索引路径，不存在则返回 None。"""
        ...


def get_modeler(name: str = "rvc", **kwargs) -> VoiceModeler:
    """按名称获取音色建模后端。"""
    if name == "rvc":
        from .rvc_modeler import RVCModeler

        return RVCModeler(**kwargs)
    raise ValueError(f"未知音色建模后端: {name}（支持: rvc）")
