"""VoiceConverter 抽象接口：输入源人声 + 目标索引，输出转换人声。"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConversionParams:
    """音色迁移可配置参数（自然度优先默认值）。"""

    index_rate: float = 0.75  # 检索注入强度 [0,1]
    f0_method: str = "rmvpe"  # 音高算法: rmvpe / crepe / pm / harvest
    pitch: int = 0  # 半音偏移（男歌女唱可 +12）
    # 分片处理
    chunk_seconds: float = 30.0
    chunk_overlap: float = 1.0
    # 检索保护：清辅音等不注入
    protect: float = 0.33


class VoiceConverter(abc.ABC):
    """音色迁移抽象基类。"""

    @abc.abstractmethod
    def convert(
        self,
        source_vocal_path: str | Path,
        index_path: str | Path,
        output_path: str | Path,
        params: ConversionParams | None = None,
    ) -> Path:
        """将源人声音色转换为目标音色。

        Args:
            source_vocal_path: 源人声音频路径
            index_path: 目标音色特征索引路径
            output_path: 输出路径
            params: 转换参数
        """
        ...


def get_converter(name: str = "external", model_name: str | None = None, **kwargs) -> VoiceConverter:
    """按名称获取音色迁移后端。

    - "external"（默认）：调用外部 RVC 推理环境，依赖解耦，推荐
    - "rvc"：内嵌 rvc-python（已废弃，仅保留接口）

    Args:
        model_name: 目标歌手的 RVC .pth 模型名（external 后端必填）
    """
    if name in ("external", "default"):
        from .rvc_converter import ExternalRVCConverter

        if not model_name:
            raise ValueError("external 后端需要 model_name（目标歌手的 .pth 模型名）")
        return ExternalRVCConverter(model_name=model_name, **kwargs)
    if name == "rvc":
        from .rvc_converter import RVCConverter

        return RVCConverter(**kwargs)
    raise ValueError(f"未知音色迁移后端: {name}（支持: external/rvc）")
