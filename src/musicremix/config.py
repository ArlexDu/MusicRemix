"""配置加载与本地模型缓存目录约定。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_cache_dir() -> Path:
    """本地模型缓存目录，默认 ~/.musicremix/。"""
    env = os.environ.get("MUSICREMIX_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".musicremix"


@dataclass
class Config:
    """全局运行配置。"""

    # 模型缓存根目录
    cache_dir: Path = field(default_factory=_default_cache_dir)
    # 工作目录（中间产物落盘）
    workdir: Path = field(default_factory=lambda: Path("./workdir"))
    # 默认输出采样率
    output_sr: int = 44100
    # 默认输出格式
    output_format: str = "wav"
    # 强制设备：auto / cuda / mps / cpu
    device: str = "auto"
    # 外部 RVC 推理环境根目录，默认指向工程内 models/rvc；由环境变量 MUSICREMIX_RVC_HOME 覆盖
    rvc_home: str = "models/rvc"
    # RVC 默认参数（自然度优先）
    index_rate: float = 0.75
    f0_method: str = "rmvpe"
    pitch: int = 0
    # 混音默认音量
    vocal_volume: float = 1.0
    accompaniment_volume: float = 1.0
    # 分片处理时长（秒），控制显存峰值
    chunk_seconds: float = 30.0
    # 分片重叠（秒），用于交叉淡入淡出
    chunk_overlap: float = 1.0

    @property
    def models_dir(self) -> Path:
        d = self.cache_dir / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def indexes_dir(self) -> Path:
        """目标歌手音色索引缓存目录。"""
        d = self.cache_dir / "indexes"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def workdir_resolved(self) -> Path:
        p = Path(self.workdir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def index_path(self, target_id: str) -> Path:
        """目标歌手音色索引文件路径。"""
        return self.indexes_dir / f"{target_id}.index"

    def ensure_workdir(self) -> Path:
        return self.workdir_resolved


_DEFAULT = Config()


def get_config() -> Config:
    """获取默认全局配置实例。"""
    return _DEFAULT


def set_config(cfg: Config) -> None:
    """替换全局配置（测试或 CLI 注入用）。"""
    global _DEFAULT
    _DEFAULT = cfg
