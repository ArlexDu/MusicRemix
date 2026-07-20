"""配置加载与本地模型缓存目录约定。"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    """定位项目根目录（config.py 位于 src/musicremix/ 下，往上三级）。"""
    return Path(__file__).resolve().parents[2]


def _default_cache_dir() -> Path:
    """运行时缓存目录，默认项目根目录下的 cache/。

    优先级：环境变量 MUSICREMIX_CACHE_DIR > 项目根/cache。
    历史数据在 ~/.musicremix/ 时，首次启动自动迁移到项目内。
    """
    env = os.environ.get("MUSICREMIX_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    return _project_root() / "cache"


def _legacy_cache_dir() -> Path:
    """历史缓存目录（~/.musicremix/），用于检测并迁移。"""
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
    # 标记是否已执行过旧目录迁移
    _migrated: bool = field(default=False, repr=False)

    def __post_init__(self):
        self._migrate_legacy_cache()

    def _migrate_legacy_cache(self):
        """首次启动时，把 ~/.musicremix/ 的历史数据迁移到项目内 cache/。"""
        if self._migrated:
            return
        self._migrated = True
        # 环境变量指定了自定义路径则不迁移
        if os.environ.get("MUSICREMIX_CACHE_DIR"):
            return
        legacy = _legacy_cache_dir()
        if not legacy.exists() or legacy == self.cache_dir:
            return
        # 仅当目标目录为空或不存在时才迁移（避免覆盖）
        target = Path(self.cache_dir)
        target.mkdir(parents=True, exist_ok=True)
        existing = [p for p in target.iterdir()] if target.exists() else []
        if existing:
            return  # 目标已有数据，不迁移
        try:
            for item in legacy.iterdir():
                dst = target / item.name
                if not dst.exists():
                    shutil.move(str(item), str(dst))
            import logging
            logging.getLogger(__name__).info(
                "已迁移历史缓存 %s -> %s", legacy, target
            )
            # 旧目录空了就删
            try:
                legacy.rmdir()
            except OSError:
                pass
        except Exception:
            import logging
            logging.getLogger(__name__).warning("历史缓存迁移失败（已忽略）", exc_info=True)

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
