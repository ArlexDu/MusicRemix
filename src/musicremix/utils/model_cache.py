"""模型懒加载与本地缓存管理。"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path

from ..config import Config, get_config

logger = logging.getLogger(__name__)


class ModelCache:
    """预训练模型本地缓存：首次按需下载，后续复用。"""

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self.models_dir = self.config.models_dir

    def resolve(self, name: str, url: str | None = None) -> Path:
        """获取模型本地路径；若不存在且提供 url 则下载。"""
        local = self.models_dir / name
        if local.exists():
            logger.debug("模型缓存命中: %s", local)
            return local
        if url is None:
            raise FileNotFoundError(
                f"模型 {name} 不在缓存 {self.models_dir} 中，且未提供下载 URL。"
                f"请手动放置或配置来源。"
            )
        self._download(url, local)
        return local

    def _download(self, url: str, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        logger.info("下载模型: %s -> %s", url, dst)
        with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:  # noqa: S310
            while True:
                chunk = resp.read(1 << 20)  # 1MB
                if not chunk:
                    break
                f.write(chunk)
        tmp.replace(dst)
        logger.info("下载完成: %s", dst)

    def list_cached(self) -> list[Path]:
        return sorted(self.models_dir.glob("*"))

    @staticmethod
    def url_hash(url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()[:12]  # noqa: S324
