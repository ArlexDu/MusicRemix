"""Audius 在线音乐源（合法免费，独立音乐人作品）。

Audius 是去中心化音乐平台，API 公开、无需 key，支持搜索与流媒体下载。
下载的音乐为独立音乐人发布的作品（非商业平台流行歌曲），可用于换音色测试。

注意：仅限 Audius 平台上可自由下载的作品，不得用于绕过商业平台版权保护。
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

from ..config import get_config

logger = logging.getLogger(__name__)

AUDIUS_HOST = "https://api.audius.co"
APP_NAME = "MusicRemix"


def _get(path: str, params: dict, timeout: int = 15) -> dict:
    qs = urllib.parse.urlencode({"app_name": APP_NAME, **params})
    url = f"{AUDIUS_HOST}{path}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": f"{APP_NAME}/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def search(query: str, limit: int = 12) -> list[dict]:
    """搜索 Audius 歌曲，返回标准化结果列表。"""
    if not query.strip():
        return []
    data = _get("/v1/tracks/search", {"query": query, "limit": limit})
    results = []
    for t in data.get("data", []):
        results.append({
            "id": t["id"],
            "title": t.get("title", ""),
            "artist": t.get("user", {}).get("name", ""),
            "artwork": t.get("artwork", {}).get("150x150", "") or t.get("artwork", {}).get("480x480", ""),
            "duration": t.get("duration", 0),
            "genre": t.get("genre", ""),
            "play_count": t.get("play_count", 0),
        })
    return results


def download(track_id: str) -> Path:
    """下载歌曲到本地缓存（幂等，已下载则直接返回路径）。"""
    cfg = get_config()
    cache_dir = cfg.cache_dir / "audius_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{track_id}.mp3"

    if dest.exists() and dest.stat().st_size > 0:
        logger.info("Audius 缓存命中: %s", dest)
        return dest

    url = f"{AUDIUS_HOST}/v1/tracks/{track_id}/stream?app_name={APP_NAME}"
    logger.info("下载 Audius 歌曲: %s", track_id)
    req = urllib.request.Request(url, headers={"User-Agent": f"{APP_NAME}/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1 << 20)  # 1MB
            if not chunk:
                break
            f.write(chunk)
    logger.info("下载完成: %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


def track_info(track_id: str) -> dict:
    """获取单首歌曲信息。"""
    data = _get(f"/v1/tracks/{track_id}", {})
    t = data.get("data", {})
    return {
        "id": t.get("id", track_id),
        "title": t.get("title", ""),
        "artist": t.get("user", {}).get("name", ""),
        "duration": t.get("duration", 0),
    }
