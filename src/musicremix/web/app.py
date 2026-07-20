"""MusicRemix Web 后端：FastAPI。

提供文件上传、任务进度查询、结果下载/试听 API。
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import Optional

from ..config import get_config
from ..conversion.base import ConversionParams
from ..remix.base import MixParams
from . import audius
from .tasks import manager

logger = logging.getLogger(__name__)

# 静态资源目录（前端单页）
_STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="MusicRemix", description="歌曲换音色工具 Web 界面", version="0.1.0")

if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# -- 页面 ----------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    p = _STATIC_DIR / "index.html"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return "<h1>MusicRemix Web</h1><p>index.html 未找到</p>"


# -- 工具 API ------------------------------------------------------------
@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/models")
async def list_models():
    """列出可用的 RVC 生成器模型（assets/weights/*.pth）。"""
    cfg = get_config()
    weights_dir = Path(cfg.rvc_home) / "assets" / "weights"
    models = []
    if weights_dir.exists():
        models = sorted(p.name for p in weights_dir.glob("*.pth"))
    return {"models": models, "weights_dir": str(weights_dir)}


@app.get("/api/config")
async def default_config():
    """返回默认参数，供前端预填。"""
    return {
        "index_rate": 0.75,
        "f0_method": "rmvpe",
        "pitch": 0,
        "vocal_volume": 1.0,
        "accompaniment_volume": 1.0,
        "output_sr": 44100,
        "f0_methods": ["rmvpe", "crepe", "pm", "harvest"],
        "default_model": "base_v2_48k.pth",
    }


# -- 在线歌曲（Audius 合法免费源）----------------------------------------
@app.get("/api/search")
async def search_tracks(q: str, limit: int = 12):
    """搜索 Audius 在线歌曲（独立音乐人作品，可自由下载）。"""
    try:
        results = audius.search(q, limit=limit)
        return {"query": q, "count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"搜索失败: {e}")


@app.get("/api/audius/download/{track_id}")
async def download_track(track_id: str):
    """下载 Audius 歌曲到本地缓存，返回 file_id 供 remix 使用。"""
    try:
        path = audius.download(track_id)
        info = audius.track_info(track_id)
        return {
            "file_id": track_id,
            "path": str(path),
            "filename": f"{info['artist']} - {info['title']}.mp3",
            "title": info["title"],
            "artist": info["artist"],
            "duration": info["duration"],
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"下载失败: {e}")


@app.api_route("/api/audius/stream/{track_id}", methods=["GET", "HEAD"])
async def stream_track(track_id: str):
    """播放 Audius 歌曲（试听，GET 返回音频，HEAD 用于预检查）。"""
    try:
        path = audius.download(track_id)  # 幂等：已缓存则直接返回
        return FileResponse(str(path), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"播放失败: {e}")


# -- 任务 API ------------------------------------------------------------
@app.post("/api/remix")
async def create_remix(
    model_name: str = Form("base_v2_48k.pth"),
    target_id: str = Form("target"),
    index_rate: float = Form(0.75),
    f0_method: str = Form("rmvpe"),
    pitch: int = Form(0),
    vocal_volume: float = Form(1.0),
    accompaniment_volume: float = Form(1.0),
    output_sr: int = Form(44100),
    device: str = Form("auto"),
    source: Optional[UploadFile] = File(None, description="原唱歌曲（上传，与 source_file_id 二选一）"),
    references: list[UploadFile] = File(default=[], description="参考歌曲（上传）"),
    source_file_id: Optional[str] = Form(None, description="原唱歌曲 Audius track_id（在线选歌）"),
    reference_file_ids: str = Form("", description="参考歌曲 Audius track_id 列表，逗号分隔"),
):
    """创建换音色任务。支持本地上传或 Audius 在线选歌（file_id）。"""
    try:
        task = manager.create()
        wd = task.workdir

        # 源歌曲：在线选歌优先，否则用上传文件
        if source_file_id:
            src_path = audius.download(source_file_id)
        elif source is not None:
            src_path = wd / f"source_{source.filename}"
            with open(src_path, "wb") as f:
                shutil.copyfileobj(source.file, f)
        else:
            raise HTTPException(status_code=400, detail="需要提供 source（上传）或 source_file_id（在线）")

        # 参考歌曲：上传 + 在线
        ref_paths: list[Path] = []
        for i, ref in enumerate(references):
            rp = wd / f"ref_{i}_{ref.filename}"
            with open(rp, "wb") as f:
                shutil.copyfileobj(ref.file, f)
            ref_paths.append(rp)
        for tid in [s.strip() for s in reference_file_ids.split(",") if s.strip()]:
            ref_paths.append(audius.download(tid))
        if not ref_paths:
            raise HTTPException(status_code=400, detail="至少需要 1 首参考歌曲")

        params = ConversionParams(
            index_rate=index_rate, f0_method=f0_method, pitch=pitch,
        )
        mix_params = MixParams(
            vocal_volume=vocal_volume, accompaniment_volume=accompaniment_volume,
            output_sr=output_sr,
        )

        manager.start(
            task, src_path, ref_paths, params, model_name, mix_params, target_id, device,
        )
        return {"task_id": task.id, "status": task.status}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("提交换音色任务失败")
        raise HTTPException(status_code=500, detail=f"提交失败: {type(e).__name__}: {e}")


@app.get("/api/tasks/{task_id}")
async def task_status(task_id: str):
    task = manager.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task.to_dict()


@app.get("/api/tasks/{task_id}/download")
async def task_download(task_id: str):
    """下载最终换音色结果。"""
    task = manager.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    final = task.outputs.get("final")
    if not final or not Path(final).exists():
        raise HTTPException(status_code=404, detail="结果尚未生成")
    return FileResponse(final, media_type="audio/wav", filename="remixed.wav")


@app.get("/api/tasks/{task_id}/file/{name}")
async def task_file(task_id: str, name: str):
    """下载/试听中间产物：vocals / accompaniment / converted / final。"""
    task = manager.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    if name not in ("vocals", "accompaniment", "converted", "final"):
        raise HTTPException(status_code=400, detail="无效的文件名")
    p = task.outputs.get(name)
    if not p or not Path(p).exists():
        raise HTTPException(status_code=404, detail=f"{name} 尚未生成")
    return FileResponse(p, media_type="audio/wav", filename=f"{name}.wav")


@app.get("/api/tasks")
async def list_tasks():
    return {"tasks": [t.to_dict() for t in manager.all()]}
