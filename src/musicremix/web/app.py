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

from ..config import get_config
from ..conversion.base import ConversionParams
from ..remix.base import MixParams
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


# -- 任务 API ------------------------------------------------------------
@app.post("/api/remix")
async def create_remix(
    source: UploadFile = File(..., description="原唱歌曲（歌手 A）"),
    references: list[UploadFile] = File(..., description="目标歌手参考歌曲（可多首）"),
    model_name: str = Form("base_v2_48k.pth"),
    target_id: str = Form("target"),
    index_rate: float = Form(0.75),
    f0_method: str = Form("rmvpe"),
    pitch: int = Form(0),
    vocal_volume: float = Form(1.0),
    accompaniment_volume: float = Form(1.0),
    output_sr: int = Form(44100),
    device: str = Form("auto"),
):
    """创建换音色任务。上传源歌曲与参考歌曲，返回 task_id。"""
    if not references:
        raise HTTPException(status_code=400, detail="至少需要 1 首参考歌曲")

    task = manager.create()
    wd = task.workdir

    # 保存上传文件
    src_path = wd / f"source_{source.filename}"
    with open(src_path, "wb") as f:
        shutil.copyfileobj(source.file, f)

    ref_paths = []
    for i, ref in enumerate(references):
        rp = wd / f"ref_{i}_{ref.filename}"
        with open(rp, "wb") as f:
            shutil.copyfileobj(ref.file, f)
        ref_paths.append(rp)

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
