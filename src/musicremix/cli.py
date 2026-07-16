"""MusicRemix 命令行入口。

命令：
  musicremix remix      一键换音色（端到端）
  musicremix separate   仅分离人声/伴奏
  musicremix model      仅构建目标音色索引
  musicremix convert    仅音色迁移
  musicremix mix        仅混音合成
  musicremix info       显示设备与缓存状态
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import Config, get_config, set_config
from .conversion.base import ConversionParams
from .pipeline import RemixPipeline, detect_resumable_stages
from .remix.base import MixParams

app = typer.Typer(
    name="musicremix",
    help="歌曲换音色工具：保留旋律与伴奏，替换演唱者音色",
    no_args_is_help=True,
)
console = Console()
logger = logging.getLogger("musicremix")


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _apply_common_opts(
    workdir: Optional[str],
    output_sr: int,
    device: str,
):
    cfg = get_config()
    if workdir:
        cfg.workdir = Path(workdir)
    cfg.output_sr = output_sr
    cfg.device = device
    set_config(cfg)
    return cfg


@app.command()
def remix(
    input: str = typer.Option(..., "--input", "-i", help="原唱歌曲路径（歌手 A）"),
    reference: list[str] = typer.Option(
        ..., "--reference", "-r", help="目标歌手参考歌曲（可多次指定）"
    ),
    model_name: str = typer.Option(
        ..., "--model-name", "-m",
        help="目标歌手 RVC 模型名（.pth，放于 models/rvc/assets/weight_root/）",
    ),
    output: str = typer.Option("output/remixed.wav", "--output", "-o", help="输出路径"),
    target_id: str = typer.Option("target", "--target-id", "-t", help="目标歌手标识（索引缓存键）"),
    workdir: Optional[str] = typer.Option(None, "--workdir", "-w", help="中间产物目录"),
    output_sr: int = typer.Option(44100, "--output-sr", help="输出采样率"),
    device: str = typer.Option("auto", "--device", help="auto/cuda/mps/cpu"),
    index_rate: float = typer.Option(0.75, "--index-rate", help="RVC 检索注入强度 [0,1]"),
    f0_method: str = typer.Option("rmvpe", "--f0-method", help="音高算法 rmvpe/crepe/pm/harvest"),
    pitch: int = typer.Option(0, "--pitch", help="半音偏移（男歌女唱可 +12）"),
    vocal_volume: float = typer.Option(1.0, "--vocal-volume", help="人声音量"),
    accompaniment_volume: float = typer.Option(1.0, "--accompaniment-volume", help="伴奏音量"),
    resume: bool = typer.Option(False, "--resume", help="自动跳过已有中间产物的阶段"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="详细日志"),
):
    """一键换音色：输入歌曲 + 参考歌曲 → 输出换音色歌曲。"""
    _setup_logging(verbose)
    cfg = _apply_common_opts(workdir, output_sr, device)

    skip = detect_resumable_stages(cfg.workdir_resolved) if resume else set()
    if skip:
        console.print(f"[yellow]续跑模式：跳过阶段 {skip}[/]")

    pipeline = RemixPipeline(device=device, model_name=model_name)
    result = pipeline.run(
        input_song=input,
        reference_songs=reference,
        output_path=output,
        target_id=target_id,
        workdir=cfg.workdir_resolved,
        conversion=ConversionParams(index_rate=index_rate, f0_method=f0_method, pitch=pitch),
        mix=MixParams(
            vocal_volume=vocal_volume,
            accompaniment_volume=accompaniment_volume,
            output_sr=output_sr,
        ),
        skip_stages=skip,
    )
    console.print(f"[green bold]完成！[/] 输出: {result.output_path}")


@app.command()
def separate(
    input: str = typer.Option(..., "--input", "-i"),
    workdir: str = typer.Option("./workdir", "--workdir", "-w"),
    output_sr: int = typer.Option(44100, "--output-sr"),
    device: str = typer.Option("auto", "--device"),
    vocals_only: bool = typer.Option(False, "--vocals-only"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """仅执行人声分离。"""
    _setup_logging(verbose)
    _apply_common_opts(workdir, output_sr, device)
    from .separation import get_separator

    paths = get_separator("demucs", device=device).separate_to_files(
        input, workdir, output_sr=output_sr, vocals_only=vocals_only, stem="",
    )
    for k, v in paths.items():
        console.print(f"{k}: {v}")


@app.command()
def model(
    reference: list[str] = typer.Option(..., "--reference", "-r"),
    target_id: str = typer.Option(..., "--target-id", "-t"),
    workdir: str = typer.Option("./workdir", "--workdir", "-w"),
    device: str = typer.Option("auto", "--device"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """仅构建目标音色索引。"""
    _setup_logging(verbose)
    _apply_common_opts(workdir, 44100, device)
    from .modeling import get_modeler

    idx = get_modeler("rvc", device=device).build_index(reference, target_id, workdir)
    console.print(f"[green]索引已构建:[/] {idx}")


@app.command()
def convert(
    vocal: str = typer.Option(..., "--vocal", help="源人声路径"),
    model_name: str = typer.Option(..., "--model-name", "-m", help="目标歌手 RVC 模型名(.pth)"),
    index: str = typer.Option("", "--index", help="目标音色索引路径（可选）"),
    output: str = typer.Option("workdir/converted.wav", "--output", "-o"),
    device: str = typer.Option("auto", "--device"),
    index_rate: float = typer.Option(0.75, "--index-rate"),
    f0_method: str = typer.Option("rmvpe", "--f0-method"),
    pitch: int = typer.Option(0, "--pitch"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """仅执行音色迁移。"""
    _setup_logging(verbose)
    from .conversion import get_converter

    out = get_converter("external", model_name=model_name, device=device).convert(
        vocal, index, output,
        ConversionParams(index_rate=index_rate, f0_method=f0_method, pitch=pitch),
    )
    console.print(f"[green]转换完成:[/] {out}")


@app.command()
def mix(
    vocal: str = typer.Option(..., "--vocal"),
    accompaniment: str = typer.Option(..., "--accompaniment"),
    output: str = typer.Option("output/mixed.wav", "--output", "-o"),
    output_sr: int = typer.Option(44100, "--output-sr"),
    vocal_volume: float = typer.Option(1.0, "--vocal-volume"),
    accompaniment_volume: float = typer.Option(1.0, "--accompaniment-volume"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """仅执行混音合成。"""
    _setup_logging(verbose)
    from .remix import get_remixer

    get_remixer("default").mix(
        vocal, accompaniment, output,
        MixParams(vocal_volume=vocal_volume, accompaniment_volume=accompaniment_volume,
                  output_sr=output_sr),
    )
    console.print(f"[green]混音完成:[/] {output}")


@app.command()
def info(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """显示设备与缓存状态。"""
    _setup_logging(verbose)

    from .utils.device import detect_device

    dev = detect_device("auto")

    table = Table(title=f"MusicRemix v{__version__} 状态")
    table.add_column("项目", style="cyan")
    table.add_column("值")
    table.add_row("计算设备", dev.value)
    table.add_row("模型缓存目录", str(get_config().cache_dir))
    table.add_row("索引缓存目录", str(get_config().indexes_dir))
    table.add_row("工作目录", str(get_config().workdir_resolved))

    # 已缓存模型
    models = get_config().models_dir.glob("*") if get_config().models_dir.exists() else []
    table.add_row("已缓存模型", ", ".join(m.name for m in models) or "（无）")
    indexes = list(get_config().indexes_dir.glob("*.index"))
    table.add_row("已构建索引", ", ".join(i.name for i in indexes) or "（无）")

    # 内嵌依赖检测
    deps = []
    for mod in ("torch", "demucs", "faiss", "transformers"):
        try:
            __import__(mod)
            deps.append(f"[green]{mod}[/]")
        except ImportError:
            deps.append(f"[red]{mod}(缺失)[/]")
    table.add_row("内嵌依赖", " ".join(deps))

    # 外部 RVC 环境检测（音色迁移必需）
    import os

    rvc_home = os.environ.get("MUSICREMIX_RVC_HOME", "")
    if rvc_home:
        rh = Path(rvc_home)
    else:
        rh = Path(get_config().rvc_home)
    infer_cli = next(
        (rh / p for p in ("tools/infer_cli.py", "infer_cli.py", "src/infer_cli.py") if (rh / p).exists()),
        None,
    )
    if infer_cli:
        # 进一步检查是否有 .pth 模型
        weights_dir = rh / "assets" / "weights"
        models = list(weights_dir.glob("*.pth")) if weights_dir.exists() else []
        m_stat = f"{len(models)}个模型" if models else "无模型"
        rvc_status = f"[green]{rh} (infer_cli ✓, {m_stat})[/]"
    else:
        rvc_status = f"[yellow]{rh} (未找到 infer_cli.py)[/]" if rh.exists() else "[red]未配置[/]"
    table.add_row("外部RVC环境", rvc_status)

    console.print(table)


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="监听地址"),
    port: int = typer.Option(8000, "--port", "-p", help="监听端口"),
    reload: bool = typer.Option(False, "--reload", help="开发热重载"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """启动 Web 界面（浏览器打开选择歌曲、配置参数、查看进度）。"""
    _setup_logging(verbose)
    import uvicorn

    console.print(f"[green bold]MusicRemix Web 启动中...[/]")
    console.print(f"访问: [cyan]http://{host}:{port}[/]")
    console.print("按 Ctrl+C 停止")
    uvicorn.run("musicremix.web:app", host=host, port=port, reload=reload, log_level="info")


if __name__ == "__main__":
    app()
