"""RVC 音色迁移后端（调用外部 RVC WebUI 环境）。

RVC 官方仓库（RVC-Project/Retrieval-based-Voice-Conversion-WebUI）通过
`tools/infer_cli.py` 提供命令行推理。本模块通过 subprocess 调用它。

RVC 推理需要：
- 预训练模型：hubert_base.pt（特征提取）、rmvpe.pt（f0 算法）—— 已随环境下载
- 生成器模型：目标歌手的 .pth（放于 assets/weights/），含 config/weight 结构
  - 可用 RVC 训练目标歌手得到，或用通用 base 模型（base_v2_48k.pth）验证流程
- ffmpeg：RVC 的 load_audio 依赖（已软链接到 venv/bin/ffmpeg）

注意：RVC 推理本质上需要一个训练完成的生成器模型（含 config 键），
纯预训练 base（仅权重）不能直接推理。本工程提供 base_v2_48k.pth 用于流程验证，
真正换音色需训练目标歌手模型。
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from ..config import Config, get_config
from ..utils.device import Device, detect_device
from .base import ConversionParams, VoiceConverter

logger = logging.getLogger(__name__)

# RVC infer_cli.py 在仓库中的相对位置
_INFER_CLI_CANDIDATES = ("tools/infer_cli.py", "infer_cli.py", "src/infer_cli.py")


def _venv_subdir() -> str:
    """虚拟环境的可执行目录：Windows 为 Scripts，Unix 为 bin。"""
    return "Scripts" if os.name == "nt" else "bin"


def _venv_python(rvc_home: Path) -> Path:
    """RVC 虚拟环境的 python 可执行文件路径（跨平台）。"""
    exe = "python.exe" if os.name == "nt" else "python"
    return rvc_home / "venv" / _venv_subdir() / exe


def _rvc_device_str(dev: Device) -> str:
    """把本工具的设备枚举映射为 RVC 能理解的字符串。

    RVC 对 Apple MPS 支持不稳定，MPS 时降级 CPU 以保证可用。
    """
    if dev == Device.CUDA:
        return "cuda"
    return "cpu"


class ExternalRVCConverter(VoiceConverter):
    """通过 subprocess 调用外部 RVC WebUI 的 infer_cli.py。

    用户需预先在 RVC 环境（默认 models/rvc）安装依赖、下载预训练模型，
   并将生成器 .pth 模型放入 assets/weights/。
    """

    def __init__(
        self,
        model_name: str,
        rvc_home: str | Path | None = None,
        config: Config | None = None,
        device: str = "auto",
        python_bin: str | None = None,
    ):
        self.model_name = model_name
        self.config = config or get_config()
        self.device = device
        self.rvc_home = Path(
            rvc_home if rvc_home else os.environ.get("MUSICREMIX_RVC_HOME", self.config.rvc_home)
        ).resolve()
        self.python_bin = python_bin or os.environ.get(
            "MUSICREMIX_RVC_PYTHON", str(_venv_python(self.rvc_home))
        )

    def _resolve_infer_cli(self) -> Path:
        for rel in _INFER_CLI_CANDIDATES:
            p = self.rvc_home / rel
            if p.exists():
                return p
        raise FileNotFoundError(
            f"未在 RVC 环境 {self.rvc_home} 找到 infer_cli.py（已检查 {list(_INFER_CLI_CANDIDATES)}）。"
            f"请确认 RVC 仓库已 clone 到 models/rvc，或设置 MUSICREMIX_RVC_HOME。"
        )

    def _resolve_model_path(self) -> Path:
        """定位目标 .pth 生成器模型（RVC 的 assets/weights/）。"""
        p = self.rvc_home / "assets" / "weights" / self.model_name
        if not p.suffix:
            p = p.with_suffix(".pth")
        return p

    def convert(
        self,
        source_vocal_path: str | Path,
        index_path: str | Path,
        output_path: str | Path,
        params: ConversionParams | None = None,
    ) -> Path:
        params = params or ConversionParams()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        infer_cli = self._resolve_infer_cli()
        dev = detect_device(self.device)
        model_path = self._resolve_model_path()

        if not model_path.exists():
            raise FileNotFoundError(
                f"RVC 模型 {model_path} 不存在。"
                f"请将生成器 .pth 放到 {self.rvc_home}/assets/weights/，"
                f"或通过 --model-name 指定正确文件名。"
                f"（流程验证可用 base_v2_48k.pth）"
            )

        # 构造 infer_cli.py 命令（参数名与 RVC 官方一致，下划线风格）
        cmd = [
            self.python_bin, str(infer_cli),
            "--f0up_key", str(params.pitch),
            "--input_path", str(source_vocal_path),
            "--f0method", params.f0_method,
            "--opt_path", str(output_path),
            "--model_name", self.model_name,
            "--index_rate", str(params.index_rate),
            "--protect", str(params.protect),
            "--device", _rvc_device_str(dev),
        ]
        # 检索索引为可选增强（index_path 可能为 None，无索引模式）
        if index_path and Path(index_path).exists():
            cmd += ["--index_path", str(index_path)]

        logger.info(
            "调用 RVC 推理: %s -> %s (model=%s, index_rate=%.2f, f0=%s, pitch=%+d, device=%s)",
            Path(source_vocal_path).name, output_path.name, self.model_name,
            params.index_rate, params.f0_method, params.pitch, _rvc_device_str(dev),
        )
        logger.debug("RVC 命令: %s", " ".join(cmd))

        # 确保 ffmpeg（位于 venv 可执行目录）在 PATH 中
        env = dict(os.environ)
        venv_bin = str(self.rvc_home / "venv" / _venv_subdir())
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

        proc = subprocess.run(
            cmd, cwd=str(self.rvc_home), capture_output=True, text=True, env=env,
        )
        if proc.returncode != 0 or not output_path.exists():
            stderr_tail = (proc.stderr or "")[-1000:]
            raise RuntimeError(
                f"RVC 推理失败 (exit {proc.returncode})。\n"
                f"命令: {' '.join(cmd)}\nstderr: {stderr_tail}\n"
                f"请确认 RVC 环境依赖与预训练模型已就绪。"
            )
        logger.info("音色迁移完成: %s", output_path)
        return output_path


class RVCConverter(VoiceConverter):
    """内嵌 rvc-python 包的音色迁移（可选，PyPI 包已废弃，仅保留接口）。"""

    def __init__(self, base_model_path=None, config=None, device="auto"):
        self.config = config or get_config()
        self.base_model_path = base_model_path
        self.device = device

    def convert(self, source_vocal_path, index_path, output_path, params=None):
        raise NotImplementedError(
            "内嵌 rvc-python 后端不可用（PyPI 包已废弃）。"
            "请使用默认的 ExternalRVCConverter（调用外部 RVC 环境）。"
        )
