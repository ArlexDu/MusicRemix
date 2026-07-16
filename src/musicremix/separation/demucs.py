"""Demucs v4 (htdemucs) 源分离后端。

将歌曲拆分为人声 (vocals) 与伴奏 (accompaniment) 两轨。
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..utils.audio import load_audio, resample, save_audio
from ..utils.device import Device, detect_device, torch_device_str
from .base import SeparationResult, SourceSeparator

logger = logging.getLogger(__name__)


class DemucsSeparator(SourceSeparator):
    """基于 Demucs htdemucs 的源分离实现。"""

    def __init__(self, model_name: str = "htdemucs", device: str = "auto", shifts: int = 1):
        self.model_name = model_name
        self.force_device = device
        self.shifts = shifts
        self._model = None
        self._device: Device | None = None

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            import torch
            from demucs.pretrained import get_model
        except ImportError as e:
            raise ImportError(
                "未安装 demucs/torch。请安装源分离依赖: pip install -e '.[ml]'"
            ) from e

        self._device = detect_device(self.force_device)
        logger.info("加载 Demucs 模型 %s (设备: %s)...", self.model_name, self._device.value)
        model = get_model(self.model_name)
        model.to(torch_device_str(self._device))
        model.eval()
        self._model = model

    def separate(
        self,
        audio_path: str | Path,
        output_sr: int = 44100,
        vocals_only: bool = False,
    ) -> SeparationResult:
        self._ensure_model()
        import torch

        # 1. 读取音频并重采样到模型所需采样率（Demucs 通常 44100，立体声）
        model_sr = self._model.samplerate  # type: ignore[union-attr]
        audio, orig_sr = load_audio(audio_path, sr=model_sr, mono=False)
        # Demucs 期望 (batch, channels, samples)
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0)
        wav = torch.from_numpy(audio).unsqueeze(0).float()  # (1, 2, samples)
        wav = wav.to(torch_device_str(self._device))  # type: ignore[union-attr]

        logger.info("开始分离 (时长 %.1fs)...", wav.shape[-1] / model_sr)
        from demucs.apply import apply_model

        with torch.no_grad():
            # apply_model 返回 (batch, sources, channels, samples)
            sources = apply_model(
                self._model, wav, shifts=self.shifts, split=True,
                overlap=0.25, progress=False,
            )
            # 移到 CPU 转 numpy
            sources = sources.cpu().numpy()[0]  # (sources, channels, samples)

        # Demucs 默认 4 源顺序: drums, bass, other, vocals (htdemucs)
        # 伴奏 = drums + bass + other
        source_names = self._model.sources  # type: ignore[union-attr]
        idx = {n: i for i, n in enumerate(source_names)}
        vocals = sources[idx["vocals"]]
        if vocals_only:
            accompaniment = np.zeros_like(vocals)
        else:
            acc = np.zeros_like(vocals)
            for name in ("drums", "bass", "other"):
                if name in idx:
                    acc += sources[idx[name]]
            accompaniment = acc

        # 2. 重采样到目标输出采样率
        if output_sr != model_sr:
            vocals = resample(vocals, model_sr, output_sr)
            if not vocals_only:
                accompaniment = resample(accompaniment, model_sr, output_sr)

        return SeparationResult(
            vocals=vocals.astype(np.float32),
            accompaniment=accompaniment.astype(np.float32),
            sample_rate=output_sr,
        )

    def separate_to_files(
        self,
        audio_path: str | Path,
        outdir: str | Path,
        output_sr: int = 44100,
        vocals_only: bool = False,
        stem: str | None = None,
    ) -> dict[str, Path]:
        """分离并落盘为文件，返回路径字典。"""
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        stem = stem or Path(audio_path).stem
        result = self.separate(audio_path, output_sr=output_sr, vocals_only=vocals_only)

        paths: dict[str, Path] = {}
        paths["vocals"] = save_audio(result.vocals, result.sample_rate, outdir / f"{stem}_vocals.wav")
        if not vocals_only:
            paths["accompaniment"] = save_audio(
                result.accompaniment, result.sample_rate, outdir / f"{stem}_accompaniment.wav"
            )
        return paths
