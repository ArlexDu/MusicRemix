"""RVC 检索式音色建模：ContentVec 特征提取 + Faiss 检索索引。

核心思想：
1. 从参考歌曲分离出干净人声（复用 SourceSeparator）
2. 用 ContentVec（HuBERT 微调）逐帧提取内容特征（256 维，RVC 降维后）
3. 用 Faiss 建立近邻检索索引，持久化为 .index 文件
4. 多首参考特征拼接融合；按 target_id 缓存复用
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..config import Config, get_config
from ..separation.base import SourceSeparator, get_separator
from ..utils.audio import get_duration, load_audio
from .base import VoiceModeler

logger = logging.getLogger(__name__)

# 参考人声有效时长下限（秒），低于此阈值告警
MIN_REFERENCE_SECONDS = 10.0
# 静音占比上限
MAX_SILENCE_RATIO = 0.6


class RVCModeler(VoiceModeler):
    """基于 ContentVec + Faiss 的目标音色建模。"""

    def __init__(
        self,
        separator: SourceSeparator | None = None,
        config: Config | None = None,
        device: str = "auto",
        contentvec_model: str = "microsoft/contentvec",
    ):
        self.config = config or get_config()
        self.separator = separator or get_separator("demucs", device=device)
        self.device = device
        self.contentvec_model = contentvec_model
        self._extractor = None

    # -- 公开接口 -----------------------------------------------------------

    def get_index(self, target_id: str) -> Path | None:
        p = self.config.index_path(target_id)
        return p if p.exists() else None

    def build_index(
        self,
        reference_audio_paths: list[str | Path],
        target_id: str,
        workdir: str | Path | None = None,
    ) -> Path:
        # 0. 缓存命中直接复用
        cached = self.get_index(target_id)
        if cached is not None:
            logger.info("音色索引缓存命中: %s", cached)
            return cached

        workdir = Path(workdir) if workdir else self.config.workdir_resolved
        workdir.mkdir(parents=True, exist_ok=True)

        # 1. 从参考歌曲提取干净人声
        ref_vocals = self._extract_reference_vocals(reference_audio_paths, target_id, workdir)

        # 2. 提取 ContentVec 特征并融合
        all_features = []
        for vocal_path in ref_vocals:
            feats = self._extract_contentvec_features(vocal_path)
            all_features.append(feats)
        if not all_features:
            raise RuntimeError("未能从参考歌曲提取到任何人声特征")
        features = np.concatenate(all_features, axis=0).astype(np.float32)
        logger.info("目标音色 %s 共提取 %d 帧特征", target_id, features.shape[0])

        # 3. 构建 Faiss 检索索引
        index_path = self._build_faiss_index(features, target_id)
        logger.info("音色索引已构建: %s", index_path)
        return index_path

    # -- 内部实现 -----------------------------------------------------------

    def _extract_reference_vocals(
        self,
        reference_paths: list[str | Path],
        target_id: str,
        workdir: Path,
    ) -> list[Path]:
        """对每首参考歌曲做人声分离，并做质量校验。"""
        out_dir = workdir / f"ref_vocals_{target_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        results: list[Path] = []
        for i, ref in enumerate(reference_paths):
            ref = Path(ref)
            logger.info("处理参考歌曲 %d/%d: %s", i + 1, len(reference_paths), ref.name)
            paths = self.separator.separate_to_files(
                ref, out_dir, output_sr=self.config.output_sr, vocals_only=True, stem=f"ref_{i}"
            )
            vocal_path = paths["vocals"]
            self._check_reference_quality(vocal_path, ref)
            results.append(vocal_path)
        return results

    def _check_reference_quality(self, vocal_path: Path, source_path: Path) -> None:
        """参考素材质量校验：时长过短/静音占比过高告警（不中断）。"""
        try:
            dur = get_duration(vocal_path)
        except Exception:
            return
        if dur < MIN_REFERENCE_SECONDS:
            logger.warning(
                "参考 %s 分离后人声仅 %.1fs（< %.1fs），可能不足以建模音色",
                source_path.name, dur, MIN_REFERENCE_SECONDS,
            )
        # 静音占比粗略估计
        try:
            y, _ = load_audio(vocal_path, sr=16000, mono=True)
            silence_ratio = float(np.mean(np.abs(y) < 1e-4))
            if silence_ratio > MAX_SILENCE_RATIO:
                logger.warning(
                    "参考 %s 静音占比 %.0f%% 偏高，可能影响音色建模质量",
                    source_path.name, silence_ratio * 100,
                )
        except Exception:
            pass

    def _ensure_extractor(self):
        if self._extractor is not None:
            return
        try:
            import torch
            from transformers import Wav2Vec2FeatureExtractor, HubertModel
        except ImportError as e:
            raise ImportError(
                "音色建模需要 torch + transformers。请安装: pip install -e '.[rvc]' transformers"
            ) from e

        from ..utils.device import detect_device, torch_device_str

        dev = detect_device(self.device)
        logger.info("加载 ContentVec 模型 %s (设备: %s)...", self.contentvec_model, dev.value)
        feat_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.contentvec_model)
        model = HubertModel.from_pretrained(self.contentvec_model)
        model.to(torch_device_str(dev))
        model.eval()
        self._extractor = (feat_extractor, model, dev)
        self._torch = torch

    def _extract_contentvec_features(self, vocal_path: Path) -> np.ndarray:
        """提取 ContentVec 逐帧特征，返回 (frames, 256)。"""
        self._ensure_extractor()
        torch = self._torch
        feat_extractor, model, dev = self._extractor

        # ContentVec 输入需 16kHz 单声道
        y, _ = load_audio(vocal_path, sr=16000, mono=True)
        y = y[0] if y.ndim > 1 else y
        inputs = feat_extractor(y, sampling_rate=16000, return_tensors="pt").input_values
        inputs = inputs.to(torch_device_str(dev))

        with torch.no_grad():
            outputs = model(inputs)
            # 最后一层隐藏态 (1, frames, 1024)
            hidden = outputs.last_hidden_state.squeeze(0).cpu().numpy()

        # RVC 使用 256 维：取后 256 维（与 RVC 官方一致）
        feats = hidden[:, -256:] if hidden.shape[1] >= 256 else hidden
        return feats.astype(np.float32)

    def _build_faiss_index(self, features: np.ndarray, target_id: str) -> Path:
        """用 Faiss 构建检索索引并持久化。"""
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "音色建模需要 faiss。请安装: pip install -e '.[rvc]'"
            ) from e

        dim = features.shape[1]
        # 量化 + 倒排索引；样本少时回退 IndexFlatL2
        n = features.shape[0]
        nlist = max(1, min(n // 39, 100))

        if n < nlist * 2:
            index = faiss.IndexFlatL2(dim)
        else:
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            index.train(features)
        index.add(features)

        index_path = self.config.index_path(target_id)
        faiss.write_index(index, str(index_path))
        return index_path
