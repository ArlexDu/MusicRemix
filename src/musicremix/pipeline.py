"""端到端换音色流水线编排器。

串联：分离 → 建模 → 迁移 → 合成
特性：中间产物保留、阶段进度输出、失败续跑、清晰报错。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config, get_config
from .conversion.base import ConversionParams, VoiceConverter, get_converter
from .conversion.rvc_converter import ExternalRVCConverter
from .modeling.base import VoiceModeler
from .modeling.rvc_modeler import RVCModeler
from .remix.base import MixParams, Remixer
from .remix.mixer import SimpleRemixer
from .separation.base import SourceSeparator, get_separator

logger = logging.getLogger(__name__)

STAGES = ("separate", "model", "convert", "mix")


@dataclass
class PipelineResult:
    """流水线结果。"""

    output_path: Path
    stage_outputs: dict[str, Path] = field(default_factory=dict)
    completed_stages: list[str] = field(default_factory=list)


class RemixPipeline:
    """端到端换音色流水线。"""

    def __init__(
        self,
        separator: SourceSeparator | None = None,
        modeler: VoiceModeler | None = None,
        converter: VoiceConverter | None = None,
        remixer: Remixer | None = None,
        config: Config | None = None,
        device: str = "auto",
        model_name: str | None = None,
    ):
        self.config = config or get_config()
        self.device = device
        self.model_name = model_name
        # 延迟初始化各组件（避免未安装重依赖时构造即失败）
        self._separator = separator
        self._modeler = modeler
        self._converter = converter
        self._remixer = remixer

    # -- 懒加载组件 ---------------------------------------------------------
    @property
    def separator(self) -> SourceSeparator:
        if self._separator is None:
            self._separator = get_separator("demucs", device=self.device)
        return self._separator

    @property
    def modeler(self) -> VoiceModeler:
        if self._modeler is None:
            self._modeler = RVCModeler(separator=self.separator, device=self.device)
        return self._modeler

    @property
    def converter(self) -> VoiceConverter:
        if self._converter is None:
            self._converter = get_converter(
                "external", model_name=self.model_name, device=self.device
            )
        return self._converter

    @property
    def remixer(self) -> Remixer:
        if self._remixer is None:
            self._remixer = SimpleRemixer()
        return self._remixer

    # -- 主流程 -------------------------------------------------------------
    def run(
        self,
        input_song: str | Path,
        reference_songs: list[str | Path],
        output_path: str | Path,
        target_id: str = "target",
        workdir: str | Path | None = None,
        conversion: ConversionParams | None = None,
        mix: MixParams | None = None,
        skip_stages: set[str] | None = None,
    ) -> PipelineResult:
        """执行完整换音色流水线。

        Args:
            input_song: 原唱歌曲路径（歌手 A）
            reference_songs: 目标歌手参考歌曲列表
            output_path: 最终输出路径
            target_id: 目标歌手标识（索引缓存键）
            workdir: 中间产物目录
            conversion: 音色迁移参数
            mix: 混音参数
            skip_stages: 要跳过的阶段（用于续跑）
        """
        workdir = Path(workdir) if workdir else self.config.workdir_resolved
        workdir.mkdir(parents=True, exist_ok=True)
        skip_stages = skip_stages or set()

        result = PipelineResult(output_path=Path(output_path))
        try:
            # 阶段 1：分离
            vocal_path, acc_path = self._stage_separate(
                input_song, workdir, skip_stages, result
            )

            # 阶段 2：建模
            index_path = self._stage_model(
                reference_songs, target_id, workdir, skip_stages, result
            )

            # 阶段 3：迁移
            converted_path = self._stage_convert(
                vocal_path, index_path, workdir, conversion, skip_stages, result
            )

            # 阶段 4：合成
            self._stage_mix(
                converted_path, acc_path, output_path, mix, skip_stages, result
            )

        except Exception as e:
            self._report_failure(e, result)
            raise

        logger.info("流水线完成！输出: %s", output_path)
        return result

    # -- 各阶段实现 ---------------------------------------------------------
    def _stage_separate(self, input_song, workdir, skip_stages, result):
        self._log_stage("separate", "分离人声与伴奏")
        vocal_path = workdir / "vocals.wav"
        acc_path = workdir / "accompaniment.wav"
        if "separate" in skip_stages and vocal_path.exists() and acc_path.exists():
            logger.info("[separate] 命中缓存，跳过")
        else:
            paths = self.separator.separate_to_files(
                input_song, workdir, output_sr=self.config.output_sr,
                vocals_only=False, stem="",
            )
            # 重命名为标准名
            vocal_path = paths.get("vocals", vocal_path)
            acc_path = paths.get("accompaniment", acc_path)
            if vocal_path != workdir / "vocals.wav":
                vocal_path.rename(workdir / "vocals.wav")
                vocal_path = workdir / "vocals.wav"
            if acc_path != workdir / "accompaniment.wav":
                acc_path.rename(workdir / "accompaniment.wav")
                acc_path = workdir / "accompaniment.wav"
        result.stage_outputs["vocals"] = vocal_path
        result.stage_outputs["accompaniment"] = acc_path
        result.completed_stages.append("separate")
        self._log_stage_done("separate")
        return vocal_path, acc_path

    def _stage_model(self, reference_songs, target_id, workdir, skip_stages, result):
        self._log_stage("model", "构建目标音色索引")
        cached = self.modeler.get_index(target_id)
        if cached is not None:
            logger.info("[model] 索引缓存命中: %s", cached)
            index_path = cached
        elif "model" in skip_stages:
            raise RuntimeError("model 阶段被要求跳过但无缓存索引，无法续跑")
        else:
            index_path = self.modeler.build_index(reference_songs, target_id, workdir)
        result.stage_outputs["index"] = index_path
        result.completed_stages.append("model")
        self._log_stage_done("model")
        return index_path

    def _stage_convert(self, vocal_path, index_path, workdir, conversion, skip_stages, result):
        self._log_stage("convert", "音色迁移")
        converted_path = workdir / "converted_vocals.wav"
        if "convert" in skip_stages and converted_path.exists():
            logger.info("[convert] 命中缓存，跳过")
        else:
            self.converter.convert(vocal_path, index_path, converted_path, conversion)
        result.stage_outputs["converted_vocals"] = converted_path
        result.completed_stages.append("convert")
        self._log_stage_done("convert")
        return converted_path

    def _stage_mix(self, converted_path, acc_path, output_path, mix, skip_stages, result):
        self._log_stage("mix", "混音合成")
        if "mix" in skip_stages:
            logger.info("[mix] 被要求跳过")
        else:
            self.remixer.mix(converted_path, acc_path, output_path, mix)
        result.stage_outputs["final"] = Path(output_path)
        result.completed_stages.append("mix")
        self._log_stage_done("mix")

    # -- 工具方法 -----------------------------------------------------------
    def _log_stage(self, stage: str, desc: str):
        idx = STAGES.index(stage) + 1
        total = len(STAGES)
        logger.info("====== [%d/%d] %s: %s ======", idx, total, stage, desc)

    def _log_stage_done(self, stage: str):
        logger.info("------ %s 完成 ------", stage)

    def _report_failure(self, err: Exception, result: PipelineResult):
        logger.error("流水线失败：%s", err)
        logger.error("已完成阶段: %s", result.completed_stages or "（无）")
        if result.completed_stages:
            logger.error(
                "中间产物已保留在 %s。修复后可用 skip_stages 跳过已完成阶段续跑。",
                self.config.workdir_resolved,
            )


def detect_resumable_stages(workdir: str | Path) -> set[str]:
    """检测工作目录中已存在中间产物的阶段，用于续跑。"""
    workdir = Path(workdir)
    skip: set[str] = set()
    if (workdir / "vocals.wav").exists() and (workdir / "accompaniment.wav").exists():
        skip.add("separate")
    if (workdir / "converted_vocals.wav").exists():
        skip.add("convert")
    return skip
