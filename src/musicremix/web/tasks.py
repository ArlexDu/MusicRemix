"""Web 任务管理器：异步执行换音色流水线，实时更新状态。"""

from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

from ..config import Config, get_config
from ..conversion.base import ConversionParams, get_converter
from ..modeling.base import get_modeler
from ..remix.base import MixParams, get_remixer
from ..separation.base import get_separator
from ..utils.audio import save_audio

# 阶段与进度区间
STAGE_PROGRESS = {
    "training": (2, 40),
    "separating": (42, 55),
    "modeling": (57, 65),
    "converting": (67, 90),
    "mixing": (92, 99),
}
STAGE_LABEL = {
    "training": "训练目标歌手模型",
    "separating": "分离人声与伴奏",
    "modeling": "构建目标音色索引",
    "converting": "音色迁移",
    "mixing": "混音合成",
}


@dataclass
class Task:
    id: str
    status: str = "queued"  # queued / running / done / error
    stage: str = ""
    progress: int = 0
    message: str = ""
    outputs: dict = field(default_factory=dict)  # name -> path(str)
    error: str = ""
    workdir: Optional[Path] = None
    created_at: float = 0.0
    finished_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "stage_label": STAGE_LABEL.get(self.stage, ""),
            "progress": self.progress,
            "message": self.message,
            "outputs": list(self.outputs.keys()),
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
        }


class TaskManager:
    """内存任务管理（本地单用户，线程安全）。"""

    def __init__(self, config: Config | None = None):
        self.config = config or get_config()
        self._tasks: dict[str, Task] = {}
        self._lock = threading.Lock()

    def create(self) -> Task:
        tid = uuid.uuid4().hex[:12]
        workdir = self.config.cache_dir / "web_tasks" / tid
        workdir.mkdir(parents=True, exist_ok=True)
        task = Task(id=tid, workdir=workdir, created_at=time.time())
        with self._lock:
            self._tasks[tid] = task
        return task

    def get(self, tid: str) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(tid)

    def all(self) -> list[Task]:
        with self._lock:
            return list(self._tasks.values())

    def _update(self, task: Task, **kw):
        with self._lock:
            for k, v in kw.items():
                setattr(task, k, v)

    def start(
        self,
        task: Task,
        source_path: str | Path,
        reference_paths: list[str | Path],
        params: ConversionParams,
        model_name: str,
        mix_params: MixParams,
        target_id: str,
        device: str,
        train_kwargs: dict | None = None,
    ) -> None:
        thread = threading.Thread(
            target=self._run,
            args=(task, source_path, reference_paths, params, model_name, mix_params, target_id, device, train_kwargs),
            daemon=True,
        )
        thread.start()

    def _run(self, task, source_path, reference_paths, params, model_name, mix_params, target_id, device, train_kwargs=None):
        try:
            self._update(task, status="running", message="任务开始")
            wd = task.workdir
            sr = mix_params.output_sr
            tk = train_kwargs or {}

            # 阶段 0：训练目标歌手模型（可选，自动复用已训练模型）
            if tk.get("auto_train", False):
                model_name = self._maybe_train(task, reference_paths, target_id, device, wd, tk, model_name)

            # 阶段 1：分离
            self._set_stage(task, "separating")
            sep = get_separator("demucs", device=device)
            res = sep.separate(source_path, output_sr=sr, vocals_only=False)
            vocals_path = wd / "vocals.wav"
            acc_path = wd / "accompaniment.wav"
            save_audio(res.vocals, res.sample_rate, vocals_path)
            save_audio(res.accompaniment, res.sample_rate, acc_path)
            task.outputs["vocals"] = str(vocals_path)
            task.outputs["accompaniment"] = str(acc_path)
            self._set_stage(task, "separating", done=True)

            # 阶段 2：建模（容错：失败则无索引模式继续，base 模型推理不依赖索引）
            self._set_stage(task, "modeling")
            modeler = get_modeler("rvc", separator=sep, device=device)
            index_path = None
            try:
                index_path = modeler.build_index(list(reference_paths), target_id, wd)
                task.outputs["index"] = str(index_path)
            except Exception as e:
                logger.warning("音色索引构建失败，以无索引模式继续: %s", e)
                self._update(task, message=f"索引构建失败({e})，以无索引模式继续")
            self._set_stage(task, "modeling", done=True)

            # 阶段 3：转换
            self._set_stage(task, "converting")
            converter = get_converter("external", model_name=model_name, device=device)
            converted_path = wd / "converted.wav"
            converter.convert(vocals_path, index_path, converted_path, params)
            task.outputs["converted"] = str(converted_path)
            self._set_stage(task, "converting", done=True)

            # 阶段 4：混音
            self._set_stage(task, "mixing")
            remixer = get_remixer("default")
            final_path = wd / "remixed.wav"
            remixer.mix(converted_path, acc_path, final_path, mix_params)
            task.outputs["final"] = str(final_path)
            self._set_stage(task, "mixing", done=True)

            self._update(
                task, status="done", stage="done", progress=100,
                message="换音色完成", finished_at=time.time(),
            )
        except Exception as e:
            self._update(
                task, status="error", error=f"{type(e).__name__}: {e}",
                message="任务失败", finished_at=time.time(),
            )
            traceback.print_exc()

    def _maybe_train(self, task, reference_paths, target_id, device, wd, tk, model_name) -> str:
        """自动训练目标歌手模型：已存在则复用，否则训练；训练失败回退原 model_name。

        Returns:
            实际用于转换的 model_name（.pth 文件名）
        """
        cfg = self.config
        weights_dir = Path(cfg.rvc_home) / "assets" / "weights"
        # 约定：target_id 对应的模型文件名 {target_id}.pth
        target_pth = weights_dir / f"{target_id}.pth"
        if target_pth.exists():
            logger.info("目标模型已存在，跳过训练: %s", target_pth)
            self._update(task, message=f"模型已存在，复用 {target_pth.name}")
            return target_pth.name

        # 需要训练
        self._set_stage(task, "training")
        lo, hi = STAGE_PROGRESS["training"]

        def on_progress(stage_desc: str, sub: int):
            p = lo + (hi - lo) * sub / 100
            self._update(
                task, progress=int(p),
                message=f"训练中 · {stage_desc}（首次较慢，请耐心等待）",
            )

        try:
            from ..training import run_training
            pth = run_training(
                list(reference_paths),
                model_name=target_id,
                workdir=wd,
                sr=tk.get("sr", "48k"),
                f0method=tk.get("f0method", "rmvpe"),
                total_epoch=tk.get("total_epoch", 10),
                batch_size=tk.get("batch_size", 4),
                save_epoch=tk.get("save_epoch", 5),
                device=device,
                n_p=tk.get("n_p", 4),
                config=cfg,
                progress_cb=on_progress,
            )
            task.outputs["trained_model"] = str(pth)
            self._set_stage(task, "training", done=True)
            logger.info("目标模型训练完成: %s", pth)
            return pth.name
        except Exception as e:
            logger.warning("自动训练失败，回退到 %s: %s", model_name, e)
            self._update(
                task,
                message=f"训练失败({type(e).__name__}: {e})，回退到 {model_name}",
            )
            # 跳过 training 阶段剩余进度，直接进入下一阶段
            return model_name

    def _set_stage(self, task: Task, stage: str, done: bool = False):
        lo, hi = STAGE_PROGRESS[stage]
        progress = hi if done else lo
        msg = f"{STAGE_LABEL[stage]}..." if not done else f"{STAGE_LABEL[stage]}完成"
        self._update(task, stage=stage, progress=progress, message=msg)


# 全局任务管理器
manager = TaskManager()
