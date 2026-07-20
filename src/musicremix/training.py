"""RVC 模型训练：用目标歌手人声微调生成器，得到该歌手音色模型。

流程：preprocess（重采样切片）→ extract_f0（提音高）→ extract_feature（提 HuBERT 特征）
      → 生成 filelist/config → train（微调生成器）→ 产出 .pth

训练产出的 .pth 放于 models/rvc/assets/weights/，可用于 convert/remix。
注意：CPU 训练慢（数小时），建议 GPU；数据需干净人声（含伴奏效果差）。
"""

from __future__ import annotations

import json
import logging
import os
import random
import subprocess
from pathlib import Path
from typing import Callable, Optional

from .config import Config, get_config
from .separation.base import get_separator
from .utils.audio import save_audio
from .utils.device import Device, detect_device

logger = logging.getLogger(__name__)

SR_DICT = {"32k": 32000, "40k": 40000, "48k": 48000}

# 进度回调签名: (stage_desc: str, sub_progress: int 0-100)
ProgressCB = Optional[Callable[[str, int], None]]


def _safe_cb(cb: ProgressCB, stage: str, sub: int):
    """安全触发进度回调（吞异常，避免回调失败中断训练）。"""
    if cb is None:
        return
    try:
        cb(stage, max(0, min(100, int(sub))))
    except Exception:
        logger.debug("进度回调异常（已忽略）", exc_info=True)


def _venv_python(rvc_home: Path) -> str:
    exe = "python.exe" if os.name == "nt" else "python"
    sub = "Scripts" if os.name == "nt" else "bin"
    return str(rvc_home / "venv" / sub / exe)


def _rvc_device(device: str) -> str:
    dev = detect_device(device)
    return "cuda" if dev == Device.CUDA else "cpu"


def prepare_dataset(
    reference_songs: list[str | Path],
    out_dir: str | Path,
    device: str = "auto",
    config: Config | None = None,
) -> Path:
    """分离参考歌曲人声到训练数据目录（RVC 训练需干净人声）。"""
    config = config or get_config()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sep = get_separator("demucs", device=device)
    for i, song in enumerate(reference_songs):
        logger.info("分离参考歌曲 %d/%d: %s", i + 1, len(reference_songs), Path(song).name)
        res = sep.separate(song, output_sr=config.output_sr, vocals_only=True)
        save_audio(res.vocals, res.sample_rate, out_dir / f"vocal_{i:03d}.wav")
    n = len(list(out_dir.glob("*.wav")))
    logger.info("训练数据准备完成: %d 个干净人声文件", n)
    return out_dir


def train_model(
    input_dir: str | Path,
    model_name: str,
    rvc_home: str | Path | None = None,
    sr: str = "48k",
    version: str = "v2",
    f0method: str = "rmvpe",
    total_epoch: int = 10,
    batch_size: int = 4,
    save_epoch: int = 5,
    device: str = "auto",
    n_p: int = 4,
    config: Config | None = None,
    progress_cb: ProgressCB = None,
) -> Path:
    """训练目标歌手 RVC 模型。

    Args:
        input_dir: 训练数据目录（干净人声 wav）
        model_name: 模型名（产出 .pth 名前缀）
        rvc_home: RVC 仓库路径（默认 models/rvc）
        sr: 采样率 32k/40k/48k
        version: v1/v2
        f0method: 音高算法 harvest/rmvpe/crepe/pm
        total_epoch: 训练总轮数（CPU 建议 5-20）
        batch_size: 批大小（CPU 建议 2-4）
        save_epoch: 每隔几轮保存一次
        device: cpu/cuda/mps（mps 会降级 cpu）
        n_p: 并发进程数
        progress_cb: 进度回调 (stage_desc, sub_progress 0-100)

    Returns:
        训练产出的 .pth 路径
    """
    config = config or get_config()
    rvc_home = Path(rvc_home).resolve() if rvc_home else Path(config.rvc_home).resolve()
    python = _venv_python(rvc_home)
    now_dir = rvc_home
    exp_dir = model_name
    logs_dir = now_dir / "logs" / exp_dir
    logs_dir.mkdir(parents=True, exist_ok=True)
    sr_int = SR_DICT[sr]
    rvc_device = _rvc_device(device)

    # RVC 的 load_audio 依赖 ffmpeg（软链接在 venv/bin），需加入 PATH
    env = dict(os.environ)
    venv_bin = str(rvc_home / "venv" / ("Scripts" if os.name == "nt" else "bin"))
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

    def run(cmd, desc):
        logger.info("[%s] %s", desc, " ".join(str(c) for c in cmd))
        proc = subprocess.run(cmd, cwd=str(now_dir), capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"{desc} 失败 (exit {proc.returncode}):\n{(proc.stderr or '')[-1500:]}")
        return proc.stdout

    # 1. preprocess（重采样切片）— 产出 0_gt_wavs
    if any((logs_dir / "0_gt_wavs").glob("*.wav")):
        logger.info("阶段 1/4: preprocess 已完成，跳过")
    else:
        _safe_cb(progress_cb, "预处理（重采样切片）", 0)
        logger.info("阶段 1/4: preprocess（重采样切片）")
        run([python, "infer/modules/train/preprocess.py", str(input_dir), str(sr_int), str(n_p),
             str(logs_dir), "False", "3.0"], "preprocess")
    _safe_cb(progress_cb, "预处理完成", 20)

    # 2. extract_f0（提取音高）— 产出 2a_f0
    if any((logs_dir / "2a_f0").glob("*.npy")):
        logger.info("阶段 2/4: extract_f0 已完成，跳过")
    else:
        _safe_cb(progress_cb, "提取音高（f0）", 20)
        logger.info("阶段 2/4: extract_f0（提取音高）")
        run([python, "infer/modules/train/extract/extract_f0_print.py", str(logs_dir), str(n_p), f0method],
            "extract_f0")
    _safe_cb(progress_cb, "音高提取完成", 40)

    # 3. extract_feature（提 HuBERT 特征）— 产出 3_feature768
    feat_subdir = "3_feature256" if version == "v1" else "3_feature768"
    if any((logs_dir / feat_subdir).glob("*.npy")):
        logger.info("阶段 3/4: extract_feature 已完成，跳过")
    else:
        _safe_cb(progress_cb, "提取 HuBERT 特征", 40)
        logger.info("阶段 3/4: extract_feature（提取 HuBERT 特征）")
        for i_part in range(n_p):
            run([python, "infer/modules/train/extract_feature_print.py", rvc_device, str(n_p), str(i_part),
                 str(logs_dir), version, "False"], f"extract_feature p{i_part}")
    _safe_cb(progress_cb, "特征提取完成", 60)

    # 4. 生成 filelist + config + 训练
    _safe_cb(progress_cb, "模型微调训练中", 60)
    logger.info("阶段 4/4: 生成 filelist/config + 训练")
    _gen_filelist(now_dir, exp_dir, sr, version)
    _gen_config(now_dir, exp_dir, sr, version)

    pretrained_G = f"assets/pretrained_v2/f0G{sr}.pth"
    pretrained_D = f"assets/pretrained_v2/f0D{sr}.pth"
    train_cmd = [python, "infer/modules/train/train.py",
                 "-e", exp_dir, "-sr", sr, "-f0", "1", "-bs", str(batch_size),
                 "-te", str(total_epoch), "-se", str(save_epoch),
                 "-pg", pretrained_G, "-pd", pretrained_D,
                 "-l", "1", "-c", "0", "-sw", "1", "-v", version]
    run(train_cmd, "train")
    _safe_cb(progress_cb, "模型微调完成", 100)

    # 查找产出 .pth（sw=1 每轮保存到 assets/weights/）
    weights_dir = now_dir / "assets" / "weights"
    candidates = sorted(
        weights_dir.glob(f"{exp_dir}*.pth"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if candidates:
        logger.info("训练完成: %s", candidates[0])
        return candidates[0]
    g_pth = logs_dir / f"G_{total_epoch}.pth"
    if g_pth.exists():
        return g_pth
    raise RuntimeError("训练完成但未找到产出 .pth（检查 logs/%s/train.log）" % exp_dir)


def _gen_filelist(now_dir: Path, exp_dir: str, sr: str, version: str):
    """复现 RVC click_train 的 filelist 生成（含 mute 静音样本）。"""
    exp_path = now_dir / "logs" / exp_dir
    gt_wavs_dir = exp_path / "0_gt_wavs"
    feature_dir = exp_path / ("3_feature256" if version == "v1" else "3_feature768")
    f0_dir = exp_path / "2a_f0"
    f0nsf_dir = exp_path / "2b-f0nsf"

    # 用 split(".")[0] 取名（RVC 官方做法），避免 .wav.npy 的 stem 不一致
    names = (
        set(p.name.split(".")[0] for p in gt_wavs_dir.glob("*.wav"))
        & set(p.name.split(".")[0] for p in feature_dir.glob("*.npy"))
        & set(p.name.split(".")[0] for p in f0_dir.glob("*.npy"))
        & set(p.name.split(".")[0] for p in f0nsf_dir.glob("*.npy"))
    )
    if not names:
        raise RuntimeError("无可用训练样本，检查 preprocess/extract 是否成功")
    spk_id = 0
    opt = [
        f"{gt_wavs_dir}/{n}.wav|{feature_dir}/{n}.npy|{f0_dir}/{n}.wav.npy|{f0nsf_dir}/{n}.wav.npy|{spk_id}"
        for n in names
    ]
    # 加 2 条 mute 静音样本（RVC 标准做法）
    fea_dim = 256 if version == "v1" else 768
    mute_gt = f"{now_dir}/logs/mute/0_gt_wavs/mute{sr}.wav"
    mute_fea = f"{now_dir}/logs/mute/3_feature{fea_dim}/mute.npy"
    mute_f0 = f"{now_dir}/logs/mute/2a_f0/mute.wav.npy"
    mute_f0nsf = f"{now_dir}/logs/mute/2b-f0nsf/mute.wav.npy"
    for _ in range(2):
        opt.append(f"{mute_gt}|{mute_fea}|{mute_f0}|{mute_f0nsf}|{spk_id}")
    random.shuffle(opt)
    (exp_path / "filelist.txt").write_text("\n".join(opt), encoding="utf-8")
    logger.info("filelist 生成: %d 条", len(opt))


def _gen_config(now_dir: Path, exp_dir: str, sr: str, version: str):
    """生成 config.json（从 configs 模板复制）。"""
    exp_path = now_dir / "logs" / exp_dir
    config_save = exp_path / "config.json"
    if config_save.exists():
        return
    rel = f"v1/{sr}.json" if version == "v1" or sr == "40k" else f"v2/{sr}.json"
    config_path = now_dir / "configs" / rel
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    with open(config_save, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4, sort_keys=True)
        f.write("\n")
    logger.info("config 生成: %s", config_save)


def run_training(
    reference_songs: list[str | Path],
    model_name: str,
    workdir: str | Path | None = None,
    progress_cb: ProgressCB = None,
    **kwargs,
) -> Path:
    """完整训练流程：分离参考歌曲人声 + RVC 训练。

    进度映射：分离人声 0-10%，RVC 训练 10-100%。
    """
    config = get_config()
    workdir = Path(workdir) if workdir else config.workdir_resolved
    train_data_dir = workdir / f"train_data_{model_name}"

    _safe_cb(progress_cb, "分离参考歌曲人声", 0)
    prepare_dataset(reference_songs, train_data_dir, device=kwargs.get("device", "auto"), config=config)
    _safe_cb(progress_cb, "参考人声分离完成", 10)

    # RVC 训练内部子阶段映射到 10-100%
    def _train_cb(stage: str, sub: int):
        _safe_cb(progress_cb, stage, 10 + int(sub * 0.9))

    return train_model(train_data_dir, model_name, config=config, progress_cb=_train_cb, **kwargs)
