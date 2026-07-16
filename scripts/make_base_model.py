#!/usr/bin/env python3
"""构造可推理的 base_v2_48k.pth（用 f0G48k 权重 + v2/48k config）。

RVC 的 pretrained_v2/f0G48k.pth 仅含模型权重（model/iteration/lr），
推理需要含 config/weight/f0/version 的完整模型。本脚本按 RVC savee 结构构造。

用法: python make_base_model.py [rvc_home]
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import torch


def main() -> None:
    rvc_home = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/rvc")
    src = rvc_home / "assets" / "pretrained_v2" / "f0G48k.pth"
    dst = rvc_home / "assets" / "weights" / "base_v2_48k.pth"

    if dst.exists():
        print(f"已存在，跳过: {dst}")
        return
    if not src.exists():
        raise FileNotFoundError(f"未找到 {src}，请先下载 pretrained_v2/f0G48k.pth")

    dst.parent.mkdir(parents=True, exist_ok=True)
    base = torch.load(src, map_location="cpu", weights_only=False)

    opt = OrderedDict()
    opt["weight"] = {k: v.float() for k, v in base["model"].items() if "enc_q" not in k}
    # config 顺序与 SynthesizerTrnMs768NSFsid 构造参数一致（v2/48k）
    opt["config"] = [
        1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
        [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        [12, 10, 2, 2], 512, [24, 20, 4, 4], 109, 256, 48000,
    ]
    opt["info"] = "base"
    opt["sr"] = "48k"
    opt["f0"] = 1
    opt["version"] = "v2"
    torch.save(opt, dst)
    print(f"构造完成: {dst} ({len(opt['weight'])} weight keys)")


if __name__ == "__main__":
    main()
