#!/usr/bin/env python3
"""patch RVC infer_cli.py，使 torch.load 默认 weights_only=False。

torch>=2.6 默认 weights_only=True，但 RVC 的 hubert_base.pt 含 fairseq 对象，
需 weights_only=False 才能加载。本脚本在 infer_cli.py 顶部插入 monkey-patch。
幂等：已 patch 则跳过。

用法: python patch_rvc_infer.py [infer_cli_path]
"""

from __future__ import annotations

import sys
from pathlib import Path

MARKER = "_orig_torch_load = torch.load"

PATCH = '''import torch

# torch>=2.6 默认 weights_only=True，RVC 的 hubert/cpt 含 fairseq 对象需 weights_only=False
_orig_torch_load = torch.load


def _torch_load_safe(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_safe
'''


def main() -> None:
    infer_cli = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/rvc/tools/infer_cli.py")
    if not infer_cli.exists():
        raise FileNotFoundError(f"未找到 {infer_cli}")

    text = infer_cli.read_text(encoding="utf-8")
    if MARKER in text:
        print(f"已 patch，跳过: {infer_cli}")
        return

    if "import torch\n" not in text:
        text = "import torch\n" + text
    text = text.replace("import torch\n", PATCH, 1)
    infer_cli.write_text(text, encoding="utf-8")
    print(f"已 patch: {infer_cli}")


if __name__ == "__main__":
    main()
