"""音色建模单元测试 (4.7)。

依赖 torch/faiss/transformers 的测试在缺依赖时自动跳过。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from musicremix.modeling import get_modeler
from musicremix.modeling.base import VoiceModeler
from musicremix.modeling.rvc_modeler import RVCModeler

faiss = pytest.importorskip("faiss", reason="faiss 未安装，跳过建模测试")
torch = pytest.importorskip("torch", reason="torch 未安装，跳过建模测试")


def test_modeler_is_voice_modeler():
    # 不触发重依赖加载（仅实例化外壳需要 separator，故用 mock）
    class MockModeler(RVCModeler):
        def __init__(self):
            pass

    assert isinstance(MockModeler(), VoiceModeler)


def test_get_modeler_unknown_raises():
    with pytest.raises(ValueError):
        get_modeler("nonexistent")


def test_get_index_returns_none_when_missing(tmp_path):
    from musicremix.config import Config

    cfg = Config(cache_dir=tmp_path)
    m = RVCModeler.__new__(RVCModeler)
    m.config = cfg
    assert m.get_index("singerX") is None


def test_get_index_returns_path_when_exists(tmp_path):
    from musicremix.config import Config

    cfg = Config(cache_dir=tmp_path)
    idx = cfg.index_path("singerB")
    idx.parent.mkdir(parents=True, exist_ok=True)
    idx.write_bytes(b"fake-index")
    m = RVCModeler.__new__(RVCModeler)
    m.config = cfg
    assert m.get_index("singerB") == idx


def test_faiss_index_buildable(tmp_path):
    """直接验证 _build_faiss_index 能产出可读索引文件。"""
    import numpy as np

    from musicremix.config import Config

    cfg = Config(cache_dir=tmp_path)
    m = RVCModeler.__new__(RVCModeler)
    m.config = cfg
    feats = np.random.randn(500, 256).astype(np.float32)
    p = m._build_faiss_index(feats, "test_singer")
    assert p.exists()
    # 可被 faiss 读回
    idx = faiss.read_index(str(p))
    assert idx.ntotal == 500
