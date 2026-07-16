"""端到端集成测试 (7.6)。

依赖重模型（demucs/rvc）的完整流程在缺依赖时跳过。
混音环节的纯逻辑由 test_remix.py 覆盖。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from musicremix.pipeline import RemixPipeline, detect_resumable_stages

demucs = pytest.importorskip("demucs", reason="demucs 未安装，跳过端到端测试")
torch = pytest.importorskip("torch", reason="torch 未安装，跳过端到端测试")
rvc = pytest.importorskip("rvc_python", reason="rvc-python 未安装，跳过端到端测试")

# 端到端集成测试依赖重模型，标记 slow
pytestmark = pytest.mark.slow


def test_pipeline_constructs():
    p = RemixPipeline(device="cpu")
    assert p is not None


def test_detect_resumable_stages_empty(tmp_path):
    assert detect_resumable_stages(tmp_path) == set()


def test_detect_resumable_stages_with_artifacts(tmp_path):
    (tmp_path / "vocals.wav").write_bytes(b"x")
    (tmp_path / "accompaniment.wav").write_bytes(b"x")
    skip = detect_resumable_stages(tmp_path)
    assert "separate" in skip


@pytest.mark.skip(reason="需要真实模型与小样音频，由人工验收 (9.4) 覆盖")
def test_full_pipeline_end_to_end():
    """占位：完整端到端由 9.4 人工验收执行。"""
