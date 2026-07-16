"""CLI 冒烟测试 (8.5)：参数解析、默认值、帮助文本。"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from musicremix.cli import app

runner = CliRunner()


def test_app_help_lists_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("remix", "separate", "model", "convert", "mix", "info"):
        assert cmd in result.output


def test_remix_help_shows_required_options():
    result = runner.invoke(app, ["remix", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--reference" in result.output
    assert "--output" in result.output


def test_remix_defaults():
    """不带必填参数应报错（exit code != 0）并提示缺失项。"""
    result = runner.invoke(app, ["remix"])
    assert result.exit_code != 0


def test_info_runs_without_error():
    """info 命令不需要重依赖，应正常输出状态表。"""
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "MusicRemix" in result.output
    assert "计算设备" in result.output


def test_mix_help():
    result = runner.invoke(app, ["mix", "--help"])
    assert result.exit_code == 0
    assert "--vocal" in result.output
    assert "--accompaniment" in result.output
