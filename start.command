#!/bin/bash
# MusicRemix 一键启动脚本（macOS 双击运行）
# 双击此文件即可启动 Web 服务并自动打开浏览器

cd "$(dirname "$0")"

echo "============================================"
echo "  MusicRemix · 歌曲换音色工具"
echo "============================================"
echo ""

# 检查虚拟环境
if [ ! -x ".venv/bin/musicremix" ]; then
    echo "[错误] 未找到 .venv/bin/musicremix"
    echo "请先完成环境安装（见 README.md 的安装章节）。"
    echo ""
    echo "按任意键退出..."
    read -n 1
    exit 1
fi

PORT=8000
echo "服务地址: http://127.0.0.1:${PORT}"
echo "2 秒后自动打开浏览器，按 Ctrl+C 停止服务"
echo ""

# 延迟打开浏览器（等服务启动）
( sleep 2 && open "http://127.0.0.1:${PORT}" ) &

# 前台启动 Web 服务
.venv/bin/musicremix web --host 127.0.0.1 --port ${PORT}

echo ""
echo "服务已停止。按任意键关闭窗口..."
read -n 1
