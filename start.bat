@echo off
chcp 65001 >nul
REM MusicRemix 一键启动脚本（Windows 双击运行）
REM 双击此文件即可启动 Web 服务并自动打开浏览器

cd /d "%~dp0"

echo ============================================
echo   MusicRemix · 歌曲换音色工具
echo ============================================
echo.

REM 检查虚拟环境
if not exist ".venv\Scripts\musicremix.exe" (
    echo [错误] 未找到 .venv\Scripts\musicremix.exe
    echo 请先运行 setup.bat 完成环境安装。
    echo.
    pause
    exit /b 1
)

set PORT=8000
echo 服务地址: http://127.0.0.1:%PORT%
echo 2 秒后自动打开浏览器，按 Ctrl+C 停止服务
echo.

REM 延迟 2 秒后打开浏览器（后台）
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://127.0.0.1:%PORT%"

REM 前台启动 Web 服务
.venv\Scripts\musicremix web --host 127.0.0.1 --port %PORT%

echo.
echo 服务已停止。按任意键关闭窗口...
pause >nul
