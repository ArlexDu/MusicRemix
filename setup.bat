@echo off
chcp 65001 >nul
REM MusicRemix 环境一键初始化（Windows）
REM 新克隆仓库后双击或运行：setup.bat
REM 获取被 .gitignore 忽略的内容：.venv / models/rvc / 预训练模型
REM 幂等：重复运行安全，已完成的步骤会跳过。

setlocal enabledelayedexpansion
cd /d "%~dp0"

REM 定位 Python 3.11
set PYTHON=python
where py >nul 2>&1 && set PYTHON=py -3.11
%PYTHON% --version 2>nul | findstr /r "3\.11" >nul
if errorlevel 1 (
    echo [错误] 未找到 Python 3.11，请先安装。
    pause
    exit /b 1
)

set MIRROR=-i https://pypi.tuna.tsinghua.edu.cn/simple
set HF_BASE=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main

echo.
echo === 1/5 主工程虚拟环境 ^(.venv^) ===
if not exist ".venv\Scripts\python.exe" (
    %PYTHON% -m venv .venv
    .venv\Scripts\python -m pip install -q -U pip wheel setuptools cython %MIRROR%
)
.venv\Scripts\python -m pip install -q -e . %MIRROR%
.venv\Scripts\python -m pip install -q torch torchaudio demucs faiss-cpu %MIRROR%
if errorlevel 1 echo [警告] 部分 ml 依赖安装失败
.venv\Scripts\python -m pip install -q transformers %MIRROR%
.venv\Scripts\python -m pip install -q "numpy<2" %MIRROR%
echo [OK] 主工程依赖就绪

echo.
echo === 2/5 clone RVC 仓库 ^(.venv^) ===
if not exist "models\rvc\.git" (
    if not exist models mkdir models
    git clone --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI models\rvc
) else (
    echo [OK] models\rvc 已存在，跳过
)

echo.
echo === 3/5 RVC 虚拟环境与依赖 ===
cd models\rvc
if not exist "venv\Scripts\python.exe" (
    %PYTHON% -m venv venv
    venv\Scripts\python -m pip install -q -U pip wheel setuptools cython %MIRROR%
)
venv\Scripts\python -m pip install -q torch torchaudio %MIRROR%
venv\Scripts\python -m pip install -q "numpy<2" %MIRROR%
echo [INFO] 安装 fairseq（需编译，耗时数分钟；Windows 需 Visual Studio Build Tools）...
venv\Scripts\python -m pip install -q --no-build-isolation "fairseq @ git+https://github.com/One-sixth/fairseq.git"
if errorlevel 1 echo [警告] fairseq 安装失败，Windows 上需安装 Visual Studio Build Tools
venv\Scripts\python -m pip install -q faiss-cpu av praat-parselmouth "torchcrepe==0.0.23" torchfcpe "librosa==0.10.2" soundfile numba ffmpeg-python onnxruntime python-dotenv resampy "pyworld==0.3.2" %MIRROR%
venv\Scripts\python -m pip install -q "setuptools<81" %MIRROR%
venv\Scripts\python -m pip install -q imageio-ffmpeg %MIRROR%
REM ffmpeg: 用 copy（Windows 软链接需管理员权限）
for /f "delims=" %%i in ('venv\Scripts\python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"') do set FFMPEG_BIN=%%i
copy /y "%FFMPEG_BIN%" "venv\Scripts\ffmpeg.exe" >nul
echo [OK] RVC 依赖就绪

echo.
echo === 4/5 下载预训练模型 ===
if not exist assets\hubert mkdir assets\hubert
if not exist assets\rmvpe mkdir assets\rmvpe
if not exist assets\pretrained_v2 mkdir assets\pretrained_v2
if not exist assets\weights mkdir assets\weights
call :dl "%HF_BASE%/hubert_base.pt" "assets\hubert\hubert_base.pt"
call :dl "%HF_BASE%/rmvpe.pt" "assets\rmvpe\rmvpe.pt"
call :dl "%HF_BASE%/pretrained_v2/f0G48k.pth" "assets\pretrained_v2\f0G48k.pth"
echo [OK] 预训练模型就绪

echo.
echo === 5/5 构造 base 模型 ^& patch infer_cli ===
venv\Scripts\python ..\..\scripts\make_base_model.py .
venv\Scripts\python ..\..\scripts\patch_rvc_infer.py tools\infer_cli.py
echo [OK] base 模型与 patch 完成

cd ..\..
echo.
echo ============================================
echo   环境初始化完成！
echo ============================================
echo.
echo 下一步：
echo   双击 start.bat    启动 Web 界面
echo   或：.venv\Scripts\musicremix web
echo   或：.venv\Scripts\musicremix info   查看状态
echo.
pause
exit /b 0

:dl
if exist "%~2" (
    for %%F in ("%~2") do if %%~zF GTR 0 ( echo [OK] 已存在: %~2 & exit /b 0 )
)
echo   下载 %~2...
curl -L -s -o "%~2" "%~1"
exit /b 0
