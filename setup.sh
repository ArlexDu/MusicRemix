#!/bin/bash
# MusicRemix 环境一键初始化
# 新克隆仓库后运行：bash setup.sh
# 完成后即可 musicremix web 或双击 start.command 使用
#
# 获取被 .gitignore 忽略的内容：
#   .venv              —— Python 虚拟环境（主工程）
#   models/rvc/        —— RVC 官方仓库 clone + 其 venv + 预训练模型
#   各 .pth/.pt 模型    —— 从 HuggingFace 下载
#
# 幂等：重复运行安全，已完成的步骤会跳过。

set -e
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3.11}
MIRROR="-i https://pypi.tuna.tsinghua.edu.cn/simple"
HF_BASE="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"

log() { echo -e "\n\033[1;36m▶ $1\033[0m"; }
ok()  { echo -e "\033[1;32m✓ $1\033[0m"; }

# 检查 python3.11
if ! command -v $PYTHON >/dev/null 2>&1; then
    echo "[错误] 未找到 $PYTHON。请先安装 Python 3.11。"
    exit 1
fi

# ============ 1. 主工程虚拟环境 ============
log "1/5  主工程虚拟环境 (.venv)"
if [ ! -d ".venv" ]; then
    $PYTHON -m venv .venv
    .venv/bin/pip install -q -U pip wheel setuptools cython $MIRROR
fi
# 安装/更新依赖
.venv/bin/pip install -q -e . $MIRROR
.venv/bin/pip install -q torch torchaudio demucs faiss-cpu $MIRROR || true
.venv/bin/pip install -q transformers $MIRROR || true
# numpy<2：faiss-cpu 在 numpy 2.x 下 ABI 冲突
.venv/bin/pip install -q "numpy<2" $MIRROR
ok "主工程依赖就绪"

# ============ 2. RVC 仓库 clone ============
log "2/5  RVC 官方仓库 (models/rvc)"
if [ ! -d "models/rvc/.git" ]; then
    mkdir -p models
    git clone --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI models/rvc
else
    ok "models/rvc 已存在，跳过 clone"
fi

# ============ 3. RVC 虚拟环境与依赖 ============
log "3/5  RVC 虚拟环境与依赖 (models/rvc/venv)"
cd models/rvc
if [ ! -d "venv" ]; then
    $PYTHON -m venv venv
    venv/bin/pip install -q -U pip wheel setuptools cython $MIRROR
fi
# torch + numpy（fairseq 构建需要）
venv/bin/pip install -q torch torchaudio $MIRROR || true
venv/bin/pip install -q "numpy<2" $MIRROR
# fairseq（One-sixth fork，需编译，耗时数分钟）
venv/bin/pip install -q --no-build-isolation "fairseq @ git+https://github.com/One-sixth/fairseq.git" || echo "[警告] fairseq 安装失败，重试或见 README"
# RVC 推理核心依赖
venv/bin/pip install -q faiss-cpu av praat-parselmouth "torchcrepe==0.0.23" torchfcpe \
    "librosa==0.10.2" soundfile numba ffmpeg-python onnxruntime python-dotenv resampy \
    "pyworld==0.3.2" $MIRROR
# setuptools<81：pyworld 依赖 pkg_resources（setuptools>=81 已移除）
venv/bin/pip install -q "setuptools<81" $MIRROR
# 训练所需：tensorboard + matplotlib<3.8（RVC 用了 tostring_rgb 旧 API）
venv/bin/pip install -q tensorboard tensorboardX "matplotlib<3.8" $MIRROR
# ffmpeg 二进制（RVC load_audio 依赖）
venv/bin/pip install -q imageio-ffmpeg $MIRROR
FFMPEG_BIN=$(venv/bin/python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
ln -sf "$FFMPEG_BIN" venv/bin/ffmpeg
ok "RVC 依赖就绪"

# ============ 4. 预训练模型下载 ============
log "4/5  预训练模型下载"
mkdir -p assets/hubert assets/rmvpe assets/pretrained_v2 assets/weights
dl() {  # dl <url> <out>
    if [ -s "$2" ]; then ok "已存在: $2"; else echo "  下载 $2..."; curl -L -s -o "$2" "$1"; fi
}
dl "$HF_BASE/hubert_base.pt" assets/hubert/hubert_base.pt
dl "$HF_BASE/rmvpe.pt" assets/rmvpe/rmvpe.pt
dl "$HF_BASE/pretrained_v2/f0G48k.pth" assets/pretrained_v2/f0G48k.pth
ok "预训练模型就绪"

# ============ 5. 构造 base 模型 + patch infer_cli ============
log "5/5  构造可推理 base 模型 & patch infer_cli"
venv/bin/python ../../scripts/make_base_model.py .
venv/bin/python ../../scripts/patch_rvc_infer.py tools/infer_cli.py
# 训练用的 extract_feature_print.py 也需 patch（加载 hubert 需 weights_only=False）
venv/bin/python ../../scripts/patch_rvc_infer.py infer/modules/train/extract_feature_print.py
ok "base 模型与 patch 完成"

cd ../..

# ============ 完成 ============
echo ""
ok "============================================"
ok "  环境初始化完成！"
ok "============================================"
echo ""
echo "下一步："
echo "  双击 start.command    启动 Web 界面"
echo "  或：.venv/bin/musicremix web"
echo "  或：.venv/bin/musicremix info   查看状态"
echo ""
echo "浏览器打开 http://127.0.0.1:8000 即可使用。"
