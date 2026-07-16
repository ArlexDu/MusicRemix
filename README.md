# MusicRemix

歌曲换音色工具：给定一首原唱歌曲（歌手 A）和若干目标歌手 B 的参考歌曲，输出"歌手 B 演唱的原曲"——保留原曲旋律、歌词与伴奏，仅替换演唱者音色。

## 工作原理

```
歌手A歌曲 ─► [Demucs 分离] ─► 人声A + 伴奏
参考歌曲B ─► [Demucs 分离] ─► 人声B ─► [RVC 特征索引]
人声A + 索引B ─► [RVC 音色迁移] ─► 人声(B音色)
人声(B音色) + 伴奏 ─► [混音合成] ─► 输出歌曲
```

核心技术：
- **源分离**：Demucs v4 (htdemucs) 拆分人声与伴奏
- **音色建模**：ContentVec 编码 + Faiss 检索索引（检索式，换歌手只换索引）
- **音色迁移**：RVC 检索式转换，保留旋律/歌词/节奏

## 安装

需要 Python 3.9+（推荐 3.11）。音色迁移为计算密集型，强烈建议有 NVIDIA CUDA GPU 或 Apple Silicon (MPS)。

本工程使用独立的虚拟环境（`.venv`）：

```bash
# 创建虚拟环境（Python 3.11）
python3.11 -m venv .venv

# 基础安装（CLI 框架 + 音频工具）
.venv/bin/pip install -e .

# 安装源分离依赖（Demucs）
.venv/bin/pip install torch torchaudio demucs faiss-cpu

# 安装音色建模依赖（ContentVec 特征）
.venv/bin/pip install transformers

# 开发工具
.venv/bin/pip install -e ".[dev]"
```

> 国内网络可加镜像源加速：`-i https://pypi.tuna.tsinghua.edu.cn/simple`

## RVC 环境准备（音色迁移必需）

音色迁移这一步调用外部 RVC 推理环境（依赖解耦，使用官方 RVC 最可靠）。PyPI 上的 `rvc-python` 包已废弃不可用，因此本工程在 `models/rvc/` 内置了 RVC 官方仓库的 clone，并已完成初始化：

- ✅ 依赖已装（fairseq、torch、faiss、praat-parselmouth、torchcrepe、torchfcpe、ffmpeg 等）
- ✅ 预训练模型已下载（hubert_base.pt、rmvpe.pt、pretrained_v2/f0G48k.pth）
- ✅ `base_v2_48k.pth` 已构造为可推理模型（放于 `assets/weights/`，用于流程验证）

```bash
# 验证 RVC 环境就绪（应显示 infer_cli ✓, 2个模型）
musicremix info

# 用 base 模型验证推理链路（音色为中性 base，非特定歌手）
musicremix convert --vocal vocals.wav --model-name base_v2_48k.pth --output out.wav
```

### 关于音色模型（重要）

RVC 推理需要一个**训练完成的生成器模型**（含 config/weight 结构，放于 `assets/weights/`）。`base_v2_48k.pth` 是通用基础模型，可用于验证流程，但音色中性、不特定于任何歌手。

要真正换某歌手 B 的音色，需用 RVC 训练流程基于该歌手的参考人声训练一个模型：

```bash
cd models/rvc
source venv/bin/activate
# 1. 准备目标歌手干净人声（可用本工程的 musicremix separate 分离）
# 2. 按 RVC WebUI 训练流程：preprocess（hubert提特征）→ extract_f0 → train
# 3. 训练产出的 .pth 自动放于 assets/weights/
# 详见 models/rvc/README.md 的训练章节
```

如需指向其他位置的 RVC，用环境变量覆盖：

```bash
export MUSICREMIX_RVC_HOME=/path/to/rvc
export MUSICREMIX_RVC_PYTHON=/path/to/rvc/venv/bin/python
```

## 快速开始

```bash
# 一键换音色（--model-name 指定目标歌手的 .pth，放于 models/rvc/assets/weight_root/）
musicremix remix \
  --input songA.wav \
  --reference ref1.wav ref2.wav ref3.wav \
  --model-name singerB.pth \
  --output output/songA_by_B.wav

# 也可分步执行
musicremix separate --input songA.wav --workdir work/
musicremix model --reference ref1.wav ref2.wav --target-id singerB --workdir work/
musicremix convert --vocal work/vocals.wav --model-name singerB.pth --index work/singerB.index --output work/converted.wav
musicremix mix --vocal work/converted.wav --accompaniment work/accompaniment.wav --output output.wav

# 查看设备与缓存状态
musicremix info

# 启动 Web 界面（浏览器操作，推荐）
musicremix web
# 浏览器打开 http://127.0.0.1:8000
```

## Web 界面

提供可视化操作界面，无需命令行即可完成换音色：

```bash
musicremix web                  # 默认 127.0.0.1:8000
musicremix web --port 8080      # 自定义端口
musicremix web --host 0.0.0.0   # 局域网可访问
```

功能：
- **上传原唱歌曲**（歌手 A）与**多首参考歌曲**（歌手 B，拖拽或点击）
- **参数配置**：音色模型选择、检索强度、音高算法、半音偏移、音量
- **实时进度**：分离→建模→转换→混音 各阶段进度条
- **结果试听**：在线试听换音色结果及各中间产物（人声/伴奏/转换人声）
- **下载**：一键下载最终结果或中间产物

## 关键参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model-name` / `-m` | 必填 | 目标歌手 RVC 模型名（.pth，放于 models/rvc/assets/weights/） |
| `--index-rate` | 0.75 | RVC 检索注入强度，越高越像目标音色但越易出机械感 |
| `--f0-method` | rmvpe | 音高提取算法（rmvpe/crepe/pm），rmvpe 最稳 |
| `--pitch` | 0 | 半音偏移，男歌女唱可设 +12 |
| `--vocal-volume` | 1.0 | 合成人声音量 |
| `--accompaniment-volume` | 1.0 | 伴奏音量 |
| `--output-sr` | 44100 | 输出采样率 |
| `--format` | wav | 输出格式 |

> 详细调优见 `docs/tuning.md`。

## 已知效果边界

- 流行歌、人声清晰、音域适中时效果良好
- 极端高/低音区、气声段、和声残留处偶有音准瑕疵
- 首版为离线批处理，非实时；情感细节（颤音幅度、气声比例）会部分丢失
- 本地优先：用户音频不上传云端

## 许可证

MIT。但集成的第三方模型（Demucs / RVC）有各自许可证，商用前请核查 `docs/licenses.md`。
