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

需要 Python 3.11（音色迁移为计算密集型，强烈建议有 NVIDIA CUDA GPU 或 Apple Silicon MPS）。支持 macOS 与 Windows。

> **关于被忽略的文件**：`.venv`、`models/rvc`（RVC 仓库 clone + 其依赖 + 预训练模型）、所有 `.pth/.pt` 大模型文件都被 `.gitignore` 忽略，不纳入仓库。新克隆后运行初始化脚本即可自动获取全部内容。

### 方式一：一键初始化（推荐，新克隆必用）

**macOS / Linux：**
```bash
git clone https://github.com/ArlexDu/MusicRemix.git
cd MusicRemix
bash setup.sh
```

**Windows：**
```bat
git clone https://github.com/ArlexDu/MusicRemix.git
cd MusicRemix
setup.bat
```

初始化脚本（`setup.sh` / `setup.bat`）会自动完成（幂等，可重复运行）：
1. 创建主工程虚拟环境 `.venv`，安装 torch/demucs/faiss/transformers 等依赖
2. clone RVC 官方仓库到 `models/rvc`
3. 创建 RVC 独立虚拟环境 `models/rvc/venv`，安装 fairseq/RVC 依赖、ffmpeg
4. 下载预训练模型（hubert_base.pt、rmvpe.pt、f0G48k.pth，约 400MB）
5. 构造可推理的 `base_v2_48k.pth` 并 patch RVC 兼容 torch≥2.6

> 国内网络已默认使用清华 PyPI 镜像加速。fairseq 需源码编译，耗时数分钟。
> **Windows 注意**：fairseq/pyworld 编译需 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)（含 C++ 桌面开发）。若编译失败，安装 Build Tools 后重跑 `setup.bat`。

### 一键启动

初始化完成后，双击启动脚本即可打开 Web 界面：
- **macOS**：双击 `start.command`
- **Windows**：双击 `start.bat`

或命令行：`.venv/bin/musicremix web`（macOS）/ `.venv\Scripts\musicremix web`（Windows）

完成后验证：
```bash
.venv/bin/musicremix info   # 应显示：内嵌依赖全 ✓，外部RVC环境 infer_cli ✓, 2个模型
```

### 方式二：手动安装（自定义场景）

```bash
# 主工程虚拟环境
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pip install torch torchaudio demucs faiss-cpu transformers "numpy<2"

# RVC 环境
git clone --depth 1 https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI models/rvc
cd models/rvc
python3.11 -m venv venv
# 详见 setup.sh 中 RVC 部分，或运行 bash setup.sh 仅补全 RVC
```

## RVC 环境说明（音色迁移必需）

音色迁移调用外部 RVC 推理环境（`models/rvc/`，由 `setup.sh` 自动准备）。环境就绪后包含：
- RVC 依赖（fairseq、torch、faiss、praat-parselmouth、torchcrepe、ffmpeg 等）
- 预训练模型（hubert_base.pt、rmvpe.pt、f0G48k.pth）
- `base_v2_48k.pth` 可推理模型（放于 `assets/weights/`，用于流程验证）

```bash
# 验证 RVC 环境就绪
musicremix info   # 应显示 infer_cli ✓, 2个模型

# 用 base 模型验证推理链路（音色为中性 base，非特定歌手）
musicremix convert --vocal vocals.wav --model-name base_v2_48k.pth --output out.wav
```

### 关于音色模型（重要）

RVC 推理需要一个**训练完成的生成器模型**（含 config/weight 结构，放于 `assets/weights/`）。`base_v2_48k.pth` 是通用基础模型，可验证流程，但音色中性。要真正换某歌手 B 的音色，需用 RVC 训练流程基于该歌手参考人声训练模型：

```bash
cd models/rvc && source venv/bin/activate
# 1. 准备目标歌手干净人声（可用 musicremix separate 分离）
# 2. 按 RVC WebUI 训练流程：preprocess → extract_f0 → train
# 3. 训练产出的 .pth 自动放于 assets/weights/，即可用于 convert/remix
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
