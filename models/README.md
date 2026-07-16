# models/ — 外部模型与引擎目录

本目录存放本工程集成的**外部模型引擎**（独立 clone 的仓库）及其预训练权重。每个子目录对应一个引擎。

## 目录结构

| 子目录 | 引擎 | 用途 | 状态 |
|--------|------|------|------|
| `rvc/` | [RVC WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | 歌唱音色迁移（检索式） | 已 clone，需按其 README 装依赖与下模型 |
| _未来_ `so-vits-svc/` | So-VITS-SVC | 歌唱音色迁移（微调式，高质量模式） | 规划中 |

## RVC 使用说明

`rvc/` 是 RVC 官方仓库的浅克隆。使用前需在其内部完成初始化：

```bash
cd models/rvc
# 1. 创建独立虚拟环境（避免依赖冲突）
python3.11 -m venv venv
source venv/bin/activate
# 2. 安装 RVC 依赖
pip install -r requirements.txt
# 3. 下载预训练模型（按 RVC 仓库 README 指引）
#    通常需 hubert_base.pth、pretrained 基础生成器等
```

完成后，回到本工程根目录，本工具会自动识别 `models/rvc/` 作为音色迁移引擎：

```bash
# 无需手动设置环境变量，默认即指向 models/rvc
musicremix info   # 应显示 "外部RVC环境: models/rvc (infer-cli ✓)"
```

如需指向其他位置的 RVC，用环境变量覆盖：

```bash
export MUSICREMIX_RVC_HOME=/path/to/your/rvc
export MUSICREMIX_RVC_PYTHON=/path/to/your/rvc/venv/bin/python
```

## 注意

- 本目录下的 clone 仓库与大权重文件（`.pth`/`.index`/`.pt` 等）已被 `.gitignore` 忽略，不纳入主仓库版本管理
- 各引擎的依赖相互独立，建议每个引擎用独立虚拟环境，避免冲突
