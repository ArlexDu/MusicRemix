## ADDED Requirements

### Requirement: 人声与伴奏分离
系统 SHALL 接收一首完整歌曲音频文件，将其分离为纯人声（vocal）轨与伴奏（accompaniment）轨，并分别输出为独立的音频文件。

#### Scenario: 正常分离单首歌曲
- **WHEN** 用户提供一首常见格式（wav/mp3/flac）的歌曲文件并触发分离
- **THEN** 系统输出两个音频文件：人声轨与伴奏轨，二者时长与原曲一致

#### Scenario: 仅输出人声
- **WHEN** 用户指定只提取人声
- **THEN** 系统仅输出人声轨文件，不输出伴奏轨

### Requirement: 输入格式兼容
系统 SHALL 支持常见音频输入格式（至少 wav、mp3、flac），并对非标准采样率做内部重采样以适配模型。

#### Scenario: 输入 mp3 文件
- **WHEN** 用户提供一个 mp3 格式歌曲
- **THEN** 系统成功解码并完成分离，不因格式报错

#### Scenario: 输入非标准采样率
- **WHEN** 输入音频采样率为 44100Hz 以外的值
- **THEN** 系统内部重采样到模型所需采样率处理后，输出可配置为目标采样率

### Requirement: GPU 加速与 CPU 回退
系统 SHALL 自动检测可用计算设备（CUDA / Apple MPS / CPU），优先使用 GPU 加速；当无 GPU 时 SHALL 回退到 CPU 并完成处理。

#### Scenario: 有 CUDA GPU 环境
- **WHEN** 运行环境存在可用 CUDA GPU
- **THEN** 系统使用 GPU 进行分离，处理速度显著快于 CPU

#### Scenario: 无 GPU 环境
- **WHEN** 运行环境无可用 GPU
- **THEN** 系统回退到 CPU 完成分离，并向用户提示预计耗时增加

### Requirement: 分离质量与低残留
系统 SHALL 在分离后的人声轨中尽量去除伴奏与和声残留，保证人声干净以供后续音色迁移使用。

#### Scenario: 人声轨干净度
- **WHEN** 对一首含明显伴奏的歌曲完成分离
- **THEN** 输出人声轨中伴奏能量显著低于原曲混合，可被人耳判定为以人声为主
