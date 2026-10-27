# Emotion Agent 应用

提供浏览器录音、文件上传、多轮对话、输入回放、语音情绪诊断和 TTS 分块播放。

完整安装步骤见 [Windows 部署](../docs/DEPLOYMENT_WINDOWS.md)，数据和模型准备见 [资源下载](../docs/DATA_AND_MODELS.md)。以下命令从仓库根目录执行：

```bat
python Emotion_Agent/scripts/run_demo.py doctor --probe
python Emotion_Agent/scripts/run_demo.py serve
```

分类头不随源码分发，需要自行训练或使用自己已有的兼容权重。没有分类头时使用 `--config Emotion_Agent/configs/no_speech_expert.yaml`；该配置不提供语音情绪识别。

| 文件 | 职责 |
|---|---|
| `src/emotion_agent/pipeline/runner.py` | 单轮推理、资源限制与结果保存 |
| `src/emotion_agent/emotion/speech.py` | 编码器和自训练分类头推理 |
| `src/emotion_agent/emotion/fusion.py` | 置信度门控与明确自述冲突检查 |
| `src/emotion_agent/dialogue/policy.py` | 对话策略和 TTS 表达指令 |
| `configs/prompts/dialogue_system.txt` | LLM 系统提示词 |
| `src/emotion_agent/tts/worker.py` | 独立 TTS 服务与加速后端 |
| `src/emotion_agent/tts/relay.py` | 分块音频中继和完整音频保存 |
| `src/emotion_agent/web/` | 录音、上传、回放及诊断页面 |

模型路径、门控阈值和会话限制位于 `configs/default.yaml`，各模块模型及服务地址位于 `configs/models.yaml`；相对路径以本目录为基准。

当前按轮处理，LLM 完成回复后再进行 TTS 分块生成，不包含实时全双工和打断。详见 [系统设计](docs/architecture.md)、[TTS 部署](docs/tts_streaming_windows.md)及[评测说明](../docs/EVALUATION.md)。
