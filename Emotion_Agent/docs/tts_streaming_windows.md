# TTS 加速与分块播放

环境安装和启动命令见 [Windows 部署](../../docs/DEPLOYMENT_WINDOWS.md)。主环境 Python 3.10；TTS 使用独立 Python 3.12，选择加速或原生环境中的一种。

- `native`：`qwen-tts==0.1.1`，提供整段 WAV。
- `faster`：`faster-qwen3-tts==0.5.4`、`qwen-tts-hf==0.1.1.post1`，提供 CUDA Graph 后端及分块生成。

两个分发包提供同名 `qwen_tts` 模块，不能混装。加速实现来自 [第三方上游](https://github.com/andimarafioti/faster-qwen3-tts)，本项目负责服务和播放适配。没有把上游显卡测量当作本项目性能成绩。

在仓库根目录、对应 TTS 环境执行：

```bat
python Emotion_Agent/scripts/serve_local_tts.py --model-dir Emotion_Agent/models/Qwen3-TTS-12Hz-1.7B-CustomVoice --device cuda:0 --backend faster --chunk-size 8
```

首次启动包含预热与 CUDA Graph 捕获。`--chunk-size` 较小可能减少首块等待，但增加解码开销；是否改善应实测。保持服务窗口运行，再在另一 CMD 执行：

```bat
curl.exe http://127.0.0.1:9880/health
```

确认 `ready: true`、`backend: faster`、`streaming: true`。启动网页后勾选“生成语音回复”及“流式播放”；浏览器限制自动播放时点击允许播放。

服务 `/synthesize` 返回完整 WAV，`/synthesize/stream` 返回分块 NDJSON。浏览器通过主应用的 `/api/tts/{trace_id}/stream` 请求已经生成的助手回复，沿用主应用认证，不直接暴露 TTS 服务。

使用 [评测说明](../../docs/EVALUATION.md)中的基准命令对照整段和分块模式。RTF 为耗时 / 音频时长；首块音频时间是 TTS 阶段指标，不是从录音结束开始的端到端时间。

内存不足、驱动不匹配、端口已占用、模型目录不完整时按具体错误排查。不要重复启动多个占用同一端口或 GPU 的模型实例。恢复原生后端时，网页取消分块播放。
