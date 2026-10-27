# 验证与评测

## 自动测试

```bat
python Emotion_Agent/scripts/run_tests.py
node --test Emotion_Agent/tests/web.test.cjs
python -m unittest discover -s Speech_model/tests -p "test_*.py"
```

应用测试使用替身 ASR、分类头、LLM 和 TTS；训练回归使用合成特征。它们验证流程、错误处理、采样和协议，不是模型质量成绩。Node.js 20 或以上用于网页测试，不是运行网页服务器的依赖。

## 真实样本

66 条功能验收方案见 [录音方案](../Emotion_Agent/docs/真人录音测试方案-66条.md)。模板在 `Emotion_Agent/data/recording_plan66/`；所有 `recorded` 初始为 false，人工标签为空，必须先录制、听评并修订实际转写。

该方案覆盖七类情绪、语义与语气冲突、弱情绪、音质变化、非语言声音和时长边界。它是功能验收设计，不能直接当作足量、独立的正式模型测试集。

自建 JSONL（音频路径相对于清单）：

```json
{"conversation_id":"demo01","audio":"audio/sample01.wav","reference":"今天的实验又失败了"}
```

录音和标注只保存在本机。不同说话人的划分、派生音频关系、环境与设备应记录清楚。

## 对话模式对照

```bat
python Emotion_Agent/scripts/evaluate.py --manifest Emotion_Agent/data/my_audio_eval.jsonl --modes text_only speech_label policy --no-tts
```

- D0 / `text_only`：ASR 后依据文本回应，不读取声学类别。
- D1 / `speech_label`：将原始声学候选作为参考，不使用置信度门控。
- D2 / `policy`：应用置信度、分差、熵和明确自述冲突规则。

报告包含 CER、模块 P50/P95、完成与降级情况，有人工期望策略时统计策略匹配。没有标注的指标为 null，不能将缺失字段当作零错误。

语音专家的 Macro-F1、混淆矩阵和拒判覆盖率须用真实标签另行核算；不能用“成功返回 HTTP 200”代替识别正确。门槛在开发集调节，测试集只用于报告。对话情绪适切性及有用性需要人工盲评。

## TTS 性能

```bat
python Emotion_Agent/scripts/benchmark_tts.py --help
python Emotion_Agent/scripts/benchmark_tts.py
python Emotion_Agent/scripts/benchmark_tts.py --stream
```

比较相同文本、音色、设备、版本及回复长度。区分冷启动和预热后稳态，记录首音延迟、总合成时间、音频时长和峰值显存。本项目 RTF 定义为合成耗时 / 音频时长，越小越快；不要与上游可能使用的倒数指标混用。

TTS 首块延迟与“用户停止说话到听见回复”的端到端延迟不同。分块播放改善等待体验，不等于 ASR、LLM 已实现同步流式推理。

仓库不附带作者的历史成绩或未经独立复核的性能承诺。发布自己的结果时注明样本量、来源、训练与测试划分、设备、依赖版本和测量方式。
