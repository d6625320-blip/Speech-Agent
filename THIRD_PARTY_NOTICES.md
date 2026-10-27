# 第三方资源说明

本仓库集成第三方库和模型接口，不重新分发数据集或模型权重，也不宣称这些预训练模型由本项目训练。用户应自行下载，并核对所用具体版本的许可证、模型卡及使用限制。

| 资源 | 上游 |
|---|---|
| FunASR / SenseVoice | [FunASR](https://github.com/modelscope/FunASR)、[SenseVoice](https://github.com/FunAudioLLM/SenseVoice) |
| emotion2vec | [官方项目](https://github.com/ddlBoJack/emotion2vec)、[模型卡](https://huggingface.co/emotion2vec/emotion2vec_plus_large) |
| Ollama / Qwen3.5 | [Ollama](https://github.com/ollama/ollama)、[所用模型](https://ollama.com/library/qwen3.5:9b) |
| Qwen3-TTS | [官方项目](https://github.com/QwenLM/Qwen3-TTS) |
| faster-qwen3-tts（可选） | [第三方加速项目](https://github.com/andimarafioti/faster-qwen3-tts) |
| qwen-tts-hf（可选） | [第三方兼容分发包](https://pypi.org/project/qwen-tts-hf/) |
| PyTorch、Transformers、FastAPI 等 | 对应安装包附带的许可证和项目文档 |

`qwen-tts-hf` 与原生 `qwen-tts` 提供相同的 Python 模块名，应使用不同环境；兼容包不应描述为 Qwen 官方另一个发行版。项目提供服务和音频传输适配，底层 CUDA Graph 等加速能力来自相应上游。

数据来源：

- MCAE-SPPS：[正式论文](https://doi.org/10.1038/s41597-026-06976-z)与[作者 OSF 仓库](https://doi.org/10.17605/OSF.IO/9JYZC)。
- 中文多说话人情绪朗读：[MagicHub 官方页](https://magichub.com/cn/datasets/multi-speaker-emotional-speech-dataset/)。当前页面注明非商业研究及开发用途；商业使用需另行授权。
- RAVDESS：[作者发布页](https://zenodo.org/records/1188976)，仅作为可选外部测试。

训练产生的权重是否可再分发，需要同时考虑基础模型与训练数据条款，本仓库没有附带或代替这类授权。论文、代码和数据的许可可能不同，不据论文的开放获取标志推断所有资源可任意使用。

项目中的录音任务文本和空模板仅用于功能验收；音频需用户自行录制。公开演示前应取得录音参与者同意并移除个人内容。
