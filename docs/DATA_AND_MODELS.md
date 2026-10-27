# 数据集与模型准备

所有数据集、模型权重、分类头及特征缓存均须用户自行准备，不随源码分发。下载时保留原资源的许可证及说明，不使用本仓库的源码许可替代模型或数据许可。

## 1. 预训练模型

| 资源 | 官方入口 | 默认本地位置 |
|---|---|---|
| SenseVoiceSmall | [ModelScope](https://modelscope.cn/models/iic/SenseVoiceSmall) / [上游项目](https://github.com/FunAudioLLM/SenseVoice) | `Emotion_Agent/models/SenseVoiceSmall/` |
| FSMN-VAD | [ModelScope](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch) | `Emotion_Agent/models/fsmn-vad/` |
| emotion2vec_plus_large | [ModelScope](https://modelscope.cn/models/iic/emotion2vec_plus_large) / [Hugging Face](https://huggingface.co/emotion2vec/emotion2vec_plus_large) | `Speech_model/models/pretrained/emotion2vec_plus_large/` |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice) / [上游项目](https://github.com/QwenLM/Qwen3-TTS) | `Emotion_Agent/models/Qwen3-TTS-12Hz-1.7B-CustomVoice/` |
| Ollama Qwen3.5-9B | [模型页](https://ollama.com/library/qwen3.5:9b) | Ollama 管理的模型目录 |

主环境安装后，在仓库根目录显式执行下载：

```bat
python Emotion_Agent/scripts/download_models.py --component all --list
python Emotion_Agent/scripts/download_models.py --component all
ollama pull qwen3.5:9b
```

`--list` 仅列出计划，不联网下载。也可分别指定 `--component asr`、`speech` 或 `tts`。下载脚本不会下载训练数据，也不会生成分类头。应下载完整目录，不能只拷贝 `model.pt` 或单个 safetensors 文件。

主程序使用这些本地路径。TTS worker 强制离线加载；Ollama 使用自己管理的权重。第三方库的缓存行为和网络访问还受其版本及配置影响，不把一次成功启动等同于完全离线审计。

## 2. 自训练分类头

默认配置期望：

```text
Speech_model/outputs/final/mcae_chinese_reading_joint_v1/final_model.pt
```

**本仓库没有提供该文件的公开下载地址，也没有打包作者本地权重。** 按 [训练复现](TRAINING.md)自行训练；或使用自己已获准使用、且由相同代码导出的 checkpoint。

checkpoint 必须包含分类头参数、七类标签顺序、embedding 均值/标准差及温度参数。原始 emotion2vec 权重不等于此分类头，不能直接改名替代。只加载可信来源的 PyTorch checkpoint。

已有 MCAE 基线头时，可修改 `Emotion_Agent/configs/default.yaml`：

```yaml
speech:
  checkpoint: outputs/final/mcae_large_class_balanced_ce/final_model.pt
```

以上展示需要修改的字段，其余 `speech` 配置保留。路径相对于 `Speech_model`。

## 3. MCAE-SPPS（训练基线）

- [正式论文与数据说明](https://doi.org/10.1038/s41597-026-06976-z)
- [作者提供的 OSF 数据仓库](https://doi.org/10.17605/OSF.IO/9JYZC)

自行下载音频与元数据，按官方说明解压情绪分类目录。当前导入配置期望以下布局；如原始文件名不同，修改 `Speech_model/configs/data_audit.yaml` 的元数据路径：

```text
Speech_model/data/raw/MCAE-SPPS/
├── Corpus information.xlsx
├── Rater information.xlsx
├── Anger/*.wav
├── Disgust/*.wav
├── Fear/*.wav
├── Happiness/*.wav
├── Neutral/*.wav
├── Sadness/*.wav
└── Surprise/*.wav
```

论文中的元数据版本可能使用 `sentence_corpus_information.xlsx` 等名称，以实际下载版本为准。导入脚本读取 `Filename`、`emotion`、`Actress` 等字段；审计还读取投票、句子和声学信息，评分者表读取 `actress` 与 `actresssex`。如果字段名称与版本不一致，需明确映射，不能通过伪造标签或忽略错误继续训练。

清单由 `prepare_mcae_manifest.py` 从这些元数据重新生成，不需要作者本机的 CSV。数据规模、标签和说话人仍由后续审计核对。不要把网页文章许可证直接当作数据集许可，使用前检查 OSF 的实际条款。

## 4. 中文多说话人情绪朗读（联合训练）

[Magic Data / MagicHub 官方数据页](https://magichub.com/cn/datasets/multi-speaker-emotional-speech-dataset/)

该数据为 10 位说话人、六类情绪、1,200 条朗读。下载通常需要登录平台。按平台实际协议使用；当前页面注明非商业学术研究与技术开发用途，商业使用另需授权。

代码使用的本地目录名为 `Chinese_Emotional_Reading_Audio_Dataset`：

```text
Emotion_Agent/data/Chinese_Emotional_Reading_Audio_Dataset/
├── Male/SPK0001/生气/SPK0001_1_S0001.wav
│   ... 其余男性说话人、类别
└── Female/SPK0006/生气/...
```

`reading_root` 必须直接包含 `Male/`、`Female/`。类别目录为 `生气、厌恶、恐惧、高兴、悲伤、惊讶`；实际文件名由清单脚本校验。目录可放到其他磁盘，再修改 `Speech_model/configs/chinese_reading_joint.yaml` 的 `paths.reading_root`。

## 5. RAVDESS（可选外部评测）

从 [作者发布的 Zenodo 页面](https://zenodo.org/records/1188976)自行下载 audio-only speech 部分，并解压为 `Speech_model/data/raw/RAVDESS/Actor_01/` 等目录。本项目的七类评估排除 `calm`，不使用歌曲或视频作为此项输入。

这是英语表演语音外部测试，不能替代中文真实对话测试。依照原资源许可使用，不随源码重新分发。

## 6. 自采测试音频

`Emotion_Agent/data/evaluation.example.jsonl` 只有手写文字示例。66 条测试模板是本项目编写的录音任务，没有附带音频或已标注真值。按 [评测说明](EVALUATION.md)自行录制，取得参与者同意，再在本机填写清单。
