# Windows 部署

本文使用 Anaconda Prompt / CMD。假设将仓库放在 `C:\work\Speech_agent`，请替换为自己的目录，进入后再执行后续命令。参考目标为 Windows Server 2019 + RTX 3090；不保证同卡上所有模型始终同时驻留或达到固定延迟。

## 1. 主应用环境

```bat
conda create -n emotion-agent python=3.10 pip -y
conda activate emotion-agent
cd /d C:\work\Speech_agent
python -m pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
cd Emotion_Agent
python -m pip install -r requirements-server.txt
cd ..
```

安装 CUDA wheel 前核对驱动，参考 [PyTorch 官方版本说明](https://pytorch.org/get-started/previous-versions/)。`requirements-server.txt` 的 editable 安装要求临时进入 `Emotion_Agent`。如只运行替身测试，可安装基础包 `python -m pip install -e ./Emotion_Agent`，无需 FunASR 或真实权重。

复制配置模板（已有 `.env` 时不要覆盖）：

```bat
if not exist Emotion_Agent\.env copy Emotion_Agent\.env.example Emotion_Agent\.env
```

默认本机模型不需要云端 API Key。不要提交填写后的 `.env`。

## 2. 下载资源

```bat
python Emotion_Agent/scripts/download_models.py --component all
```

下载内容及分类头准备见 [数据与模型](DATA_AND_MODELS.md)。此命令不下载训练集，不训练分类头。

## 3. Ollama LLM

从 [Ollama 官方 Windows 页面](https://ollama.com/download/windows)安装后打开新 CMD：

```bat
ollama --version
ollama pull qwen3.5:9b
ollama list
```

若 Windows 应用已经提供服务，无需再运行 `serve`。便携版需要在单独 CMD 启动并保持运行：

```bat
set "OLLAMA_HOST=127.0.0.1:11434"
ollama serve
```

默认使用 `http://127.0.0.1:11434/api/chat`，关闭思考输出，上下文 4096，单轮最多 300 Token；配置见 `Emotion_Agent/configs/models.yaml`。若此端口已占用，请核实已有服务，或自行选择其他端口并在主应用 `.env` 设置 `OLLAMA_BASE_URL`。

## 4. 独立 TTS 环境

选择下面一种，两个包体系不要装到同一环境。

### 加速与分块生成

```bat
conda create -n emotion-tts-fast python=3.12 pip -y
conda activate emotion-tts-fast
cd /d C:\work\Speech_agent
python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r Emotion_Agent/requirements-tts-fast.txt
python Emotion_Agent/scripts/serve_local_tts.py --model-dir Emotion_Agent/models/Qwen3-TTS-12Hz-1.7B-CustomVoice --device cuda:0 --backend faster --chunk-size 8
```

此环境使用第三方 `faster-qwen3-tts` 及兼容分发包 `qwen-tts-hf`；它们不是本项目自己实现的底层加速算法。首次启动包含预热与 CUDA Graph 捕获，看到服务启动成功后再使用。版本、模型与驱动不兼容时需按实际错误定位。

### 原生整段合成

```bat
conda create -n emotion-tts python=3.12 pip -y
conda activate emotion-tts
cd /d C:\work\Speech_agent
python -m pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r Emotion_Agent/requirements-tts.txt
python Emotion_Agent/scripts/serve_local_tts.py --model-dir Emotion_Agent/models/Qwen3-TTS-12Hz-1.7B-CustomVoice --device cuda:0 --backend native
```

TTS 窗口保持运行。在另一个 CMD 检查：

```bat
curl.exe http://127.0.0.1:9880/health
```

`ready: true` 表示 TTS 服务已就绪，不代表整套对话系统已经验收。分块生成还需要 `streaming: true`。端口被占用时先检查已有 TTS，避免重复加载多个模型实例。

## 5. 启动网页

另开 Anaconda Prompt：

```bat
conda activate emotion-agent
cd /d C:\work\Speech_agent
python Emotion_Agent/scripts/run_demo.py doctor --probe
python Emotion_Agent/scripts/run_demo.py serve
```

浏览器打开 `http://127.0.0.1:8000`。`doctor` 检查文件、依赖和服务，不等于真实音频验收。缺少自训练分类头时，先使用：

```bat
python Emotion_Agent/scripts/run_demo.py --config Emotion_Agent/configs/no_speech_expert.yaml serve
```

勾选“生成语音回复”；使用加速服务时可再勾选“流式播放”。先发短文字，再上传一段正常录音，核对转写、输入回放、专家原始分数、策略和回复。

## 6. 麦克风与配置

录音使用打开网页的浏览器所在电脑的麦克风。`Requested device not found` 表示无可用输入设备；远程桌面内运行浏览器时，需要远程音频录制重定向。上传文件不依赖麦克风。

`localhost` 可用于浏览器录音；其他普通 HTTP 地址通常不满足安全上下文。本文只描述本机访问，不配置公网分享。

`configs/default.yaml` 以 `Emotion_Agent` 为路径基准；语音专家的 `root` 为 `../Speech_model`。修改模型路径后重启主应用使内存模型重新加载。TTS 和 LLM 服务需要按各自修改重启。

内存不足时先缩短回复、保持单请求；再考虑较小 LLM 或把 ASR 放到 CPU。0.6B TTS 与 1.7B 的表达能力不同，换用前核对是否支持指令控制。
