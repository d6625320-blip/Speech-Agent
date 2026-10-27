from __future__ import annotations

import base64
import re
from pathlib import Path
from threading import Lock

from ..config import Settings
from ..errors import ConfigurationError, ProviderError
from ..network import JSONClient, api_key
from ..schemas import ASRResult


class SenseVoiceASR:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.config = settings.data["models"]["asr"]
        self.model = None
        self.lock = Lock()

    def transcribe(self, audio: Path) -> ASRResult:
        with self.lock:
            if self.model is None:
                model_dir = self.settings.path(self.config["model_dir"])
                vad_dir = self.settings.path(self.config["vad_dir"])
                if not (model_dir / "model.pt").is_file() or not (vad_dir / "model.pt").is_file():
                    raise ConfigurationError("ASR/VAD 权重未准备，请先运行 scripts/download_models.py --component asr")
                try:
                    from funasr import AutoModel
                except ImportError as exc:
                    raise ConfigurationError("本地 ASR 需要 funasr，请安装主环境 requirements-server.txt") from exc
                self.model = AutoModel(
                    model=str(model_dir), vad_model=str(vad_dir),
                    device=self.config["device"], disable_update=True,
                    vad_kwargs={"max_single_segment_time": 30000},
                )
            results = self.model.generate(
                input=str(audio), cache={}, language=self.config.get("language", "auto"),
                use_itn=True, batch_size_s=60, merge_vad=True, merge_length_s=15,
            )
        # SenseVoice's own emotion tags are discarded; emotion2vec remains the expert.
        text = "".join(re.sub(r"<\|[^|]*\|>", "", str(row.get("text", ""))) for row in results)
        return ASRResult(text.strip())


class CloudASR:
    def __init__(self, settings: Settings, client: JSONClient):
        self.settings, self.client = settings, client

    def transcribe(self, audio: Path) -> ASRResult:
        config = self.settings.data["models"]["asr"]
        encoded = base64.b64encode(audio.read_bytes()).decode("ascii")
        if len(encoded) > 10 * 1024 * 1024:
            raise ProviderError("ASR Base64 输入超过接口限制，请缩短音频")
        payload = {
            "model": config["model"], "stream": False,
            "messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": {
                "data": "data:audio/wav;base64," + encoded,
            }}]}],
            "asr_options": {"enable_itn": config.get("enable_itn", False)},
        }
        if config.get("language"):
            payload["asr_options"]["language"] = config["language"]
        data = self.client.post(self.settings.cloud_url("/compatible-mode/v1/chat/completions"), payload, api_key())
        try:
            text = data["choices"][0]["message"]["content"]
            if not isinstance(text, str):
                raise TypeError
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("ASR 响应缺少识别文本") from exc
        return ASRResult(text.strip(), data.get("usage", {}))
