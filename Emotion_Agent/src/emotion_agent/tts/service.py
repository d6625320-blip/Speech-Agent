from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from urllib.parse import urlparse

import httpx
import soundfile as sf

from ..config import Settings
from ..errors import ProviderError
from ..network import JSONClient, api_key
from ..schemas import TTSResult


class SpeechSynthesis:
    def __init__(self, settings: Settings, client: JSONClient):
        self.settings, self.client = settings, client

    def _validate_and_write(self, data: bytes, output: Path):
        if not data or len(data) > self.settings.data["tts"]["max_audio_bytes"]:
            raise ProviderError("TTS 音频为空或超过大小限制")
        try:
            with sf.SoundFile(io.BytesIO(data)) as audio:
                if audio.frames == 0 or audio.format != "WAV":
                    raise ValueError
        except (RuntimeError, ValueError) as exc:
            raise ProviderError("TTS 返回的内容不是有效 WAV 音频") from exc
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".part")
        temporary.write_bytes(data)
        temporary.replace(output)

    def synthesize(self, text: str, instruction: str, output: Path) -> TTSResult:
        c = self.settings.data["models"]["tts"]
        style = self.settings.data["tts"]["instruction_control"]
        if c["provider"] == "local_qwen":
            base = os.getenv("LOCAL_TTS_BASE_URL", "").strip() or c["base_url"]
            try:
                response = self.client.client.post(base.rstrip("/") + "/synthesize", json={
                    "text": text, "speaker": c["voice"], "instruction": instruction if style else "",
                })
            except (httpx.NetworkError, httpx.TimeoutException) as exc:
                raise ProviderError("本地 TTS 服务未启动或超时，请启动 scripts/serve_local_tts.py") from exc
            if response.is_error:
                raise ProviderError(f"本地 TTS 返回 HTTP {response.status_code}，请检查 TTS 服务日志")
            data, usage = response.content, {}
            style = style and response.headers.get("x-style-applied") == "true"
        else:
            if c["provider"] == "qwen_audio":
                request_input = {"text": text, "voice": c["voice"], "format": "wav", "sample_rate": c["sample_rate"]}
                if style:
                    request_input["instruction"] = instruction
                suffix = "/api/v1/services/audio/tts/SpeechSynthesizer"
            else:
                request_input = {"text": text, "voice": c["voice"], "language_type": "Chinese"}
                suffix = "/api/v1/services/aigc/multimodal-generation/generation"
            result = self.client.post(self.settings.cloud_url(suffix), {"model": c["model"], "input": request_input}, api_key())
            try:
                audio = result["output"]["audio"]
                if audio.get("url"):
                    url = audio["url"]
                    if urlparse(url).scheme != "https":
                        raise ProviderError("TTS 音频地址必须为 HTTPS")
                    # Never attach the API Authorization header to the download URL.
                    with self.client.client.stream("GET", url) as response:
                        response.raise_for_status()
                        chunks, size = [], 0
                        for chunk in response.iter_bytes():
                            size += len(chunk)
                            if size > self.settings.data["tts"]["max_audio_bytes"]:
                                raise ProviderError("TTS 下载超过大小限制")
                            chunks.append(chunk)
                    data = b"".join(chunks)
                else:
                    data = base64.b64decode(audio["data"], validate=True)
            except (KeyError, TypeError, ValueError, httpx.HTTPError) as exc:
                raise ProviderError("TTS 音频响应无效或下载失败") from exc
            usage = result.get("usage", {})
        self._validate_and_write(data, output)
        return TTSResult(str(output), usage, style)
