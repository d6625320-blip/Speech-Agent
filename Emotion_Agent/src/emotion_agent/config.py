from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from .errors import ConfigurationError

APP_ROOT = Path(__file__).resolve().parents[2]


def load_env(path: Path) -> None:
    """Load simple KEY=value entries without executing or expanding their contents.

    Existing process environment wins. Supports blank lines, full-line comments,
    and paired quotes; inline comments/interpolation/export are intentionally absent.
    """
    if not path.is_file():
        return
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key.replace("_", "a").isalnum() or key[0].isdigit():
            raise ConfigurationError(f".env 第 {number} 行格式错误，请使用 KEY=value")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def read_yaml(path: Path) -> dict[str, Any]:
    try:
        result = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"无法读取 YAML：{path}") from exc
    if not isinstance(result, dict):
        raise ConfigurationError(f"配置根节点必须是映射：{path}")
    return result


class Settings:
    def __init__(self, data: dict[str, Any], root: Path = APP_ROOT):
        self.data = copy.deepcopy(data)
        self.root = root.resolve()
        self.validate()

    def path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    def validate(self) -> None:
        d = self.data
        if d.get("mode") not in {"text_only", "speech_label", "policy"}:
            raise ConfigurationError("mode 只支持 text_only / speech_label / policy")
        for section in ("audio", "speech", "dialogue", "tts", "network", "models"):
            if not isinstance(d.get(section), dict):
                raise ConfigurationError(f"缺少配置段：{section}")
        for section, keys in {
            "audio": ["sample_rate", "max_seconds", "min_seconds", "max_bytes"],
            "dialogue": ["max_history_turns", "max_text_chars", "max_prompt_chars", "max_reply_chars", "max_sessions", "session_ttl_seconds"],
            "network": ["timeout_seconds"],
            "tts": ["max_audio_bytes"],
        }.items():
            for key in keys:
                value = d[section].get(key)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                    raise ConfigurationError(f"{section}.{key} 必须大于零")
        if d["audio"]["sample_rate"] != 16000:
            raise ConfigurationError("当前语音专家固定使用 16000 Hz 输入")
        if d["audio"]["min_seconds"] >= d["audio"]["max_seconds"] or d["audio"]["max_seconds"] > 120:
            raise ConfigurationError("音频时长范围错误；当前最大支持 120 秒")
        for key in ("confidence_threshold", "margin_threshold", "max_normalized_entropy"):
            if not 0 <= d["speech"].get(key, -1) <= 1:
                raise ConfigurationError(f"speech.{key} 必须在 [0, 1] 范围内")
        if d["network"].get("retries") not in (0, 1, 2):
            raise ConfigurationError("network.retries 必须为 0、1 或 2")
        for section, providers in {"asr": {"funasr", "dashscope"}, "llm": {"ollama", "compatible"}, "tts": {"local_qwen", "qwen_audio", "qwen3"}}.items():
            if d["models"].get(section, {}).get("provider") not in providers:
                raise ConfigurationError(f"{section} provider 不支持")
        if d["models"]["tts"]["provider"] == "qwen3" and d["tts"].get("instruction_control"):
            raise ConfigurationError("qwen3-tts-flash 不支持指令控制，请关闭 tts.instruction_control")

    def cloud_url(self, suffix: str) -> str:
        base = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com").strip().rstrip("/")
        if urlparse(base).scheme != "https" or not urlparse(base).hostname:
            raise ConfigurationError("DASHSCOPE_BASE_URL 必须是 HTTPS 域名根地址")
        if urlparse(base).path not in ("", "/"):
            raise ConfigurationError("DASHSCOPE_BASE_URL 仅填写域名，不包含 /api/v1 等路径")
        return base + suffix


def load_settings(config: str | Path | None = None) -> Settings:
    load_env(APP_ROOT / ".env")
    path = Path(config).expanduser().resolve() if config else APP_ROOT / "configs/default.yaml"
    data = read_yaml(path)
    model_path = Path(data["models_config"])
    data["models"] = read_yaml(model_path if model_path.is_absolute() else APP_ROOT / model_path)
    for section, env_key in (("asr", "ASR_MODEL"), ("llm", "LLM_MODEL"), ("tts", "TTS_MODEL")):
        if os.getenv(env_key, "").strip():
            data["models"][section]["model"] = os.environ[env_key].strip()
    if os.getenv("TTS_VOICE", "").strip():
        data["models"]["tts"]["voice"] = os.environ["TTS_VOICE"].strip()
    return Settings(data)
