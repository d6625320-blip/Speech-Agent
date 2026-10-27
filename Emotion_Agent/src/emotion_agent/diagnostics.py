from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import httpx

from .config import Settings


def doctor(settings: Settings, probe: bool = False) -> dict:
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    c = settings.data
    for package in ("numpy", "scipy", "soundfile", "yaml", "httpx", "fastapi", "uvicorn"):
        add(package, importlib.util.find_spec(package) is not None, "Python dependency")
    if c["speech"]["enabled"]:
        root = settings.path(c["speech"]["root"])
        for path in (root / c["speech"]["checkpoint"], root / c["speech"]["encoder"] / "model.pt", root / "src/mcae_large_final_common.py"):
            add("speech_file", path.is_file(), str(path))
    if c["models"]["asr"]["provider"] == "funasr":
        for key in ("model_dir", "vad_dir"):
            path = settings.path(c["models"]["asr"][key]) / "model.pt"
            add("asr_" + key, path.is_file(), str(path))
        add("funasr", importlib.util.find_spec("funasr") is not None, "Local ASR dependency")
    if c["models"]["tts"]["provider"] == "local_qwen":
        path = settings.path("models/" + c["models"]["tts"]["model"])
        add("tts_local_directory", (path / "config.json").is_file(), str(path))
    for component in ("asr", "llm", "tts"):
        provider = c["models"][component]["provider"]
        if provider in {"dashscope", "qwen_audio", "qwen3"} or (provider == "compatible" and not os.getenv("LLM_API_KEY")):
            add("cloud_api_key", bool(os.getenv("DASHSCOPE_API_KEY", "").strip()), "Set/unset only; value hidden")
    if probe:
        if importlib.util.find_spec("torch") and (c["speech"]["enabled"] or c["models"]["asr"]["provider"] == "funasr"):
            import torch

            cuda = torch.cuda.is_available()
            add("main_cuda", cuda, torch.cuda.get_device_name(0) if cuda else "CUDA unavailable in this environment")
        with httpx.Client(timeout=10, trust_env=False) as client:
            if c["models"]["llm"]["provider"] == "ollama":
                base = os.getenv("OLLAMA_BASE_URL", "").strip() or c["models"]["llm"]["base_url"]
                try:
                    response = client.get(base.rstrip("/") + "/api/tags")
                    response.raise_for_status()
                    names = {m["name"] for m in response.json()["models"]}
                    add("ollama_model", c["models"]["llm"]["model"] in names, "Available models: " + ", ".join(sorted(names)))
                except (httpx.HTTPError, ValueError, KeyError):
                    add("ollama_model", False, "Start Ollama and pull the configured model")
            if c["models"]["tts"]["provider"] == "local_qwen":
                base = os.getenv("LOCAL_TTS_BASE_URL", "").strip() or c["models"]["tts"]["base_url"]
                try:
                    response = client.get(base.rstrip("/") + "/health")
                    response.raise_for_status()
                    info = response.json()
                    add("local_tts", info.get("ready"), info.get("model", "unknown"))
                except (httpx.HTTPError, ValueError):
                    add("local_tts", False, "Start scripts/serve_local_tts.py in its isolated environment")
    return {"ready": all(check["ok"] for check in checks), "probe": probe, "checks": checks,
            "note": "File/dependency/service checks only; run one real audio turn on the target server for acceptance."}
