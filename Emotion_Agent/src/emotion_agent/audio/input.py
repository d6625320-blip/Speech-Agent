from pathlib import Path


def require_audio(value: str) -> Path:
    return Path(value).expanduser().resolve()
