from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from ..errors import AudioError


@dataclass
class AudioInfo:
    path: Path
    duration_seconds: float
    rms: float
    silent: bool


def prepare_audio(source: Path, target: Path, config: dict) -> AudioInfo:
    if not source.is_file():
        raise AudioError("音频文件不存在")
    if source.stat().st_size > config["max_bytes"]:
        raise AudioError("音频文件过大，最大允许 10 MiB")
    try:
        with sf.SoundFile(str(source)) as handle:
            duration = handle.frames / handle.samplerate
            if not config["min_seconds"] <= duration <= config["max_seconds"]:
                raise AudioError(f"音频时长应为 {config['min_seconds']}–{config['max_seconds']} 秒")
            if handle.channels > 8 or not 8000 <= handle.samplerate <= 192000:
                raise AudioError("不支持的声道数或采样率")
            values = handle.read(dtype="float32", always_2d=True)
            rate = handle.samplerate
    except (RuntimeError, ValueError, sf.LibsndfileError) as exc:
        raise AudioError("无法解码音频，请上传 WAV、FLAC 或 OGG 文件") from exc
    if not np.isfinite(values).all():
        raise AudioError("音频包含非有限数值")
    values = values.mean(axis=1)
    if rate != config["sample_rate"]:
        divisor = gcd(rate, config["sample_rate"])
        values = resample_poly(values, config["sample_rate"] // divisor, rate // divisor)
    values = np.clip(values, -1, 1)
    rms = float(np.sqrt(np.mean(values.astype(np.float64) ** 2)))
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), values, config["sample_rate"], subtype="PCM_16")
    return AudioInfo(target, len(values) / config["sample_rate"], rms, rms < config["silence_rms"])
