from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

LABELS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprised")


@dataclass
class SpeechEmotion:
    available: bool = False
    label: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    margin: float | None = None
    normalized_entropy: float | None = None
    calibrated: bool = False
    reason: str = "disabled"


@dataclass
class EmotionState:
    status: str
    speech: SpeechEmotion
    reliable: bool = False
    label: str | None = None
    conflict: bool = False
    reason: str = ""


@dataclass
class DialoguePolicy:
    strategy: str
    guidance: str
    tts_instruction: str
    fixed_reply: str | None = None


@dataclass
class ASRResult:
    text: str
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResult:
    text: str
    usage: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False


@dataclass
class TTSResult:
    audio_path: str
    usage: dict[str, Any] = field(default_factory=dict)
    style_applied: bool = False


@dataclass
class TurnResult:
    trace_id: str
    session_id: str
    mode: str
    transcript: str
    reply: str
    emotion: EmotionState
    policy: DialoguePolicy
    audio_path: str | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    status: str = "ok"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
