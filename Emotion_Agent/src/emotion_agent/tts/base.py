from pathlib import Path
from typing import Protocol

from ..schemas import TTSResult


class TTS(Protocol):
    def synthesize(self, text: str, instruction: str, output: Path) -> TTSResult: ...
