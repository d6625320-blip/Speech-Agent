from pathlib import Path
from typing import Protocol

from ..schemas import ASRResult


class ASR(Protocol):
    def transcribe(self, audio: Path) -> ASRResult: ...
