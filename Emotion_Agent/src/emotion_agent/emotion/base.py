from pathlib import Path
from typing import Protocol

from ..schemas import SpeechEmotion


class SpeechExpert(Protocol):
    def predict(self, audio: Path) -> SpeechEmotion: ...
