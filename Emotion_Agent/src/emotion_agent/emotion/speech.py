from __future__ import annotations

import importlib.util
from pathlib import Path
from threading import Lock

import numpy as np

from ..config import Settings
from ..errors import ConfigurationError
from ..schemas import LABELS, SpeechEmotion


class LocalSpeechExpert:
    """Reuse the retained encoder, head and normalization without retraining."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = Lock()
        self.loaded = False

    def _load(self):
        import torch

        c = self.settings.data["speech"]
        root = self.settings.path(c["root"])
        module_path = root / "src/mcae_large_final_common.py"
        checkpoint = root / c["checkpoint"]
        encoder = root / c["encoder"]
        for path in (module_path, checkpoint, encoder / "model.pt"):
            if not path.is_file():
                raise ConfigurationError(f"语音专家文件不存在：{path}")
        spec = importlib.util.spec_from_file_location("_retained_speech_common", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if tuple(module.LABELS) != LABELS:
            raise ConfigurationError("已有语音专家与应用的标签顺序不一致")
        self.common, self.torch = module, torch
        self.device = module.choose_device(c["device"])
        self.head, _, self.mean, self.std, self.temperature = module.load_final_head(checkpoint, self.device)
        self.extractor = module.Emotion2VecEmbeddingExtractor(encoder, self.device, quiet=False)
        self.loaded = True

    def predict(self, audio: Path) -> SpeechEmotion:
        if not self.settings.data["speech"]["enabled"]:
            return SpeechEmotion()
        with self.lock:
            if not self.loaded:
                self._load()
            embedding = self.extractor.extract_one(audio)
            if embedding.shape != self.mean.shape or not np.isfinite(embedding).all():
                raise ConfigurationError("语音特征维度或数值异常")
            feature = self.torch.from_numpy(((embedding - self.mean) / self.std).astype(np.float32)).to(self.device)
            with self.torch.inference_mode():
                logits = self.head(feature).float().cpu().numpy()
            calibrated = self.settings.data["speech"]["use_calibration"]
            p = self.common.softmax(logits, self.temperature if calibrated else 1.0)[0]
        order = np.argsort(p)[::-1]
        entropy = float(-(p * np.log(np.clip(p, 1e-12, 1))).sum() / np.log(len(LABELS)))
        return SpeechEmotion(
            available=True, label=LABELS[int(order[0])],
            probabilities=dict(zip(LABELS, map(float, p))), confidence=float(p[order[0]]),
            margin=float(p[order[0]] - p[order[1]]), normalized_entropy=entropy,
            calibrated=calibrated, reason="model_prediction",
        )
