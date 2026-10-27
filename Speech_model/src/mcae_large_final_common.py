"""MCAE 最终分类头训练、外部评估和部署推理共用代码。"""

from __future__ import annotations

import contextlib
import io
import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]


class ClassificationHead(nn.Module):
    """把冻结的 emotion2vec utterance embedding 映射到七种情感。"""

    def __init__(self, input_dim: int, hidden_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, len(LABELS)),
            )
        else:
            self.net = nn.Linear(input_dim, len(LABELS))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def choose_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    return device


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = values.astype(np.float64) / float(temperature)
    scaled -= scaled.max(axis=1, keepdims=True)
    exponential = np.exp(scaled)
    return exponential / exponential.sum(axis=1, keepdims=True)


def load_final_head(
    checkpoint_path: Path, device: torch.device
) -> tuple[ClassificationHead, dict[str, Any], np.ndarray, np.ndarray, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("labels") != LABELS:
        raise ValueError(f"checkpoint 标签顺序错误: {checkpoint.get('labels')}")
    model = ClassificationHead(
        int(checkpoint["input_dim"]),
        int(checkpoint.get("hidden_dim", 0)),
        float(checkpoint.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    mean = np.asarray(checkpoint["embedding_mean"], dtype=np.float32).reshape(1, -1)
    std = np.asarray(checkpoint["embedding_std"], dtype=np.float32).reshape(1, -1)
    temperature = float(checkpoint["temperature"])
    if mean.shape[1] != int(checkpoint["input_dim"]) or std.shape != mean.shape:
        raise ValueError("checkpoint 中 embedding 标准化参数的维度错误")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"无效 temperature: {temperature}")
    return model, checkpoint, mean, std, temperature


class Emotion2VecEmbeddingExtractor:
    """按 MCAE 特征提取时的同一官方接口生成 utterance embedding。"""

    def __init__(self, model_dir: Path, device: torch.device, quiet: bool = True):
        from funasr import AutoModel

        self.quiet = quiet
        with self._quiet_context():
            self.model = AutoModel(
                model=str(model_dir),
                device=str(device),
                disable_update=True,
                log_level="ERROR",
            )

    @contextlib.contextmanager
    def _quiet_context(self):
        """部署时隐藏第三方模型的进度条和普通日志，异常仍会正常抛出。"""
        if not self.quiet:
            yield
            return
        previous_disable = logging.root.manager.disable
        buffer = io.StringIO()
        logging.disable(logging.CRITICAL)
        try:
            with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                yield
        finally:
            logging.disable(previous_disable)

    def extract_one(self, audio_path: Path) -> np.ndarray:
        with self._quiet_context():
            results = self.model.generate(
                input=str(audio_path),
                granularity="utterance",
                extract_embedding=True,
            )
        if len(results) != 1 or "feats" not in results[0]:
            raise ValueError(f"无法提取 embedding: {audio_path}")
        embedding = np.asarray(results[0]["feats"], dtype=np.float32).reshape(1, -1)
        if embedding.size == 0 or not np.isfinite(embedding).all():
            raise ValueError(f"embedding 无效: {audio_path}")
        return embedding

    def extract_many(self, audio_paths: list[Path], batch_size: int) -> np.ndarray:
        keys = [f"utt{index:06d}" for index in range(len(audio_paths))]
        with tempfile.TemporaryDirectory(prefix="mcae_final_") as temp_dir:
            scp_path = Path(temp_dir) / "audio.scp"
            with scp_path.open("w", encoding="utf-8") as handle:
                for key, audio_path in zip(keys, audio_paths):
                    handle.write(f"{key} {audio_path}\n")
            with self._quiet_context():
                results = self.model.generate(
                    input=str(scp_path),
                    batch_size=batch_size,
                    granularity="utterance",
                    extract_embedding=True,
                )
        by_key = {str(result.get("key", "")): result for result in results}
        if len(results) != len(keys) or set(by_key) != set(keys):
            raise ValueError("批量 embedding 的输出 key 与输入不一致")
        if any("feats" not in by_key[key] for key in keys):
            raise ValueError("批量输出中缺少 embedding")
        embeddings = [np.asarray(by_key[key]["feats"], dtype=np.float32).reshape(-1) for key in keys]
        dimensions = {len(embedding) for embedding in embeddings}
        if len(dimensions) != 1 or not np.isfinite(np.stack(embeddings)).all():
            raise ValueError(f"批量 embedding 无效或维度不一致: {dimensions}")
        return np.stack(embeddings)
