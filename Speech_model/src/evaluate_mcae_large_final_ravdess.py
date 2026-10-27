"""在 RAVDESS 上仅执行一次最终外部七分类评估。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score

from config_utils import load_config, resolve_project_path
from mcae_large_final_common import (
    Emotion2VecEmbeddingExtractor,
    LABELS,
    choose_device,
    load_final_head,
    softmax,
)
from train_mcae_large_final import calibration_metrics, write_json


RAVDESS_LABELS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprised",
}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ravdess(root: Path) -> tuple[list[dict[str, Any]], int]:
    if not root.is_dir():
        raise FileNotFoundError(f"RAVDESS 目录不存在: {root}")
    rows: list[dict[str, Any]] = []
    calm_count = 0
    for audio_path in sorted(root.rglob("*.wav")):
        parts = audio_path.stem.split("-")
        if len(parts) != 7 or parts[0] != "03" or parts[1] != "01":
            raise ValueError(f"发现非 RAVDESS audio-only speech 文件: {audio_path}")
        label = RAVDESS_LABELS.get(parts[2])
        if label is None:
            raise ValueError(f"未知 RAVDESS 情感编码: {audio_path}")
        if label == "calm":
            calm_count += 1
            continue
        actor = int(parts[6])
        rows.append(
            {
                "sample_id": audio_path.stem,
                "audio_path": str(audio_path.resolve()),
                "true_label": label,
                "actor_id": actor,
                "gender": "male" if actor % 2 == 1 else "female",
                "intensity": "normal" if parts[3] == "01" else "strong",
                "statement": int(parts[4]),
                "repetition": int(parts[5]),
            }
        )
    if len(rows) != 1248 or calm_count != 192 or {row["actor_id"] for row in rows} != set(range(1, 25)):
        raise ValueError(f"RAVDESS 数量异常: seven_class={len(rows)} calm_excluded={calm_count}")
    return rows, calm_count


def metrics(probabilities: np.ndarray, truths: np.ndarray, bins: int) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    report = classification_report(
        truths, predictions, labels=list(range(len(LABELS))), target_names=LABELS,
        output_dict=True, zero_division=0,
    )
    return {
        "samples": len(truths),
        "accuracy": float(accuracy_score(truths, predictions)),
        "macro_f1": float(f1_score(truths, predictions, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "balanced_accuracy": float(recall_score(truths, predictions, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "classification_report": report,
        "true_distribution": dict(Counter(LABELS[index] for index in truths)),
        "prediction_distribution": dict(Counter(LABELS[index] for index in predictions)),
        "probability_quality": calibration_metrics(probabilities, truths, bins),
        "confusion_matrix": confusion_matrix(truths, predictions, labels=list(range(len(LABELS)))).tolist(),
    }


def save_confusion(output_dir: Path, matrix_value: list[list[int]]) -> None:
    matrix = np.asarray(matrix_value, dtype=np.int64)
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    write_csv(
        output_dir / "confusion_counts.csv",
        [{"true_label": label, **dict(zip(LABELS, values.tolist()))} for label, values in zip(LABELS, matrix)],
        ["true_label", *LABELS],
    )
    write_csv(
        output_dir / "confusion_row_normalized.csv",
        [{"true_label": label, **dict(zip(LABELS, values.tolist()))} for label, values in zip(LABELS, normalized)],
        ["true_label", *LABELS],
    )
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for values, filename, number_format in (
        (matrix, "confusion_counts.png", "d"),
        (normalized, "confusion_row_normalized.png", ".2f"),
    ):
        figure, axis = plt.subplots(figsize=(8, 7))
        image = axis.imshow(values, cmap="Blues", vmin=0)
        axis.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right")
        axis.set_yticks(range(len(LABELS)), LABELS)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        for row in range(len(LABELS)):
            for column in range(len(LABELS)):
                value = values[row, column]
                text = format(int(value), number_format) if number_format == "d" else format(float(value), number_format)
                axis.text(column, row, text, ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="最终模型 RAVDESS 单次外部评估")
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    inference = config["inference"]
    root = resolve_project_path(paths["ravdess_root"])
    checkpoint_path = resolve_project_path(paths["final_output_dir"]) / "final_model.pt"
    model_dir = resolve_project_path(paths["model_dir"])
    output_dir = resolve_project_path(paths["external_output_dir"])
    started_marker = output_dir / "EVALUATION_STARTED.json"
    complete_marker = output_dir / "EVALUATION_COMPLETE.json"
    if started_marker.exists() or complete_marker.exists():
        raise RuntimeError(f"RAVDESS 外部评估已启动或完成，禁止重复运行: {output_dir}")
    rows, calm_count = parse_ravdess(root)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"最终 checkpoint 不存在: {checkpoint_path}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"emotion2vec_plus_large 不存在: {model_dir}")
    print(f"validated RAVDESS seven_class={len(rows)} calm_excluded={calm_count}")
    if args.validate_only:
        return

    device = choose_device(str(inference["device"]))
    head, checkpoint, mean, std, temperature = load_final_head(checkpoint_path, device)
    extractor = Emotion2VecEmbeddingExtractor(model_dir, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "samples": len(rows),
        "rule": "This external test may be run only once and is never used for tuning.",
    }
    write_json(started_marker, started)

    embeddings = extractor.extract_many(
        [Path(row["audio_path"]) for row in rows], int(inference["embedding_batch_size"])
    )
    if embeddings.shape[1] != mean.shape[1]:
        raise ValueError(f"RAVDESS embedding 维度错误: {embeddings.shape[1]} != {mean.shape[1]}")
    standardized = ((embeddings - mean) / std).astype(np.float32)
    logits_parts: list[np.ndarray] = []
    head_batch_size = int(inference["head_batch_size"])
    with torch.inference_mode():
        for begin in range(0, len(standardized), head_batch_size):
            batch = torch.from_numpy(standardized[begin : begin + head_batch_size]).to(device)
            logits_parts.append(head(batch).float().cpu().numpy())
    logits = np.concatenate(logits_parts)
    raw_probabilities = softmax(logits)
    calibrated_probabilities = softmax(logits, temperature)
    truths = np.asarray([LABELS.index(row["true_label"]) for row in rows], dtype=np.int64)
    raw_metrics = metrics(raw_probabilities, truths, int(config["calibration"]["ece_bins"]))
    calibrated_metrics = metrics(calibrated_probabilities, truths, int(config["calibration"]["ece_bins"]))

    predictions: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        prediction_id = int(calibrated_probabilities[index].argmax())
        output = {
            **source,
            "pred_label": LABELS[prediction_id],
            "correct": int(prediction_id == truths[index]),
            "temperature": temperature,
            "raw_confidence": float(raw_probabilities[index].max()),
            "calibrated_confidence": float(calibrated_probabilities[index].max()),
            "calibrated_entropy": float(-(calibrated_probabilities[index] * np.log(np.clip(calibrated_probabilities[index], 1e-12, 1.0))).sum()),
        }
        output.update({f"logit_{label}": float(logits[index, label_id]) for label_id, label in enumerate(LABELS)})
        output.update({f"raw_prob_{label}": float(raw_probabilities[index, label_id]) for label_id, label in enumerate(LABELS)})
        output.update({f"prob_{label}": float(calibrated_probabilities[index, label_id]) for label_id, label in enumerate(LABELS)})
        predictions.append(output)
    write_csv(output_dir / "predictions.csv", predictions)
    write_csv(
        output_dir / "focus_errors.csv",
        [row for row in predictions if row["true_label"] in {"fear", "disgust", "surprised"} and not row["correct"]],
        list(predictions[0]),
    )
    save_confusion(output_dir, calibrated_metrics["confusion_matrix"])
    result = {
        "protocol": "single independent external evaluation; no tuning on RAVDESS",
        "checkpoint": str(checkpoint_path),
        "checkpoint_training_samples": checkpoint["training_samples"],
        "temperature_source": "MCAE six-fold LOSO out-of-fold logits",
        "temperature": temperature,
        "ravdess_calm_excluded": calm_count,
        "uncalibrated": raw_metrics,
        "calibrated": calibrated_metrics,
        "note": "Scalar temperature scaling changes probabilities, not argmax labels.",
    }
    write_json(output_dir / "metrics.json", result)
    write_json(
        complete_marker,
        {**started, "completed_utc": datetime.now(timezone.utc).isoformat(), "metrics_file": str(output_dir / "metrics.json")},
    )
    print(json.dumps({key: calibrated_metrics[key] for key in ("samples", "accuracy", "macro_f1", "balanced_accuracy")}, ensure_ascii=False, indent=2))
    print(f"result={output_dir}")


if __name__ == "__main__":
    main()
