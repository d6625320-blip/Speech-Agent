"""使用原始 emotion2vec_plus_large 完整测试 MCAE-SPPS。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss, recall_score

from predict_emotion2vec_plus_large import NATIVE_LABELS, SEVEN_LABELS, parse_result


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_DIR / "models" / "pretrained" / "emotion2vec_plus_large"
DEFAULT_MANIFEST = PROJECT_DIR / "data" / "manifests" / "mcae_spps_7class_manifest.csv"
DEFAULT_AUDIO_DIR = PROJECT_DIR / "data" / "raw" / "MCAE-SPPS"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "outputs" / "evaluation" / "mcae_emotion2vec_plus_large"


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, PROJECT_DIR / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def read_mcae(manifest_path: Path, audio_dir: Path) -> tuple[list[dict[str, str]], list[str]]:
    """读取清单，并将旧电脑中的绝对音频路径替换成服务器本地路径。"""
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        source_rows = list(reader)
    if not source_rows:
        raise ValueError(f"MCAE 清单为空: {manifest_path}")

    emotion_dirs = [
        path
        for path in audio_dir.iterdir()
        if path.is_dir() and path.name != "Codes and Raw Data"
    ]
    local_files: dict[str, Path] = {}
    for directory in emotion_dirs:
        for path in directory.glob("*.wav"):
            if path.name in local_files:
                raise ValueError(f"MCAE 中存在同名音频: {path.name}")
            local_files[path.name] = path

    rows: list[dict[str, str]] = []
    for source in source_rows:
        label = source["label"].strip().lower()
        if label not in SEVEN_LABELS:
            raise ValueError(f"MCAE 出现未知标签: {label}")
        original = Path(source["audio_path"])
        audio_path = original if original.is_file() else local_files.get(original.name)
        if audio_path is None:
            raise FileNotFoundError(f"找不到 MCAE 音频: {original.name}")
        row = dict(source)
        row["manifest_audio_path"] = source["audio_path"]
        row["audio_path"] = str(audio_path.resolve())
        row["label"] = label
        rows.append(row)

    if len(rows) != 6675:
        raise ValueError(f"MCAE 应包含 6675 条音频，实际读取 {len(rows)} 条")
    if {row["label"] for row in rows} != set(SEVEN_LABELS):
        raise ValueError("MCAE 没有完整覆盖七类情感")
    if "manifest_audio_path" not in fieldnames:
        fieldnames.append("manifest_audio_path")
    return rows, fieldnames


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def rectangular_confusion(
    true_labels: list[str], predicted_labels: list[str], columns: list[str]
) -> list[list[int]]:
    matrix = [[0 for _ in columns] for _ in SEVEN_LABELS]
    row_ids = {label: index for index, label in enumerate(SEVEN_LABELS)}
    column_ids = {label: index for index, label in enumerate(columns)}
    for truth, prediction in zip(true_labels, predicted_labels):
        matrix[row_ids[truth]][column_ids[prediction]] += 1
    return matrix


def matrix_rows(matrix: list[list[int]], columns: list[str]) -> list[dict[str, Any]]:
    return [
        {"true_label": label, **dict(zip(columns, values))}
        for label, values in zip(SEVEN_LABELS, matrix)
    ]


def save_matrix_png(matrix: list[list[int]], columns: list[str], path: Path, title: str) -> None:
    """如果服务器装有 matplotlib，同时保存可视化混淆矩阵。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，已跳过混淆矩阵 PNG；CSV 不受影响。")
        return

    values = np.asarray(matrix, dtype=np.int64)
    width = max(8.0, len(columns) * 1.05)
    fig, axis = plt.subplots(figsize=(width, 7.2))
    image = axis.imshow(values, cmap="Blues")
    axis.set_xticks(range(len(columns)), columns, rotation=35, ha="right")
    axis.set_yticks(range(len(SEVEN_LABELS)), SEVEN_LABELS)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(title)
    threshold = float(values.max()) / 2.0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def calculate_metrics(
    true_labels: list[str],
    predicted_labels: list[str],
    probabilities: np.ndarray | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "samples": len(true_labels),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(
            f1_score(
                true_labels,
                predicted_labels,
                labels=SEVEN_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "uar": float(
            recall_score(
                true_labels,
                predicted_labels,
                labels=SEVEN_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "prediction_counts": dict(Counter(predicted_labels)),
        "classification_report": classification_report(
            true_labels,
            predicted_labels,
            labels=SEVEN_LABELS,
            target_names=SEVEN_LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }
    if probabilities is not None:
        true_ids = np.asarray([SEVEN_LABELS.index(label) for label in true_labels])
        metrics["nll"] = float(log_loss(true_ids, probabilities, labels=range(len(SEVEN_LABELS))))
        metrics["mean_confidence"] = float(probabilities.max(axis=1).mean())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="完整测试原始 emotion2vec_plus_large 在 MCAE 上的表现")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--audio-dir", default=str(DEFAULT_AUDIO_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    model_dir = resolve_path(args.model_dir)
    manifest_path = resolve_path(args.manifest)
    audio_dir = resolve_path(args.audio_dir)
    output_dir = resolve_path(args.output_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"模型目录不存在: {model_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MCAE 清单不存在: {manifest_path}")
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"MCAE 音频目录不存在: {audio_dir}")
    if args.batch_size <= 0:
        raise ValueError("batch-size 必须大于零")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, manifest_fields = read_mcae(manifest_path, audio_dir)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法使用 CUDA")

    scp_path = output_dir / "mcae.scp"
    keys = [f"mcae{index:06d}" for index in range(len(rows))]
    with scp_path.open("w", encoding="utf-8") as handle:
        for key, row in zip(keys, rows):
            handle.write(f"{key} {row['audio_path']}\n")

    from funasr import AutoModel

    print(f"device={device} samples={len(rows)} batch_size={args.batch_size}")
    print(f"model={model_dir}")
    model = AutoModel(
        model=str(model_dir),
        device=device,
        disable_update=True,
        log_level="ERROR",
    )
    raw_results = model.generate(
        input=str(scp_path),
        batch_size=args.batch_size,
        granularity="utterance",
        extract_embedding=False,
    )
    result_by_key = {str(result.get("key", "")): result for result in raw_results}
    if len(result_by_key) != len(rows) or set(result_by_key) != set(keys):
        raise ValueError("模型输出数量或样本 key 与 MCAE 输入不一致")

    predictions: list[dict[str, Any]] = []
    true_labels: list[str] = []
    native_predictions: list[str] = []
    forced_predictions: list[str] = []
    forced_probabilities = np.empty((len(rows), len(SEVEN_LABELS)), dtype=np.float64)

    for index, (key, source) in enumerate(zip(keys, rows)):
        native, forced = parse_result(result_by_key[key])
        native_prediction = max(native, key=native.get)
        forced_prediction = max(forced, key=forced.get)
        true_label = source["label"]
        forced_probabilities[index] = [forced[label] for label in SEVEN_LABELS]

        output: dict[str, Any] = dict(source)
        output.update(
            {
                "native_prediction": native_prediction,
                "native_correct": int(native_prediction == true_label),
                "native_confidence": max(native.values()),
                "forced7_prediction": forced_prediction,
                "forced7_correct": int(forced_prediction == true_label),
                "forced7_confidence": max(forced.values()),
            }
        )
        output.update({f"native_prob_{label}": native[label] for label in NATIVE_LABELS})
        output.update({f"forced7_prob_{label}": forced[label] for label in SEVEN_LABELS})
        predictions.append(output)
        true_labels.append(true_label)
        native_predictions.append(native_prediction)
        forced_predictions.append(forced_prediction)

    prediction_fields = [
        *manifest_fields,
        "native_prediction",
        "native_correct",
        "native_confidence",
        "forced7_prediction",
        "forced7_correct",
        "forced7_confidence",
        *[f"native_prob_{label}" for label in NATIVE_LABELS],
        *[f"forced7_prob_{label}" for label in SEVEN_LABELS],
    ]
    write_csv(output_dir / "predictions.csv", predictions, prediction_fields)

    forced_matrix = rectangular_confusion(true_labels, forced_predictions, SEVEN_LABELS)
    native_matrix = rectangular_confusion(true_labels, native_predictions, NATIVE_LABELS)
    write_csv(
        output_dir / "confusion_matrix_forced7.csv",
        matrix_rows(forced_matrix, SEVEN_LABELS),
        ["true_label", *SEVEN_LABELS],
    )
    write_csv(
        output_dir / "confusion_matrix_native9.csv",
        matrix_rows(native_matrix, NATIVE_LABELS),
        ["true_label", *NATIVE_LABELS],
    )
    save_matrix_png(
        forced_matrix,
        SEVEN_LABELS,
        output_dir / "confusion_matrix_forced7.png",
        "emotion2vec_plus_large on MCAE (forced 7-class)",
    )
    save_matrix_png(
        native_matrix,
        NATIVE_LABELS,
        output_dir / "confusion_matrix_native9.png",
        "emotion2vec_plus_large on MCAE (native 9-class)",
    )

    metrics = {
        "model": "emotion2vec_plus_large",
        "model_dir": str(model_dir),
        "dataset": "MCAE-SPPS complete set",
        "samples": len(rows),
        "true_label_counts": dict(Counter(true_labels)),
        "device": device,
        "batch_size": args.batch_size,
        "native9": {
            **calculate_metrics(true_labels, native_predictions),
            "other_unknown_prediction_rate": float(
                sum(label in {"other", "unknown"} for label in native_predictions) / len(rows)
            ),
        },
        "forced7": calculate_metrics(
            true_labels,
            forced_predictions,
            forced_probabilities,
        ),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    native = metrics["native9"]
    forced = metrics["forced7"]
    print(
        "native9: "
        f"accuracy={native['accuracy']:.4f} macro_f1={native['macro_f1']:.4f} "
        f"uar={native['uar']:.4f} other_unknown={native['other_unknown_prediction_rate']:.4f}"
    )
    print(
        "forced7: "
        f"accuracy={forced['accuracy']:.4f} macro_f1={forced['macro_f1']:.4f} "
        f"uar={forced['uar']:.4f}"
    )
    print(f"结果已保存: {output_dir}")


if __name__ == "__main__":
    main()
