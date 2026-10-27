"""仅在中文长句的保留测试说话人上评估联合训练模型。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from config_utils import load_config, resolve_project_path
from chinese_reading_joint_utils import (
    PIPELINE_VERSION, encoder_signature, manifest_identity, require_matching_manifests,
    resolve_feature_path, sha256, validate_reading_rows, verify_reading_cache,
)
from mcae_large_final_common import LABELS, choose_device, load_final_head, softmax


READING_LABELS = ("angry", "disgust", "fear", "happy", "sad", "surprised")
PROJECT_DIR = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 MCAE + 中文长句联合训练模型")
    parser.add_argument("--config", required=True)
    parser.add_argument("--use-calibration", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    output_dir = resolve_project_path(paths["test_output_dir"])
    metrics_path = output_dir / "metrics.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"为避免覆盖已有评估，已停止: {output_dir}")
    all_rows = read_csv(resolve_project_path(paths["reading_feature_manifest"]))
    source_rows = read_csv(resolve_project_path(paths["reading_manifest"]))
    validate_reading_rows(source_rows, config)
    validate_reading_rows(all_rows, config)
    require_matching_manifests(source_rows, all_rows)
    rows = [row for row in all_rows if row["split"] == "test"]
    if len(rows) != 240 or {row["label"] for row in rows} != set(READING_LABELS):
        raise ValueError("测试集应为 240 条、且包含六类非中性情绪")
    feature_paths = [resolve_feature_path(row["feature_path"]) for row in rows]
    if any(not path.is_file() for path in feature_paths):
        raise FileNotFoundError("测试集缺少 embedding；请先完成特征提取")
    device = choose_device(str(config["training"]["device"]))
    checkpoint_path = resolve_project_path(paths["output_dir"]) / "final_model.pt"
    model, checkpoint, mean, std, temperature = load_final_head(checkpoint_path, device)
    if checkpoint.get("pipeline_version") != PIPELINE_VERSION:
        raise ValueError("模型来自旧采样代码；请修正后重新训练")
    if checkpoint.get("reading_manifest_identity") != manifest_identity(all_rows):
        raise ValueError("测试清单与模型训练时使用的划分不一致")
    fingerprint = encoder_signature(resolve_project_path(paths["model_dir"]))
    if checkpoint.get("encoder_sha256") != fingerprint:
        raise ValueError("当前编码器与训练记录不一致")
    verify_reading_cache(rows, fingerprint, source_rows)
    if args.use_calibration and not checkpoint.get("calibration_fitted", False):
        raise ValueError("本轮没有拟合校准温度，请去掉 --use-calibration")
    features = np.stack([np.load(path, allow_pickle=False).astype(np.float32).reshape(-1) for path in feature_paths])
    if (features.shape[1] != mean.shape[1] or not np.isfinite(features).all()
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError("测试 embedding 维度与 checkpoint 不一致")
    features = ((features - mean) / std).astype(np.float32)
    import torch
    with torch.inference_mode():
        logits = model(torch.from_numpy(features).to(device)).float().cpu().numpy()
    probabilities = softmax(logits, temperature if args.use_calibration else 1.0)
    truth = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    predicted = probabilities.argmax(axis=1)
    target_ids = [LABELS.index(label) for label in READING_LABELS]
    prediction_rows: list[dict[str, Any]] = []
    for row, probabilities_one, prediction in zip(rows, probabilities, predicted):
        prediction_rows.append({
            **row, "prediction": LABELS[int(prediction)], "confidence": float(probabilities_one[prediction]),
            **{f"prob_{label}": float(probabilities_one[index]) for index, label in enumerate(LABELS)},
        })
    matrix = confusion_matrix(truth, predicted, labels=list(range(len(LABELS))))
    metrics = {
        "pipeline_version": PIPELINE_VERSION, "checkpoint_sha256": sha256(checkpoint_path),
        "protocol": "Speaker-disjoint diagnostic evaluation; these speakers were inspected in earlier experiments.",
        "samples": len(rows), "test_speakers": sorted({row["speaker_id"] for row in rows}),
        "calibrated": bool(args.use_calibration), "accuracy": float(accuracy_score(truth, predicted)),
        "six_class_macro_f1": float(f1_score(truth, predicted, labels=target_ids, average="macro", zero_division=0)),
        "classification_report_six_classes": classification_report(
            truth, predicted, labels=target_ids, target_names=list(READING_LABELS), output_dict=True, zero_division=0
        ),
        "label_order": LABELS, "confusion_matrix_all_predictions": matrix.tolist(),
        "note": "The target test corpus has no neutral recordings; do not interpret this as target-domain seven-class performance.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "predictions.csv", prediction_rows)
    write_csv(output_dir / "confusion_counts.csv", [
        {"true_label": LABELS[row], **{LABELS[column]: int(matrix[row, column]) for column in range(len(LABELS))}}
        for row in range(len(LABELS))
    ])
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"six_class_macro_f1={metrics['six_class_macro_f1']:.4f} accuracy={metrics['accuracy']:.4f}")
    print(f"metrics={metrics_path}")


if __name__ == "__main__":
    main()
