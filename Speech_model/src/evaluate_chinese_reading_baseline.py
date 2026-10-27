"""Evaluate the original MCAE-only head and optional joint head on identical cached features."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from config_utils import load_config, resolve_project_path
from chinese_reading_joint_utils import (
    PIPELINE_VERSION, READING_LABELS, encoder_signature, manifest_identity,
    require_matching_manifests, resolve_feature_path, sha256,
    validate_reading_rows, verify_reading_cache,
)
from mcae_large_final_common import LABELS, choose_device, load_final_head, softmax


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(truth: np.ndarray, probabilities: np.ndarray, rows: list[dict]) -> dict:
    predicted = probabilities.argmax(axis=1)
    target_ids = [LABELS.index(label) for label in READING_LABELS]
    matrix = confusion_matrix(truth, predicted, labels=list(range(len(LABELS))))
    confidence = probabilities.max(axis=1)
    disgust = LABELS.index("disgust")
    return {
        "samples": len(rows), "accuracy": float(accuracy_score(truth, predicted)),
        "six_class_macro_f1": float(f1_score(truth, predicted, labels=target_ids, average="macro", zero_division=0)),
        "classification_report_six_classes": classification_report(
            truth, predicted, labels=target_ids, target_names=list(READING_LABELS),
            output_dict=True, zero_division=0),
        "confusion_matrix_all_predictions": matrix.tolist(),
        "disgust_prediction_counts": {label: int(matrix[disgust, i]) for i, label in enumerate(LABELS)},
        "high_confidence_0_9": {"samples": int((confidence >= .9).sum()),
                                  "errors": int(((confidence >= .9) & (predicted != truth)).sum())},
        "per_speaker": {speaker: {
            "samples": sum(r["speaker_id"] == speaker for r in rows),
            "accuracy": float(np.mean([predicted[i] == truth[i] for i, r in enumerate(rows)
                                        if r["speaker_id"] == speaker]))}
            for speaker in sorted({r["speaker_id"] for r in rows})},
    }


def predict_checkpoint(path: Path, features: np.ndarray, device: torch.device) -> tuple[np.ndarray, dict]:
    model, checkpoint, mean, std, _ = load_final_head(path, device)
    if (features.shape[1] != mean.shape[1] or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError(f"Feature dimension or normalization mismatch: {path}")
    standardized = ((features - mean) / std).astype(np.float32)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(features), 256):
            logits = model(torch.from_numpy(standardized[start:start + 256]).to(device))
            outputs.append(logits.float().cpu().numpy())
    logits = np.concatenate(outputs)
    if not np.isfinite(logits).all():
        raise ValueError(f"Non-finite model outputs: {path}")
    # Both heads use raw softmax for a fair comparison; MCAE temperature is not applied.
    return softmax(logits), checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="原 MCAE 模型在中文长句上的基线及联合模型对照评估")
    parser.add_argument("--config", default="configs/chinese_reading_joint.yaml")
    parser.add_argument("--split", choices=("diagnostics", "train", "validation", "test", "all"),
                        default="diagnostics", help="默认只诊断 train+validation，不读取 test 特征")
    parser.add_argument("--original-checkpoint", default="outputs/final/mcae_large_class_balanced_ce/final_model.pt")
    parser.add_argument("--original-only", action="store_true", help="只评估原模型，不加载联合模型")
    parser.add_argument("--output-dir", help="相对于 Speech_model 的独立输出目录；默认按 split 分开")
    parser.add_argument("--device", default="cpu", help="分类头计算设备，默认 CPU，无需加载语音编码器")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    output_dir = resolve_project_path(args.output_dir or f"outputs/evaluation/mcae_original_chinese_reading/{args.split}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output already exists; select a new --output-dir: {output_dir}")
    all_rows = read_csv(resolve_project_path(paths["reading_feature_manifest"]))
    source_rows = read_csv(resolve_project_path(paths["reading_manifest"]))
    validate_reading_rows(source_rows, config)
    validate_reading_rows(all_rows, config)
    require_matching_manifests(source_rows, all_rows)
    selected_splits = ({"train", "validation"} if args.split == "diagnostics" else
                       {"train", "validation", "test"} if args.split == "all" else {args.split})
    rows = [r for r in all_rows if r["split"] in selected_splits]
    if not rows:
        raise ValueError(f"No samples for split {args.split}")
    print(f"split={args.split} samples={len(rows)} speakers={sorted({r['speaker_id'] for r in rows})}")
    print("Checking cached feature and encoder fingerprints; no encoder inference or training.", flush=True)
    fingerprint = encoder_signature(resolve_project_path(paths["model_dir"]))
    verify_reading_cache(rows, fingerprint, source_rows)
    arrays = [np.load(resolve_feature_path(r["feature_path"]), allow_pickle=False).astype(np.float32).reshape(-1)
              for r in rows]
    if len({a.size for a in arrays}) != 1 or not arrays[0].size:
        raise ValueError("Inconsistent or empty cached feature vectors")
    features = np.stack(arrays)
    if not np.isfinite(features).all():
        raise ValueError("Non-finite cached features")
    truth = np.asarray([int(r["label_id"]) for r in rows], dtype=np.int64)
    device = choose_device(args.device)
    original_path = resolve_project_path(args.original_checkpoint)
    original_p, _ = predict_checkpoint(original_path, features, device)
    original = metrics(truth, original_p, rows)
    report = {
        "protocol": "Original MCAE-only head on cached Chinese reading speech; paired joint comparison uses identical features.",
        "split": args.split, "samples": len(rows), "label_order": LABELS, "calibrated": False,
        "class_distribution": dict(Counter(r["label"] for r in rows)),
        "encoder_sha256": fingerprint, "reading_manifest_identity": manifest_identity(all_rows),
        "original_checkpoint": str(original_path), "original_checkpoint_sha256": sha256(original_path),
        "original": original,
        "note": "No target neutral samples. These are six-class metrics from seven-class outputs. "
                "Top-level mixed-split metrics are only descriptive; compare each split separately below. "
                "Joint train scores measure fitting; validation is used for model selection; test is diagnostic, previously inspected.",
    }
    joint_path = resolve_project_path(paths["output_dir"]) / "final_model.pt"
    joint_p = None
    if not args.original_only and joint_path.is_file():
        joint_p, checkpoint = predict_checkpoint(joint_path, features, device)
        if (checkpoint.get("pipeline_version") != PIPELINE_VERSION
                or checkpoint.get("reading_manifest_identity") != manifest_identity(all_rows)
                or checkpoint.get("encoder_sha256") != fingerprint):
            raise ValueError("Joint checkpoint does not match corrected pipeline/manifest/encoder; use --original-only")
        joint = metrics(truth, joint_p, rows)
        report.update({"joint_checkpoint": str(joint_path), "joint_checkpoint_sha256": sha256(joint_path),
                       "joint": joint, "joint_minus_original": {
                           key: joint[key] - original[key] for key in ("accuracy", "six_class_macro_f1")}})
        changes = {
            "original_wrong_joint_right": int(((original_p.argmax(1) != truth) & (joint_p.argmax(1) == truth)).sum()),
            "original_right_joint_wrong": int(((original_p.argmax(1) == truth) & (joint_p.argmax(1) != truth)).sum()),
        }
        report["paired_changes"] = changes
    elif not args.original_only:
        raise FileNotFoundError(f"Joint checkpoint not found: {joint_path}; use --original-only for baseline only")
    report["per_split"] = {}
    for split in sorted({r["split"] for r in rows}):
        mask = np.asarray([r["split"] == split for r in rows])
        subset = [r for r in rows if r["split"] == split]
        report["per_split"][split] = {"original": metrics(truth[mask], original_p[mask], subset)}
        if joint_p is not None:
            report["per_split"][split]["joint"] = metrics(truth[mask], joint_p[mask], subset)
    predictions = []
    for i, row in enumerate(rows):
        item = {"sample_id": row["sample_id"], "speaker_id": row["speaker_id"], "split": row["split"],
                "label": row["label"], "original_prediction": LABELS[int(original_p[i].argmax())],
                "original_confidence": float(original_p[i].max()),
                **{f"original_prob_{label}": float(original_p[i, j]) for j, label in enumerate(LABELS)}}
        if joint_p is not None:
            item.update({"joint_prediction": LABELS[int(joint_p[i].argmax())], "joint_confidence": float(joint_p[i].max()),
                         **{f"joint_prob_{label}": float(joint_p[i, j]) for j, label in enumerate(LABELS)}})
        predictions.append(item)
    comparison = []
    for split, models in report["per_split"].items():
        for name, value in models.items():
            comparison.append({"split": split, "model": name, "samples": value["samples"],
                               "accuracy": value["accuracy"], "six_class_macro_f1": value["six_class_macro_f1"],
                               **{f"{label}_f1": value["classification_report_six_classes"][label]["f1-score"]
                                  for label in READING_LABELS}})
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "predictions.csv", predictions)
    write_csv(output_dir / "comparison.csv", comparison)
    for split, models in report["per_split"].items():
        for name, value in models.items():
            matrix = value["confusion_matrix_all_predictions"]
            write_csv(output_dir / f"{split}_{name}_confusion_counts.csv", [
                {"true_label": label, **dict(zip(LABELS, matrix[i]))} for i, label in enumerate(LABELS)])
            write_csv(output_dir / f"{split}_{name}_class_metrics.csv", [
                {"label": label, **value["classification_report_six_classes"][label]} for label in READING_LABELS])
    for row in comparison:
        print(f"{row['split']} {row['model']}: samples={row['samples']} accuracy={row['accuracy']:.2%} "
              f"six_class_macro_f1={row['six_class_macro_f1']:.2%}")
    for split, models in report["per_split"].items():
        if "joint" in models:
            delta = {key: models['joint'][key] - models['original'][key]
                     for key in ("accuracy", "six_class_macro_f1")}
            models["joint_minus_original"] = delta
            print(f"{split} Joint - original (percentage points): " + json.dumps({
                key: round(100 * value, 3) for key, value in delta.items()}))
    # Persist split-level deltas as well as the full reports.
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results={output_dir}")


if __name__ == "__main__":
    main()
