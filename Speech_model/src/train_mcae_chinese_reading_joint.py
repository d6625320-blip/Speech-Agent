"""冻结 emotion2vec encoder，联合 MCAE 与中文长句数据训练新的七分类头。"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from config_utils import load_config, resolve_project_path
from chinese_reading_joint_utils import (
    PIPELINE_VERSION, build_sampling_weights, encoder_signature, manifest_identity,
    require_matching_manifests, reset_results, resolve_feature_path, sampling_summary,
    validate_reading_rows, verify_reading_cache,
)
from mcae_large_final_common import ClassificationHead, LABELS, choose_device, softmax


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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_features(rows: list[dict[str, str]], name: str) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError(f"{name} 为空")
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    if any(not 0 <= label < len(LABELS) or LABELS[label] != row["label"] for row, label in zip(rows, labels)):
        raise ValueError(f"{name} 的 label 与 label_id 不一致")
    original_paths = [Path(row["feature_path"]) for row in rows]
    paths = [resolve_feature_path(row["feature_path"]) for row in rows]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{name} 缺少 embedding，首个为: {missing[0]}")
    relocated = sum(original != resolved for original, resolved in zip(original_paths, paths))
    if relocated:
        print(f"{name}: 已重定位 {relocated} 个旧 embedding 路径到 {PROJECT_DIR}")
    features = [np.load(path, allow_pickle=False).astype(np.float32).reshape(-1) for path in paths]
    dimensions = {len(feature) for feature in features}
    if len(dimensions) != 1 or 0 in dimensions or not np.isfinite(np.stack(features)).all():
        raise ValueError(f"{name} embedding 无效或维度不一致: {dimensions}")
    return np.stack(features), labels


def class_weights(labels: np.ndarray, beta: float) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(LABELS)).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"训练集缺少类别: {counts.tolist()}")
    weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def predict(model: nn.Module, features: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start:start + batch_size]).to(device)
            outputs.append(model(batch).float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_tensor = torch.tensor(logits, dtype=torch.float64)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    log_temperature = nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 10.0)
        loss = F.cross_entropy(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().clamp(0.05, 10.0).item())


def probability_metrics(probabilities: np.ndarray, labels: np.ndarray, bins: int) -> dict[str, float]:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        upper = lower + 1.0 / bins
        mask = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    selected = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    return {"nll": float(-np.log(selected).mean()), "ece": float(ece)}


def main() -> None:
    parser = argparse.ArgumentParser(description="联合 MCAE 与中文长句 embedding 训练分类头")
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--reset-results", action="store_true", help="删除本轮联合实验的旧模型和评估后重新训练")
    args = parser.parse_args()
    if args.validate_only and args.reset_results:
        parser.error("--validate-only 与 --reset-results 不能一起使用")
    config = load_config(args.config)
    paths, training = config["paths"], config["training"]
    mcae_rows = read_csv(resolve_project_path(paths["mcae_feature_manifest"]))
    reading_rows = read_csv(resolve_project_path(paths["reading_feature_manifest"]))
    source_rows = read_csv(resolve_project_path(paths["reading_manifest"]))
    validate_reading_rows(source_rows, config)
    validate_reading_rows(reading_rows, config)
    require_matching_manifests(source_rows, reading_rows)
    if (len(mcae_rows) != 6675 or len({r["sample_id"] for r in mcae_rows}) != 6675
            or {row["label"] for row in mcae_rows} != set(LABELS)):
        raise ValueError("MCAE feature manifest 必须是完整的 6,675 条七分类清单")
    train_rows = [row for row in reading_rows if row["split"] == "train"]
    validation_rows = [row for row in reading_rows if row["split"] == "validation"]
    test_rows = [row for row in reading_rows if row["split"] == "test"]
    if len(train_rows) != 720 or len(validation_rows) != 240 or len(test_rows) != 240:
        raise ValueError("中文朗读特征清单的 split 数量应为 train=720, validation=240, test=240")
    if {row["label"] for row in reading_rows} != set(READING_LABELS):
        raise ValueError("中文朗读数据应只含六类非中性情绪")
    if set(row["speaker_id"] for row in train_rows) & set(row["speaker_id"] for row in validation_rows):
        raise ValueError("训练与验证说话人重叠")
    if set(row["speaker_id"] for row in train_rows + validation_rows) & set(row["speaker_id"] for row in test_rows):
        raise ValueError("测试说话人泄漏到训练或验证集")
    fingerprint = encoder_signature(resolve_project_path(paths["model_dir"]))
    verify_reading_cache(train_rows + validation_rows, fingerprint, source_rows)
    mcae_x, mcae_y = load_features(mcae_rows, "MCAE")
    reading_train_x, reading_train_y = load_features(train_rows, "中文朗读训练集")
    validation_x, validation_y = load_features(validation_rows, "中文朗读验证集")
    if len({mcae_x.shape[1], reading_train_x.shape[1], validation_x.shape[1]}) != 1:
        raise ValueError("MCAE、训练、验证 embedding 维度不一致")
    if mcae_x.shape[1] != int(config["model"].get("embedding_dim", 1024)):
        raise ValueError("特征维度不符合 emotion2vec_plus_large；请勿混用 base 特征")
    print(
        f"validated mcae={len(mcae_y)} reading_train={len(reading_train_y)} "
        f"reading_validation={len(validation_y)} dim={mcae_x.shape[1]}"
    )
    if args.validate_only:
        return
    if args.reset_results:
        reset_results(config)
    output_dir = resolve_project_path(paths["output_dir"])
    checkpoint_path = output_dir / "final_model.pt"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"为避免覆盖已有实验，已停止: {checkpoint_path}")
    set_seed(int(config["experiment"]["seed"]))
    combined_x = np.concatenate([mcae_x, reading_train_x], axis=0)
    combined_y = np.concatenate([mcae_y, reading_train_y], axis=0)
    mean = combined_x.mean(axis=0, keepdims=True)
    std = combined_x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    combined_x = ((combined_x - mean) / std).astype(np.float32)
    validation_x = ((validation_x - mean) / std).astype(np.float32)
    source_mass, target_mass = float(training["mcae_mass"]), float(training["reading_mass"])
    sampling_weights = build_sampling_weights(len(mcae_y), len(reading_train_y), source_mass, target_mass)
    print(f"sampling probabilities: MCAE={sampling_weights[:len(mcae_y)].sum():.2%} "
          f"reading={sampling_weights[len(mcae_y):].sum():.2%}")
    generator = torch.Generator().manual_seed(int(config["experiment"]["seed"]))
    sampler = WeightedRandomSampler(
        torch.from_numpy(sampling_weights.astype(np.float64)),
        num_samples=int(training["samples_per_epoch"]), replacement=True, generator=generator,
    )
    dataset = TensorDataset(torch.from_numpy(combined_x), torch.from_numpy(combined_y))
    device = choose_device(str(training["device"]))
    model = ClassificationHead(combined_x.shape[1], int(config["model"]["hidden_dim"]),
                               float(config["model"]["dropout"])).to(device)
    weights = class_weights(combined_y, float(training["class_weight_beta"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["head_learning_rate"]),
                                  weight_decay=float(training["weight_decay"]))
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"device={device} trainable_parameters={parameter_count} head_init=random encoder=frozen_cached_features")
    target_ids = [LABELS.index(label) for label in READING_LABELS]
    best_score, best_epoch, stale_epochs = -1.0, 0, 0
    best_state: dict[str, torch.Tensor] | None = None
    best_logits: np.ndarray | None = None
    history: list[dict[str, Any]] = []
    sampling_history: list[dict[str, Any]] = []
    seen_reading: set[int] = set()
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(training["epochs"]) + 1):
        started = time.perf_counter()
        indices = list(sampler)
        realized = sampling_summary(indices, len(mcae_y), combined_y)
        seen_reading.update(i for i in indices if i >= len(mcae_y))
        sampling_history.append({"epoch": epoch, **realized, "reading_unique_seen": len(seen_reading)})
        loader = DataLoader(dataset, batch_size=int(training["batch_size"]), sampler=indices,
                            num_workers=int(training["num_workers"]), pin_memory=device.type == "cuda")
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(batch_x), batch_y, weight=weights)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; inspect features and learning rate")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_y)
            total_samples += len(batch_y)
        logits = predict(model, validation_x, device, int(training["batch_size"]))
        predicted = logits.argmax(axis=1)
        score = float(f1_score(validation_y, predicted, labels=target_ids, average="macro", zero_division=0))
        accuracy = float(accuracy_score(validation_y, predicted))
        report = classification_report(validation_y, predicted, labels=target_ids,
                                       target_names=list(READING_LABELS), output_dict=True, zero_division=0)
        history.append({"epoch": epoch, "train_loss": total_loss / total_samples,
                        "validation_accuracy": accuracy, "validation_six_macro_f1": score,
                        "mcae_draws": realized["mcae_draws"], "reading_draws": realized["reading_draws"],
                        "reading_unique": realized["reading_unique"], "reading_unique_seen": len(seen_reading),
                        "epoch_seconds": time.perf_counter() - started,
                        **{f"val_{label}_{metric}": report[label][metric]
                           for label in READING_LABELS for metric in ("precision", "recall", "f1-score")}})
        write_csv(output_dir / "training_history.csv", history)
        write_json(output_dir / "sampling_history.json", sampling_history)
        print(f"epoch={epoch} train_loss={total_loss / total_samples:.6f} val_acc={accuracy:.4f} "
              f"val_six_f1={score:.4f} MCAE={realized['mcae_draws']} reading={realized['reading_draws']} "
              f"reading_unique={realized['reading_unique']} val_disgust_recall={report['disgust']['recall']:.4f}")
        if score > best_score:
            best_score, best_epoch, stale_epochs = score, epoch, 0
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            best_logits = logits.copy()
        else:
            stale_epochs += 1
            if stale_epochs >= int(training["early_stopping_patience"]):
                print(f"early_stopping epoch={epoch} best_epoch={best_epoch}")
                break
    assert best_state is not None and best_logits is not None
    calibrate = bool(config["calibration"].get("enabled", False))
    temperature = fit_temperature(best_logits, validation_y) if calibrate else 1.0
    before, after = softmax(best_logits), softmax(best_logits, temperature)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "pipeline_version": PIPELINE_VERSION,
        "reading_manifest_identity": manifest_identity(reading_rows),
        "encoder_sha256": fingerprint,
        "calibration_fitted": calibrate,
        "model_state": best_state,
        "input_dim": int(combined_x.shape[1]),
        "hidden_dim": int(config["model"]["hidden_dim"]),
        "dropout": float(config["model"]["dropout"]),
        "labels": LABELS,
        "embedding_mean": mean.reshape(-1),
        "embedding_std": std.reshape(-1),
        "temperature": temperature,
        "training_samples": int(len(combined_y)),
        "training_epochs": best_epoch,
        "class_weights": weights.detach().cpu().numpy(),
        "config": config,
    }
    torch.save(checkpoint, checkpoint_path)
    write_csv(output_dir / "training_history.csv", history)
    write_json(output_dir / "calibration.json", {
        "enabled": calibrate, "method": "scalar_temperature_scaling" if calibrate else "none",
        "source": "epoch-selection validation set (six classes only)" if calibrate else None,
        "note": "If fitted, before/after are in-sample diagnostics, not independent calibration validation.",
        "samples": len(validation_y), "temperature": temperature,
        "before": probability_metrics(before, validation_y, int(config["calibration"]["ece_bins"])),
        "after": probability_metrics(after, validation_y, int(config["calibration"]["ece_bins"])),
        "argmax_unchanged": bool(np.array_equal(before.argmax(axis=1), after.argmax(axis=1))),
    })
    write_json(output_dir / "training_summary.json", {
        "pipeline_version": PIPELINE_VERSION, "head_initialization": "random",
        "encoder_frozen": True, "trainable_parameters": parameter_count,
        "mcae_samples": len(mcae_y), "reading_train_samples": len(reading_train_y),
        "reading_validation_samples": len(validation_y), "reading_test_samples_not_loaded": len(test_rows),
        "best_epoch": best_epoch, "best_validation_six_macro_f1": best_score,
        "sampling": {"mcae_mass": source_mass, "reading_mass": target_mass,
                     "samples_per_epoch": int(training["samples_per_epoch"]),
                     "verified_reading_probability": float(sampling_weights[len(mcae_y):].sum()),
                     "realized_mcae_draws": sum(r["mcae_draws"] for r in sampling_history),
                     "realized_reading_draws": sum(r["reading_draws"] for r in sampling_history)},
        "reading_train_distribution": dict(Counter(row["label"] for row in train_rows)),
        "validation_speakers": sorted({row["speaker_id"] for row in validation_rows}),
        "test_speakers": sorted({row["speaker_id"] for row in test_rows}),
        "note": "Target-domain data has no neutral class; final target test reports six-class metrics.",
    })
    write_json(output_dir / "validation_metrics.json", {
        "best_epoch": best_epoch, "label_order": LABELS,
        "classification_report": classification_report(validation_y, best_logits.argmax(axis=1),
            labels=target_ids, target_names=list(READING_LABELS), output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(validation_y, best_logits.argmax(axis=1), labels=list(range(7))).tolist(),
        "note": "Epoch-selection validation data, not an independent test.",
    })
    write_json(output_dir / "resolved_config.json", config)
    print(f"final_checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
