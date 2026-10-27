"""用全部 MCAE embedding 训练最终 class-balanced CE 分类头。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config_utils import load_config, resolve_project_path
from mcae_large_final_common import ClassificationHead, LABELS, choose_device, softmax


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_balanced_weights(labels: np.ndarray, beta: float) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(LABELS)).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"MCAE 缺少类别: {counts.tolist()}")
    weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


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


def calibration_metrics(probabilities: np.ndarray, labels: np.ndarray, bins: int) -> dict[str, float]:
    selected = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        upper = lower + 1.0 / bins
        mask = (confidence >= lower) & (confidence < upper if upper < 1.0 else confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    one_hot = np.eye(len(LABELS), dtype=np.float64)[labels]
    return {
        "nll": float(-np.log(selected).mean()),
        "ece": float(ece),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
    }


def load_mcae(feature_manifest: Path) -> tuple[list[dict[str, str]], np.ndarray, np.ndarray]:
    rows = read_csv(feature_manifest)
    if len(rows) != 6675 or len({row["sample_id"] for row in rows}) != 6675:
        raise ValueError("MCAE feature manifest 必须含 6675 个唯一样本")
    if {row["label"] for row in rows} != set(LABELS):
        raise ValueError("MCAE 标签集合错误")
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    if any(LABELS[label_id] != row["label"] for row, label_id in zip(rows, labels)):
        raise ValueError("MCAE label 与 label_id 不一致")
    features = [np.load(row["feature_path"]).astype(np.float32).reshape(-1) for row in rows]
    dimensions = {len(feature) for feature in features}
    if len(dimensions) != 1:
        raise ValueError(f"embedding 维度不一致: {dimensions}")
    return rows, np.stack(features), labels


def calibrate_from_oof(
    oof_path: Path, mcae_by_id: dict[str, dict[str, str]], bins: int
) -> tuple[float, dict[str, Any]]:
    rows = read_csv(oof_path)
    required = {"sample_id", "true_label", "fold", *{f"logit_{label}" for label in LABELS}}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"LOSO 折外预测缺少字段: {sorted(required - set(rows[0] if rows else []))}")
    if len(rows) != 6675 or {row["sample_id"] for row in rows} != set(mcae_by_id):
        raise ValueError("校准文件必须与 6675 条 MCAE 样本一一对应")
    if {int(row["fold"]) for row in rows} != set(range(1, 7)):
        raise ValueError("校准文件必须包含完整的 6 个 LOSO 折")
    for row in rows:
        source = mcae_by_id[row["sample_id"]]
        if row["true_label"] != source["label"] or int(row["fold"]) != int(source["actor_id"]):
            raise ValueError(f"LOSO 标签或 test actor 不一致: {row['sample_id']}")
    logits = np.asarray(
        [[float(row[f"logit_{label}"]) for label in LABELS] for row in rows], dtype=np.float64
    )
    labels = np.asarray([LABELS.index(row["true_label"]) for row in rows], dtype=np.int64)
    temperature = fit_temperature(logits, labels)
    before = softmax(logits)
    after = softmax(logits, temperature)
    return temperature, {
        "method": "scalar_temperature_scaling",
        "source": "MCAE six-fold LOSO out-of-fold logits; RAVDESS not used",
        "samples": len(rows),
        "temperature": temperature,
        "before": calibration_metrics(before, labels, bins),
        "after": calibration_metrics(after, labels, bins),
        "argmax_unchanged": bool(np.array_equal(before.argmax(axis=1), after.argmax(axis=1))),
        "accuracy": float(accuracy_score(labels, after.argmax(axis=1))),
        "macro_f1": float(f1_score(labels, after.argmax(axis=1), average="macro")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="用全部 MCAE 训练最终分类头")
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    training = config["training"]
    calibration = config["calibration"]
    feature_manifest = resolve_project_path(paths["feature_manifest"])
    oof_path = resolve_project_path(paths["oof_predictions"])
    output_dir = resolve_project_path(paths["final_output_dir"])
    checkpoint_path = output_dir / "final_model.pt"

    rows, embeddings, labels = load_mcae(feature_manifest)
    temperature, calibration_result = calibrate_from_oof(
        oof_path, {row["sample_id"]: row for row in rows}, int(calibration["ece_bins"])
    )
    print(
        f"validated MCAE={len(rows)} dim={embeddings.shape[1]} "
        f"labels={dict(Counter(row['label'] for row in rows))} temperature={temperature:.6f}"
    )
    if args.validate_only:
        return
    if checkpoint_path.exists():
        raise FileExistsError(f"最终模型已存在，为避免误覆盖已停止: {checkpoint_path}")

    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    device = choose_device(str(training["device"]))
    mean = embeddings.mean(axis=0, keepdims=True)
    std = embeddings.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    features = ((embeddings - mean) / std).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=int(training["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    model = ClassificationHead(
        features.shape[1], int(config["model"]["hidden_dim"]), float(config["model"]["dropout"])
    ).to(device)
    weights = class_balanced_weights(labels, float(training["class_weight_beta"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["head_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    history: list[dict[str, Any]] = []
    epochs = int(training["epochs"])
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_features)
            loss = F.cross_entropy(logits, batch_labels, weight=weights)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_labels)
            total_samples += len(batch_labels)
        epoch_loss = total_loss / total_samples
        history.append({"epoch": epoch, "train_loss": epoch_loss})
        print(f"epoch={epoch}/{epochs} train_loss={epoch_loss:.6f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "input_dim": int(features.shape[1]),
        "hidden_dim": int(config["model"]["hidden_dim"]),
        "dropout": float(config["model"]["dropout"]),
        "labels": LABELS,
        "embedding_mean": mean.reshape(-1),
        "embedding_std": std.reshape(-1),
        "temperature": temperature,
        "training_samples": len(rows),
        "training_epochs": epochs,
        "class_weights": weights.detach().cpu().numpy(),
        "config": config,
    }
    torch.save(checkpoint, checkpoint_path)
    write_csv(output_dir / "training_history.csv", history)
    write_json(output_dir / "calibration.json", calibration_result)
    write_json(output_dir / "resolved_config.json", config)
    write_json(
        output_dir / "training_summary.json",
        {
            "samples": len(rows),
            "class_distribution": dict(Counter(row["label"] for row in rows)),
            "epochs": epochs,
            "selection_rule": "fixed before external test from LOSO median best epoch; no RAVDESS selection",
            "feature_manifest_sha256": sha256(feature_manifest),
            "oof_predictions_sha256": sha256(oof_path),
            "checkpoint": str(checkpoint_path),
        },
    )
    print(f"final_checkpoint={checkpoint_path}")
    print(f"calibration={output_dir / 'calibration.json'}")


if __name__ == "__main__":
    main()
