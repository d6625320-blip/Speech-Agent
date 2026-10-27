"""在冻结的 emotion2vec_plus_large embedding 上执行六折 LOSO。"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from config_utils import load_config, resolve_project_path
from mcae_large_final_common import ClassificationHead, LABELS, choose_device, softmax


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_features(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    features = [np.load(row["feature_path"]).astype(np.float32).reshape(-1) for row in rows]
    dimensions = {len(feature) for feature in features}
    if len(dimensions) != 1:
        raise ValueError(f"embedding 维度不一致: {dimensions}")
    labels = np.asarray([int(row["label_id"]) for row in rows], dtype=np.int64)
    if any(LABELS[label_id] != row["label"] for row, label_id in zip(rows, labels)):
        raise ValueError("label 与 label_id 不一致")
    return np.stack(features), labels


def split_indices(rows: list[dict[str, str]], test_actor: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    actors = sorted({int(row["actor_id"]) for row in rows})
    validation_actor = actors[(actors.index(test_actor) + 1) % len(actors)]
    actor_ids = np.asarray([int(row["actor_id"]) for row in rows])
    train = np.flatnonzero((actor_ids != test_actor) & (actor_ids != validation_actor))
    validation = np.flatnonzero(actor_ids == validation_actor)
    test = np.flatnonzero(actor_ids == test_actor)
    return validation_actor, train, validation, test


def class_balanced_weights(labels: np.ndarray, beta: float) -> torch.Tensor:
    counts = np.bincount(labels, minlength=len(LABELS)).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"训练折缺少类别: {counts.tolist()}")
    weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def metrics(true_ids: list[int], pred_ids: list[int]) -> dict[str, Any]:
    ids = list(range(len(LABELS)))
    report = classification_report(
        true_ids, pred_ids, labels=ids, target_names=LABELS, output_dict=True, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(true_ids, pred_ids)),
        "macro_f1": float(f1_score(true_ids, pred_ids, labels=ids, average="macro", zero_division=0)),
        "balanced_accuracy": float(recall_score(true_ids, pred_ids, labels=ids, average="macro", zero_division=0)),
        "classification_report": report,
        "true_distribution": dict(Counter(LABELS[index] for index in true_ids)),
        "prediction_distribution": dict(Counter(LABELS[index] for index in pred_ids)),
        "confusion_matrix": confusion_matrix(true_ids, pred_ids, labels=ids).tolist(),
    }


@torch.inference_mode()
def infer(
    model: ClassificationHead,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features), torch.from_numpy(labels)),
        batch_size=batch_size,
        shuffle=False,
    )
    logits: list[np.ndarray] = []
    model.eval()
    for batch_features, _ in loader:
        logits.append(model(batch_features.to(device)).float().cpu().numpy())
    logits_array = np.concatenate(logits)
    probabilities = softmax(logits_array)
    result = metrics(labels.tolist(), probabilities.argmax(axis=1).tolist())
    result["loss"] = float(F.cross_entropy(torch.from_numpy(logits_array), torch.from_numpy(labels)).item())
    return result, logits_array


def save_confusion(output_dir: Path, metric: dict[str, Any]) -> None:
    matrix = np.asarray(metric["confusion_matrix"], dtype=np.int64)
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fields = ["true_label", *LABELS]
    write_csv(
        output_dir / "confusion_counts.csv",
        [{"true_label": label, **dict(zip(LABELS, values.tolist()))} for label, values in zip(LABELS, matrix)],
        fields,
    )
    write_csv(
        output_dir / "confusion_row_normalized.csv",
        [{"true_label": label, **dict(zip(LABELS, values.tolist()))} for label, values in zip(LABELS, normalized)],
        fields,
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


def aggregate(folds: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("accuracy", "macro_f1", "balanced_accuracy"):
        values = np.asarray([fold[name] for fold in folds], dtype=np.float64)
        result[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    result["per_class"] = {}
    for label in LABELS:
        result["per_class"][label] = {}
        for name in ("precision", "recall", "f1-score"):
            values = np.asarray([fold["classification_report"][label][name] for fold in folds])
            result["per_class"][label][name] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="MCAE frozen-embedding LOSO")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 7))
    args = parser.parse_args()
    config = load_config(args.config)
    training = config["training"]
    model_config = config["model"]
    manifest_path = resolve_project_path(config["paths"]["feature_manifest"])
    output_dir = resolve_project_path(config["paths"]["output_dir"])
    rows = read_csv(manifest_path)
    if len(rows) != 6675 or len({row["sample_id"] for row in rows}) != 6675:
        raise ValueError("feature manifest 必须包含 6675 个唯一样本")
    if sorted({int(row["actor_id"]) for row in rows}) != list(range(1, 7)):
        raise ValueError("LOSO 要求 actor 1..6")
    loss_name = str(training["loss"])
    if loss_name not in {"cross_entropy", "class_balanced_ce"}:
        raise ValueError(f"精简后的主线只支持 cross_entropy/class_balanced_ce: {loss_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "resolved_config.json", config)
    all_features, all_labels = load_features(rows)
    device = choose_device(str(training["device"]))
    folds = [args.fold] if args.fold else list(range(1, 7))
    fold_results: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []

    for fold in folds:
        seed = int(config["experiment"]["seed"]) + fold
        set_seed(seed)
        validation_actor, train_indices, validation_indices, test_indices = split_indices(rows, fold)
        mean = all_features[train_indices].mean(axis=0, keepdims=True)
        std = all_features[train_indices].std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        train_features = ((all_features[train_indices] - mean) / std).astype(np.float32)
        validation_features = ((all_features[validation_indices] - mean) / std).astype(np.float32)
        test_features = ((all_features[test_indices] - mean) / std).astype(np.float32)
        train_labels = all_labels[train_indices]
        validation_labels = all_labels[validation_indices]
        test_labels = all_labels[test_indices]
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels)),
            batch_size=int(training["batch_size"]),
            shuffle=True,
            generator=generator,
        )
        model = ClassificationHead(
            train_features.shape[1], int(model_config["hidden_dim"]), float(model_config["dropout"])
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training["head_learning_rate"]),
            weight_decay=float(training["weight_decay"]),
        )
        class_weights = (
            class_balanced_weights(train_labels, float(training["class_weight_beta"])).to(device)
            if loss_name == "class_balanced_ce"
            else None
        )
        fold_dir = output_dir / f"fold_{fold:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = fold_dir / "best_model.pt"
        history: list[dict[str, Any]] = []
        best_macro_f1 = -1.0
        no_improvement = 0
        for epoch in range(1, int(training["epochs"]) + 1):
            model.train()
            total_loss = 0.0
            total_samples = 0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(batch_features), batch_labels, weight=class_weights)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(batch_labels)
                total_samples += len(batch_labels)
            validation_metric, _ = infer(
                model, validation_features, validation_labels, int(training["batch_size"]), device
            )
            improved = validation_metric["macro_f1"] > best_macro_f1 + float(training["min_delta"])
            if improved:
                best_macro_f1 = validation_metric["macro_f1"]
                no_improvement = 0
                torch.save(
                    {
                        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "input_dim": train_features.shape[1],
                        "hidden_dim": int(model_config["hidden_dim"]),
                        "dropout": float(model_config["dropout"]),
                        "labels": LABELS,
                        "epoch": epoch,
                        "test_actor": fold,
                        "validation_actor": validation_actor,
                        "embedding_mean": mean.reshape(-1),
                        "embedding_std": std.reshape(-1),
                        "validation_metrics": validation_metric,
                        "config": config,
                    },
                    checkpoint_path,
                )
            else:
                no_improvement += 1
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total_loss / total_samples,
                    "val_accuracy": validation_metric["accuracy"],
                    "val_macro_f1": validation_metric["macro_f1"],
                    "val_balanced_accuracy": validation_metric["balanced_accuracy"],
                    "best": int(improved),
                }
            )
            print(f"fold={fold} epoch={epoch} val_macro_f1={validation_metric['macro_f1']:.4f}")
            if no_improvement >= int(training["patience"]):
                break
        write_csv(fold_dir / "history.csv", history)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        test_metric, test_logits = infer(model, test_features, test_labels, int(training["batch_size"]), device)
        test_probabilities = softmax(test_logits)
        predictions: list[dict[str, Any]] = []
        for position, global_index in enumerate(test_indices):
            source = rows[int(global_index)]
            prediction_id = int(test_probabilities[position].argmax())
            output: dict[str, Any] = {
                **source,
                "fold": fold,
                "validation_actor": validation_actor,
                "true_label": LABELS[int(test_labels[position])],
                "pred_label": LABELS[prediction_id],
                "correct": int(prediction_id == int(test_labels[position])),
                "confidence": float(test_probabilities[position].max()),
            }
            output.update({f"logit_{label}": float(test_logits[position, index]) for index, label in enumerate(LABELS)})
            output.update({f"prob_{label}": float(test_probabilities[position, index]) for index, label in enumerate(LABELS)})
            predictions.append(output)
        write_csv(fold_dir / "predictions.csv", predictions)
        save_confusion(fold_dir, test_metric)
        fold_result = {
            "fold": fold,
            "test_actor": fold,
            "validation_actor": validation_actor,
            "train_samples": len(train_indices),
            "validation_samples": len(validation_indices),
            "test_samples": len(test_indices),
            "best_epoch": int(checkpoint["epoch"]),
            **test_metric,
        }
        write_json(fold_dir / "metrics.json", fold_result)
        fold_results.append(fold_result)
        all_predictions.extend(predictions)

    write_csv(output_dir / "predictions.csv", all_predictions)
    pooled = metrics(
        [LABELS.index(row["true_label"]) for row in all_predictions],
        [LABELS.index(row["pred_label"]) for row in all_predictions],
    )
    save_confusion(output_dir, pooled)
    summary = {
        "experiment": config["experiment"]["name"],
        "completed_folds": folds,
        "fold_metrics": fold_results,
        "fold_mean_std": aggregate(fold_results),
        "pooled_loso": pooled,
    }
    write_json(output_dir / "metrics_summary.json", summary)
    print(json.dumps(summary["fold_mean_std"], ensure_ascii=False, indent=2))
    print(f"result={output_dir}")


if __name__ == "__main__":
    main()
