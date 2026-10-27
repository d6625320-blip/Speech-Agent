"""Optional asymmetric confusion penalty; original hard seven-class labels are retained."""

from __future__ import annotations

import math

import numpy as np
import torch

from mcae_large_final_common import LABELS


def cost_settings(config: dict) -> dict:
    value = config.get("confusion_cost", {})
    if not value.get("enabled", False):
        return {"enabled": False, "strength": 0.0, "costs": [0.0] * len(LABELS)}
    strength = float(value.get("strength", .3))
    costs = value.get("disgust_prediction_costs", {})
    if set(costs) != set(LABELS):
        raise ValueError("confusion_cost.disgust_prediction_costs must specify all seven labels")
    vector = [float(costs[label]) for label in LABELS]
    if not math.isfinite(strength) or strength <= 0 or not np.isfinite(vector).all() or min(vector) < 0:
        raise ValueError("Confusion costs and strength must be finite and nonnegative (strength > 0)")
    if vector[LABELS.index("disgust")] != 0 or any(c <= 0 for i, c in enumerate(vector) if LABELS[i] != "disgust"):
        raise ValueError("Correct disgust must have zero cost; every wrong label must have positive cost")
    if vector[LABELS.index("surprised")] <= vector[LABELS.index("angry")]:
        raise ValueError("This experiment requires disgust->surprised cost greater than disgust->angry")
    return {"enabled": True, "strength": strength, "costs": vector,
            "scope": "new-reading training samples with true label disgust only"}


def confusion_penalty(logits: torch.Tensor, truth: torch.Tensor,
                      is_reading: torch.Tensor, settings: dict) -> torch.Tensor:
    if not settings["enabled"]:
        return logits.sum() * 0.0
    mask = is_reading.bool() & (truth == LABELS.index("disgust"))
    if not mask.any():
        return logits.sum() * 0.0
    costs = logits.new_tensor(settings["costs"])
    # Conditional mean: strength controls the extra penalty per target disgust example,
    # independently of how many such examples happen to occur in this minibatch.
    return (logits[mask].softmax(dim=-1) * costs).sum(dim=-1).mean()


def disgust_diagnostics(logits: np.ndarray, truth: np.ndarray, settings: dict) -> dict:
    prediction = logits.argmax(axis=1)
    mask = truth == LABELS.index("disgust")
    if not mask.any():
        raise ValueError("Cannot compute disgust diagnostics without true disgust samples")
    rates = {label: float((prediction[mask] == index).mean()) for index, label in enumerate(LABELS)}
    other = ~mask
    return {"samples": int(mask.sum()), "recall": rates["disgust"],
            "to_angry": rates["angry"], "to_surprised": rates["surprised"],
            "to_other_wrong": sum(rates[label] for label in ("fear", "happy", "neutral", "sad")),
            "mean_error_cost": sum(rates[label] * settings["costs"][index]
                                   for index, label in enumerate(LABELS)),
            "false_disgust_rate": float((prediction[other] == LABELS.index("disgust")).mean()) if other.any() else 0.0,
            "prediction_rates": rates}


def candidate_rank(macro_f1: float, diagnostics: dict, selection: dict, reference: dict | None) -> tuple:
    tradeoff = float(selection.get("cost_tradeoff", 0.0))
    tolerance = float(selection.get("max_macro_f1_drop", 0.0))
    if not math.isfinite(tradeoff) or not math.isfinite(tolerance) or min(tradeoff, tolerance) < 0:
        raise ValueError("Selection tradeoff and tolerance must be finite and nonnegative")
    eligible = (reference is None or (
        macro_f1 >= reference["macro_f1"] - tolerance
        and diagnostics["recall"] >= reference["disgust"]["recall"]
    ))
    score = macro_f1 - tradeoff * diagnostics["mean_error_cost"]
    # Feasible candidates outrank infeasible ones; no test data is consulted.
    return bool(eligible), float(score), float(macro_f1)
