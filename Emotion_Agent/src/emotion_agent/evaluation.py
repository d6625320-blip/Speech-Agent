from __future__ import annotations

import unicodedata

import numpy as np


def normalize_text(text: str) -> str:
    return "".join(c.lower() for c in unicodedata.normalize("NFKC", text)
                   if not c.isspace() and not unicodedata.category(c).startswith("P"))


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        row = [i]
        for j, right in enumerate(hypothesis, 1):
            row.append(min(row[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = row
    return previous[-1]


def character_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    reference, hypothesis = normalize_text(reference), normalize_text(hypothesis)
    return edit_distance(reference, hypothesis), len(reference)


def latency_summary(values: list[float]) -> dict:
    return {"n": len(values), "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95))} if values else {"n": 0}
