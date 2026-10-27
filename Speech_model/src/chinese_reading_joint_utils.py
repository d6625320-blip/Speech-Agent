"""Validation and provenance helpers for the independent joint-head experiment."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path, PureWindowsPath

import numpy as np

from config_utils import PROJECT_DIR, resolve_project_path
from mcae_large_final_common import LABELS

READING_LABELS = ("angry", "disgust", "fear", "happy", "sad", "surprised")
PIPELINE_VERSION = "joint_head_sampler_fixed_20261026"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encoder_signature(model_dir: Path) -> str:
    """Fingerprint the local encoder weights and configuration once per run."""
    if not (model_dir / "model.pt").is_file():
        raise FileNotFoundError(f"Encoder weights not found: {model_dir / 'model.pt'}")
    digest = hashlib.sha256(b"emotion2vec_utterance_extract_embedding_v1")
    for name in ("model.pt", "config.yaml"):
        path = model_dir / name
        if path.is_file():
            digest.update(name.encode())
            digest.update(sha256(path).encode())
    return digest.hexdigest()


def resolve_feature_path(value: str) -> Path:
    """Prefer the current project for features after a Windows directory move."""
    path = Path(value)
    parts = PureWindowsPath(value).parts if "\\" in value else path.parts
    for index, part in enumerate(parts):
        if part.lower() == "features":
            candidate = PROJECT_DIR.joinpath(*parts[index:])
            if candidate.is_file() or not path.is_file():
                return candidate
    return path if path.is_absolute() else resolve_project_path(path)


def validate_reading_rows(rows: list[dict], config: dict) -> None:
    splits = config["splits"]
    if set(splits) != {"train", "validation", "test"} or any(not v for v in splits.values()):
        raise ValueError("splits must contain nonempty train, validation and test speaker lists")
    speakers = [speaker for group in splits.values() for speaker in group]
    if len(speakers) != len(set(speakers)):
        raise ValueError("Speaker leakage: configured split speaker lists overlap")
    if len(rows) != 1200 or len({r["sample_id"] for r in rows}) != 1200:
        raise ValueError("Reading manifest must have 1,200 unique sample IDs")
    expected = {speaker: split for split, group in splits.items() for speaker in group}
    counts = Counter()
    for row in rows:
        if expected.get(row["speaker_id"]) != row["split"]:
            raise ValueError(f"Manifest split disagrees with YAML: {row['sample_id']}; rebuild manifests")
        label_id = int(row["label_id"])
        if not 0 <= label_id < len(LABELS) or LABELS[label_id] != row["label"]:
            raise ValueError(f"Invalid label/label_id: {row['sample_id']}")
        if row["label"] not in READING_LABELS:
            raise ValueError(f"Unexpected reading label: {row['label']}")
        if not np.isfinite(float(row["duration"])) or float(row["duration"]) <= 0:
            raise ValueError(f"Invalid audio duration: {row['sample_id']}")
        counts[row["speaker_id"]] += 1
    if set(counts) != set(expected) or any(count != 120 for count in counts.values()):
        raise ValueError("Each configured reading speaker must have exactly 120 recordings")
    for split in splits:
        if {r["label"] for r in rows if r["split"] == split} != set(READING_LABELS):
            raise ValueError(f"Split is missing an emotion class: {split}")


def require_matching_manifests(source_rows: list[dict], feature_rows: list[dict]) -> None:
    source = {r["sample_id"]: r for r in source_rows}
    if len(source) != len(source_rows) or len(feature_rows) != len(source):
        raise ValueError("Audio and feature manifests differ in sample IDs")
    if len({r["sample_id"] for r in feature_rows}) != len(feature_rows):
        raise ValueError("Duplicate feature manifest sample IDs")
    for row in feature_rows:
        previous = source.get(row["sample_id"])
        if previous is None or any(previous[k] != row[k] for k in
                                   ("speaker_id", "split", "label", "label_id", "prompt_id")):
            raise ValueError(f"Stale feature manifest: {row['sample_id']}; rerun extraction")


def manifest_identity(rows: list[dict]) -> str:
    """Identity independent of server drive letters and CSV ordering."""
    keys = ("sample_id", "speaker_id", "split", "label", "label_id", "prompt_id")
    canonical = sorted([str(row[key]) for key in keys] for row in rows)
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=True).encode()).hexdigest()


def build_sampling_weights(n_mcae: int, n_reading: int, mcae_mass: float, reading_mass: float) -> np.ndarray:
    if min(n_mcae, n_reading) <= 0:
        raise ValueError("Both training sources must be nonempty")
    masses = np.asarray([mcae_mass, reading_mass], dtype=float)
    if not np.isfinite(masses).all() or (masses <= 0).any() or not np.isclose(masses.sum(), 1):
        raise ValueError("Source sampling masses must be finite, positive and sum to one")
    # Combined dataset order is MCAE first, reading second. Never swap these blocks.
    weights = np.concatenate([np.full(n_mcae, mcae_mass / n_mcae),
                              np.full(n_reading, reading_mass / n_reading)])
    if not np.allclose([weights[:n_mcae].sum(), weights[n_mcae:].sum()], masses):
        raise AssertionError("Source sampling probability check failed")
    return weights


def sampling_summary(indices: list[int], n_mcae: int, labels: np.ndarray) -> dict:
    ids = np.asarray(indices, dtype=np.int64)
    reading = ids >= n_mcae
    result = {}
    for source, mask in (("mcae", ~reading), ("reading", reading)):
        selected = ids[mask]
        result[f"{source}_draws"] = int(len(selected))
        result[f"{source}_fraction"] = float(len(selected) / len(ids))
        result[f"{source}_unique"] = int(len(np.unique(selected)))
        result[f"{source}_class_counts"] = {
            label: int((labels[selected] == i).sum()) for i, label in enumerate(LABELS)
        }
    return result


def verify_reading_cache(rows: list[dict], encoder_hash: str, source_rows: list[dict]) -> None:
    audio_by_id = {r["sample_id"]: Path(r["audio_path"]) for r in source_rows}
    paths = []
    for row in rows:
        path = resolve_feature_path(row["feature_path"])
        audio = audio_by_id[row["sample_id"]]
        if row.get("encoder_sha256") != encoder_hash:
            raise ValueError("Missing/stale encoder fingerprint; rerun extract_chinese_reading_embeddings.py")
        if not path.is_file() or row.get("feature_sha256") != sha256(path):
            raise ValueError(f"Missing/corrupt cached feature: {path}; rerun extraction")
        if not audio.is_file() or row.get("audio_sha256") != sha256(audio):
            raise ValueError(f"Audio changed or missing: {audio}; rebuild manifest and rerun extraction")
        paths.append(str(path.resolve()).casefold())
    if len(set(paths)) != len(paths):
        raise ValueError("Different samples reference the same feature file")


def reset_results(config: dict) -> None:
    """Explicit reset is restricted to these two experiment result directories."""
    expected = {
        "output_dir": PROJECT_DIR / "outputs/final/mcae_chinese_reading_joint_v1",
        "test_output_dir": PROJECT_DIR / "outputs/evaluation/mcae_chinese_reading_joint_v1",
    }
    targets = []
    for key, allowed in expected.items():
        target = resolve_project_path(config["paths"][key])
        if target.resolve() != allowed.absolute() or target.is_symlink():
            raise ValueError(f"Refusing to reset unexpected results directory: {target}")
        if target.exists() and any(p.is_symlink() or
                                   (hasattr(p, 'is_junction') and p.is_junction())
                                   for p in [target, *target.rglob('*')]):
            raise ValueError(f"Refusing to reset a linked directory: {target}")
        targets.append(target)
    for target in targets:
        if target.exists():
            shutil.rmtree(target)
            print(f"Removed this experiment's previous results: {target}")
