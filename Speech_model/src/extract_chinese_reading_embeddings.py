"""为中文情绪朗读数据提取与部署一致的 emotion2vec_plus_large embedding。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from config_utils import load_config, resolve_project_path
from chinese_reading_joint_utils import encoder_signature, sha256, validate_reading_rows
from mcae_large_final_common import Emotion2VecEmbeddingExtractor, choose_device


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
    parser = argparse.ArgumentParser(description="提取中文情绪朗读数据 embedding")
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    paths, extraction = config["paths"], config["extraction"]
    manifest = resolve_project_path(paths["reading_manifest"])
    feature_dir = resolve_project_path(paths["reading_feature_dir"])
    feature_manifest = resolve_project_path(paths["reading_feature_manifest"])
    model_dir = resolve_project_path(paths["model_dir"])
    rows = read_csv(manifest)
    validate_reading_rows(rows, config)
    missing = [row["audio_path"] for row in rows if not Path(row["audio_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"音频不存在，首个为: {missing[0]}")
    if not (model_dir / "model.pt").is_file():
        raise FileNotFoundError(f"编码器不存在: {model_dir / 'model.pt'}")
    print(f"validated samples={len(rows)} model={model_dir}")
    if args.validate_only:
        return
    fingerprint = encoder_signature(model_dir)
    pending: list[dict[str, str]] = []
    complete: list[dict[str, Any]] = []
    for row in rows:
        feature_path = feature_dir / row["split"] / row["speaker_id"] / row["label"] / f"{row['sample_id']}.npy"
        audio_hash = sha256(Path(row["audio_path"]))
        metadata_path = feature_path.with_suffix(".json")
        cached = None
        if feature_path.is_file() and metadata_path.is_file() and not bool(extraction["overwrite"]):
            try:
                meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                if (meta.get("encoder_sha256") == fingerprint and meta.get("audio_sha256") == audio_hash
                        and meta.get("feature_sha256") == sha256(feature_path)):
                    embedding = np.load(feature_path, allow_pickle=False).astype(np.float32).reshape(-1)
                    if embedding.size and np.isfinite(embedding).all():
                        cached = {**row, "feature_path": str(feature_path.resolve()),
                                  "feature_dim": len(embedding), **meta}
            except (ValueError, OSError, EOFError, TypeError):
                pass  # Stale/incomplete caches are re-extracted rather than silently trusted.
        if cached is not None:
            complete.append(cached)
        else:
            pending.append({**row, "audio_sha256": audio_hash, "_feature_path": str(feature_path.resolve())})
    batch_size = int(extraction["batch_size"])
    if batch_size <= 0:
        raise ValueError("extraction.batch_size 必须大于 0")
    print(f"verified_cache={len(complete)} reextract={len(pending)} encoder={fingerprint[:12]}")
    extractor = (Emotion2VecEmbeddingExtractor(model_dir, choose_device(str(extraction["device"])), quiet=False)
                 if pending else None)
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        embeddings = extractor.extract_many([Path(row["audio_path"]) for row in batch], batch_size=len(batch))
        for row, embedding in zip(batch, embeddings):
            feature_path = Path(row.pop("_feature_path"))
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = feature_path.with_suffix(".tmp")
            with temporary.open("wb") as handle:
                np.save(handle, embedding.astype(np.float32), allow_pickle=False)
            temporary.replace(feature_path)
            meta = {"audio_sha256": row["audio_sha256"], "encoder_sha256": fingerprint,
                    "feature_sha256": sha256(feature_path)}
            metadata_path = feature_path.with_suffix(".json")
            meta_temp = metadata_path.with_suffix(".tmp")
            meta_temp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            meta_temp.replace(metadata_path)
            complete.append({**row, "feature_path": str(feature_path.resolve()),
                             "feature_dim": len(embedding), **meta})
        print(f"saved={min(start + len(batch), len(pending))}/{len(pending)} dim={embeddings.shape[1]}")
    by_id = {row["sample_id"]: row for row in complete}
    ordered = [by_id[row["sample_id"]] for row in rows]
    dimensions = {int(row["feature_dim"]) for row in ordered}
    if len(dimensions) != 1:
        raise ValueError(f"embedding 维度不一致: {dimensions}")
    write_csv(feature_manifest, ordered)
    print(f"feature_manifest={feature_manifest} samples={len(ordered)} dim={next(iter(dimensions))}")


if __name__ == "__main__":
    main()
