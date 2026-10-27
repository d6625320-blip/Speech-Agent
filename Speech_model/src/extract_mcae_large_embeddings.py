"""一次提取 MCAE 的 emotion2vec_plus_large utterance embedding。"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config_utils import load_config, resolve_project_path


LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6675:
        raise ValueError(f"MCAE manifest 应为 6675 条，实际 {len(rows)}")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("MCAE manifest 存在重复 sample_id")
    if {row["label"] for row in rows} != set(LABELS):
        raise ValueError("MCAE manifest 标签集合错误")
    missing = [row["audio_path"] for row in rows if not Path(row["audio_path"]).is_file()]
    if missing:
        raise FileNotFoundError(f"MCAE manifest 中有 {len(missing)} 个音频不存在，首个为 {missing[0]}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="提取 MCAE large utterance embedding")
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    extraction = config["extraction"]
    manifest_path = resolve_project_path(paths["manifest"])
    model_dir = resolve_project_path(paths["model_dir"])
    feature_dir = resolve_project_path(paths["feature_dir"])
    output_manifest = resolve_project_path(paths["feature_manifest"])
    rows = read_rows(manifest_path)
    print(
        f"validated samples={len(rows)} actors={sorted({int(row['actor_id']) for row in rows})} "
        f"labels={dict(Counter(row['label'] for row in rows))}"
    )
    if args.validate_only:
        return

    device_value = str(extraction["device"])
    device = "cuda" if device_value == "auto" and torch.cuda.is_available() else device_value
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    feature_dir.mkdir(parents=True, exist_ok=True)
    overwrite = bool(extraction["overwrite"])

    existing_rows: list[dict[str, Any]] = []
    pending_rows: list[dict[str, str]] = []
    for row in rows:
        feature_path = feature_dir / f"actor_{int(row['actor_id']):02d}" / row["label"] / f"{row['sample_id']}.npy"
        if feature_path.is_file() and not overwrite:
            embedding = np.load(feature_path).astype(np.float32).reshape(-1)
            existing_rows.append({**row, "feature_path": str(feature_path.resolve()), "feature_dim": len(embedding)})
        else:
            pending_rows.append({**row, "_feature_path": str(feature_path.resolve())})

    if pending_rows:
        from funasr import AutoModel

        scp_path = feature_dir / "pending.scp"
        keys = [f"utt{index:06d}" for index in range(len(pending_rows))]
        with scp_path.open("w", encoding="utf-8") as handle:
            for key, row in zip(keys, pending_rows):
                handle.write(f"{key} {row['audio_path']}\n")
        model = AutoModel(model=str(model_dir), device=device, disable_update=True, log_level="ERROR")
        results = model.generate(
            input=str(scp_path),
            batch_size=int(extraction["batch_size"]),
            granularity="utterance",
            extract_embedding=True,
        )
        by_key = {str(result.get("key", "")): result for result in results}
        if set(by_key) != set(keys):
            raise ValueError("embedding 输出 key 与输入不一致")
        for index, (key, row) in enumerate(zip(keys, pending_rows), start=1):
            if "feats" not in by_key[key]:
                raise ValueError(f"模型没有返回 embedding: {row['sample_id']}")
            embedding = np.asarray(by_key[key]["feats"], dtype=np.float32).reshape(-1)
            if embedding.size == 0 or not np.isfinite(embedding).all():
                raise ValueError(f"embedding 无效: {row['sample_id']}")
            feature_path = Path(row.pop("_feature_path"))
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(feature_path, embedding)
            existing_rows.append(
                {**row, "feature_path": str(feature_path.resolve()), "feature_dim": len(embedding)}
            )
            if index % 500 == 0 or index == len(pending_rows):
                print(f"saved={index}/{len(pending_rows)} dim={len(embedding)}")

    by_id = {row["sample_id"]: row for row in existing_rows}
    ordered = [by_id[row["sample_id"]] for row in rows]
    dimensions = {int(row["feature_dim"]) for row in ordered}
    if len(dimensions) != 1:
        raise ValueError(f"embedding 维度不一致: {dimensions}")
    write_csv(output_manifest, ordered)
    print(f"feature_manifest={output_manifest} samples={len(ordered)} dim={next(iter(dimensions))}")


if __name__ == "__main__":
    main()
