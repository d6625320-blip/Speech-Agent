"""Build the audit input manifest from independently downloaded MCAE metadata."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from audit_mcae_large import LABELS, LABEL_TO_FOLDER, SOURCE_TO_LABEL, excel_rows, sample_id
from config_utils import load_config, resolve_project_path


def prepare(config: dict, overwrite: bool = False) -> Path:
    paths = config["paths"]
    output = resolve_project_path(paths["existing_manifest"])
    if output.exists() and not overwrite:
        raise FileExistsError(f"Manifest already exists: {output}; use --overwrite deliberately")
    root = resolve_project_path(paths["audio_dir"])
    rows, seen = [], set()
    for source in excel_rows(resolve_project_path(paths["metadata_xlsx"])):
        sid = sample_id(source["Filename"])
        if not sid or sid in seen:
            raise ValueError(f"Empty or duplicate sample ID: {sid}")
        seen.add(sid)
        label = SOURCE_TO_LABEL[str(source["emotion"]).strip().lower()]
        audio = root / LABEL_TO_FOLDER[label] / f"{sid}.wav"
        if not audio.is_file():
            raise FileNotFoundError(f"Missing downloaded audio: {audio}")
        rows.append({"sample_id": sid, "audio_path": str(audio.resolve()),
                     "label": label, "label_id": LABELS.index(label),
                     "actor_id": int(source["Actress"])})
    if not rows:
        raise ValueError("Metadata contains no samples")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(prepare(load_config(args.config), args.overwrite))


if __name__ == "__main__":
    main()
