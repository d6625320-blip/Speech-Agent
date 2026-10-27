"""为中文情绪朗读数据建立严格按说话人切分的清单。"""

from __future__ import annotations

import argparse
import csv
import re
import wave
from collections import Counter
from pathlib import Path
from typing import Any

from config_utils import load_config, resolve_project_path
from chinese_reading_joint_utils import validate_reading_rows
from mcae_large_final_common import LABELS


SOURCE_TO_LABEL = {
    "生气": "angry",
    "厌恶": "disgust",
    "恐惧": "fear",
    "高兴": "happy",
    "悲伤": "sad",
    "惊讶": "surprised",
}
GENDERS = {**{f"SPK{index:04d}": "male" for index in range(1, 6)},
           **{f"SPK{index:04d}": "female" for index in range(6, 11)}}
FILENAME = re.compile(r"^(SPK\d{4})_1_(S\d{4})\.wav$", re.IGNORECASE)


def duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() <= 0:
            raise ValueError(f"无效采样率: {path}")
        return handle.getnframes() / handle.getframerate()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成中文情绪朗读数据集的说话人隔离清单")
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    root = resolve_project_path(config["paths"]["reading_root"])
    manifest = resolve_project_path(config["paths"]["reading_manifest"])
    split_by_speaker = {
        speaker: split
        for split, speakers in config["splits"].items()
        for speaker in speakers
    }
    expected_speakers = set(GENDERS)
    if set(split_by_speaker) != expected_speakers or len(split_by_speaker) != sum(
        len(value) for value in config["splits"].values()
    ):
        raise ValueError("splits 必须恰好覆盖 SPK0001–SPK0010，且每位说话人只能出现一次")
    rows: list[dict[str, Any]] = []
    for gender_dir in (root / "Male", root / "Female"):
        if not gender_dir.is_dir():
            raise FileNotFoundError(f"缺少目录: {gender_dir}")
        for speaker_dir in sorted(path for path in gender_dir.iterdir() if path.is_dir()):
            speaker = speaker_dir.name
            if speaker not in split_by_speaker:
                raise ValueError(f"未配置的说话人: {speaker}")
            for source_label, label in SOURCE_TO_LABEL.items():
                class_dir = speaker_dir / source_label
                if not class_dir.is_dir():
                    raise FileNotFoundError(f"缺少类别目录: {class_dir}")
                for audio in sorted(class_dir.glob("*.wav")):
                    match = FILENAME.match(audio.name)
                    if not match or match.group(1).upper() != speaker:
                        raise ValueError(f"不符合预期文件名: {audio}")
                    rows.append({
                        "sample_id": f"reading_{speaker}_{match.group(2).upper()}",
                        "audio_path": str(audio.resolve()),
                        "dataset": "Chinese_Emotional_Reading_Audio_Dataset",
                        "speaker_id": speaker,
                        "gender": GENDERS[speaker],
                        "split": split_by_speaker[speaker],
                        "source_label": source_label,
                        "label": label,
                        "label_id": LABELS.index(label),
                        "prompt_id": match.group(2).upper(),
                        "duration": f"{duration_seconds(audio):.6f}",
                    })
    if len(rows) != 1200 or len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError(f"数据集应有 1200 个唯一样本，实际 {len(rows)}")
    if {row["label"] for row in rows} != set(SOURCE_TO_LABEL.values()):
        raise ValueError("中文朗读数据类别错误")
    validate_reading_rows(rows, config)
    distribution = Counter((row["split"], row["label"]) for row in rows)
    for split in ("train", "validation", "test"):
        print(split, {label: distribution[(split, label)] for label in SOURCE_TO_LABEL.values()})
    if args.validate_only:
        return
    write_csv(manifest, rows)
    print(f"manifest={manifest} samples={len(rows)}")


if __name__ == "__main__":
    main()
