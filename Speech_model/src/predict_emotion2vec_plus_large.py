"""使用原始 emotion2vec_plus_large 预测音频情感概率。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = PROJECT_DIR / "models" / "pretrained" / "emotion2vec_plus_large"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "predictions" / "emotion2vec_plus_large.csv"
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}

SEVEN_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]
NATIVE_LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "other",
    "sad",
    "surprised",
    "unknown",
]
LABEL_ZH = {
    "angry": "生气",
    "disgust": "厌恶",
    "fear": "恐惧",
    "happy": "高兴",
    "neutral": "中性",
    "other": "其他",
    "sad": "悲伤",
    "surprised": "惊讶",
    "unknown": "未知",
}
LABEL_ALIASES = {
    "angry": "angry",
    "anger": "angry",
    "生气": "angry",
    "disgust": "disgust",
    "disgusted": "disgust",
    "厌恶": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "恐惧": "fear",
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "开心": "happy",
    "neutral": "neutral",
    "中立": "neutral",
    "other": "other",
    "其他": "other",
    "sad": "sad",
    "sadness": "sad",
    "难过": "sad",
    "surprise": "surprised",
    "surprised": "surprised",
    "吃惊": "surprised",
    "unknown": "unknown",
    "unk": "unknown",
}


def resolve_path(value: str | Path, *, must_exist: bool = True) -> Path:
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, PROJECT_DIR / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    resolved = candidates[0].resolve()
    if must_exist:
        raise FileNotFoundError(f"路径不存在: {resolved}")
    return resolved


def collect_audio(inputs: list[str]) -> list[Path]:
    audio_paths: list[Path] = []
    for value in inputs:
        path = resolve_path(value)
        if path.is_file():
            if path.suffix.lower() not in AUDIO_SUFFIXES:
                raise ValueError(f"不支持的音频格式: {path}")
            audio_paths.append(path)
        elif path.is_dir():
            audio_paths.extend(
                item.resolve()
                for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in AUDIO_SUFFIXES
            )
    # 保持输入顺序，同时去除重复路径。
    return list(dict.fromkeys(audio_paths))


def normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace("<|", "").replace("|>", "").replace("<", "").replace(">", "")
    candidates = [text, *text.replace("\\", "/").split("/")]
    for candidate in reversed(candidates):
        compact = candidate.strip().replace("-", "").replace("_", "").replace(" ", "")
        if compact in LABEL_ALIASES:
            return LABEL_ALIASES[compact]
    raise ValueError(f"无法识别模型输出标签: {value!r}")


def normalize_scores(scores: Any) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"模型返回了无效分数: {scores!r}")
    total = float(values.sum())
    if np.all(values >= 0.0) and total > 0.0:
        return values / total
    values = values - float(values.max())
    probabilities = np.exp(values)
    return probabilities / float(probabilities.sum())


def parse_result(result: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    labels = result.get("labels")
    scores = result.get("scores")
    if labels is None or scores is None:
        raise ValueError(f"模型结果缺少 labels 或 scores: {sorted(result)}")

    probabilities = normalize_scores(scores)
    if len(labels) != len(probabilities):
        raise ValueError("模型返回的标签数量和分数数量不一致")

    native = {label: 0.0 for label in NATIVE_LABELS}
    observed: set[str] = set()
    for raw_label, probability in zip(labels, probabilities.tolist()):
        label = normalize_label(raw_label)
        observed.add(label)
        native[label] += float(probability)

    missing = [label for label in SEVEN_LABELS if label not in observed]
    if missing:
        raise ValueError(f"模型结果缺少七分类标签: {missing}")

    seven_total = sum(native[label] for label in SEVEN_LABELS)
    if seven_total <= 0.0:
        raise ValueError("七类情感的概率总和为零")
    seven = {label: native[label] / seven_total for label in SEVEN_LABELS}
    return native, seven


class Emotion2VecLargePredictor:
    """加载原始 large 模型，并返回官方九类和项目七类概率。"""

    def __init__(self, model_dir: Path, device: str):
        from funasr import AutoModel

        self.model = AutoModel(
            model=str(model_dir),
            device=device,
            disable_update=True,
            log_level="ERROR",
        )

    def predict(self, audio_path: Path) -> dict[str, Any]:
        results = self.model.generate(
            input=str(audio_path),
            granularity="utterance",
            extract_embedding=False,
        )
        if len(results) != 1:
            raise ValueError(f"模型应返回一条结果，实际返回 {len(results)} 条: {audio_path}")
        native, seven = parse_result(results[0])
        return {
            "audio_path": str(audio_path),
            "native_prediction": max(native, key=native.get),
            "native_probabilities": native,
            "seven_prediction": max(seven, key=seven.get),
            "seven_probabilities": seven,
        }


def print_prediction(result: dict[str, Any], index: int, total: int) -> None:
    native = result["native_probabilities"]
    seven = result["seven_probabilities"]
    print(f"\n[{index}/{total}] {result['audio_path']}")
    print(
        f"七类预测: {LABEL_ZH[result['seven_prediction']]} "
        f"({result['seven_prediction']}) {max(seven.values()):.2%}"
    )
    for label in SEVEN_LABELS:
        print(f"  {LABEL_ZH[label]:<4} {label:<10} {seven[label]:7.2%}")

    print(
        f"官方九类预测: {LABEL_ZH[result['native_prediction']]} "
        f"({result['native_prediction']}) {max(native.values()):.2%}"
    )
    for label in NATIVE_LABELS:
        print(f"  {LABEL_ZH[label]:<4} {label:<10} {native[label]:7.2%}")


def save_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "audio_path",
        "seven_prediction",
        "seven_confidence",
        "native_prediction",
        "native_confidence",
        *[f"seven_prob_{label}" for label in SEVEN_LABELS],
        *[f"native_prob_{label}" for label in NATIVE_LABELS],
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            seven = result["seven_probabilities"]
            native = result["native_probabilities"]
            row: dict[str, Any] = {
                "audio_path": result["audio_path"],
                "seven_prediction": result["seven_prediction"],
                "seven_confidence": max(seven.values()),
                "native_prediction": result["native_prediction"],
                "native_confidence": max(native.values()),
            }
            row.update({f"seven_prob_{label}": seven[label] for label in SEVEN_LABELS})
            row.update({f"native_prob_{label}": native[label] for label in NATIVE_LABELS})
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="使用原始 emotion2vec_plus_large 预测情感概率")
    parser.add_argument("--input", nargs="+", required=True, help="一个或多个音频文件/文件夹")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="本地 large 模型目录")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="结果 CSV 保存位置")
    args = parser.parse_args()

    audio_paths = collect_audio(args.input)
    if not audio_paths:
        raise FileNotFoundError("输入路径中没有找到支持的音频文件")

    model_dir = resolve_path(args.model_dir)
    if not model_dir.is_dir():
        raise NotADirectoryError(f"模型目录无效: {model_dir}")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法使用 CUDA")

    print(f"device={device} model={model_dir} samples={len(audio_paths)}")
    predictor = Emotion2VecLargePredictor(model_dir, device)
    results: list[dict[str, Any]] = []
    for index, audio_path in enumerate(audio_paths, start=1):
        result = predictor.predict(audio_path)
        results.append(result)
        print_prediction(result, index, len(audio_paths))

    output_path = resolve_path(args.output, must_exist=False)
    save_csv(results, output_path)
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
