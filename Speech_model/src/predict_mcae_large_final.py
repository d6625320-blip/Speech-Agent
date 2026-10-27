"""加载最终分类头，对一条实际音频输出七类概率。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from config_utils import load_config, resolve_project_path
from mcae_large_final_common import (
    Emotion2VecEmbeddingExtractor,
    LABELS,
    choose_device,
    load_final_head,
    softmax,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="最终 MCAE large 七分类模型推理")
    parser.add_argument("--config", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--use-calibration",
        action="store_true",
        help="使用仅在 MCAE LOSO 折外数据上拟合的温度校准",
    )
    parser.add_argument("--verbose", action="store_true", help="显示 FunASR 详细日志")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"音频不存在: {audio_path}")
    device = choose_device(str(config["inference"]["device"]))
    checkpoint_path = resolve_project_path(paths["final_output_dir"]) / "final_model.pt"
    model_dir = resolve_project_path(paths["model_dir"])
    head, _, mean, std, temperature = load_final_head(checkpoint_path, device)
    extractor = Emotion2VecEmbeddingExtractor(model_dir, device, quiet=not args.verbose)
    embedding = extractor.extract_one(audio_path)
    if embedding.shape[1] != mean.shape[1]:
        raise ValueError(f"embedding 维度错误: {embedding.shape[1]} != {mean.shape[1]}")
    feature = torch.from_numpy(((embedding - mean) / std).astype(np.float32)).to(device)
    with torch.inference_mode():
        logits = head(feature).float().cpu().numpy()
    raw = softmax(logits)[0]
    calibrated = softmax(logits, temperature)[0]
    probabilities = calibrated if args.use_calibration else raw
    applied_temperature = temperature if args.use_calibration else 1.0
    probability_mode = "mcae_temperature_scaled" if args.use_calibration else "uncalibrated"
    order = np.argsort(probabilities)[::-1]
    entropy = float(-(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum())
    result = {
        "audio_path": str(audio_path),
        "prediction": LABELS[int(order[0])],
        "confidence": float(probabilities[order[0]]),
        "margin": float(probabilities[order[0]] - probabilities[order[1]]),
        "entropy": entropy,
        "normalized_entropy": entropy / float(np.log(len(LABELS))),
        "probability_mode": probability_mode,
        "temperature_applied": applied_temperature,
        "stored_mcae_temperature": temperature,
        "probabilities": {label: float(probabilities[index]) for index, label in enumerate(LABELS)},
        "calibrated_probabilities": {label: float(calibrated[index]) for index, label in enumerate(LABELS)},
        "uncalibrated_probabilities": {label: float(raw[index]) for index, label in enumerate(LABELS)},
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    prediction = result["prediction"]
    print(f"prediction: {prediction}")
    print(f"probability_mode: {probability_mode}")
    print(f"confidence: {result['confidence']:.2%}")
    print(f"margin: {result['margin']:.2%}")
    print(f"normalized_entropy: {result['normalized_entropy']:.4f} (lower is more certain)")
    print("probabilities:")
    for index in order:
        label = LABELS[int(index)]
        print(f"  {label:<10} {probabilities[index]:7.2%}")
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"完整 JSON：{output_path}")


if __name__ == "__main__":
    main()
