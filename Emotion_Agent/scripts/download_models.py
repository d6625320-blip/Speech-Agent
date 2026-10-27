"""Explicit model downloads. Datasets and trained classifiers are not bundled."""
from __future__ import annotations

import argparse
from pathlib import Path


def download_plan(component: str, output: Path, speech_output: Path, tts_size: str):
    entries = []
    if component in {'asr', 'all'}:
        entries.extend([
            ('iic/SenseVoiceSmall', output / 'SenseVoiceSmall'),
            ('iic/speech_fsmn_vad_zh-cn-16k-common-pytorch', output / 'fsmn-vad'),
        ])
    if component in {'speech', 'all'}:
        entries.append(('iic/emotion2vec_plus_large', speech_output))
    if component in {'tts', 'all'}:
        name = f'Qwen3-TTS-12Hz-{tts_size}-CustomVoice'
        entries.append(('Qwen/' + name, output / name))
    return entries


def main():
    app = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description='从 ModelScope 显式下载预训练模型；不下载数据集或自训练分类头')
    parser.add_argument('--component', choices=('asr', 'speech', 'tts', 'all'), required=True)
    parser.add_argument('--tts-size', choices=('1.7B', '0.6B'), default='1.7B')
    parser.add_argument('--output-dir', type=Path, default=app / 'models', help='ASR 和 TTS 下载目录')
    parser.add_argument('--speech-output-dir', type=Path,
                        default=app.parent / 'Speech_model/models/pretrained/emotion2vec_plus_large')
    parser.add_argument('--list', action='store_true', help='只列出计划，不联网或创建下载目录')
    args = parser.parse_args()
    entries = download_plan(args.component, args.output_dir, args.speech_output_dir, args.tts_size)
    for model_id, destination in entries:
        print(f'{model_id} -> {destination}', flush=True)
    if not args.list:
        from modelscope import snapshot_download
        for model_id, destination in entries:
            destination.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(model_id, local_dir=str(destination))
    print('LLM: ollama pull qwen3.5:9b')
    print('分类头不随源码提供，请自行训练；数据集另行下载，见 docs/DATA_AND_MODELS.md。')


if __name__ == '__main__':
    main()
