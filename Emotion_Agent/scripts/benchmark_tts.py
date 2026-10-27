"""Benchmark the already running isolated worker; no LLM or model download."""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.request
import wave
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="比较 TTS 首块、总耗时与实时因子；RTF 越小越快")
    parser.add_argument("--base-url", default="http://127.0.0.1:9880")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--text", default="你好，今天想聊些什么？我会认真听你说。")
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--instruction", default="语气自然、友好，保持清晰。")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not 1 <= args.repeat <= 20 or not 1 <= len(args.text) <= 500:
        parser.error("repeat 需为 1–20，text 长度需为 1–500")
    folder = args.output_dir or Path(__file__).resolve().parents[1] / "outputs" / f"tts_benchmark_{time.time_ns()}"
    folder.mkdir(parents=True, exist_ok=True)
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = args.base_url.rstrip("/")
    with client.open(base + "/health", timeout=20) as response:
        health = json.load(response)
    payload = json.dumps({"text": args.text, "speaker": args.speaker, "instruction": args.instruction}).encode("utf-8")
    rows = []
    for index in range(args.repeat):
        request = urllib.request.Request(base + ("/synthesize/stream" if args.stream else "/synthesize"),
            data=payload, headers={"Content-Type": "application/json"})
        start, first, rate, sequence, done = time.perf_counter(), None, None, 0, False
        data = bytearray()
        with client.open(request, timeout=180) as response:
            if args.stream:
                for line in response:
                    if len(line) > 131072:
                        raise RuntimeError("TTS event too large")
                    event = json.loads(line)
                    if event["type"] == "error":
                        raise RuntimeError(event["detail"])
                    if event["type"] == "start":
                        if event.get("sample_rate") != 24000 or event.get("encoding") != "pcm_s16le":
                            raise RuntimeError("Unexpected PCM format")
                        rate = event["sample_rate"]
                    elif event["type"] == "audio":
                        if rate is None or event.get("sequence") != sequence:
                            raise RuntimeError("Out-of-order PCM event")
                        if first is None:
                            first = (time.perf_counter() - start) * 1000
                        data.extend(base64.b64decode(event["pcm"], validate=True))
                        sequence += 1
                    elif event["type"] == "done":
                        done = True
                    if len(data) > 20 * 1024 * 1024:
                        raise RuntimeError("Audio too large")
                if not done or not data or len(data) % 2:
                    raise RuntimeError("Incomplete TTS stream")
            else:
                data.extend(response.read(20 * 1024 * 1024 + 1))
                if len(data) > 20 * 1024 * 1024:
                    raise RuntimeError("Audio too large")
        elapsed = (time.perf_counter() - start) * 1000
        output = folder / f"run_{index + 1:02d}.wav"
        if args.stream:
            seconds = len(data) / 2 / rate
            with wave.open(str(output), "wb") as handle:
                handle.setnchannels(1); handle.setsampwidth(2); handle.setframerate(rate); handle.writeframes(data)
        else:
            with wave.open(io.BytesIO(data)) as handle:
                seconds = handle.getnframes() / handle.getframerate()
            output.write_bytes(data)
        if seconds <= 0:
            raise RuntimeError("Empty audio")
        row = {"run": index + 1, "first_audio_ms": first, "total_ms": elapsed,
               "audio_seconds": seconds, "rtf": elapsed / 1000 / seconds, "audio": str(output)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        (folder / "report.json").write_text(json.dumps({"worker": health, "text": args.text,
            "streaming": args.stream, "rows": rows, "notes": ["第一轮可能仍含编解码预热，比较后两轮",
            "RTF=合成耗时/音频时长，越小越快；不是播放完成时间", "不是 3090 实测承诺"]},
            ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(folder / "report.json"))


if __name__ == "__main__":
    main()
