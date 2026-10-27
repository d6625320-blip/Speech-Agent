"""Run in an independent qwen-tts environment; no dependency on FunASR."""
from __future__ import annotations

import argparse
import base64
import io
import logging
import os
import time
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from .streaming import closing_stream, event_line

logger = logging.getLogger(__name__)


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    speaker: str = Field(default="Vivian", max_length=40)
    instruction: str = Field(default="", max_length=300)


def create_tts_app(model_dir: Path, device: str, dtype: str = "auto", *, backend: str = "native",
                   chunk_size: int = 8, warmup: bool = True) -> FastAPI:
    # Download is a separate explicit command, never an inference side effect.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import soundfile as sf
    import numpy as np

    if backend not in {"native", "faster"} or not 1 <= chunk_size <= 24:
        raise ValueError("backend 必须为 native/faster，chunk_size 必须为 1–24")

    if not (model_dir / "config.json").is_file():
        raise RuntimeError(f"TTS 模型目录不存在：{model_dir}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("TTS 环境的 PyTorch 无法使用 CUDA，请检查驱动及 torch 安装")
    if dtype == "auto":
        dtype = "bfloat16" if device.startswith("cuda") and torch.cuda.is_bf16_supported() else ("float16" if device.startswith("cuda") else "float32")
    if backend == "faster":
        from faster_qwen3_tts import FasterQwen3TTS

        if not device.startswith("cuda"):
            raise RuntimeError("faster 后端需要 CUDA，请使用 --device cuda:0")
        # v0.5.4's Torch loader accepts a local directory but does not forward
        # local_files_only. The offline variables above prevent network fallback.
        model = FasterQwen3TTS.from_pretrained(str(model_dir), device=device, dtype=getattr(torch, dtype),
                                               backend="torch", attn_implementation="sdpa")
        speaker_model = model.model
        if warmup:
            with torch.inference_mode():
                model.warmup()
    else:
        from qwen_tts import Qwen3TTSModel

        model = Qwen3TTSModel.from_pretrained(
            str(model_dir), device_map=device, dtype=getattr(torch, dtype), attn_implementation="sdpa", local_files_only=True,
        )
        speaker_model = model
    speakers = {name.lower(): name for name in speaker_model.get_supported_speakers()}
    style_supported = "1.7B" in model_dir.name
    lock = Lock()
    app = FastAPI(title="Local Qwen3-TTS", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health():
        return {"ready": True, "model": model_dir.name, "device": device, "style_supported": style_supported,
                "backend": backend, "streaming": backend == "faster", "dtype": dtype,
                "warmup_completed": backend == "faster" and warmup, "chunk_size": chunk_size}

    @app.post("/synthesize")
    def synthesize(request: SynthesisRequest):
        if request.speaker.lower() not in speakers:
            raise HTTPException(400, "不支持该音色，请检查模型支持的 speaker")
        if not lock.acquire(blocking=False):
            raise HTTPException(503, "TTS 正忙，请稍后重试")
        try:
            with torch.inference_mode():
                wavs, sample_rate = model.generate_custom_voice(
                    text=request.text, language="Chinese", speaker=speakers[request.speaker.lower()],
                    instruct=request.instruction if style_supported else "", max_new_tokens=2048,
                )
            buffer = io.BytesIO()
            sf.write(buffer, wavs[0], sample_rate, format="WAV", subtype="PCM_16")
            return Response(buffer.getvalue(), media_type="audio/wav", headers={
                "X-Style-Applied": str(bool(request.instruction) and style_supported).lower(),
            })
        finally:
            lock.release()

    @app.post("/synthesize/stream")
    def synthesize_stream(request: SynthesisRequest):
        if backend != "faster":
            raise HTTPException(501, "原生后端不提供音频块流式输出；请使用 --backend faster，或取消网页的流式播放")
        if request.speaker.lower() not in speakers:
            raise HTTPException(400, "不支持该音色")

        def generate():
            if not lock.acquire(blocking=False):
                yield event_line({"type": "error", "detail": "TTS 正忙，请稍后重试"})
                return
            source = None
            start, first, samples, rate = time.perf_counter(), None, 0, None
            try:
                source = model.generate_custom_voice_streaming(
                    text=request.text, language="Chinese", speaker=speakers[request.speaker.lower()],
                    instruct=request.instruction if style_supported else "", max_new_tokens=2048, chunk_size=chunk_size,
                )
                sequence = 0
                while True:
                    # Do not keep a thread-local inference_mode context active
                    # across yields: ASGI can resume the generator on another thread.
                    with torch.inference_mode():
                        try:
                            audio, sample_rate, _ = next(source)
                        except StopIteration:
                            break
                    values = np.asarray(audio, dtype=np.float32).reshape(-1)
                    if not len(values):
                        continue
                    if not np.isfinite(values).all() or sample_rate != 24000:
                        raise ValueError("流式音频采样率或数值异常")
                    if rate is None:
                        rate = int(sample_rate)
                        yield event_line({"type": "start", "sample_rate": rate, "channels": 1,
                                          "encoding": "pcm_s16le", "backend": backend,
                                          "style_applied": bool(request.instruction) and style_supported})
                    if samples + len(values) > 24000 * 180:
                        raise ValueError("流式输出超过 180 秒限制")
                    # Bound each network event even if a backend yields a large block.
                    pcm = np.rint(np.clip(values, -1, 1) * 32767).astype("<i2").tobytes()
                    for offset in range(0, len(pcm), 32768):
                        if first is None:
                            first = (time.perf_counter() - start) * 1000
                        yield event_line({"type": "audio", "sequence": sequence,
                                          "pcm": base64.b64encode(pcm[offset:offset + 32768]).decode("ascii")})
                        sequence += 1
                    samples += len(values)
                if not samples:
                    raise ValueError("TTS 未生成音频")
                elapsed = (time.perf_counter() - start) * 1000
                yield event_line({"type": "done", "usage": {"backend": backend, "first_audio_ms": first,
                    "wall_ms": elapsed, "audio_seconds": samples / rate, "rtf": elapsed / 1000 / (samples / rate)}})
            except Exception:
                logger.exception("Streaming TTS failed")
                yield event_line({"type": "error", "detail": "流式合成失败，请查看 TTS 服务窗口；文字回复已保留"})
            finally:
                try:
                    if source is not None:
                        source.close()
                finally:
                    lock.release()

        return StreamingResponse(closing_stream(generate()), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="自托管 Qwen3-TTS，仅绑定服务器回环地址")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--port", type=int, default=9880)
    parser.add_argument("--backend", choices=("native", "faster"), default="native")
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()
    uvicorn.run(create_tts_app(args.model_dir.resolve(), args.device, args.dtype, backend=args.backend,
                              chunk_size=args.chunk_size, warmup=not args.no_warmup),
                host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
