from __future__ import annotations

import hmac
import json
import os
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .config import Settings
from .errors import AgentError
from .pipeline.runner import Agent
from .tts.relay import stream_saved_reply
from .tts.streaming import closing_stream

WEB = Path(__file__).parent / "web"


class TextTurn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64)
    mode: str = "policy"
    synthesize: bool = True


def create_app(settings: Settings, agent: Agent | None = None) -> FastAPI:
    agent = agent or Agent(settings)

    @asynccontextmanager
    async def lifespan(app):
        yield
        agent.close()

    app = FastAPI(title="Emotion Agent", lifespan=lifespan)
    app.state.agent = agent
    token = os.getenv("AGENT_ACCESS_TOKEN", "").strip()

    @app.middleware("http")
    async def protect(request: Request, call_next):
        if request.url.path.startswith("/api/") and token:
            value = request.headers.get("Authorization", "").removeprefix("Bearer ")
            if not hmac.compare_digest(value, token):
                return JSONResponse({"detail": "请输入正确的访问口令"}, status_code=401)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(AgentError)
    async def expected_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        return JSONResponse({"detail": "服务处理失败，请查看服务器日志和 outputs 下的 error.json"}, status_code=500)

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/health")
    def health():
        models = settings.data["models"]
        return {"status": "running", "models": {k: {"provider": v["provider"], "model": v["model"]} for k, v in models.items()},
                "note": "进程存活不代表各模型已加载；请运行 doctor --probe"}

    def public_result(result):
        data = result.to_dict() if hasattr(result, "to_dict") else dict(result)
        trace_id = data["trace_id"]
        data["audio_url"] = f"/api/audio/{trace_id}" if data.get("audio_path") else None
        input_path = settings.path(settings.data["output_dir"]) / trace_id / "input.wav"
        data["input_audio_url"] = f"/api/input-audio/{trace_id}" if input_path.is_file() else None
        data["emotion_gate"] = {key: settings.data["speech"][key] for key in (
            "confidence_threshold", "margin_threshold", "max_normalized_entropy",
        )}
        data.pop("audio_path", None)
        return data

    @app.post("/api/text")
    def text_turn(body: TextTurn):
        return public_result(agent.run(text=body.text, session_id=body.session_id, mode=body.mode, synthesize=body.synthesize))

    @app.post("/api/turn")
    async def audio_turn(request: Request, session_id: str | None = None, mode: str = "policy", synthesize: bool = True):
        limit = settings.data["audio"]["max_bytes"]
        length = 0
        with tempfile.TemporaryDirectory(prefix="emotion_upload_") as folder:
            path = Path(folder) / "input.audio"
            with path.open("wb") as handle:
                async for chunk in request.stream():
                    length += len(chunk)
                    if length > limit:
                        raise HTTPException(413, "上传音频超过 10 MiB")
                    handle.write(chunk)
            result = await run_in_threadpool(agent.run, audio=path, session_id=session_id, mode=mode, synthesize=synthesize)
        return public_result(result)

    def audio_file(trace_id: str, filename: str):
        if not re.fullmatch(r"[a-f0-9]{32}", trace_id):
            raise HTTPException(404)
        path = settings.path(settings.data["output_dir"]) / trace_id / filename
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="audio/wav")

    @app.get("/api/audio/{trace_id}")
    def audio_output(trace_id: str):
        return audio_file(trace_id, "reply.wav")

    @app.get("/api/input-audio/{trace_id}")
    def audio_input(trace_id: str):
        return audio_file(trace_id, "input.wav")

    @app.post("/api/tts/{trace_id}/stream")
    def tts_stream(trace_id: str):
        if not re.fullmatch(r"[a-f0-9]{32}", trace_id):
            raise HTTPException(404)
        folder = settings.path(settings.data["output_dir"]) / trace_id
        if not (folder / "turn.json").is_file():
            raise HTTPException(404)
        if not settings.data["tts"]["enabled"] or settings.data["models"]["tts"]["provider"] != "local_qwen":
            raise HTTPException(400, "流式播放需要启用本地 TTS")
        return StreamingResponse(closing_stream(stream_saved_reply(agent, folder, public_result)),
            media_type="application/x-ndjson", headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"})

    @app.delete("/api/session/{session_id}")
    def clear_session(session_id: str):
        agent.memory.clear(session_id)
        return {"cleared": True}

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app
