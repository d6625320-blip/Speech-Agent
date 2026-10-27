"""Stream saved assistant text through the isolated TTS worker and persist WAV."""
from __future__ import annotations

import base64
import json
import logging
import os
import time
import wave
from pathlib import Path

from ..errors import ProviderError
from .streaming import event_line

logger = logging.getLogger(__name__)


def write_turn(path: Path, turn: dict):
    temporary = path.with_suffix(".json.part")
    temporary.write_text(json.dumps(turn, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def stream_saved_reply(agent, folder: Path, public_result):
    """One GPU operation at a time, including between separate HTTP requests.

    Only text saved by Agent.run can be synthesized. A client cannot inject a
    different text or style into this second phase.
    """
    if not agent.slot.acquire(blocking=False):
        yield event_line({"type": "error", "detail": "服务器正在处理另一轮，请稍后重试"})
        return
    part = folder / "reply.stream.part"
    handle, completed, started, turn = None, False, False, None
    turn_path = folder / "turn.json"
    try:
        turn = json.loads(turn_path.read_text(encoding="utf-8"))
        if (folder / "reply.wav").is_file() and turn.get("audio_path"):
            yield event_line({"type": "done", "cached": True, "result": public_result(turn)})
            completed = True
            return
        config = agent.settings.data
        voice = config["models"]["tts"]
        base = os.getenv("LOCAL_TTS_BASE_URL", "").strip() or voice["base_url"]
        style = config["tts"]["instruction_control"]
        turn["tts_stream_status"] = "generating"
        write_turn(turn_path, turn)
        started = True
        start, first, size, rate, sequence = time.perf_counter(), None, 0, None, 0
        metadata, worker_done = None, None
        with agent.client.client.stream("POST", base.rstrip("/") + "/synthesize/stream", json={
            "text": turn["reply"], "speaker": voice["voice"],
            "instruction": turn["policy"]["tts_instruction"] if style else "",
        }) as response:
            if response.is_error:
                response.read()
                if response.status_code == 501:
                    raise ProviderError("TTS 仍是原生后端，请启动 --backend faster；或取消网页的流式播放")
                raise ProviderError(f"TTS 服务返回 HTTP {response.status_code}，请检查 TTS 服务窗口")
            for line in response.iter_lines():
                if not line.strip():
                    continue
                if len(line) > 131072:
                    raise ProviderError("TTS 流事件过大")
                event = json.loads(line)
                kind = event.get("type")
                if worker_done is not None:
                    raise ProviderError("TTS 完成事件后仍返回数据")
                if kind == "error":
                    raise ProviderError(str(event.get("detail", "TTS 流式生成失败")))
                if kind == "start":
                    if metadata is not None or event.get("sample_rate") != 24000 or event.get("channels") != 1 or event.get("encoding") != "pcm_s16le":
                        raise ProviderError("TTS 流格式不正确")
                    metadata, rate = event, event["sample_rate"]
                    handle = wave.open(str(part), "wb")
                    handle.setnchannels(1)
                    handle.setsampwidth(2)
                    handle.setframerate(rate)
                    yield event_line(event)
                elif kind == "audio":
                    if metadata is None or event.get("sequence") != sequence:
                        raise ProviderError("TTS 流顺序不正确")
                    pcm = base64.b64decode(event["pcm"], validate=True)
                    if not pcm or len(pcm) % 2 or size + len(pcm) + 44 > config["tts"]["max_audio_bytes"]:
                        raise ProviderError("TTS 音频块为空、损坏或超过大小限制")
                    handle.writeframesraw(pcm)
                    size += len(pcm)
                    sequence += 1
                    if first is None:
                        first = (time.perf_counter() - start) * 1000
                    yield event_line(event)
                elif kind == "done":
                    worker_done = event
                else:
                    raise ProviderError("未知 TTS 流事件")
        if worker_done is None or not size:
            raise ProviderError("TTS 连接提前结束，未收到完整音频")
        handle.close()
        handle = None
        part.replace(folder / "reply.wav")
        wall_ms, audio_seconds = (time.perf_counter() - start) * 1000, size / 2 / rate
        turn["audio_path"] = str(folder / "reply.wav")
        # Total here is the sum of two server processing phases, not browser RTT.
        turn["timings_ms"]["tts"] = wall_ms
        turn["timings_ms"]["tts_first_audio"] = first
        turn["timings_ms"]["total"] += wall_ms
        turn["usage"]["tts"] = {"backend": metadata.get("backend"), "streaming": True,
            "first_audio_ms": first, "wall_ms": wall_ms, "audio_seconds": audio_seconds,
            "rtf": wall_ms / 1000 / audio_seconds, "worker": worker_done.get("usage", {})}
        turn["usage"]["tts_style_applied"] = bool(metadata.get("style_applied"))
        if style and not metadata.get("style_applied"):
            turn["warnings"].append("当前 TTS 未应用表达指令，已使用普通语音")
        turn["tts_stream_status"] = "completed"
        write_turn(turn_path, turn)
        completed = True
        yield event_line({"type": "done", "cached": False, "result": public_result(turn)})
    except Exception as exc:
        logger.exception("TTS stream relay failed: %s", folder.name)
        detail = str(exc) if isinstance(exc, ProviderError) else "流式语音处理中断，请检查服务器日志；文字回复已保留"
        if started and turn is not None:
            turn["tts_stream_status"] = "failed"
            write_turn(turn_path, turn)
        yield event_line({"type": "error", "detail": detail})
    finally:
        try:
            if handle is not None:
                handle.close()
            part.unlink(missing_ok=True)
            if started and not completed and turn is not None and turn.get("tts_stream_status") == "generating":
                turn["tts_stream_status"] = "cancelled"
                write_turn(turn_path, turn)
        finally:
            agent.slot.release()
