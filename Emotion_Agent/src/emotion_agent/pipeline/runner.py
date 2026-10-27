from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from threading import BoundedSemaphore
from uuid import uuid4

from ..asr.service import CloudASR, SenseVoiceASR
from ..audio.preprocess import prepare_audio
from ..config import Settings
from ..dialogue.llm import DialogueLLM
from ..dialogue.memory import Memory
from ..dialogue.policy import choose_policy
from ..emotion.fusion import assemble_state
from ..emotion.speech import LocalSpeechExpert
from ..errors import AgentError
from ..network import JSONClient
from ..schemas import SpeechEmotion, TurnResult
from ..tts.service import SpeechSynthesis

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, settings: Settings, *, asr=None, speech=None, llm=None, tts=None, client=None):
        self.settings = settings
        c = settings.data
        self.client = client or JSONClient(c["network"]["timeout_seconds"], c["network"]["retries"])
        self.asr = asr or (SenseVoiceASR(settings) if c["models"]["asr"]["provider"] == "funasr" else CloudASR(settings, self.client))
        self.speech = speech or LocalSpeechExpert(settings)
        self.llm = llm or DialogueLLM(settings, self.client)
        self.tts = tts or SpeechSynthesis(settings, self.client)
        self.memory = Memory(c["dialogue"])
        self.slot = BoundedSemaphore(1)  # Bound GPU peak usage on the single 3090.

    def close(self):
        self.client.close()

    def run(self, *, audio: Path | None = None, text: str | None = None, session_id: str | None = None,
            mode: str | None = None, synthesize: bool = True) -> TurnResult:
        if (audio is None) == (text is None):
            raise AgentError("请提供音频或文本中的一种")
        mode = mode or self.settings.data["mode"]
        if mode not in {"text_only", "speech_label", "policy"}:
            raise AgentError("不支持该消融模式")
        if text is not None and (not text.strip() or len(text) > self.settings.data["dialogue"]["max_text_chars"]):
            raise AgentError("文本为空或超过配置的最大长度")
        if not self.slot.acquire(blocking=False):
            raise AgentError("服务器正在处理另一轮，请稍后重试")
        try:
            with self.memory.session(session_id) as (sid, conversation):
                return self._run(audio, text, sid, conversation, mode, synthesize)
        finally:
            self.slot.release()

    def _run(self, audio, text, sid, conversation, mode, synthesize):
        c = self.settings.data
        trace_id = uuid4().hex
        folder = self.settings.path(c["output_dir"]) / trace_id
        folder.mkdir(parents=True, exist_ok=False)
        start = time.perf_counter()
        timings, usage, warnings = {}, {}, []
        speech = SpeechEmotion(reason="text_input")
        fixed_reply = None

        def timed(name, function):
            before = time.perf_counter()
            try:
                return function()
            finally:
                timings[name] = (time.perf_counter() - before) * 1000

        try:
            if audio is not None:
                prepared = timed("audio", lambda: prepare_audio(audio, folder / "input.wav", c["audio"]))
                usage["input_audio_seconds"] = prepared.duration_seconds
                if prepared.silent:
                    text, fixed_reply = "", "没有听到清晰的声音，请靠近麦克风再说一次。"
                    speech = SpeechEmotion(reason="silent_audio")
                else:
                    recognized = timed("asr", lambda: self.asr.transcribe(prepared.path))
                    text, usage["asr"] = recognized.text.strip(), recognized.usage
                    if not text:
                        fixed_reply = "没有识别到清晰的话语，请再说一次，或者输入文字。"
                        speech = SpeechEmotion(reason="empty_transcript")
                    elif mode != "text_only" and c["speech"]["enabled"]:
                        try:
                            speech = timed("speech", lambda: self.speech.predict(prepared.path))
                        except Exception as exc:
                            # Inference failure may degrade emotion; never invent a prediction.
                            speech = SpeechEmotion(reason="expert_unavailable")
                            logger.exception("Speech expert failed; trace_id=%s", trace_id)
                            warnings.append(f"语音情感专家不可用（{type(exc).__name__}），本轮依据文本回应")
            text = text.strip()
            if len(text) > c["dialogue"]["max_text_chars"]:
                raise AgentError("识别文本过长，请缩短输入")
            state = assemble_state(speech, c["speech"], text)
            if fixed_reply:
                state.status = "no_audio" if speech.reason == "silent_audio" else "unclear_speech"
            policy = choose_policy(state, mode, text)
            reply = fixed_reply or policy.fixed_reply
            if reply is None:
                generated = timed("llm", lambda: self.llm.generate(text, list(conversation.messages), policy))
                reply, usage["llm"] = generated.text, generated.usage
                if generated.truncated:
                    warnings.append("回复达到长度限制，已保留可见内容")
            audio_path = None
            if synthesize and c["tts"]["enabled"]:
                try:
                    result = timed("tts", lambda: self.tts.synthesize(reply, policy.tts_instruction, folder / "reply.wav"))
                    audio_path, usage["tts"] = result.audio_path, result.usage
                    usage["tts_style_applied"] = result.style_applied
                    if c["tts"]["instruction_control"] and not result.style_applied:
                        warnings.append("当前 TTS 未应用表达指令，已使用普通语音")
                except Exception as exc:
                    logger.exception("TTS failed; trace_id=%s", trace_id)
                    warnings.append(f"语音合成失败（{type(exc).__name__}），文字回复已保留")
            # Missing speech and fixed clarification turns should not contaminate memory.
            if text and not fixed_reply:
                self.memory.commit(conversation, text, reply)
            timings["total"] = (time.perf_counter() - start) * 1000
            result = TurnResult(trace_id, sid, mode, text, reply, state, policy, audio_path,
                                timings, usage, warnings, "degraded" if warnings or fixed_reply else "ok")
            (folder / "turn.json").write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            return result
        except Exception as exc:
            (folder / "error.json").write_text(json.dumps({
                "trace_id": trace_id, "error_type": type(exc).__name__, "timings_ms": timings,
                "total_ms": (time.perf_counter() - start) * 1000,
            }, indent=2), encoding="utf-8")
            raise
