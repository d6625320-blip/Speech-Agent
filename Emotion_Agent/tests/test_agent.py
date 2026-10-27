from __future__ import annotations

import base64
import io
import importlib.util
import json
import math
import os
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from emotion_agent.asr.service import CloudASR, SenseVoiceASR
from emotion_agent.audio.preprocess import prepare_audio
from emotion_agent.config import APP_ROOT, Settings, load_settings
from emotion_agent.dialogue.llm import DialogueLLM, clean_reply
from emotion_agent.dialogue.memory import Memory
from emotion_agent.dialogue.policy import choose_policy
from emotion_agent.emotion.fusion import assemble_state
from emotion_agent.errors import AgentError, AudioError, ConfigurationError, ProviderError
from emotion_agent.evaluation import character_errors
from emotion_agent.network import JSONClient
from emotion_agent.pipeline.runner import Agent
from emotion_agent.schemas import ASRResult, LABELS, LLMResult, SpeechEmotion, TTSResult
from emotion_agent.server import create_app
from emotion_agent.tts.service import SpeechSynthesis
from emotion_agent.tts.worker import create_tts_app, SynthesisRequest


def emotion(label="sad", confidence=0.85):
    probabilities = {key: (1 - confidence) / 6 for key in LABELS}
    probabilities[label] = confidence
    p = np.array(list(probabilities.values()))
    return SpeechEmotion(True, label, probabilities, confidence, confidence - (1 - confidence) / 6,
                         float(-(p * np.log(p)).sum() / np.log(7)), False, "test_fixture")


class FakeASR:
    def __init__(self, text="今天的实验失败了"):
        self.text, self.calls = text, 0

    def transcribe(self, path):
        self.calls += 1
        return ASRResult(self.text)


class FakeSpeech:
    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def predict(self, path):
        self.calls += 1
        if self.fail:
            raise RuntimeError("simulated failure")
        return emotion()


class FakeLLM:
    def __init__(self, fail=False):
        self.histories, self.calls, self.fail = [], 0, fail

    def generate(self, text, history, policy):
        self.calls += 1
        self.histories.append(list(history))
        if self.fail:
            raise ProviderError("simulated LLM error")
        return LLMResult("可以先检查数据和实验记录。", {"prompt_tokens": 10, "completion_tokens": 8})


class FakeTTS:
    def __init__(self, fail=False):
        self.fail = fail

    def synthesize(self, text, instruction, output):
        if self.fail:
            raise ProviderError("simulated TTS failure")
        sf.write(str(output), np.sin(np.arange(2400) * 0.1) * 0.1, 24000, subtype="PCM_16")
        return TTSResult(str(output), style_applied=True)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.settings = load_settings()
        self.settings.data["output_dir"] = str(self.directory / "outputs")
        self.source = self.directory / "source.wav"
        sf.write(str(self.source), np.sin(np.arange(16000) * 0.08) * 0.1, 16000)

    def agent(self, **kwargs):
        agent = Agent(self.settings, asr=kwargs.get("asr", FakeASR()), speech=kwargs.get("speech", FakeSpeech()),
                      llm=kwargs.get("llm", FakeLLM()), tts=kwargs.get("tts", FakeTTS()))
        self.addCleanup(agent.close)
        return agent

    def test_default_uses_local_providers(self):
        agent = Agent(self.settings)
        self.addCleanup(agent.close)
        self.assertIsInstance(agent.asr, SenseVoiceASR)
        self.assertEqual(self.settings.data["models"]["llm"]["provider"], "ollama")
        self.assertEqual(self.settings.data["models"]["tts"]["provider"], "local_qwen")

    def test_voice_end_to_end_and_persisted_trace(self):
        result = self.agent().run(audio=self.source)
        self.assertEqual(result.emotion.status, "known_emotion")
        self.assertEqual(result.policy.strategy, "support")
        self.assertTrue(Path(result.audio_path).is_file())
        saved = json.loads((Path(result.audio_path).parent / "turn.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["trace_id"], result.trace_id)
        self.assertGreater(saved["timings_ms"]["total"], 0)

    def test_silence_bypasses_asr_speech_and_llm(self):
        sf.write(str(self.source), np.zeros(16000), 16000)
        asr, speech, llm = FakeASR(), FakeSpeech(), FakeLLM()
        result = self.agent(asr=asr, speech=speech, llm=llm).run(audio=self.source)
        self.assertEqual(result.emotion.status, "no_audio")
        self.assertEqual((asr.calls, speech.calls, llm.calls), (0, 0, 0))

    def test_empty_transcript_does_not_invent_dialogue(self):
        llm = FakeLLM()
        result = self.agent(asr=FakeASR(""), llm=llm).run(audio=self.source)
        self.assertEqual(result.emotion.status, "unclear_speech")
        self.assertEqual(llm.calls, 0)

    def test_expert_failure_keeps_real_transcript(self):
        with self.assertLogs("emotion_agent.pipeline.runner", level="ERROR"):
            result = self.agent(speech=FakeSpeech(fail=True)).run(audio=self.source)
        self.assertEqual(result.transcript, "今天的实验失败了")
        self.assertFalse(result.emotion.reliable)
        self.assertEqual(result.policy.strategy, "neutral_fallback")
        self.assertEqual(result.status, "degraded")

    def test_tts_failure_preserves_reply_and_memory(self):
        agent = self.agent(tts=FakeTTS(fail=True))
        with self.assertLogs("emotion_agent.pipeline.runner", level="ERROR"):
            result = agent.run(text="你好", session_id="one")
        self.assertIsNone(result.audio_path)
        self.assertTrue(result.reply)
        self.assertEqual(len(agent.memory.sessions["one"].messages), 2)

    def test_llm_failure_does_not_commit_history(self):
        agent = self.agent(llm=FakeLLM(fail=True))
        with self.assertRaises(ProviderError):
            agent.run(text="你好", session_id="one")
        self.assertEqual(agent.memory.sessions["one"].messages, [])
        self.assertEqual(len(list((self.directory / "outputs").glob("*/error.json"))), 1)

    def test_multiturn_memory_and_session_isolation(self):
        llm = FakeLLM()
        agent = self.agent(llm=llm)
        agent.run(text="你好", session_id="a", synthesize=False)
        agent.run(text="继续", session_id="a", synthesize=False)
        agent.run(text="另一段对话", session_id="b", synthesize=False)
        self.assertEqual(len(llm.histories[1]), 2)
        self.assertEqual(llm.histories[2], [])

    def test_text_only_ablation_skips_speech(self):
        speech = FakeSpeech()
        result = self.agent(speech=speech).run(audio=self.source, mode="text_only", synthesize=False)
        self.assertEqual(speech.calls, 0)
        self.assertEqual(result.policy.strategy, "text_only")

    def test_invalid_session_and_input_rejected(self):
        agent = self.agent()
        with self.assertRaises(AgentError):
            agent.run(text="hello", session_id="../bad")
        with self.assertRaises(AgentError):
            agent.run(audio=self.source, text="both")

    def test_low_confidence_and_conflict_policy(self):
        state = assemble_state(emotion(confidence=0.2), self.settings.data["speech"])
        self.assertFalse(state.reliable)
        self.assertIsNone(state.label)
        self.assertEqual(state.speech.label, "sad")  # Raw prediction survives the policy gate.
        self.assertEqual(state.reason, "speech_gate_failed")
        self.assertEqual(len(state.speech.probabilities), 7)
        state = assemble_state(emotion("happy"), self.settings.data["speech"], "我很难过")
        self.assertTrue(state.conflict)
        self.assertEqual(choose_policy(state, "policy", "我很难过").strategy, "neutral_fallback")

    def test_emergency_policy_does_not_call_llm(self):
        llm = FakeLLM()
        result = self.agent(llm=llm).run(text="我要自杀", synthesize=False)
        self.assertEqual(llm.calls, 0)
        self.assertEqual(result.policy.strategy, "immediate_safety")

    def test_audio_stereo_resampling_and_length_guard(self):
        wave = np.stack([np.ones(48000) * .1, np.ones(48000) * .2], axis=1)
        sf.write(str(self.source), wave, 48000)
        target = self.directory / "converted.wav"
        info = prepare_audio(self.source, target, self.settings.data["audio"])
        self.assertEqual(sf.info(str(target)).samplerate, 16000)
        self.assertEqual(sf.info(str(target)).channels, 1)
        self.assertAlmostEqual(info.rms, .15, places=2)
        sf.write(str(self.source), np.zeros(100), 16000)
        with self.assertRaises(AudioError):
            prepare_audio(self.source, target, self.settings.data["audio"])

    def test_memory_bound_and_busy_session(self):
        memory = Memory(self.settings.data["dialogue"])
        with memory.session("a") as (_, conversation):
            with self.assertRaises(AgentError):
                with memory.session("a"):
                    pass
            for i in range(20):
                memory.commit(conversation, str(i), "reply")
            self.assertEqual(len(conversation.messages), 12)

    def test_http_auth_audio_upload_and_traversal(self):
        with patch.dict(os.environ, {"AGENT_ACCESS_TOKEN": "test-token"}):
            with TestClient(create_app(self.settings, self.agent())) as client:
                self.assertEqual(client.get("/").status_code, 200)
                self.assertEqual(client.get("/api/health").status_code, 401)
                headers = {"Authorization": "Bearer test-token"}
                result = client.post("/api/turn?session_id=web", content=self.source.read_bytes(), headers=headers)
                self.assertEqual(result.status_code, 200, result.text)
                self.assertNotIn("audio_path", result.json())
                audio_url = result.json()["audio_url"]
                self.assertEqual(client.get(audio_url, headers=headers).status_code, 200)
                self.assertEqual(client.get("/api/audio/not-a-trace", headers=headers).status_code, 404)
                input_url = result.json()["input_audio_url"]
                self.assertEqual(client.get(input_url).status_code, 401)
                original = client.get(input_url, headers=headers)
                self.assertEqual(original.status_code, 200)
                waveform, sample_rate = sf.read(io.BytesIO(original.content))
                self.assertEqual(sample_rate, 16000)
                self.assertEqual(len(waveform), 16000)
                self.assertEqual(result.json()["emotion_gate"]["confidence_threshold"], 0.55)
                self.assertEqual(client.get("/api/input-audio/not-a-trace", headers=headers).status_code, 404)
                self.assertEqual(client.get("/api/input-audio/" + "0" * 32, headers=headers).status_code, 404)
                text = client.post("/api/text", json={"text": "你好", "synthesize": False}, headers=headers)
                self.assertEqual(text.status_code, 200)
                self.assertIsNone(text.json()["input_audio_url"])

    def test_config_rejects_unsupported_tts_style(self):
        self.settings.data["models"]["tts"]["provider"] = "qwen3"
        with self.assertRaises(ConfigurationError):
            self.settings.validate()

    def test_large_history_is_trimmed_without_dropping_system(self):
        settings = self.settings
        def respond(request):
            messages = json.loads(request.content)["messages"]
            self.assertEqual(messages[0]["role"], "system")
            self.assertEqual(messages[-1]["content"], "当前问题")
            self.assertLessEqual(sum(len(m["content"]) for m in messages), settings.data["dialogue"]["max_prompt_chars"])
            return httpx.Response(200, json={"message": {"content": "当前回复。"}})
        client = JSONClient(5, transport=httpx.MockTransport(respond))
        self.addCleanup(client.close)
        history = [{"role": role, "content": "旧消息" * 500} for role in ("user", "assistant", "user", "assistant")]
        policy = choose_policy(assemble_state(SpeechEmotion(), settings.data["speech"]), "policy", "当前问题")
        DialogueLLM(settings, client).generate("当前问题", history, policy)

    def test_tts_worker_contract_without_real_weights(self):
        directory = self.directory / "Qwen3-TTS-12Hz-1.7B-CustomVoice"
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        received = {}
        class Model:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                received.update(kwargs)
                return cls()
            def get_supported_speakers(self):
                return ["Vivian"]
            def generate_custom_voice(self, **kwargs):
                received["generation"] = kwargs
                return [np.zeros(2400)], 24000
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True),
                                bfloat16="fixture-bf16", inference_mode=nullcontext)
        with patch.dict("sys.modules", {"torch": torch, "qwen_tts": SimpleNamespace(Qwen3TTSModel=Model)}), patch.dict(os.environ):
            app = create_tts_app(directory, "cuda:0")
            with TestClient(app) as client:
                self.assertTrue(client.get("/health").json()["ready"])
                response = client.post("/synthesize", json={"text": "你好", "speaker": "Vivian", "instruction": "温和"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["x-style-applied"], "true")
                self.assertEqual(received["generation"]["language"], "Chinese")
                self.assertEqual(received["attn_implementation"], "sdpa")
                self.assertTrue(received["local_files_only"])
                self.assertEqual(client.post("/synthesize", json={"text": "你好", "speaker": "unknown"}).status_code, 400)

    def test_cer_and_reply_cleaning(self):
        self.assertEqual(character_errors("你好，世界！", "你好世间"), (1, 4))
        self.assertEqual(clean_reply("<think>hidden</think>你好。", 20), ("你好。", False))
        with self.assertRaises(ProviderError):
            clean_reply("<think>unfinished", 20)

    def test_batch_evaluation_releases_completed_sessions(self):
        path = APP_ROOT / "scripts/evaluate.py"
        spec = importlib.util.spec_from_file_location("evaluation_script_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        manifest = self.directory / "manifest.jsonl"
        manifest.write_text("\n".join(json.dumps({"text": f"问题{i}"}) for i in range(40)), encoding="utf-8")
        output = self.directory / "report.json"
        agent = self.agent()
        arguments = ["evaluate.py", "--manifest", str(manifest), "--output", str(output), "--no-tts"]
        with patch.object(module, "Agent", return_value=agent), patch("sys.argv", arguments), patch("sys.stdout", io.StringIO()):
            module.main()
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(report["summary"]["policy"]["completed"], 40)
        self.assertEqual(report["summary"]["policy"]["asr_cer"], None)
        self.assertEqual(agent.memory.sessions, {})


class ProviderContractTests(unittest.TestCase):
    def test_local_ollama_payload_does_not_use_cloud(self):
        settings = load_settings()
        calls = []
        def respond(request):
            calls.append(request)
            body = json.loads(request.content)
            self.assertFalse(body["think"])
            self.assertFalse(body["stream"])
            self.assertNotIn("Authorization", request.headers)
            self.assertEqual(body["options"]["num_ctx"], 4096)
            return httpx.Response(200, json={"message": {"content": "你好。"}, "eval_count": 3})
        client = JSONClient(5, transport=httpx.MockTransport(respond))
        self.addCleanup(client.close)
        policy = choose_policy(assemble_state(SpeechEmotion(), settings.data["speech"]), "policy", "你好")
        self.assertEqual(DialogueLLM(settings, client).generate("你好", [], policy).text, "你好。")
        self.assertEqual(calls[0].url.host, "127.0.0.1")

    def test_cloud_asr_exact_audio_schema(self):
        settings = load_settings(APP_ROOT / "configs/cloud.yaml")
        def respond(request):
            body = json.loads(request.content)
            audio = body["messages"][0]["content"][0]
            self.assertEqual(audio["type"], "input_audio")
            self.assertTrue(audio["input_audio"]["data"].startswith("data:audio/wav;base64,"))
            self.assertEqual(request.headers["Authorization"], "Bearer fixture")
            return httpx.Response(200, json={"choices": [{"message": {"content": "你好"}}]})
        client = JSONClient(5, transport=httpx.MockTransport(respond))
        self.addCleanup(client.close)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"DASHSCOPE_API_KEY": "fixture"}):
            audio = Path(directory) / "input.wav"
            sf.write(str(audio), np.zeros(16000), 16000)
            self.assertEqual(CloudASR(settings, client).transcribe(audio).text, "你好")

    def test_local_tts_wav_and_style_header(self):
        settings = load_settings()
        data = io.BytesIO()
        sf.write(data, np.zeros(2400), 24000, format="WAV")
        def respond(request):
            self.assertEqual(request.url.path, "/synthesize")
            self.assertEqual(json.loads(request.content)["speaker"], "Vivian")
            return httpx.Response(200, content=data.getvalue(), headers={"X-Style-Applied": "true"})
        client = JSONClient(5, transport=httpx.MockTransport(respond))
        self.addCleanup(client.close)
        with tempfile.TemporaryDirectory() as directory:
            result = SpeechSynthesis(settings, client).synthesize("你好", "温和", Path(directory) / "reply.wav")
            self.assertTrue(Path(result.audio_path).is_file())
            self.assertTrue(result.style_applied)

    def test_ambiguous_timeout_is_not_retried(self):
        calls = []
        def respond(request):
            calls.append(request)
            raise httpx.ReadTimeout("fixture", request=request)
        client = JSONClient(5, retries=2, transport=httpx.MockTransport(respond))
        self.addCleanup(client.close)
        with self.assertRaises(ProviderError):
            client.post("http://127.0.0.1/test", {})
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
