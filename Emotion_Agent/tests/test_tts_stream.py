from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import tempfile
import unittest
import wave
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np
from fastapi.testclient import TestClient

import test_agent as fixtures
from emotion_agent.config import load_settings
from emotion_agent.network import JSONClient
from emotion_agent.pipeline.runner import Agent
from emotion_agent.server import create_app
from emotion_agent.tts.relay import stream_saved_reply
from emotion_agent.tts.streaming import closing_stream, event_line
from emotion_agent.tts.worker import create_tts_app


def worker_events(*, done=True, wrong_sequence=False):
    events = [{"type": "start", "sample_rate": 24000, "channels": 1,
               "encoding": "pcm_s16le", "backend": "faster", "style_applied": True},
              {"type": "audio", "sequence": 9 if wrong_sequence else 0,
               "pcm": base64.b64encode(b"\x00\x40\x00\xc0" * 1200).decode()}]
    if done:
        events.append({"type": "done", "usage": {"first_audio_ms": 10}})
    return events


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.settings = load_settings()
        self.settings.data["output_dir"] = str(self.directory / "outputs")

    def agent(self, events=None, status=200):
        self.requests = []
        def respond(request):
            self.requests.append(request)
            return httpx.Response(status, content=b"".join(event_line(event) for event in
                (events if events is not None else worker_events())))
        client = JSONClient(5, transport=httpx.MockTransport(respond))
        agent = Agent(self.settings, client=client, asr=fixtures.FakeASR(), speech=fixtures.FakeSpeech(),
                      llm=fixtures.FakeLLM(), tts=fixtures.FakeTTS())
        self.addCleanup(agent.close)
        return agent

    def saved(self, agent):
        result = agent.run(text="今天实验失败了", synthesize=False)
        return result, self.settings.path(self.settings.data["output_dir"]) / result.trace_id

    def test_proxy_stream_writes_wav_and_uses_saved_text_and_style(self):
        agent = self.agent()
        with patch.dict(os.environ, {"AGENT_ACCESS_TOKEN": "test-token"}):
            with TestClient(create_app(self.settings, agent)) as client:
                headers = {"Authorization": "Bearer test-token"}
                reply = client.post("/api/text", json={"text": "你好", "synthesize": False}, headers=headers).json()
                url = f"/api/tts/{reply['trace_id']}/stream"
                self.assertEqual(client.post(url).status_code, 401)
                response = client.post(url, headers=headers)
                events = [json.loads(line) for line in response.text.splitlines()]
                self.assertEqual([event["type"] for event in events], ["start", "audio", "done"])
                final = events[-1]["result"]
                self.assertNotIn("audio_path", final)
                self.assertTrue(final["usage"]["tts"]["streaming"])
                self.assertTrue(final["usage"]["tts_style_applied"])
                self.assertAlmostEqual(final["usage"]["tts"]["audio_seconds"], .1)
                audio = client.get(final["audio_url"], headers=headers)
                with wave.open(io.BytesIO(audio.content)) as wav:
                    self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getnframes()), (24000, 1, 2400))
                payload = json.loads(self.requests[0].content)
                self.assertEqual(payload["text"], reply["reply"])
                self.assertEqual(payload["instruction"], reply["policy"]["tts_instruction"])
                self.assertEqual(client.post("/api/tts/not-a-trace/stream", headers=headers).status_code, 404)
                replay = client.post(url, headers=headers)
                self.assertTrue(json.loads(replay.text)["cached"])
                self.assertEqual(len(self.requests), 1)

    def test_audio_is_yielded_before_completion_and_disconnect_releases_gpu_slot(self):
        agent = self.agent()
        _, folder = self.saved(agent)
        source = stream_saved_reply(agent, folder, lambda turn: turn)
        self.assertEqual(json.loads(next(source))["type"], "start")
        self.assertEqual(json.loads(next(source))["type"], "audio")
        self.assertTrue((folder / "reply.stream.part").is_file())
        self.assertFalse((folder / "reply.wav").exists())
        self.assertFalse(agent.slot.acquire(blocking=False))
        source.close()
        self.assertFalse((folder / "reply.stream.part").exists())
        self.assertEqual(json.loads((folder / "turn.json").read_text(encoding="utf-8"))["tts_stream_status"], "cancelled")
        self.assertTrue(agent.slot.acquire(blocking=False))
        agent.slot.release()

    def test_truncated_or_out_of_order_stream_never_commits_partial_wav(self):
        for events in (worker_events(done=False), worker_events(wrong_sequence=True)):
            with self.subTest(events=events):
                agent = self.agent(events)
                _, folder = self.saved(agent)
                with self.assertLogs("emotion_agent.tts.relay", level="ERROR"):
                    output = [json.loads(line) for line in stream_saved_reply(agent, folder, lambda turn: turn)]
                self.assertEqual(output[-1]["type"], "error")
                self.assertFalse((folder / "reply.wav").exists())
                self.assertFalse((folder / "reply.stream.part").exists())
                self.assertTrue(agent.slot.acquire(blocking=False))
                agent.slot.release()

    def test_busy_slot_and_native_worker_have_explicit_errors(self):
        agent = self.agent(status=501)
        _, folder = self.saved(agent)
        agent.slot.acquire()
        try:
            output = list(stream_saved_reply(agent, folder, lambda turn: turn))
            self.assertIn("正在处理另一轮", json.loads(output[0])["detail"])
        finally:
            agent.slot.release()
        with self.assertLogs("emotion_agent.tts.relay", level="ERROR"):
            output = list(stream_saved_reply(agent, folder, lambda turn: turn))
        self.assertIn("--backend faster", json.loads(output[-1])["detail"])

    def test_async_adapter_closes_sync_iterator_on_disconnect(self):
        closed = []
        def source():
            try:
                yield b"first\n"
                yield b"second\n"
            finally:
                closed.append(True)
        async def consume():
            stream = closing_stream(source())
            self.assertEqual(await stream.__anext__(), b"first\n")
            await stream.aclose()
        asyncio.run(consume())
        self.assertEqual(closed, [True])

    def test_faster_worker_contract_preserves_instruct_and_warms_up(self):
        directory = self.directory / "Qwen3-TTS-12Hz-1.7B-CustomVoice"
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        calls = {}
        class Model:
            model = SimpleNamespace(get_supported_speakers=lambda: ["Vivian"])
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                calls["load"] = kwargs
                return cls()
            def warmup(self):
                calls["warmup"] = True
            def generate_custom_voice_streaming(self, **kwargs):
                calls["generate"] = kwargs
                yield np.ones(1200, dtype=np.float32) * .2, 24000, {}
                yield np.ones(1200, dtype=np.float32) * .1, 24000, {}
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True),
                                bfloat16="bf16-fixture", inference_mode=nullcontext)
        with patch.dict("sys.modules", {"torch": torch, "faster_qwen3_tts": SimpleNamespace(FasterQwen3TTS=Model)}), patch.dict(os.environ):
            app = create_tts_app(directory, "cuda:0", backend="faster", chunk_size=4)
            with TestClient(app) as client:
                self.assertTrue(client.get("/health").json()["streaming"])
                response = client.post("/synthesize/stream", json={"text": "你好", "instruction": "温和地说"})
                events = [json.loads(line) for line in response.text.splitlines()]
                self.assertEqual([event["type"] for event in events], ["start", "audio", "audio", "done"])
                self.assertEqual(calls["generate"]["instruct"], "温和地说")
                self.assertEqual(calls["generate"]["chunk_size"], 4)
                self.assertEqual(calls["load"]["backend"], "torch")
                self.assertEqual(calls["load"]["attn_implementation"], "sdpa")
                self.assertTrue(calls["warmup"])
                self.assertAlmostEqual(events[-1]["usage"]["audio_seconds"], .1)
                self.assertTrue(events[0]["style_applied"])


if __name__ == "__main__":
    unittest.main()
