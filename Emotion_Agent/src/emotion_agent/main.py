from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .config import load_settings
from .errors import AgentError


def main():
    parser = argparse.ArgumentParser(description="Emotion Agent：Windows 3090 自托管语音对话")
    parser.add_argument("--config", help="默认 configs/default.yaml；cloud.yaml 必须显式指定")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="执行一轮真实音频或文本对话")
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio", type=Path)
    source.add_argument("--text")
    run.add_argument("--session-id")
    run.add_argument("--mode", choices=("text_only", "speech_label", "policy"))
    run.add_argument("--no-tts", action="store_true")
    serve = commands.add_parser("serve", help="启动录音/上传/文字演示界面")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    inspect = commands.add_parser("doctor", help="检查依赖、权重路径和模型服务")
    inspect.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    try:
        settings = load_settings(args.config)
        if args.command == "doctor":
            from .diagnostics import doctor

            result = doctor(settings, args.probe)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(0 if result["ready"] else 1)
        if args.command == "serve":
            import uvicorn
            from .server import create_app

            if args.host not in {"127.0.0.1", "localhost", "::1"} and not os.getenv("AGENT_ACCESS_TOKEN", "").strip():
                raise AgentError("对外绑定前请在 .env 设置 AGENT_ACCESS_TOKEN；或保留回环地址并使用 SSH 转发")
            uvicorn.run(create_app(settings), host=args.host, port=args.port, workers=1)
        else:
            from .pipeline.runner import Agent

            agent = Agent(settings)
            try:
                result = agent.run(audio=args.audio, text=args.text, session_id=args.session_id,
                                   mode=args.mode, synthesize=not args.no_tts)
                print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            finally:
                agent.close()
    except AgentError as exc:
        parser.exit(2, f"错误：{exc}\n")
