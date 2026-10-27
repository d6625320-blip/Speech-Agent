from __future__ import annotations

import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from uuid import uuid4

from ..errors import AgentError


@dataclass
class Conversation:
    messages: list[dict[str, str]] = field(default_factory=list)
    touched: float = field(default_factory=time.monotonic)
    lock: Lock = field(default_factory=Lock)


class Memory:
    def __init__(self, config: dict):
        self.config, self.sessions, self.lock = config, {}, Lock()

    @contextmanager
    def session(self, session_id: str | None = None):
        session_id = session_id or uuid4().hex
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", session_id):
            raise AgentError("session_id 只能包含 1–64 位字母、数字、下划线或连字符")
        with self.lock:
            now = time.monotonic()
            for key, value in list(self.sessions.items()):
                if now - value.touched > self.config["session_ttl_seconds"] and not value.lock.locked():
                    del self.sessions[key]
            if session_id not in self.sessions:
                if len(self.sessions) >= self.config["max_sessions"]:
                    raise AgentError("活动会话已达上限，请稍后重试或清理会话")
                self.sessions[session_id] = Conversation()
            conversation = self.sessions[session_id]
            # Acquiring inside registry lock prevents expiration between lookup and use.
            if not conversation.lock.acquire(blocking=False):
                raise AgentError("该会话仍在处理上一轮，请稍后重试")
            conversation.touched = now
        try:
            yield session_id, conversation
        finally:
            conversation.touched = time.monotonic()
            conversation.lock.release()

    def commit(self, conversation: Conversation, user: str, assistant: str):
        conversation.messages.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
        limit = int(self.config["max_history_turns"]) * 2
        conversation.messages[:] = conversation.messages[-limit:]

    def clear(self, session_id: str):
        with self.lock:
            conversation = self.sessions.get(session_id)
            if conversation and conversation.lock.locked():
                raise AgentError("会话正在处理，暂时不能清除")
            self.sessions.pop(session_id, None)
