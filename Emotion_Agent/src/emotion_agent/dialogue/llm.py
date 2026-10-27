from __future__ import annotations

import os
import re

from ..config import Settings
from ..errors import AgentError, ProviderError
from ..network import JSONClient, api_key
from ..schemas import DialoguePolicy, LLMResult


def clean_reply(text: str, limit: int) -> tuple[str, bool]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    if "<think>" in text:
        raise ProviderError("LLM 未返回完整的用户可见回复，请关闭思考模式或增加输出上限")
    text = re.sub(r"[*#`]", "", text).strip()
    if not text:
        raise ProviderError("LLM 返回了空回复")
    truncated = len(text) > limit
    if truncated:
        prefix = text[:limit]
        end = max(prefix.rfind(mark) for mark in "。！？.!?")
        text = prefix[:end + 1] if end >= limit // 3 else prefix + "……"
    return text, truncated


class DialogueLLM:
    def __init__(self, settings: Settings, client: JSONClient):
        self.settings, self.client = settings, client
        self.prompt = settings.path(settings.data["system_prompt"]).read_text(encoding="utf-8")

    def generate(self, text: str, history: list[dict], policy: DialoguePolicy) -> LLMResult:
        c = self.settings.data["models"]["llm"]
        messages = [
            {"role": "system", "content": self.prompt + f"\n本轮策略：{policy.guidance}\n回复不超过 {self.settings.data['dialogue']['max_reply_chars']} 字。"},
            *history, {"role": "user", "content": text},
        ]
        # Keep the system message and current user intact; remove whole old turns.
        # This is a conservative character budget, not an exact tokenizer count.
        budget = self.settings.data["dialogue"]["max_prompt_chars"]
        while sum(len(m["content"]) for m in messages) > budget and len(messages) > 2:
            del messages[1:3]
        if sum(len(m["content"]) for m in messages) > budget:
            raise AgentError("本轮文本超过上下文预算，请缩短输入")
        if c["provider"] == "ollama":
            base = os.getenv("OLLAMA_BASE_URL", "").strip() or c["base_url"]
            result = self.client.post(base.rstrip("/") + "/api/chat", {
                "model": c["model"], "messages": messages, "stream": False, "think": False,
                "keep_alive": c.get("keep_alive", "5m"),
                "options": {"temperature": c["temperature"], "num_predict": c["max_tokens"], "num_ctx": c["num_ctx"]},
            })
            try:
                reply = result["message"]["content"]
            except (KeyError, TypeError) as exc:
                raise ProviderError("Ollama 响应缺少回复文本") from exc
            usage = {"prompt_tokens": result.get("prompt_eval_count", 0), "completion_tokens": result.get("eval_count", 0)}
            provider_truncated = result.get("done_reason") == "length"
        else:
            base = os.getenv("LLM_BASE_URL", "").strip()
            url = base.rstrip("/") + "/chat/completions" if base else self.settings.cloud_url("/compatible-mode/v1/chat/completions")
            key = api_key("LLM_API_KEY") if os.getenv("LLM_API_KEY", "").strip() else api_key()
            result = self.client.post(url, {
                **c.get("extra_body", {}), "model": c["model"], "messages": messages,
                "stream": False, "temperature": c["temperature"], "max_tokens": c["max_tokens"],
            }, key)
            try:
                reply = result["choices"][0]["message"]["content"]
                provider_truncated = result["choices"][0].get("finish_reason") == "length"
            except (KeyError, IndexError, TypeError) as exc:
                raise ProviderError("LLM 响应缺少回复文本") from exc
            usage = result.get("usage", {})
        if not isinstance(reply, str):
            raise ProviderError("LLM 回复类型异常")
        reply, truncated = clean_reply(reply, self.settings.data["dialogue"]["max_reply_chars"])
        return LLMResult(reply, usage, truncated or provider_truncated)
