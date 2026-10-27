from __future__ import annotations

import os
import time
from typing import Any

import httpx

from .errors import ConfigurationError, ProviderError


def api_key(name: str = "DASHSCOPE_API_KEY") -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(f"未设置 {name}；本地模式无需此项，云端模式请填写 .env")
    return value


class JSONClient:
    """Bounded requests; never surface request bodies, credentials or audio data."""

    def __init__(self, timeout: float, retries: int = 0, transport=None):
        self.retries = retries
        self.client = httpx.Client(timeout=timeout, transport=transport, follow_redirects=False, trust_env=False)

    def post(self, url: str, payload: dict[str, Any], key: str | None = None) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        for attempt in range(self.retries + 1):
            try:
                response = self.client.post(url, json=payload, headers=headers)
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                # Do not retry ambiguous POST timeouts: the server may have completed it.
                raise ProviderError("模型服务连接失败或超时；请检查服务地址、启动状态和显存") from exc
            if response.status_code in (429, 502, 503, 504) and attempt < self.retries:
                time.sleep(min(2 ** attempt, 2))
                continue
            if response.is_error:
                request_id = response.headers.get("x-request-id", "")[:80]
                raise ProviderError(f"模型服务返回 HTTP {response.status_code}，request_id={request_id or 'unknown'}")
            try:
                result = response.json()
            except ValueError as exc:
                raise ProviderError("模型服务未返回有效 JSON") from exc
            if not isinstance(result, dict) or result.get("error"):
                raise ProviderError("模型服务返回错误或无效响应，请查看服务端日志")
            return result
        raise ProviderError("模型服务暂时不可用")

    def close(self) -> None:
        self.client.close()
