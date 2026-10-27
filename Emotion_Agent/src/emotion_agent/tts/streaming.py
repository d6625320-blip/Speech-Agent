"""Small, explicit PCM/NDJSON protocol shared by worker and application."""
from __future__ import annotations

import json

import anyio
from starlette.concurrency import run_in_threadpool


def event_line(event: dict) -> bytes:
    return (json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _pull(iterator):
    try:
        return True, next(iterator)
    except StopIteration:
        return False, None


async def closing_stream(iterator):
    """Close synchronous HTTP/model generators even on a browser disconnect."""
    try:
        while True:
            found, item = await run_in_threadpool(_pull, iterator)
            if not found:
                break
            yield item
    finally:
        with anyio.CancelScope(shield=True):
            await run_in_threadpool(iterator.close)
