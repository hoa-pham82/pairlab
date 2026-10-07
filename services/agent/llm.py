"""Chat client for an OpenAI-compatible server (llama.cpp)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import httpx


@dataclass(frozen=True)
class ChatResult:
    """One assistant turn and what it cost."""

    message: dict
    model: str = "unknown"
    input_tokens: int = 0
    output_tokens: int = 0
    round_trip_seconds: float = 0.0
    ttft_seconds: float | None = None

    @property
    def tool_calls(self) -> list[dict]:
        return self.message.get("tool_calls") or []

    @property
    def content(self) -> str:
        return self.message.get("content") or ""


class ChatClient(Protocol):
    """Sends a conversation and tool schemas, returns the assistant's turn."""

    async def chat(self, messages: list[dict], tools: list[dict]) -> ChatResult:
        """Return the next assistant message."""


@dataclass
class OpenAICompatClient:
    """Calls ``/v1/chat/completions`` with deterministic sampling settings."""

    client: httpx.AsyncClient
    base_url: str = "http://localhost:8085"
    model: str = "qwen2.5-3b-instruct"
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 512
    timeout_seconds: float = 120.0
    extra_body: dict = field(default_factory=dict)

    async def chat(self, messages: list[dict], tools: list[dict]) -> ChatResult:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            **self.extra_body,
        }
        if tools:
            body["tools"] = tools
        started = time.perf_counter()
        response = await self.client.post(
            f"{self.base_url}/v1/chat/completions", json=body, timeout=self.timeout_seconds
        )
        elapsed = time.perf_counter() - started
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage") or {}
        prompt_ms = (data.get("timings") or {}).get("prompt_ms")
        return ChatResult(
            message=data["choices"][0]["message"],
            model=data.get("model", self.model),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            round_trip_seconds=elapsed,
            ttft_seconds=prompt_ms / 1000 if prompt_ms is not None else None,
        )
