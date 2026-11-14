"""Provider interface and an OpenAI-compatible HTTP implementation (Groq, Gemini, OpenAI...)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

import httpx

from .errors import AuthError, LLMError, RateLimitError, TransientError


class Message(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


def system(text: str) -> Message:
    return {"role": "system", "content": text}


def user(text: str) -> Message:
    return {"role": "user", "content": text}


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    data: dict[str, Any] | None = None  # parsed JSON, set by LLMClient.complete_json


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(
        self,
        messages: list[Message],
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """One request. Raise RateLimitError / TransientError for retryable failures and
        LLMError (or AuthError) for the rest."""
        ...


class OpenAICompatibleProvider:
    """Chat-completions over HTTP for any provider that speaks the OpenAI wire format."""

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key: str,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ):
        self.name, self.model = name, model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=timeout)

    def __repr__(self) -> str:  # never print the key
        return f"OpenAICompatibleProvider(name={self.name!r}, model={self.model!r})"

    def complete(self, messages, *, json_mode=False, max_tokens=None, temperature=0.0) -> LLMResponse:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        try:
            resp = self._client.post(self._url, json=payload, headers={"Authorization": f"Bearer {self._api_key}"})
        except httpx.TimeoutException as exc:
            raise TransientError(f"{self.name}: timeout") from exc
        except httpx.TransportError as exc:
            raise TransientError(f"{self.name}: {exc.__class__.__name__}") from exc

        status = resp.status_code
        if status == 429:
            raise RateLimitError(f"{self.name}: rate limited", _retry_after(resp))
        if status >= 500 or status in (408, 409):
            raise TransientError(f"{self.name}: HTTP {status}")
        if status in (401, 403):
            raise AuthError(f"{self.name}: HTTP {status}, check the API key")
        if status >= 400:
            raise LLMError(f"{self.name}: HTTP {status}: {resp.text[:300]}")
        try:
            text = resp.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.name}: unexpected response shape") from exc
        if not isinstance(text, str):
            raise LLMError(f"{self.name}: response had no text content")
        return LLMResponse(text, self.name, self.model)


def _retry_after(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after")
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:  # an HTTP-date; not worth parsing, fall back to backoff
        return None
