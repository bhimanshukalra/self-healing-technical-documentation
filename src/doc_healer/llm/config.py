"""Provider presets and construction of an LLMClient from plain settings."""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from .client import LLMClient
from .provider import OpenAICompatibleProvider

# name -> (base_url, default model). Model names and free-tier limits change often, so
# treat the defaults as starting points and override `model` when a provider retires one.
PRESETS: dict[str, tuple[str, str]] = {
    "groq": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash"),
    "openai": ("https://api.openai.com/v1", "gpt-4o"),
}


@dataclass(frozen=True)
class ProviderConfig:
    provider: str  # a PRESETS key, or any name when base_url is given
    api_key: str
    model: str | None = None  # default: the preset's model
    base_url: str | None = None  # any OpenAI-compatible endpoint

    def __repr__(self) -> str:  # never print the key
        return f"ProviderConfig(provider={self.provider!r}, model={self.model!r})"


@dataclass(frozen=True)
class LLMConfig:
    primary: ProviderConfig
    fallback: ProviderConfig | None = None  # opt-in: free tiers may use requests for training
    max_calls: int | None = None
    max_retries: int = 3


def build_provider(cfg: ProviderConfig, http: httpx.Client | None = None) -> OpenAICompatibleProvider:
    preset = PRESETS.get(cfg.provider)
    base_url = cfg.base_url or (preset[0] if preset else None)
    if base_url is None:
        raise ValueError(f"unknown provider {cfg.provider!r}; use one of {sorted(PRESETS)} or pass base_url")
    model = cfg.model or (preset[1] if preset else None)
    if model is None:
        raise ValueError(f"provider {cfg.provider!r} needs an explicit model")
    if not cfg.api_key:
        raise ValueError(f"no API key for provider {cfg.provider!r}")
    return OpenAICompatibleProvider(cfg.provider, base_url, model, cfg.api_key, client=http)


def build_client(cfg: LLMConfig, http: httpx.Client | None = None, **client_kwargs) -> LLMClient:
    return LLMClient(
        build_provider(cfg.primary, http),
        build_provider(cfg.fallback, http) if cfg.fallback else None,
        max_calls=cfg.max_calls,
        max_retries=cfg.max_retries,
        **client_kwargs,
    )
