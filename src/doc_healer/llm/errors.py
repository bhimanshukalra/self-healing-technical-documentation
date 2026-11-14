"""LLM error types. Callers mostly care about two cases: "can't verify right now"
(LLMUnavailable) and "something is misconfigured or malformed" (LLMError)."""

from __future__ import annotations


class LLMError(Exception):
    """A request failed in a way retrying or switching providers won't fix."""


class AuthError(LLMError):
    """The API key was rejected."""


class RateLimitError(Exception):
    """The provider throttled us (HTTP 429)."""

    def __init__(self, message: str = "rate limited", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after  # seconds the provider asked us to wait, if it said


class TransientError(Exception):
    """A server error, timeout or connection failure that may succeed on retry."""


class LLMUnavailable(LLMError):
    """No provider could serve the request. Callers should flag the item for human review
    instead of failing the whole run (NFR-3a)."""


class BudgetExceeded(LLMUnavailable):
    """The per-run cap on LLM calls was reached (NFR-2)."""
