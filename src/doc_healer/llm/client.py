"""LLMClient: retries, provider fallback, a call budget, and JSON replies on top of providers."""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from .errors import BudgetExceeded, LLMError, LLMUnavailable, RateLimitError, TransientError
from .provider import LLMProvider, LLMResponse, Message, user

log = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


@dataclass
class ClientStats:
    calls: int = 0  # logical calls, the unit the budget counts
    retries: int = 0
    fallback_calls: int = 0
    by_provider: Counter = field(default_factory=Counter)  # successful responses per provider


class _Exhausted(Exception):
    """A provider is out of retries (or told us to wait too long)."""


class LLMClient:
    def __init__(
        self,
        primary: LLMProvider,
        fallback: LLMProvider | None = None,
        *,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_wait: float = 60.0,
        max_calls: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.primary, self.fallback = primary, fallback
        self.max_retries, self.base_delay, self.max_wait = max_retries, base_delay, max_wait
        self.max_calls, self._sleep = max_calls, sleep
        self.stats = ClientStats()
        self._primary_exhausted = False  # once rate-limited out, stay on the fallback (NFR-3)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """One logical call: retry with backoff, then fall back, within the call budget.

        Raises LLMUnavailable (or its subclass BudgetExceeded) when nothing can serve it;
        callers should flag the item for human review rather than fail the run.
        """
        if self.max_calls is not None and self.stats.calls >= self.max_calls:
            raise BudgetExceeded(f"call budget of {self.max_calls} reached")
        self.stats.calls += 1
        return self._dispatch(
            list(messages), dict(json_mode=json_mode, max_tokens=max_tokens, temperature=temperature)
        )

    def _dispatch(self, messages: list[Message], kwargs: dict) -> LLMResponse:
        """Send to the primary, else the fallback. Doesn't touch the call budget."""
        if not self._primary_exhausted:
            try:
                return self._record(self._with_retries(self.primary, messages, kwargs), fallback=False)
            except _Exhausted as exc:
                if self.fallback is None:
                    raise LLMUnavailable(f"{self.primary.name} unavailable: {exc}") from exc
                self._primary_exhausted = True
                log.warning("%s exhausted (%s); using fallback %s for the rest of the run",
                            self.primary.name, exc, self.fallback.name)
        if self.fallback is None:
            raise LLMUnavailable(f"{self.primary.name} is exhausted and no fallback is configured")
        try:
            return self._record(self._with_retries(self.fallback, messages, kwargs), fallback=True)
        except _Exhausted as exc:
            raise LLMUnavailable(f"all providers unavailable: {exc}") from exc

    def complete_json(
        self,
        messages: Sequence[Message],
        *,
        required_keys: Sequence[str] = (),
        max_tokens: int | None = None,
        json_retries: int = 1,
    ) -> LLMResponse:
        """Like ``complete`` but the reply must be a JSON object containing ``required_keys``.

        The parsed object is on ``.data``. A malformed reply is re-asked up to
        ``json_retries`` times (within the same budgeted call) before raising LLMError.
        """
        conversation = list(messages)
        for attempt in range(json_retries + 1):
            if attempt == 0:
                response = self.complete(conversation, json_mode=True, max_tokens=max_tokens)
            else:  # a re-ask belongs to the same logical call, so it spends no extra budget
                response = self._dispatch(
                    list(conversation), dict(json_mode=True, max_tokens=max_tokens, temperature=0.0)
                )
            data, problem = _parse_json_object(response.text, required_keys)
            if problem is None:
                return replace(response, data=data)
            log.warning("%s returned unusable JSON (%s)", response.provider, problem)
            conversation += [
                {"role": "assistant", "content": response.text},
                user(f"That reply was not usable: {problem}. Reply with only a JSON object"
                     + (f" containing the keys {list(required_keys)}." if required_keys else ".")),
            ]
        raise LLMError(f"no valid JSON after {json_retries + 1} attempts: {problem}")

    # -- internals ---------------------------------------------------------------------

    def _record(self, response: LLMResponse, *, fallback: bool) -> LLMResponse:
        self.stats.by_provider[response.provider] += 1
        self.stats.fallback_calls += fallback
        return response

    def _with_retries(self, provider: LLMProvider, messages: list[Message], kwargs: dict) -> LLMResponse:
        for attempt in range(self.max_retries + 1):
            try:
                return provider.complete(messages, **kwargs)
            except (RateLimitError, TransientError) as exc:
                retry_after = getattr(exc, "retry_after", None)
                if retry_after is not None and retry_after > self.max_wait:
                    raise _Exhausted(f"{provider.name} asked us to wait {retry_after:.0f}s") from exc
                if attempt == self.max_retries:
                    raise _Exhausted(f"{provider.name} still failing after {attempt} retries: {exc}") from exc
                delay = retry_after if retry_after is not None else min(self.base_delay * 2**attempt, self.max_wait)
                log.info("%s: %s; retrying in %.1fs", provider.name, exc, delay)
                self.stats.retries += 1
                self._sleep(delay)
        raise AssertionError("unreachable")


def _parse_json_object(text: str, required_keys: Sequence[str]) -> tuple[dict | None, str | None]:
    """Parse a model reply as a JSON object, tolerating a markdown code fence around it."""
    body = text.strip()
    if fence := _FENCE_RE.match(body):
        body = fence.group(1)
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON ({exc.msg})"
    if not isinstance(data, dict):
        return None, "the JSON was not an object"
    if missing := [k for k in required_keys if k not in data]:
        return None, f"missing keys {missing}"
    return data, None
