import pytest
from conftest import ScriptedProvider

from doc_healer.llm import (
    BudgetExceeded, LLMClient, LLMError, LLMUnavailable, RateLimitError, TransientError, user,
)

MSGS = [user("hi")]


Scripted = ScriptedProvider


def client(primary, fallback=None, **kw):
    sleeps = []
    kw.setdefault("sleep", sleeps.append)
    c = LLMClient(primary, fallback, **kw)
    c.sleeps = sleeps
    return c


def test_success_needs_no_retry():
    c = client(Scripted("groq", "ok"))
    assert c.complete(MSGS).text == "ok"
    assert (c.stats.calls, c.stats.retries, c.sleeps) == (1, 0, [])
    assert c.stats.by_provider["groq"] == 1


def test_retries_with_exponential_backoff_then_succeeds():
    p = Scripted("groq", TransientError("x"), TransientError("x"), "ok")
    c = client(p, base_delay=1.0)
    assert c.complete(MSGS).text == "ok"
    assert c.sleeps == [1.0, 2.0] and c.stats.retries == 2 and p.calls == 3


def test_backoff_is_capped_by_max_wait():
    p = Scripted("groq", *[TransientError("x")] * 5, "ok")
    c = client(p, base_delay=10.0, max_wait=25.0, max_retries=5)
    c.complete(MSGS)
    assert c.sleeps == [10.0, 20.0, 25.0, 25.0, 25.0]


def test_retry_after_is_honored_over_backoff():
    c = client(Scripted("groq", RateLimitError(retry_after=7), "ok"), base_delay=1.0)
    c.complete(MSGS)
    assert c.sleeps == [7]


def test_retry_after_longer_than_max_wait_skips_straight_to_fallback():
    primary = Scripted("groq", RateLimitError(retry_after=3600))
    fb = Scripted("gemini", "from fallback")
    c = client(primary, fb, max_wait=60)
    resp = c.complete(MSGS)
    assert (resp.text, resp.provider) == ("from fallback", "gemini")
    assert c.sleeps == [] and primary.calls == 1 and c.stats.fallback_calls == 1


def test_exhausted_retries_fall_back_and_primary_stays_disabled_for_the_run():
    primary = Scripted("groq", RateLimitError())
    fb = Scripted("gemini", "fb")
    c = client(primary, fb, max_retries=2)
    assert c.complete(MSGS).provider == "gemini"
    assert primary.calls == 3  # first try + 2 retries
    for _ in range(3):
        assert c.complete(MSGS).provider == "gemini"
    assert primary.calls == 3  # never hammered again
    assert c.stats.by_provider["gemini"] == 4 and c.stats.fallback_calls == 4


def test_no_fallback_means_unavailable():
    c = client(Scripted("groq", RateLimitError()), max_retries=1)
    with pytest.raises(LLMUnavailable):
        c.complete(MSGS)
    with pytest.raises(LLMUnavailable):  # and stays so for later calls
        c.complete(MSGS)


def test_all_providers_exhausted_is_unavailable():
    c = client(Scripted("groq", RateLimitError()), Scripted("gemini", TransientError("down")), max_retries=1)
    with pytest.raises(LLMUnavailable, match="all providers unavailable"):
        c.complete(MSGS)


def test_permanent_errors_propagate_without_retry_or_fallback():
    primary = Scripted("groq", LLMError("bad request"))
    fb = Scripted("gemini", "fb")
    c = client(primary, fb)
    with pytest.raises(LLMError, match="bad request") as exc:
        c.complete(MSGS)
    assert not isinstance(exc.value, LLMUnavailable)
    assert (primary.calls, fb.calls, c.sleeps) == (1, 0, [])


def test_call_budget():
    c = client(Scripted("groq", "ok"), max_calls=2)
    c.complete(MSGS), c.complete(MSGS)
    with pytest.raises(BudgetExceeded):
        c.complete(MSGS)
    assert isinstance(BudgetExceeded("x"), LLMUnavailable)  # callers can treat both alike
    assert c.stats.calls == 2


def test_complete_json_parses_plain_and_fenced_replies():
    assert client(Scripted("groq", '{"stale": true}')).complete_json(MSGS).data == {"stale": True}
    fenced = client(Scripted("groq", '```json\n{"stale": false}\n```'))
    assert fenced.complete_json(MSGS, required_keys=["stale"]).data == {"stale": False}


def test_complete_json_reasks_once_without_spending_extra_budget():
    p = Scripted("groq", "not json at all", '{"stale": true}')
    c = client(p, max_calls=1)
    resp = c.complete_json(MSGS, required_keys=["stale"])
    assert resp.data == {"stale": True}
    assert p.calls == 2 and c.stats.calls == 1
    roles = [m["role"] for m in p.last_messages]
    assert roles == ["user", "assistant", "user"]
    assert "stale" in p.last_messages[-1]["content"]


@pytest.mark.parametrize("bad", ['["a list"]', '{"other": 1}', "plain text"])
def test_complete_json_gives_up_after_the_retry(bad):
    c = client(Scripted("groq", bad))
    with pytest.raises(LLMError, match="no valid JSON"):
        c.complete_json(MSGS, required_keys=["stale"])


def test_json_calls_use_fallback_when_primary_is_exhausted():
    c = client(Scripted("groq", RateLimitError()), Scripted("gemini", '{"ok": 1}'), max_retries=0)
    resp = c.complete_json(MSGS)
    assert (resp.provider, resp.data) == ("gemini", {"ok": 1})
