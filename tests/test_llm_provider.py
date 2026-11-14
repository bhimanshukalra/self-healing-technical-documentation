import json

import httpx
import pytest

from doc_healer.llm import (
    AuthError, LLMConfig, LLMError, OpenAICompatibleProvider, ProviderConfig, RateLimitError,
    TransientError, build_client, build_provider, system, user,
)

OK_BODY = {"choices": [{"message": {"content": "hello"}}]}


def provider(handler):
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAICompatibleProvider("groq", "https://api.example.com/v1/", "m1", "sk-secret", client=http)


def test_success_sends_expected_request():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"], seen["auth"] = str(request.url), request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_BODY)

    resp = provider(handler).complete([system("s"), user("u")], json_mode=True, max_tokens=50, temperature=0.2)
    assert (resp.text, resp.provider, resp.model) == ("hello", "groq", "m1")
    assert seen["url"] == "https://api.example.com/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-secret"
    assert seen["body"] == {
        "model": "m1", "temperature": 0.2, "max_tokens": 50,
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        "response_format": {"type": "json_object"},
    }


def test_plain_request_omits_optional_fields():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_BODY)

    provider(handler).complete([user("u")])
    assert "response_format" not in seen["body"] and "max_tokens" not in seen["body"]


def test_429_carries_retry_after():
    with pytest.raises(RateLimitError) as exc:
        provider(lambda r: httpx.Response(429, headers={"Retry-After": "12"})).complete([user("u")])
    assert exc.value.retry_after == 12.0


@pytest.mark.parametrize("headers", [{}, {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}])
def test_429_without_usable_retry_after(headers):
    with pytest.raises(RateLimitError) as exc:
        provider(lambda r: httpx.Response(429, headers=headers)).complete([user("u")])
    assert exc.value.retry_after is None


@pytest.mark.parametrize("status", [500, 502, 503, 408])
def test_server_errors_are_transient(status):
    with pytest.raises(TransientError):
        provider(lambda r: httpx.Response(status)).complete([user("u")])


def test_timeouts_and_connection_errors_are_transient():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request):
        raise httpx.ConnectError("no route", request=request)

    for handler in (timeout, refused):
        with pytest.raises(TransientError):
            provider(handler).complete([user("u")])


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures(status):
    with pytest.raises(AuthError):
        provider(lambda r: httpx.Response(status)).complete([user("u")])


def test_bad_request_is_a_permanent_error():
    with pytest.raises(LLMError, match="HTTP 400"):
        provider(lambda r: httpx.Response(400, text="context too long")).complete([user("u")])


@pytest.mark.parametrize("body", [{}, {"choices": []}, {"choices": [{"message": {"content": None}}]}])
def test_malformed_success_bodies(body):
    with pytest.raises(LLMError):
        provider(lambda r: httpx.Response(200, json=body)).complete([user("u")])


def test_repr_never_contains_the_key():
    p = provider(lambda r: httpx.Response(200, json=OK_BODY))
    assert "sk-secret" not in repr(p)
    assert "sk-secret" not in repr(ProviderConfig("groq", "sk-secret"))


def test_presets_and_overrides():
    groq = build_provider(ProviderConfig("groq", "k"))
    assert groq.name == "groq" and groq._url == "https://api.groq.com/openai/v1/chat/completions"
    assert build_provider(ProviderConfig("gemini", "k", model="other")).model == "other"
    custom = build_provider(ProviderConfig("local", "k", model="m", base_url="http://localhost:8000/v1"))
    assert custom._url == "http://localhost:8000/v1/chat/completions"


@pytest.mark.parametrize("cfg", [
    ProviderConfig("nope", "k"),                              # unknown, no base_url
    ProviderConfig("local", "k", base_url="http://x/v1"),     # custom endpoint, no model
    ProviderConfig("groq", ""),                               # missing key
])
def test_invalid_configs(cfg):
    with pytest.raises(ValueError):
        build_provider(cfg)


def test_build_client_wires_primary_fallback_and_limits():
    client = build_client(LLMConfig(
        ProviderConfig("groq", "k1"), ProviderConfig("gemini", "k2"), max_calls=5, max_retries=1))
    assert (client.primary.name, client.fallback.name) == ("groq", "gemini")
    assert (client.max_calls, client.max_retries) == (5, 1)
    assert build_client(LLMConfig(ProviderConfig("groq", "k1"))).fallback is None
