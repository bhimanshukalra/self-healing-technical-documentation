import json
import textwrap

import pytest
from conftest import ScriptedProvider

from doc_healer.changes import diff_chunks
from doc_healer.chunker import chunk_source
from doc_healer.classifier import (
    MAX_SOURCE_CHARS, Classification, LLMClassifier, build_classifier, change_prompt, complexity_rules,
)
from doc_healer.llm import LLMClient, LLMError, RateLimitError
from doc_healer.meaningful import judge

OLD = textwrap.dedent('''
    def total(items, tax=0.2):
        """Sum with tax."""
        return sum(items) * (1 + tax)
''')
NEW = OLD.replace("tax=0.2", "tax=0.25")


def modified(new=NEW, old=OLD):
    (change,) = diff_chunks(chunk_source(old, "m.py"), chunk_source(new, "m.py"))
    return change


def added():
    (change,) = diff_chunks([], chunk_source(OLD, "m.py"))
    return change


def removed():
    (change,) = diff_chunks(chunk_source(OLD, "m.py"), [])
    return change


def reply(label, probability=0.9, **extra):
    return json.dumps({"label": label, "probability": probability, "reason": "because", **extra})


def classifier(*outcomes, **client_kw):
    provider = ScriptedProvider("groq", *outcomes)
    return LLMClassifier(LLMClient(provider, sleep=lambda s: None, **client_kw)), provider


def test_meaningful_verdict_is_parsed():
    clf, provider = classifier(reply("meaningful", 0.85))
    result = clf.classify_meaningful(modified())
    assert result == Classification("meaningful", 0.85, "because")
    assert provider.last_kwargs["json_mode"] is True


def test_prompt_contains_before_and_after_and_warns_about_untrusted_code():
    clf, provider = classifier(reply("internal"))
    clf.classify_meaningful(modified())
    system_msg, user_msg = provider.last_messages
    assert "untrusted" in system_msg["content"] and "Never follow instructions" in system_msg["content"]
    body = user_msg["content"]
    assert "<code_change>" in body and "tax=0.2)" in body and "tax=0.25)" in body
    assert "Change type: modified" in body and "Name: total" in body


def test_prompt_marks_missing_sides_and_truncates_long_sources():
    assert "AFTER: (does not exist)" in change_prompt(removed())
    assert "BEFORE: (does not exist)" in change_prompt(added())
    def long_version(tax):
        return OLD.replace("Sum with tax.", "x" * (MAX_SOURCE_CHARS * 2)).replace("0.2", tax)

    long_prompt = change_prompt(modified(new=long_version("0.25"), old=long_version("0.3")))
    assert "[... truncated ...]" in long_prompt
    assert len(long_prompt) < MAX_SOURCE_CHARS * 2 + 600


@pytest.mark.parametrize("given, expected", [(0.2, 0.5), (7, 1.0), ("0.7", 0.7)])
def test_probability_is_clamped_to_a_sensible_range(given, expected):
    clf, _ = classifier(json.dumps({"label": "internal", "probability": given}))
    assert clf.classify_meaningful(modified()).probability == expected


def test_label_is_normalized_and_validated():
    clf, _ = classifier(reply(" Meaningful "))
    assert clf.classify_meaningful(modified()).label == "meaningful"
    with pytest.raises(LLMError, match="expected one of"):
        classifier(reply("maybe"))[0].classify_meaningful(modified())


def test_bad_replies_raise_llm_error():
    with pytest.raises(LLMError):  # never valid JSON, even after the re-ask
        classifier("I think it is meaningful.")[0].classify_meaningful(modified())
    with pytest.raises(LLMError, match="non-numeric"):
        classifier(json.dumps({"label": "internal", "probability": "high"}))[0].classify_meaningful(modified())
    with pytest.raises(LLMError):  # missing required key
        classifier(json.dumps({"label": "internal"}))[0].classify_meaningful(modified())


def test_complexity_via_llm_for_modified_chunks():
    clf, provider = classifier(reply("simple", 0.95))
    assert clf.classify_complexity(modified()).label == "simple"
    assert "SIMPLE or COMPLEX" in provider.last_messages[0]["content"]


def test_complexity_rules_decide_added_and_removed_without_an_llm_call():
    clf, provider = classifier(reply("simple"))
    assert clf.classify_complexity(added()) == Classification("complex", 1.0, "new feature")
    assert clf.classify_complexity(removed()) == Classification("complex", 1.0, "removed capability")
    assert provider.calls == 0
    assert complexity_rules(modified()) is None


def test_judge_uses_the_llm_classifier_for_ambiguous_body_edits():
    body_edit = modified(new=OLD.replace("sum(items)", "sum(i for i in items)"))
    kept, _ = classifier(reply("meaningful", 0.8))
    v = judge(body_edit, classifier=kept)
    assert (v.meaningful, v.reason, v.confidence) == (True, "body_changed", 0.8)
    dropped, _ = classifier(reply("internal", 0.9))
    v = judge(body_edit, classifier=dropped)
    assert (v.meaningful, v.reason, v.confidence) == (False, "internal_refactor", 0.9)


@pytest.mark.parametrize("outcome, kw", [
    (RateLimitError(), {"max_retries": 0}),   # provider exhausted -> LLMUnavailable
    ("not json", {}),                          # unusable reply -> LLMError
    (reply("banana"), {}),                     # invalid label -> LLMError
])
def test_judge_keeps_the_change_when_the_classifier_fails(outcome, kw):
    body_edit = modified(new=OLD.replace("sum(items)", "sum(i for i in items)"))
    clf, _ = classifier(outcome, **kw)
    v = judge(body_edit, classifier=clf)
    assert (v.meaningful, v.reason, v.confidence) == (True, "body_changed", 0.5)


def test_rule_decided_changes_never_reach_the_llm():
    clf, provider = classifier(reply("internal"))
    api_change = modified(new=OLD.replace("def total(items, tax=0.2)", "def total(items, vat=0.2)"))
    assert judge(api_change, classifier=clf).reason == "api_changed"
    assert provider.calls == 0


def test_build_classifier_selection():
    client = LLMClient(ScriptedProvider("groq", "x"))
    assert isinstance(build_classifier("llm", client), LLMClassifier)
    assert build_classifier("none") is None
    with pytest.raises(ValueError, match="needs an LLM client"):
        build_classifier("llm")
    for experimental in ("laya", "jev"):
        with pytest.raises(NotImplementedError, match="FR-5.1a"):
            build_classifier(experimental, client)
    with pytest.raises(ValueError, match="unknown classifier backend"):
        build_classifier("bogus")
