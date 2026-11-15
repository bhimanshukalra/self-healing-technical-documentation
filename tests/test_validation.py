import json

import pytest
from conftest import ScriptedProvider
from test_repair import DOC, STALE_QUOTE, edits, stale
from test_verification import reply as verify_reply

from doc_healer.analysis import analyze_changes
from doc_healer.llm import LLMClient, LLMError, RateLimitError
from doc_healer.pipeline import build_index
from doc_healer.repair import Correction, Edit, repair_all, repair_section
from doc_healer.validation import lint_correction, validate_all, validate_correction, validation_prompt
from doc_healer.verification import verify_suspects

FIXED = DOC.replace("20%", "25%")


def verdict(accurate=True, preserved=True, style=True, confidence=0.9, issues=()):
    return json.dumps({"accurate": accurate, "preserved": preserved, "style_consistent": style,
                       "confidence": confidence, "issues": list(issues)})


def llm(*outcomes, **kw):
    provider = ScriptedProvider("groq", *outcomes)
    return LLMClient(provider, sleep=lambda s: None, **kw), provider


def make_correction(original=DOC, corrected=FIXED, section_id="d.md#Pricing"):
    v = stale(statements=(STALE_QUOTE,), content=original)
    edit = Edit(0, len(original), original, corrected)
    c = Correction(v, "corrected", original, corrected, (edit,))
    return c


def test_all_three_checks_passing_passes():
    client, provider = llm(verdict(confidence=0.85))
    v = validate_correction(make_correction(), client)
    assert (v.status, v.passed) == ("passed", True)
    assert (v.accurate, v.preserved, v.style_consistent) == (True, True, True)
    assert (v.confidence, v.issues, v.lint_problems) == (0.85, (), ())
    assert (v.provider, v.model, v.section_id) == ("groq", "m", "d.md#Pricing")
    assert provider.last_kwargs["json_mode"] is True


@pytest.mark.parametrize("flags, failing", [
    (dict(accurate=False), "accurate"),
    (dict(preserved=False), "preserved"),
    (dict(style=False), "style_consistent"),
])
def test_any_failing_check_fails_the_validation(flags, failing):
    client, _ = llm(verdict(issues=["the tax in the second sentence is still wrong"], **flags))
    v = validate_correction(make_correction(), client)
    assert v.status == "failed" and not v.passed
    assert getattr(v, failing) is False
    assert v.issues == ("the tax in the second sentence is still wrong",)


def test_prompt_contains_both_versions_the_diff_the_code_and_the_rubric():
    client, provider = llm(verdict())
    c = make_correction()
    validate_correction(c, client)
    system_msg, user_msg = provider.last_messages
    rules = " ".join(system_msg["content"].split())
    for phrase in ("strict reviewer", "accurate:", "preserved:", "style_consistent:", "untrusted data"):
        assert phrase in rules
    body = user_msg["content"]
    assert f"<original_section>\n{DOC}\n</original_section>" in body
    assert f"<corrected_section>\n{FIXED}\n</corrected_section>" in body
    assert "-The default tax rate is 20%." in body and "tax=0.25)" in body
    assert validation_prompt(c) == body


def test_confidence_is_clamped_and_issues_cleaned():
    client, _ = llm(json.dumps({"accurate": True, "preserved": True, "style_consistent": True,
                                "confidence": 3, "issues": ["  ", 7, "real"]}))
    v = validate_correction(make_correction(), client)
    assert v.confidence == 1.0 and v.issues == ("real",)


@pytest.mark.parametrize("bad", [
    json.dumps({"accurate": "yes", "preserved": True, "style_consistent": True, "confidence": 0.9}),
    json.dumps({"accurate": True, "preserved": True, "style_consistent": True, "confidence": "high"}),
    json.dumps({"accurate": True, "preserved": True, "confidence": 0.9}),
])
def test_malformed_verdicts_raise(bad):
    with pytest.raises(LLMError):
        validate_correction(make_correction(), llm(bad)[0])


def test_only_corrected_corrections_can_be_validated():
    failed = Correction(stale(), "failed", DOC, DOC, note="x")
    with pytest.raises(ValueError, match="only corrected"):
        validate_correction(failed, llm(verdict())[0])


# -- lint ------------------------------------------------------------------------------------

FENCED = "Run it:\n\n```bash\ntotal 1 2\n```\n\nThe default tax rate is 20%."


def test_lint_accepts_a_structure_preserving_fix():
    assert lint_correction(make_correction(FENCED, FENCED.replace("20%", "25%"))) == []


@pytest.mark.parametrize("corrected, problem", [
    (FENCED.replace("```bash\n", "").replace("\n```", ""), "code fences"),
    (FENCED + "\n```\nstray", "code fences"),
    (FENCED + "\n\n## A new heading", "headings"),
    (FENCED + "\nSee https://example.com/new for details.", "introduces URLs"),
])
def test_lint_catches_structural_damage(corrected, problem):
    problems = lint_correction(make_correction(FENCED, corrected))
    assert len(problems) >= 1 and problem in " ".join(problems)


def test_lint_ignores_hash_comments_inside_fences_and_urls_already_present():
    original = "See https://example.com/a.\n\n```bash\n# a shell comment\nrun\n```"
    assert lint_correction(make_correction(original, original.replace("run", "run --fast"))) == []
    assert lint_correction(make_correction(original, original + " Again: https://example.com/a.")) == []


def test_a_lint_failure_fails_without_spending_an_llm_call():
    client, provider = llm(verdict())
    v = validate_correction(make_correction(FENCED, FENCED + "\nhttps://evil.example/x"), client)
    assert v.status == "failed" and "evil.example" in v.lint_problems[0]
    assert (v.accurate, v.preserved, v.style_consistent) == (None, None, None)
    assert provider.calls == 0 and client.stats.calls == 0


def test_changed_fraction_reports_how_much_was_replaced():
    c = make_correction()
    v = validate_correction(c, llm(verdict())[0])
    assert v.changed_fraction == 1.0  # the helper's single edit spans the whole section
    real = repair_section(stale(), llm(edits((STALE_QUOTE, "The default tax rate is 25%.")))[0])
    v2 = validate_correction(real, llm(verdict())[0])
    assert v2.changed_fraction == round(len(STALE_QUOTE) / len(DOC), 3)


# -- validate_all ----------------------------------------------------------------------------

def test_validate_all_checks_only_corrected_ones_in_order():
    corrected = [make_correction(), make_correction()]
    skipped = Correction(stale(), "skipped", DOC, DOC, note="x")
    client, provider = llm(verdict(), verdict(accurate=False, issues=["wrong"]))
    out = validate_all([skipped, corrected[0], Correction(stale(), "failed", DOC, DOC), corrected[1]], client)
    assert [v.status for v in out] == ["passed", "failed"] and provider.calls == 2


def test_a_malformed_verdict_leaves_only_that_correction_unchecked():
    client, _ = llm("junk", "junk", verdict())
    out = validate_all([make_correction(), make_correction()], client)
    assert [v.status for v in out] == ["unchecked", "passed"] and "not validated" in out[0].note


def test_running_out_of_capacity_leaves_the_rest_unchecked_but_still_lints():
    provider = ScriptedProvider("groq", verdict(), RateLimitError())
    client = LLMClient(provider, max_retries=0, sleep=lambda s: None)
    broken = make_correction(FENCED, FENCED + "\nhttps://evil.example/x")
    out = validate_all([make_correction(), make_correction(), broken, make_correction()], client)
    assert [v.status for v in out] == ["passed", "unchecked", "failed", "unchecked"]
    assert provider.calls == 2  # nothing after the failed attempt, not even for the lint failure
    assert out[2].lint_problems and "not validated" in out[3].note


def test_call_budget_applies():
    client, _ = llm(verdict(), max_calls=1)
    out = validate_all([make_correction(), make_correction()], client)
    assert [v.status for v in out] == ["passed", "unchecked"] and "budget" in out[1].note


def test_end_to_end_detect_verify_repair_validate_on_the_taskr_project(taskr_repo):
    repo, base = taskr_repo
    index = build_index(repo.path, use_embeddings=False).index
    repo.edit("taskr/tasks.py", "def add_task(self, title: str, priority: int = 1)",
              "def add_task(self, name: str, priority: int = 1)")
    repo.edit("taskr/tasks.py", '{"title": title,', '{"title": name,')
    report = analyze_changes(repo.path, base, repo.commit("rename"), index).report

    verified = verify_suspects(report.suspects, llm(
        verify_reply(False, 0.8, "unaffected"),
        verify_reply(True, 0.9, "now called name", ["Use `add_task(title, priority=1)`"]),
    )[0])
    corrections = repair_all(verified, llm(edits(("`add_task(title, priority=1)`", "`add_task(name, priority=1)`")))[0])
    good = validate_all(corrections, llm(verdict(confidence=0.9))[0])
    assert [(v.section_id, v.status) for v in good] == [("docs/usage.md#Usage > Adding tasks", "passed")]

    rejected = validate_all(corrections, llm(verdict(preserved=False, issues=["dropped the shell example"]))[0])
    assert rejected[0].status == "failed" and rejected[0].issues == ("dropped the shell example",)
