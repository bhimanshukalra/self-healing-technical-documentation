import json

import pytest
from conftest import ScriptedProvider

from doc_healer.analysis import analyze_changes
from doc_healer.llm import LLMClient, LLMError, RateLimitError
from doc_healer.pipeline import build_index
from doc_healer.repair import repair_all, repair_prompt, repair_section
from doc_healer.verification import Verification, VerificationReport, verify_suspects

from test_verification import DOC, make_change, make_suspect, reply as verify_reply

STALE_QUOTE = "The default tax rate is 20%."


def stale(statements=(STALE_QUOTE,), content=DOC, explanation="the default tax is now 25%", **kw):
    return Verification(make_suspect(content=content, **kw), "stale", explanation, tuple(statements), 0.9)


def edits(*pairs, summary="updated the default"):
    return json.dumps({"edits": [{"find": f, "replace": r} for f, r in pairs], "summary": summary})


def llm(*outcomes, **kw):
    provider = ScriptedProvider("groq", *outcomes)
    return LLMClient(provider, sleep=lambda s: None, **kw), provider


def test_edit_is_applied_and_everything_else_is_preserved_exactly():
    client, _ = llm(edits((STALE_QUOTE, "The default tax rate is 25%.")))
    c = repair_section(stale(), client)
    assert c.status == "corrected"
    assert c.corrected == DOC.replace("20%", "25%")
    assert c.corrected.startswith("Call `total(items)` to add up prices.\n") and c.corrected.endswith("Results are rounded.")
    assert [(e.find, e.replace) for e in c.edits] == [(STALE_QUOTE, "The default tax rate is 25%.")]
    assert (c.summary, c.provider, c.model, c.anchored, c.rejected) == ("updated the default", "groq", "m", True, ())
    assert c.original == DOC and c.section.id == "d.md#Pricing"


def test_diff_shows_only_the_changed_line():
    client, _ = llm(edits((STALE_QUOTE, "The default tax rate is 25%.")))
    diff = repair_section(stale(), client).diff()
    assert "-The default tax rate is 20%." in diff and "+The default tax rate is 25%." in diff
    assert "Results are rounded" not in "".join(l for l in diff.splitlines() if l.startswith(("+", "-")))
    assert diff.startswith("--- a/d.md#Pricing")


def test_edits_apply_regardless_of_the_order_the_model_lists_them():
    doc = "Alpha is 1.\nBeta is 2.\nGamma is 3."
    v = stale(statements=("Alpha is 1.", "Gamma is 3."), content=doc)
    client, _ = llm(edits(("Gamma is 3.", "Gamma is 30."), ("Alpha is 1.", "Alpha is 10.")))
    assert repair_section(v, client).corrected == "Alpha is 10.\nBeta is 2.\nGamma is 30."


def test_find_text_tolerates_whitespace_differences():
    doc = "The default tax rate\nis 20% for all\norders."
    v = stale(statements=("default tax rate is 20%",), content=doc)
    client, _ = llm(edits(("default tax rate is 20% for all orders.", "default tax rate is 25% for all orders.")))
    c = repair_section(v, client)
    assert c.status == "corrected" and c.corrected == "The default tax rate is 25% for all orders."


def test_replacement_can_delete_text():
    client, _ = llm(edits((STALE_QUOTE + "\n", "")))
    assert repair_section(stale(), client).corrected == "Call `total(items)` to add up prices.\nResults are rounded."


@pytest.mark.parametrize("bad_edit, reason", [
    (("The tax is never mentioned", "x"), "not found"),
    (("", "x"), "empty 'find'"),
    (("  ", "x"), "empty 'find'"),
    ((STALE_QUOTE, STALE_QUOTE), "no-op"),
])
def test_unusable_edits_are_rejected_and_a_lone_one_fails_the_correction(bad_edit, reason):
    c = repair_section(stale(), llm(edits(bad_edit))[0])
    assert c.status == "failed" and c.corrected == DOC and c.edits == ()
    assert reason in c.rejected[0].reason and "every proposed edit was rejected" in c.note


def test_ambiguous_find_is_rejected():
    doc = "Tax is 20%. Really, tax is 20%."
    v = stale(statements=("Tax is 20%.",), content=doc)
    c = repair_section(v, llm(edits(("20%", "25%")))[0])
    assert c.status == "failed" and "matches 2 places" in c.rejected[0].reason


def test_edits_outside_the_diagnosed_statements_are_rejected():
    client, _ = llm(edits((STALE_QUOTE, "The default tax rate is 25%."), ("Results are rounded.", "Results are rounded up.")))
    c = repair_section(stale(), client)
    assert c.status == "corrected" and c.corrected == DOC.replace("20%", "25%")
    assert "did not call wrong" in c.rejected[0].reason and c.rejected[0].find == "Results are rounded."


def test_an_edit_may_extend_beyond_the_quote_as_long_as_it_overlaps_it():
    client, _ = llm(edits(("add up prices.\nThe default tax rate is 20%.", "add up prices.\nThe default tax rate is 25%.")))
    assert repair_section(stale(), client).corrected == DOC.replace("20%", "25%")


def test_overlapping_edits_keep_only_the_first():
    client, _ = llm(edits((STALE_QUOTE, "A."), ("tax rate is 20%", "B")))
    c = repair_section(stale(), client)
    assert [e.replace for e in c.edits] == ["A."] and "overlaps" in c.rejected[0].reason


def test_without_a_grounded_diagnosis_edits_are_not_anchored():
    v = stale(statements=())  # the verifier could not quote anything
    c = repair_section(v, llm(edits(("Results are rounded.", "Results are rounded up.")))[0])
    assert c.status == "corrected" and c.anchored is False


def test_empty_edit_list_means_the_model_found_nothing_to_change():
    c = repair_section(stale(), llm(edits())[0])
    assert c.status == "failed" and "no edits" in c.note


def test_malformed_edits_raise_llm_error():
    for bad in [json.dumps({"edits": "nope"}), json.dumps({"edits": [{"find": "x"}]}), json.dumps({"edits": [["x", "y"]]})]:
        with pytest.raises(LLMError):
            repair_section(stale(), llm(bad)[0])


def test_only_stale_verifications_can_be_repaired():
    accurate = Verification(make_suspect(), "accurate", "fine", (), 0.9)
    with pytest.raises(ValueError, match="only stale"):
        repair_section(accurate, llm(edits())[0])


def test_prompt_contains_section_diagnosis_code_and_guardrails():
    client, provider = llm(edits((STALE_QUOTE, "The default tax rate is 25%.")))
    repair_section(stale(), client)
    system_msg, user_msg = provider.last_messages
    rules = " ".join(system_msg["content"].split())  # the prompt is line-wrapped
    for phrase in ("smallest set of text edits", "Do not rewrite, reword, reorder", "Never invent", "untrusted data"):
        assert phrase in rules
    body = user_msg["content"]
    assert DOC in body and "the default tax is now 25%" in body and f"- {STALE_QUOTE}" in body
    assert "tax=0.25)" in body and "Section: Pricing" in body
    assert repair_prompt(stale()) == body
    assert provider.last_kwargs["json_mode"] is True


def test_repair_all_handles_only_stale_results_and_isolates_bad_replies():
    accurate = Verification(make_suspect("d.md#Fine"), "accurate", "fine", (), 0.9)
    unverified = Verification(make_suspect("d.md#Unknown"), "unverified", "no key")
    good = stale()
    bad = Verification(make_suspect("d.md#Bad", content=DOC), "stale", "x", (STALE_QUOTE,), 0.9)
    client, _ = llm("garbage", "garbage", edits((STALE_QUOTE, "The default tax rate is 25%.")))
    out = repair_all([accurate, bad, unverified, good], client)
    assert [(c.section.id, c.status) for c in out] == [("d.md#Bad", "failed"), ("d.md#Pricing", "corrected")]
    assert "not repaired" in out[0].note and out[0].corrected == DOC


def test_repair_all_skips_the_rest_when_capacity_runs_out():
    provider = ScriptedProvider("groq", edits((STALE_QUOTE, "The default tax rate is 25%.")), RateLimitError())
    client = LLMClient(provider, max_retries=0, sleep=lambda s: None)
    vs = [Verification(make_suspect(f"d.md#S{i}"), "stale", "x", (STALE_QUOTE,), 0.9) for i in range(4)]
    out = repair_all(vs, client)
    assert [c.status for c in out] == ["corrected", "skipped", "skipped", "skipped"]
    assert provider.calls == 2 and "not repaired" in out[1].note
    assert all(c.corrected == DOC == c.original for c in out[1:])  # unrepaired means unchanged


def test_repair_all_respects_the_call_budget():
    client, _ = llm(edits((STALE_QUOTE, "The default tax rate is 25%.")), max_calls=1)
    vs = [Verification(make_suspect(f"d.md#S{i}"), "stale", "x", (STALE_QUOTE,), 0.9) for i in range(2)]
    out = repair_all(vs, client)
    assert [c.status for c in out] == ["corrected", "skipped"] and "budget" in out[1].note


def test_end_to_end_detect_verify_repair_on_the_taskr_project(taskr_repo):
    repo, base = taskr_repo
    index = build_index(repo.path, use_embeddings=False).index
    repo.edit("taskr/tasks.py", "def add_task(self, title: str, priority: int = 1)",
              "def add_task(self, name: str, priority: int = 1)")
    repo.edit("taskr/tasks.py", '{"title": title,', '{"title": name,')
    report = analyze_changes(repo.path, base, repo.commit("rename"), index).report

    verify_client, _ = llm(
        verify_reply(False, 0.8, "README only names the method"),
        verify_reply(True, 0.9, "the parameter is now called name", ["Use `add_task(title, priority=1)`"]),
    )
    verified = verify_suspects(report.suspects, verify_client)
    assert [r.status for r in verified.results] == ["accurate", "stale"]

    repair_client, _ = llm(edits(("`add_task(title, priority=1)`", "`add_task(name, priority=1)`")))
    (correction,) = repair_all(verified, repair_client)
    usage = correction.section.content
    assert correction.status == "corrected"
    assert correction.corrected == usage.replace("add_task(title, priority=1)", "add_task(name, priority=1)")
    assert "The default priority is 1." in correction.corrected and "calls the `add` command" in correction.corrected
