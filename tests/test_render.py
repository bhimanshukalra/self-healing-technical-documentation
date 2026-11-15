from dataclasses import replace

from test_modes import correction, passed, verification

from doc_healer.apply import AppliedFix, ApplyResult
from doc_healer.modes import Decision
from doc_healer.render import (
    FIX_MARKER, REVIEW_MARKER, LinkContext, ReviewItem, _fenced, _plain, render_fix_pr, render_review_comment,
    section_link,
)

CTX = LinkContext("octo/docs", "abc1234def")


def fix_decision(section_id="d.md#Pricing", explanation="the default tax is now 25%"):
    v = replace(verification(section_id=section_id), explanation=explanation)
    return Decision(v, "auto_fix", 0.85, "simple", (), correction(v).corrected, correction(v), passed(correction(v)))


def applied_for(*decisions):
    return ApplyResult([AppliedFix(d.section_id, "d.md") for d in decisions], [], ["d.md"])


def test_section_link_points_at_the_heading_line():
    section = fix_decision().verification.suspect.section
    link = section_link(section, CTX)
    assert link == "[`d.md` – Pricing](https://github.com/octo/docs/blob/abc1234def/d.md#L1)"
    assert section_link(section, LinkContext()) == "`d.md` – Pricing (line 1)"


def test_section_link_encodes_awkward_paths_and_refs():
    section = replace(fix_decision().verification.suspect.section, file_path="my docs/a b.md")
    url = section_link(section, LinkContext("o/r", "feature/x")).split("](")[1]
    assert "my%20docs/a%20b.md" in url and "/blob/feature%2Fx/" in url


def test_fix_pr_has_title_marker_why_confidence_trigger_and_diff():
    d = fix_decision()
    title, body = render_fix_pr([d], applied_for(d), CTX, pr_number=42)
    assert title == "docs: update 1 section for changes in #42"
    assert body.startswith(FIX_MARKER)
    assert "fixed 1 section automatically" in body and "changes in #42" in body
    assert "**What was wrong:** the default tax is now 25%" in body
    assert "**Confidence:** 0.85 · validation passed" in body
    assert "`m.py::total` (api changed)" in body
    assert "<details><summary>Diff</summary>" in body and "```diff" in body
    assert "-The default tax rate is 20%." in body and "+The default tax rate is 25%." in body


def test_fix_pr_uses_the_commit_when_there_is_no_pr_number_and_counts_correctly():
    a, b = fix_decision("d.md#A"), fix_decision("d.md#B")
    title, body = render_fix_pr([a, b], applied_for(a, b), CTX)
    assert title == "docs: update 2 sections for changes in abc1234" and "#### 2." in body


def test_fix_pr_lists_only_sections_that_were_actually_applied():
    a, b = fix_decision("d.md#A"), fix_decision("d.md#B")
    title, body = render_fix_pr([a, b], applied_for(a), CTX)
    assert "1 section" in title and "d.md#B" not in body and "#### 2." not in body


def test_llm_text_cannot_ping_link_or_inject_html():
    evil = "ping @maintainer [click](http://evil.example) <img src=x onerror=1> <!-- hidden -->"
    d = fix_decision(explanation=evil)
    _, body = render_fix_pr([d], applied_for(d), CTX)
    why = next(l for l in body.splitlines() if l.startswith("**What was wrong:**"))
    assert "@maintainer" not in why and "@​maintainer" in why
    assert "[click](" not in why and "\\[click\\]" in why
    assert "<img" not in why and "<!--" not in why and "&lt;img" in why


def test_plain_flattens_and_limits():
    assert _plain("a\n\n  b\tc") == "a b c"
    assert len(_plain("x" * 1000)) == 400


def test_fences_grow_past_any_backticks_in_the_text():
    assert _fenced("plain", "diff") == "```diff\nplain\n```"
    inner = "+```bash\n+run\n+```"
    fenced = _fenced(inner, "diff")
    assert fenced.startswith("````diff\n") and fenced.endswith("\n````")


def test_review_comment_lists_each_item_with_link_status_reasons_and_suggestion():
    section = fix_decision().verification.suspect.section
    items = [
        ReviewItem(section, "draft", "the tax changed", ("complex change", "low confidence"), 0.62, "-a\n+b"),
        ReviewItem(replace(section, id="d.md#B", heading_path=("B",)), "unverified", "", ("not verified: rate limited",)),
        ReviewItem(section, "not_applied", "x", ("the section's text no longer matches",), 0.9),
    ]
    text = render_review_comment(items, CTX, fix_pr_url="https://github.com/octo/docs/pull/9")
    assert text.startswith(REVIEW_MARKER) and "https://github.com/octo/docs/pull/9" in text
    assert "### Needs human review (3)" in text
    assert "#### 1. [`d.md` – Pricing](https://github.com/octo/docs/blob/abc1234def/d.md#L1)" in text
    assert "**Status:** draft for review · confidence 0.62" in text
    assert "**Why it needs a person:** complex change; low confidence" in text
    assert "<details><summary>Suggested edit</summary>" in text and "```diff\n-a\n+b\n```" in text
    assert "**Status:** could not be checked" in text and "not verified: rate limited" in text
    assert "fix could not be applied" in text


def test_review_comment_with_nothing_to_review_but_a_fix_pr():
    text = render_review_comment([], CTX, "https://github.com/octo/docs/pull/9")
    assert "fixed automatically in https://github.com/octo/docs/pull/9" in text and "Needs human review" not in text
