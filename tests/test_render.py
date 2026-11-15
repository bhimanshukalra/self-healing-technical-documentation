from dataclasses import replace

from test_modes import correction, passed, verification

from doc_healer.apply import AppliedFix, ApplyResult
from doc_healer.modes import Decision
from doc_healer.render import (
    COMMENT_MARKER, FIX_MARKER, MAX_COMMENT_CHARS, MAX_DIFF_CHARS, CommentData, Entry, LinkContext, ReviewItem,
    _fenced, _plain, render_fix_pr, render_summary_comment, section_link, summary_line,
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


def test_nested_headings_keep_a_readable_separator_but_each_title_is_escaped():
    section = replace(fix_decision().verification.suspect.section, heading_path=("Guide", "A <b> & [x]"))
    label = section_link(section, LinkContext()).split(" (line")[0]
    assert label == "`d.md` – Guide > A &lt;b&gt; & \\[x\\]"


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


def _section(sid="d.md#Pricing", heading="Pricing"):
    return replace(fix_decision().verification.suspect.section, id=sid, heading_path=(heading,))


def data(**kw):
    base = dict(accurate=[], fixed=[], review=[], already_updated=[])
    base.update(kw)
    return CommentData(**base)


def test_the_headline_matches_the_specified_wording():
    d = data(
        accurate=[Entry(_section())] * 3, fixed=[Entry(_section())],
        review=[ReviewItem(_section(), "flag", "", ())] * 2, fix_ref="[#42](https://github.com/o/r/pull/42)",
    )
    assert summary_line(d) == (
        "**Doc Check Results:** 3 sections verified accurate, 1 auto-fixed (see [#42](https://github.com/o/r/pull/42)), "
        "2 flagged for review."
    )


def test_the_headline_pluralizes_and_leaves_out_empty_parts():
    assert summary_line(data(accurate=[Entry(_section())])) == "**Doc Check Results:** 1 section verified accurate."
    assert summary_line(data(already_updated=[Entry(_section())] * 2)) == (
        "**Doc Check Results:** 2 already updated in this PR."
    )
    assert summary_line(data(fixed=[Entry(_section())])) == "**Doc Check Results:** 1 auto-fixed."
    assert "no documentation sections were affected" in summary_line(data())


def test_the_full_comment_lists_every_category_with_links():
    review = [
        ReviewItem(_section(), "draft", "the tax changed", ("complex change", "low confidence"), 0.62, "-a\n+b"),
        ReviewItem(_section("d.md#B", "B"), "unverified", "", ("not verified: rate limited",)),
        ReviewItem(_section(), "not_applied", "x", ("the section's text no longer matches",), 0.9),
    ]
    d = data(
        review=review, accurate=[Entry(_section("d.md#C", "C"), "still correct")],
        fixed=[Entry(_section("d.md#D", "D"), "renamed a parameter")],
        already_updated=[Entry(_section("d.md#E", "E"))], unresolved_links=2,
        llm_calls={"by_provider": {"groq": 4}}, fix_ref="[#9](https://github.com/octo/docs/pull/9)",
    )
    text = render_summary_comment(d, CTX)
    assert text.startswith(COMMENT_MARKER) and "## doc-healer" in text
    assert "(see [#9](https://github.com/octo/docs/pull/9))" in text
    assert "### Needs human review (3)" in text
    assert "#### 1. [`d.md` – Pricing](https://github.com/octo/docs/blob/abc1234def/d.md#L1)" in text
    assert "**Status:** draft for review · confidence 0.62" in text
    assert "**Why it needs a person:** complex change; low confidence" in text
    assert "<details><summary>Suggested edit</summary>" in text and "```diff\n-a\n+b\n```" in text
    assert "**Status:** could not be checked" in text and "fix could not be applied" in text
    for title in ("Auto-fixed (1)", "Verified accurate (1)", "Already updated in this PR (1)"):
        assert f"<summary>{title}</summary>" in text
    assert "`d.md` – C](" in text and "– still correct" in text and "– renamed a parameter" in text
    assert "refers to 2 documentation link(s) that no longer exist" in text
    assert "_Checked with `groq` (4)._" in text and "fallback" not in text


def test_provider_tags_and_the_fallback_note_appear_only_when_more_than_one_provider_was_used():
    entry = Entry(_section("d.md#C", "C"), "ok", "gemini")
    one = render_summary_comment(data(accurate=[entry], llm_calls={"by_provider": {"gemini": 2}}), CTX)
    assert "via `" not in one
    many = render_summary_comment(data(
        accurate=[entry], llm_calls={"by_provider": {"groq": 3, "gemini": 2}, "fallback_calls": 2}), CTX)
    assert "via `gemini`" in many and "`gemini` (2), `groq` (3)" in many
    assert "fallback provider was used" in many


def test_it_says_so_when_no_llm_was_configured():
    text = render_summary_comment(data(review=[ReviewItem(_section(), "unverified", "", ("no LLM",))], llm_configured=False), CTX)
    assert "No LLM was configured" in text and "could not be checked" in text


def test_a_huge_result_is_cut_back_to_fit_githubs_comment_limit():
    big_diff = "\n".join(f"+line {i} " + "x" * 80 for i in range(400))
    items = [ReviewItem(_section(f"d.md#S{i}", f"S{i}"), "draft", "why " * 50, ("r",), 0.7, big_diff) for i in range(80)]
    text = render_summary_comment(data(review=items, accurate=[Entry(_section(f"d.md#A{i}", f"A{i}")) for i in range(200)]), CTX)
    assert len(text) <= MAX_COMMENT_CHARS and text.startswith(COMMENT_MARKER)
    assert "Needs human review (80)" in text and "more (the list was shortened" in text


def test_long_diffs_are_clipped_in_the_fix_pr_and_the_comment():
    long_diff = "\n".join(f"+line {i}" for i in range(2000))
    item = ReviewItem(_section(), "draft", "x", ("r",), 0.7, long_diff)
    text = render_summary_comment(data(review=[item]), CTX)
    assert "[... diff shortened ...]" in text and len(text) < MAX_DIFF_CHARS + 2000
    d = fix_decision()
    d = replace(d, correction=replace(d.correction, original="a\n", corrected="\n".join(f"line {i}" for i in range(2000))))
    _, body = render_fix_pr([d], applied_for(d), CTX)
    assert "[... diff shortened ...]" in body and len(body) < MAX_DIFF_CHARS + 2500
