import json
from dataclasses import replace

import pytest
from conftest import ScriptedProvider
from test_repair import DOC, STALE_QUOTE, edits
from test_validation import FENCED, verdict
from test_verification import make_change, make_suspect, reply as verify_reply

from doc_healer.analysis import analyze_changes
from doc_healer.changes import ChunkChange
from doc_healer.classifier import Classification
from doc_healer.llm import LLMClient, LLMError
from doc_healer.modes import add_todo_markers, assess_complexity, decide_modes
from doc_healer.pipeline import build_index
from doc_healer.repair import Correction, Edit, RejectedEdit, repair_all
from doc_healer.validation import Validation, lint_correction, validate_all
from doc_healer.verification import Verification, verify_suspects

FIXED = DOC.replace("20%", "25%")


class FakeClassifier:
    def __init__(self, label="simple", probability=0.95, error=None):
        self.result, self.error, self.calls = Classification(label, probability), error, []

    def classify_meaningful(self, change):
        raise AssertionError("not used here")

    def classify_complexity(self, change):
        self.calls.append(change.chunk_id)
        if self.error:
            raise self.error
        return self.result


def verification(confidence=0.9, grounded=True, section_id="d.md#Pricing", changes=None, reason="api_changed"):
    suspect = make_suspect(section_id, changes=changes, reason=reason)
    return Verification(suspect, "stale", "the default tax is now 25%", (STALE_QUOTE,), confidence, grounded)


def correction(v, rejected=(), anchored=True, original=DOC, corrected=FIXED):
    edit = Edit(original.index(STALE_QUOTE), original.index(STALE_QUOTE) + len(STALE_QUOTE), STALE_QUOTE,
                "The default tax rate is 25%.")
    return Correction(v, "corrected", original, corrected, (edit,), tuple(rejected), anchored=anchored)


def passed(c, confidence=0.85):
    return Validation(c, "passed", True, True, True, confidence)


def decide(v=None, c=None, val=None, classifier=None, **kw):
    v = v or verification()
    c = c if c is not None else correction(v)
    val = val if val is not None else passed(c)
    (d,) = decide_modes([v], [c], [val], classifier or FakeClassifier(), **kw)
    return d


# -- auto-fix and its gates --------------------------------------------------------------

def test_simple_confident_validated_change_is_auto_fixed():
    d = decide()
    assert (d.mode, d.complexity, d.new_content) == ("auto_fix", "simple", FIXED)
    assert d.confidence == 0.85  # the lowest of verifier 0.9, validator 0.85, classifier 0.95
    assert "simple change" in d.reasons and "validation passed" in d.reasons
    assert d.section_id == "d.md#Pricing" and d.correction.status == "corrected"


def test_confidence_is_the_minimum_of_the_signals():
    assert decide(v=verification(confidence=0.7)).confidence == 0.7
    assert decide(classifier=FakeClassifier("simple", 0.6)).confidence == 0.6


def test_below_threshold_becomes_a_draft():
    d = decide(val=passed(correction(verification()), confidence=0.7))
    assert d.mode == "draft" and any("below the threshold" in r for r in d.reasons)
    assert "<!-- TODO(doc-healer)" in d.new_content


def test_threshold_is_configurable():
    assert decide(threshold=0.9).mode == "draft"
    assert decide(threshold=0.5).mode == "auto_fix"


def test_a_complex_change_is_always_a_draft_even_when_every_score_is_high():
    d = decide(classifier=FakeClassifier("complex", 0.99), val=passed(correction(verification()), 1.0))
    assert (d.mode, d.complexity) == ("draft", "complex")
    assert any("complex change" in r for r in d.reasons)


def test_removed_and_added_chunks_are_complex_by_rule_without_asking_the_classifier():
    clf = FakeClassifier("simple")
    removed = make_change(new=None)
    d = decide(v=verification(changes=[removed], reason="removed"), classifier=clf)
    assert (d.mode, d.complexity) == ("draft", "complex") and clf.calls == []


def test_without_a_classifier_a_modified_change_is_never_auto_fixed():
    v = verification()
    c = correction(v)
    (d,) = decide_modes([v], [c], [passed(c)], classifier=None)
    assert (d.mode, d.complexity) == ("draft", "unknown")
    assert any("could not establish" in r for r in d.reasons)


def test_a_classifier_error_is_treated_as_unknown_not_simple():
    d = decide(classifier=FakeClassifier(error=LLMError("down")))
    assert (d.mode, d.complexity) == ("draft", "unknown")


@pytest.mark.parametrize("kwargs, text", [
    (dict(v=verification(grounded=False)), "did not quote text"),
])
def test_an_ungrounded_diagnosis_is_a_draft(kwargs, text):
    d = decide(**kwargs)
    assert d.mode == "draft" and any(text in r for r in d.reasons)


def test_rejected_or_unanchored_edits_make_it_a_draft():
    v = verification()
    d = decide(v=v, c=correction(v, rejected=[RejectedEdit("x", "y", "not found")]))
    assert d.mode == "draft" and any("rejected" in r for r in d.reasons)
    d2 = decide(v=v, c=correction(v, anchored=False))
    assert d2.mode == "draft" and any("not tied" in r for r in d2.reasons)


def test_every_concern_is_listed():
    v = verification(confidence=0.6, grounded=False)
    d = decide(v=v, classifier=FakeClassifier("complex"))
    assert d.mode == "draft" and len([r for r in d.reasons if "complex" in r or "below" in r or "quote" in r]) == 3


# -- flags -------------------------------------------------------------------------------

def test_failed_validation_is_flagged_with_the_issues_and_no_text():
    v = verification()
    c = correction(v)
    bad = Validation(c, "failed", True, False, True, 0.9, issues=("dropped the example",))
    d = decide(v=v, c=c, val=bad)
    assert (d.mode, d.new_content, d.confidence) == ("flag", None, 0.0)
    assert "dropped the example" in d.reasons and d.correction is c  # the suggestion stays attached


def test_lint_failures_are_flagged():
    v = verification()
    c = correction(v)
    bad = Validation(c, "failed", lint_problems=("changes the number of code fences",))
    assert "changes the number of code fences" in decide(v=v, c=c, val=bad).reasons


def test_unchecked_validation_is_flagged_never_auto_fixed():
    v = verification()
    c = correction(v)
    d = decide(v=v, c=c, val=Validation(c, "unchecked", note="not validated: budget reached"))
    assert d.mode == "flag" and "budget reached" in " ".join(d.reasons)


@pytest.mark.parametrize("status", ["failed", "skipped"])
def test_unusable_corrections_are_flagged(status):
    v = verification()
    c = Correction(v, status, DOC, DOC, note="the model proposed no edits")
    (d,) = decide_modes([v], [c], [], FakeClassifier())
    assert d.mode == "flag" and "no edits" in " ".join(d.reasons)


def test_missing_correction_or_validation_is_flagged():
    v = verification()
    assert decide_modes([v], [], [], FakeClassifier())[0].reasons == ("no correction was generated",)
    c = correction(v)
    assert decide_modes([v], [c], [], FakeClassifier())[0].reasons == ("the correction was not validated",)


# -- bookkeeping -------------------------------------------------------------------------

def test_decisions_follow_the_stale_verifications_in_order_and_match_by_section():
    a, b = verification(section_id="d.md#A"), verification(section_id="d.md#B")
    accurate = Verification(make_suspect("d.md#C"), "accurate", "fine", (), 0.9)
    ca, cb = correction(a), correction(b)
    out = decide_modes([accurate, b, a], [ca, cb], [passed(cb), passed(ca)], FakeClassifier())
    assert [d.section_id for d in out] == ["d.md#B", "d.md#A"]
    assert all(d.mode == "auto_fix" for d in out)


def test_complexity_is_judged_once_per_changed_chunk_across_sections():
    clf = FakeClassifier()
    change = make_change()
    vs = [verification(section_id=f"d.md#S{i}", changes=[change]) for i in range(3)]
    cs = [correction(v) for v in vs]
    decide_modes(vs, cs, [passed(c) for c in cs], clf)
    assert clf.calls == ["m.py::total"]


def test_a_section_is_simple_only_if_every_change_behind_it_is_simple():
    other = make_change(old="def other(a):\n    return a\n", new="def other(b):\n    return b\n")
    v = verification(changes=[make_change(), other])
    label, confidence, _ = assess_complexity(v, FakeClassifier("simple", 0.9))
    assert (label, confidence) == ("simple", 0.9)

    class Mixed(FakeClassifier):
        def classify_complexity(self, change):
            return Classification("complex" if change.chunk_id.endswith("other") else "simple", 0.9)

    assert assess_complexity(v, Mixed())[0] == "complex"


# -- TODO markers ------------------------------------------------------------------------

def edited(original, corrected, *pairs):
    """A correction whose edits replace each (find, replace) pair, computed against ``original``."""
    v = verification()
    es = []
    for find, replace in pairs:
        i = original.index(find)
        es.append(Edit(i, i + len(find), find, replace))
    return Correction(v, "corrected", original, corrected, tuple(es))


def marker_lines(text):
    return [i for i, line in enumerate(text.split("\n")) if line.startswith("<!-- TODO(doc-healer)")]


def strip_markers(text):
    return "\n".join(l for l in text.split("\n") if not l.startswith("<!-- TODO(doc-healer)"))


def test_a_marker_sits_above_the_edited_line_and_nothing_else_changes():
    c = edited(DOC, FIXED, (STALE_QUOTE, "The default tax rate is 25%."))
    out = add_todo_markers(c, ["complex change"])
    lines = out.split("\n")
    assert marker_lines(out) == [1] and lines[2] == "The default tax rate is 25%."
    assert "complex change" in lines[1] and "the default tax is now 25%" in lines[1]
    assert strip_markers(out) == FIXED


def test_two_edits_on_one_line_get_one_marker_and_edits_on_two_lines_get_two():
    orig = "Alpha is 1 and beta is 2.\nGamma is 3."
    new = "Alpha is 10 and beta is 20.\nGamma is 30."
    one = edited(orig, "Alpha is 10 and beta is 20.\nGamma is 3.", ("1 and", "10 and"), ("2.", "20."))
    assert len(marker_lines(add_todo_markers(one, ["r"]))) == 1
    two = edited(orig, new, ("1 and", "10 and"), ("3.", "30."))
    out = add_todo_markers(two, ["r"])
    assert len(marker_lines(out)) == 2 and strip_markers(out) == new


def test_an_edit_inside_a_code_block_puts_the_marker_above_the_block():
    orig = "Run this:\n\n```bash\ntotal --tax 0.2\n```\n\nDone."
    new = orig.replace("0.2", "0.25")
    out = add_todo_markers(edited(orig, new, ("0.2", "0.25")), ["r"])
    lines = out.split("\n")
    assert marker_lines(out) == [2] and lines[3] == "```bash"
    assert strip_markers(out) == new
    assert lint_correction(edited(orig, out)) == []  # fences and headings intact


def test_an_edit_in_a_table_puts_the_marker_above_the_table():
    orig = "Rates:\n\n| item | tax |\n|------|-----|\n| tea  | 20% |\n| milk | 5%  |\n"
    new = orig.replace("20%", "25%")
    out = add_todo_markers(edited(orig, new, ("20%", "25%")), ["r"])
    assert marker_lines(out) == [2] and out.split("\n")[3].startswith("| item")
    assert strip_markers(out) == new


def test_marker_text_cannot_break_out_of_the_comment():
    v = replace(verification(), explanation="x --> <script>alert(1)</script> --")
    c = replace(edited(DOC, FIXED, (STALE_QUOTE, "The default tax rate is 25%.")), verification=v)
    (line,) = [l for l in add_todo_markers(c, ["a -->b"]).split("\n") if "TODO" in l]
    assert line.count("-->") == 1 and line.endswith("-->") and "--" not in line[4:-3]


# -- end to end --------------------------------------------------------------------------

def llm(*outcomes):
    return LLMClient(ScriptedProvider("groq", *outcomes), sleep=lambda s: None)


def test_end_to_end_auto_fix_and_draft_on_the_taskr_project(taskr_repo):
    repo, base = taskr_repo
    index = build_index(repo.path, use_embeddings=False).index
    repo.edit("taskr/tasks.py", "def add_task(self, title: str, priority: int = 1)",
              "def add_task(self, name: str, priority: int = 1)")
    repo.edit("taskr/tasks.py", '{"title": title,', '{"title": name,')
    report = analyze_changes(repo.path, base, repo.commit("rename"), index).report

    def run(classifier):
        verified = verify_suspects(report.suspects, llm(
            verify_reply(False, 0.8, "the README only names the method"),
            verify_reply(True, 0.9, "the parameter is now called name", ["Use `add_task(title, priority=1)`"]),
        ))
        corrections = repair_all(verified, llm(
            edits(("`add_task(title, priority=1)`", "`add_task(name, priority=1)`")),
        ))
        validations = validate_all(corrections, llm(verdict(confidence=0.9)))
        return decide_modes(verified, corrections, validations, classifier)

    (fix,) = run(FakeClassifier("simple", 0.95))
    assert (fix.mode, fix.section_id) == ("auto_fix", "docs/usage.md#Usage > Adding tasks")
    assert "add_task(name, priority=1)" in fix.new_content and "TODO" not in fix.new_content
    assert "The default priority is 1." in fix.new_content  # untouched text survives

    (draft,) = run(FakeClassifier("complex", 0.9))
    assert draft.mode == "draft" and "TODO(doc-healer)" in draft.new_content
    assert draft.new_content.replace("\n", " ").count("add_task(name, priority=1)") == 1
