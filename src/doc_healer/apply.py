"""Write decided corrections into the documentation files."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .modes import Decision


@dataclass(frozen=True)
class AppliedFix:
    section_id: str
    file_path: str


@dataclass(frozen=True)
class SkippedFix:
    section_id: str
    reason: str


@dataclass(frozen=True)
class ApplyResult:
    applied: list[AppliedFix]
    skipped: list[SkippedFix]
    changed_files: list[str]


def apply_decisions(
    root: Path,
    decisions: Iterable[Decision],
    modes: tuple[str, ...] = ("auto_fix",),
    dry_run: bool = False,
) -> ApplyResult:
    """Replace each section's text with the decision's ``new_content`` in the files under ``root``.

    ``root`` must be a checkout of the commit the analysis ran on. A section is replaced only if its
    text is still exactly what the analysis saw, so a file that moved on is skipped rather than
    corrupted. With ``dry_run`` nothing is written. Decisions in other modes (or with no text) are ignored.
    """
    root = root.resolve()
    by_file: dict[str, list[Decision]] = {}
    for d in decisions:
        if d.mode in modes and d.new_content is not None:
            by_file.setdefault(d.verification.suspect.section.file_path, []).append(d)

    applied: list[AppliedFix] = []
    skipped: list[SkippedFix] = []
    changed: list[str] = []
    for rel, items in sorted(by_file.items()):
        path = (root / rel).resolve()
        if root not in path.parents or not path.is_file():
            skipped += [SkippedFix(d.section_id, f"{rel} is not a file inside the repository") for d in items]
            continue
        with path.open(encoding="utf-8", newline="") as f:  # newline="" keeps CRLF intact
            raw = f.read()
        crlf = "\r\n" in raw
        lines = (raw.replace("\r\n", "\n") if crlf else raw).split("\n")

        file_changed = False
        # Bottom to top, so replacing one section never shifts the line numbers of the ones above it.
        for d in sorted(items, key=lambda d: -d.verification.suspect.section.start_line):
            section = d.verification.suspect.section
            region = "\n".join(lines[section.start_line:section.end_line])  # the body under the heading
            if region.count(section.content) != 1:
                skipped.append(SkippedFix(d.section_id, "the section's text no longer matches the analyzed version"))
                continue
            lines[section.start_line:section.end_line] = region.replace(section.content, d.new_content, 1).split("\n")
            applied.append(AppliedFix(d.section_id, rel))
            file_changed = True

        if file_changed:
            changed.append(rel)
            if not dry_run:
                text = "\n".join(lines)
                with path.open("w", encoding="utf-8", newline="") as f:
                    f.write(text.replace("\n", "\r\n") if crlf else text)
    applied.sort(key=lambda a: (a.file_path, a.section_id))
    return ApplyResult(applied, skipped, changed)
