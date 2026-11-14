"""Split markdown docs into heading-delimited sections and extract code references."""

from __future__ import annotations

import re
from pathlib import Path

from .chunker import IGNORED_DIRS
from .models import DocSection

HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
INLINE_CODE_RE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")
CALL_RE = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*\(")
IDENT_RE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
ROUTE_RE = re.compile(r"^(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+(/\S*)$")
QUOTED_RE = re.compile(r"""(".*?"|'.*?')""")
FLAG_RE = re.compile(r"(?<!\w)--?[A-Za-z][\w-]*")


def is_doc_path(rel_path: str) -> bool:
    """True for a project-relative markdown path outside vendored/test directories."""
    path = Path(rel_path)
    return path.suffix == ".md" and not any(part in IGNORED_DIRS for part in path.parts[:-1])


def iter_markdown_files(root: Path):
    for path in sorted(root.rglob("*.md")):
        if is_doc_path(path.relative_to(root).as_posix()):
            yield path


def parse_docs(root: Path) -> list[DocSection]:
    """Parse every markdown file under ``root``."""
    sections: list[DocSection] = []
    for path in iter_markdown_files(root):
        sections.extend(parse_doc(path, root))
    return sections


def parse_doc(path: Path, root: Path) -> list[DocSection]:
    return parse_markdown(path.read_text(encoding="utf-8"), path.relative_to(root).as_posix())


def parse_markdown(text: str, rel: str) -> list[DocSection]:
    """Split markdown ``text`` into sections, as if it lived at project-relative path ``rel``."""
    lines = text.splitlines()

    sections: list[DocSection] = []
    stack: list[tuple[int, str]] = []  # (level, title)
    current: tuple[tuple[str, ...], int] | None = None  # (heading_path, heading line)
    body: list[str] = []
    fence: str | None = None

    def flush(end_line: int) -> None:
        if current is None:
            return
        content = "\n".join(body).strip()
        if content:
            path_, start = current
            sections.append(
                DocSection(
                    id=f"{rel}#{' > '.join(path_)}",
                    file_path=rel,
                    heading_path=path_,
                    content=content,
                    code_refs=extract_code_refs(content),
                    start_line=start,
                    end_line=end_line,
                )
            )

    for lineno, line in enumerate(lines, start=1):
        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence is None:
                fence = marker[0]
            elif marker[0] == fence:
                fence = None
            body.append(line)
            continue
        heading = None if fence else HEADING_RE.match(line)
        if heading:
            flush(lineno - 1)
            level, title = len(heading.group(1)), heading.group(2)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            current = (tuple(t for _, t in stack), lineno)
            body = []
        else:
            body.append(line)
    flush(len(lines))
    return sections


def extract_code_refs(content: str) -> tuple[str, ...]:
    """Names mentioned in inline code spans and fenced code blocks, in first-seen order.

    Candidates are not checked against the codebase here; the linker keeps only the
    ones that exist. Call-like spans (``add_task(title)``) yield just the callee.
    """
    refs: dict[str, None] = {}

    def add(name: str) -> None:
        refs.setdefault(name)

    for span in _code_spans(content):
        span = span.strip()
        if route := ROUTE_RE.match(span):
            add(f"{route.group(1)} {route.group(2)}")
        elif call := CALL_RE.match(span):
            add(call.group(1))
        else:
            cleaned = FLAG_RE.sub(" ", QUOTED_RE.sub(" ", span))
            for ident in IDENT_RE.findall(cleaned):
                add(ident)
    return tuple(refs)


def _code_spans(content: str):
    """Yield inline code spans, and each call expression found inside fenced blocks."""
    fence: str | None = None
    prose: list[str] = []
    for line in content.splitlines():
        m = FENCE_RE.match(line)
        if m:
            fence = None if fence == m.group(1)[0] else m.group(1)[0]
            continue
        if fence:
            for call in re.finditer(r"([A-Za-z_][\w.]*)\(", line):
                yield f"{call.group(1)}("
        else:
            prose.append(line)
    for m in INLINE_CODE_RE.finditer("\n".join(prose)):
        yield m.group(2)
