"""Parse Python source into semantic code chunks."""

from __future__ import annotations

import ast
import copy
import hashlib
import logging
import textwrap
from dataclasses import replace
from pathlib import Path

from .models import CodeChunk

log = logging.getLogger(__name__)

IGNORED_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__", "build", "dist",
    "site-packages", ".tox", ".mypy_cache", ".pytest_cache", "tests", "test",
}
HTTP_METHODS = {"get", "post", "put", "delete", "patch", "head", "options"}
CLI_DECORATORS = {"command", "group"}
CONFIG_BASES = {"BaseModel", "BaseSettings"}

FuncNode = ast.FunctionDef | ast.AsyncFunctionDef


def is_test_file(path: Path) -> bool:
    name = path.name
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


def is_code_path(rel_path: str) -> bool:
    """True for a project-relative path that holds non-test, non-vendored Python code."""
    path = Path(rel_path)
    return (
        path.suffix == ".py"
        and not any(part in IGNORED_DIRS for part in path.parts[:-1])
        and not is_test_file(path)
    )


def iter_python_files(root: Path):
    for path in sorted(root.rglob("*.py")):
        if is_code_path(path.relative_to(root).as_posix()):
            yield path


def chunk_project(root: Path) -> list[CodeChunk]:
    """Chunk every non-test Python file under ``root``."""
    chunks: list[CodeChunk] = []
    for path in iter_python_files(root):
        chunks.extend(chunk_file(path, root))
    return chunks


def chunk_file(path: Path, root: Path) -> list[CodeChunk]:
    rel = path.relative_to(root).as_posix()
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        log.warning("Skipping %s: %s", rel, exc)
        return []
    return chunk_source(source, rel)


def chunk_source(source: str, rel: str) -> list[CodeChunk]:
    """Chunk Python ``source`` as if it lived at project-relative path ``rel``."""
    try:
        tree = ast.parse(source, filename=rel)
    except SyntaxError as exc:
        log.warning("Skipping %s: %s", rel, exc)
        return []

    chunks: list[CodeChunk] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(_function_chunk(node, source, rel, qualname=node.name))
        elif isinstance(node, ast.ClassDef):
            chunks.append(_class_chunk(node, source, rel))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunks.append(
                        _function_chunk(child, source, rel, qualname=f"{node.name}.{child.name}")
                    )
    return chunks


def _decorator_name(dec: ast.expr) -> str:
    """Last attribute/name of a decorator, ignoring call args: ``app.get(...)`` -> ``get``."""
    if isinstance(dec, ast.Call):
        dec = dec.func
    if isinstance(dec, ast.Attribute):
        return dec.attr
    if isinstance(dec, ast.Name):
        return dec.id
    return ""


def _first_str_arg(dec: ast.expr) -> str | None:
    if isinstance(dec, ast.Call) and dec.args:
        arg = dec.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    return None


def _function_chunk(node: FuncNode, source: str, rel: str, qualname: str) -> CodeChunk:
    kind = "function"
    extra: dict[str, str] = {}
    for dec in node.decorator_list:
        name = _decorator_name(dec)
        if name in HTTP_METHODS and _first_str_arg(dec) is not None:
            kind = "endpoint"
            extra = {"http_method": name.upper(), "route": _first_str_arg(dec) or ""}
            break
        if name in CLI_DECORATORS:
            kind = "cli"
            extra = {"command": node.name}
            break
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    signature = f"{prefix} {node.name}({ast.unparse(node.args)}){returns}"
    return _build(node, source, rel, qualname, kind, signature, extra)


def _class_chunk(node: ast.ClassDef, source: str, rel: str) -> CodeChunk:
    bases = ", ".join(ast.unparse(b) for b in node.bases)
    signature = f"class {node.name}({bases})" if bases else f"class {node.name}"
    is_config = any(_decorator_name(d) == "dataclass" for d in node.decorator_list) or any(
        _decorator_name(b) in CONFIG_BASES for b in node.bases
    )
    extra: dict[str, str] = {}
    if is_config:
        fields = [
            stmt.target.id
            for stmt in node.body
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
        ]
        extra["fields"] = ",".join(fields)
    chunk = _build(
        node, source, rel, node.name, "config" if is_config else "class", signature, extra,
        drop_methods=True,
    )
    # The class body minus its methods, so editing a method doesn't mark the class changed.
    lines = source.splitlines()
    method_lines: set[int] = set()
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            first = min([child.lineno] + [d.lineno for d in child.decorator_list])
            method_lines.update(range(first, (child.end_lineno or child.lineno) + 1))
    own = [lines[n - 1] for n in range(chunk.start_line, chunk.end_line + 1) if n not in method_lines]
    return replace(chunk, own_source="\n".join(own))


def _build(node, source, rel, qualname, kind, signature, extra, drop_methods=False) -> CodeChunk:
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    end = node.end_lineno or node.lineno
    return CodeChunk(
        id=f"{rel}::{qualname}",
        kind=kind,
        file_path=rel,
        name=qualname,
        signature=signature,
        docstring=ast.get_docstring(node),
        source=textwrap.dedent("\n".join(source.splitlines()[start - 1 : end])),
        start_line=start,
        end_line=end,
        extra=extra,
        decorators=tuple(ast.unparse(d) for d in node.decorator_list),
        fingerprint=_fingerprint(node, drop_methods),
    )


def _fingerprint(node: ast.AST, drop_methods: bool) -> str:
    """Hash of the code's structure, blind to comments, whitespace and docstrings.

    Two versions of a chunk with the same fingerprint behave identically. For classes,
    ``drop_methods`` leaves the methods out (they are chunks of their own).
    """
    clone = copy.deepcopy(node)
    body = clone.body
    if drop_methods:
        body = [s for s in body if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    clone.body = body or [ast.Pass()]
    return hashlib.sha256(ast.dump(clone).encode()).hexdigest()[:16]
