"""Parse Python source into semantic code chunks."""

from __future__ import annotations

import ast
import logging
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


def iter_python_files(root: Path):
    for path in sorted(root.rglob("*.py")):
        rel_parts = path.relative_to(root).parts
        if any(part in IGNORED_DIRS for part in rel_parts[:-1]) or is_test_file(path):
            continue
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
        tree = ast.parse(source, filename=rel)
    except (SyntaxError, UnicodeDecodeError) as exc:
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
    return _build(node, source, rel, node.name, "config" if is_config else "class", signature, {})


def _build(node, source, rel, qualname, kind, signature, extra) -> CodeChunk:
    start = min([node.lineno] + [d.lineno for d in node.decorator_list])
    return CodeChunk(
        id=f"{rel}::{qualname}",
        kind=kind,
        file_path=rel,
        name=qualname,
        signature=signature,
        docstring=ast.get_docstring(node),
        source=ast.get_source_segment(source, node) or "",
        start_line=start,
        end_line=node.end_lineno or node.lineno,
        extra=extra,
    )
