"""Command line entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .linker import DEFAULT_SIMILARITY_THRESHOLD
from .pipeline import build_index

STATE_DIR = ".doc-healer"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="doc-healer")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build the code-to-docs link index")
    build.add_argument("root", type=Path, help="project root containing code and markdown docs")
    build.add_argument("--index", type=Path, help=f"index file (default: <root>/{STATE_DIR}/index.json)")
    build.add_argument("--store", type=Path, help=f"vector cache dir (default: <root>/{STATE_DIR}/chroma)")
    build.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD)
    build.add_argument("--no-embeddings", action="store_true", help="name-based links only; no model")

    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"{args.root} is not a directory")
    index_path = args.index or root / STATE_DIR / "index.json"
    store_path = args.store or root / STATE_DIR / "chroma"

    result = build_index(
        root,
        index_path=index_path,
        store_path=None if args.no_embeddings else store_path,
        use_embeddings=not args.no_embeddings,
        threshold=args.threshold,
    )
    print(
        f"{len(result.chunks)} code chunks, {len(result.sections)} doc sections -> "
        f"{result.heuristic_links} heuristic + {result.embedding_links} embedding links "
        f"({result.embedded_chunks} chunks embedded)"
    )
    print(f"index written to {index_path}")
    return 0
