"""Command line entry point."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

from .analysis import analyze_changes
from .classifier import LLMClassifier
from .driver import DEFAULT_INDEX_PATH, RunConfig, run_check
from .index import LinkIndex
from .linker import DEFAULT_SIMILARITY_THRESHOLD
from .modes import DEFAULT_CONFIDENCE_THRESHOLD
from .llm import LLMClient, build_client, llm_config_from_env
from .pipeline import build_index
from .verification import verify_suspects

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

    check = sub.add_parser("check", help="list doc sections that code changes between two refs may have made stale")
    check.add_argument("root", type=Path, help="git repository root")
    check.add_argument("--base", required=True, help="base ref, e.g. the PR's target branch")
    check.add_argument("--head", default="HEAD", help="head ref (default: HEAD)")
    check.add_argument("--index", type=Path, help=f"index built at base (default: <root>/{STATE_DIR}/index.json)")
    check.add_argument(
        "--verify", action="store_true",
        help="ask an LLM whether each suspect is really stale (needs LLM_API_KEY; see llm_config_from_env)",
    )

    run = sub.add_parser("run", help="run the full pipeline on a git range and report what it found")
    run.add_argument("root", type=Path, help="git repository root")
    run.add_argument("--base", required=True, help="base ref, e.g. the PR's target branch")
    run.add_argument("--head", default="HEAD", help="head ref (default: HEAD)")
    run.add_argument("--index", default=DEFAULT_INDEX_PATH, help="index path inside the repo, read at --base")
    run.add_argument("--classifier", default="llm", choices=["llm", "none"], help="backend for ambiguous changes")
    run.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD, help="confidence needed to auto-fix")
    run.add_argument("--no-embeddings", action="store_true", help="when the index must be built: name links only")
    run.add_argument("--output", type=Path, help="write the full result as JSON to this file")

    args = parser.parse_args(argv)
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"{args.root} is not a directory")
    if args.command == "check":
        return _check(args, root)
    if args.command == "run":
        return _run(args, root)
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


def _make_client() -> LLMClient:
    return build_client(llm_config_from_env(os.environ))


def _check(args, root: Path) -> int:
    index_path = args.index or root / STATE_DIR / "index.json"
    if not index_path.is_file():
        print(f"no index at {index_path}; run 'doc-healer build' first")
        return 2
    client = None
    if args.verify:
        try:
            client = _make_client()
        except ValueError as exc:
            print(f"cannot verify: {exc}")
            return 2
    analysis = analyze_changes(
        root, args.base, args.head, LinkIndex.load(index_path),
        classifier=LLMClassifier(client) if client else None,
    )
    kept = sum(v.meaningful for v in analysis.verdicts)
    report = analysis.report
    print(
        f"{len(analysis.changes)} chunks changed, {kept} meaningful -> "
        f"{len(report.suspects)} suspect doc sections"
    )
    for suspect in report.suspects:
        print(f"  {suspect.priority:.2f}  {suspect.section.id}")
        for t in suspect.triggers:
            print(f"        {t.verdict.reason}: {t.verdict.change.chunk_id} ({t.link.method} {t.link.score:.2f})")
    for suspect in report.already_updated:
        print(f"  (already updated in this range) {suspect.section.id}")
    for link in report.unresolved:
        print(f"  (index links to missing section) {link.doc_section_id}")

    if client is not None:
        verified = verify_suspects(report.suspects, client)
        print(
            f"verification: {len(verified.stale)} stale, {len(verified.accurate)} accurate, "
            f"{len(verified.unverified)} unverified"
        )
        for r in verified.results:
            print(f"  [{r.status}] {r.section_id}")
            if r.explanation:
                print(f"        {r.explanation}")
            for quote in r.stale_statements:
                print(f"        > {quote}")
    return 0


def _run(args, root: Path) -> int:
    client = None
    try:
        client = _make_client()
    except ValueError as exc:
        print(f"{exc}: listing suspect sections only, with no verification or repair")
    config = RunConfig(
        base=args.base, head=args.head, index_path=args.index, classifier_backend=args.classifier,
        confidence_threshold=args.threshold, use_embeddings=not args.no_embeddings,
        store_path=root / STATE_DIR / "chroma",
    )
    result = run_check(root, config, client)
    s = result.summary
    print(f"index: {result.index_source}")
    print(
        f"{s.suspects} suspect sections: {s.accurate} accurate, {s.stale} stale "
        f"({s.auto_fix} auto-fix, {s.draft} draft, {s.flag} flagged), {s.unverified} unverified"
        + (f", {s.already_updated} already updated in this range" if s.already_updated else "")
    )
    for section in result.to_dict()["sections"]:
        label = section["mode"] or section["status"]
        print(f"  [{label}] {section['section']}")
        if section["explanation"]:
            print(f"        {section['explanation']}")
    if args.output:
        args.output.write_text(result.to_json(), encoding="utf-8")
        print(f"result written to {args.output}")
    return 0
